"""Robust audio matcher using full-sequence feature alignment.

This version keeps the current DTW-backed approach, but changes region creation
so edges come from actual contiguous path support instead of a synthetic span.
It also adds a light local edge-refinement pass near each coarse region start
and end.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import librosa
import numpy as np
from scipy.spatial.distance import cdist

from .alignment_map import AlignmentRegion, RegionType
from .matcher import MatchConfig

logger = logging.getLogger(__name__)


@dataclass
class _PathPoint:
    dvd_idx: int
    tv_idx: int
    cost: float


class RobustFingerprintMatcher:
    """Piecewise audio aligner with DTW-backed segment extraction."""

    def __init__(self, config: MatchConfig = MatchConfig()):
        self.config = config

        # Feature / time resolution.
        self.feature_hop_seconds = 0.5
        self.feature_window_seconds = 2.0

        # Region extraction tuning.
        self.max_local_cost = 0.50
        self.bad_cost_tolerance = 0.68
        self.max_short_bad_run = 8
        self.max_anchor_gap_seconds = 20.0
        self.max_balanced_gap_seconds = 45.0
        self.max_gap_mismatch_seconds = 12.0
        self.min_region_duration_seconds = 12.0
        self.min_region_points = 12
        self.edge_deemphasis_seconds = 30.0

        # Region fit / merge tuning.
        self.max_fit_residual_seconds = 2.0
        self.max_offset_jump_seconds = 5.0
        self.max_edge_prediction_error_seconds = 6.0
        self.merge_gap_seconds = 120.0
        self.merge_speed_tolerance = 0.05

        # Boundary refinement.
        self.refine_window_seconds = 8.0
        self.refine_probe_seconds = 5.0
        self.refine_edge_trim_seconds = 0.5

    def find_matches(
        self,
        dvd_audio: np.ndarray,
        tv_audio: np.ndarray,
        sample_rate: int,
    ) -> List[AlignmentRegion]:
        """Find long matched regions between DVD and TV audio."""
        logger.info(
            "Robust piecewise matching: DVD %s, TV %s, sr=%s",
            dvd_audio.shape,
            tv_audio.shape,
            sample_rate,
        )

        dvd_feat, step = self._extract_features(dvd_audio, sample_rate)
        tv_feat, _ = self._extract_features(tv_audio, sample_rate)
        if dvd_feat.shape[1] < 10 or tv_feat.shape[1] < 10:
            logger.warning("Insufficient features for matching")
            return []

        logger.info(
            "Feature matrices: DVD %s, TV %s, step %.3fs",
            dvd_feat.shape,
            tv_feat.shape,
            step,
        )

        cost = self._compute_cost_matrix(dvd_feat, tv_feat)
        logger.info("Cost matrix shape: %s", cost.shape)

        path = self._compute_warping_path(cost)
        if not path:
            logger.warning("No DTW path produced")
            return []

        logger.info("Warping path points: %d", len(path))
        raw_regions = self._path_to_regions(path, cost, step)
        merged_regions = self._merge_regions(raw_regions)
        logger.info("Produced %d matched regions", len(merged_regions))
        return merged_regions

    def _extract_features(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> Tuple[np.ndarray, float]:
        """Extract robust low-rate features for full-sequence alignment."""
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0)

        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 1e-9:
            audio = audio / peak

        hop_length = max(256, int(sample_rate * self.feature_hop_seconds))
        win_length = max(hop_length * 2, int(sample_rate * self.feature_window_seconds))
        n_fft = 1
        while n_fft < win_length:
            n_fft *= 2

        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sample_rate,
            n_mfcc=13,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
        )
        delta = librosa.feature.delta(mfcc)
        rms = librosa.feature.rms(y=audio, frame_length=win_length, hop_length=hop_length)
        contrast = librosa.feature.spectral_contrast(
            y=audio,
            sr=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
        )
        zcr = librosa.feature.zero_crossing_rate(
            y=audio,
            frame_length=win_length,
            hop_length=hop_length,
        )

        features = np.vstack([mfcc, delta, contrast, rms, zcr]).astype(np.float32)
        mean = np.mean(features, axis=1, keepdims=True)
        std = np.std(features, axis=1, keepdims=True)
        std[std < 1e-6] = 1.0
        features = (features - mean) / std
        return features, hop_length / sample_rate

    def _compute_cost_matrix(self, dvd_feat: np.ndarray, tv_feat: np.ndarray) -> np.ndarray:
        """Cosine distance over frame embeddings."""
        cost = cdist(dvd_feat.T, tv_feat.T, metric="cosine").astype(np.float32)
        cost[~np.isfinite(cost)] = 1.0

        if cost.shape[0] > 2 and cost.shape[1] > 2:
            kernel = (
                cost[:-2, :-2]
                + cost[:-2, 1:-1]
                + cost[:-2, 2:]
                + cost[1:-1, :-2]
                + cost[1:-1, 1:-1]
                + cost[1:-1, 2:]
                + cost[2:, :-2]
                + cost[2:, 1:-1]
                + cost[2:, 2:]
            ) / 9.0
            smooth = cost.copy()
            smooth[1:-1, 1:-1] = kernel
            cost = smooth
        return cost

    def _compute_warping_path(self, cost: np.ndarray) -> List[_PathPoint]:
        """Run DTW and return forward-ordered path points."""
        _, wp = librosa.sequence.dtw(C=cost, subseq=False, backtrack=True)
        if wp is None or len(wp) == 0:
            return []
        return [
            _PathPoint(dvd_idx=int(i), tv_idx=int(j), cost=float(cost[int(i), int(j)]))
            for i, j in reversed(wp.tolist())
        ]

    def _path_to_regions(
        self,
        path: List[_PathPoint],
        cost: np.ndarray,
        step: float,
    ) -> List[AlignmentRegion]:
        """Convert the DTW path into matched regions."""
        if not path:
            return []

        anchors = self._collect_forward_anchors(path)
        if len(anchors) < self.min_region_points:
            return []

        costs = np.array([p.cost for p in anchors], dtype=np.float32)
        if len(costs) >= 7:
            smooth_costs = np.convolve(costs, np.ones(7, dtype=np.float32) / 7.0, mode="same")
        else:
            smooth_costs = costs

        regions: List[AlignmentRegion] = []
        current: List[_PathPoint] = []
        bad_run = 0
        total_duration = anchors[-1].dvd_idx * step if anchors else 0.0

        for idx, point in enumerate(anchors):
            local_cost = float(smooth_costs[idx])
            point_time = point.dvd_idx * step
            is_near_edge = (
                point_time < self.edge_deemphasis_seconds
                or point_time > max(0.0, total_duration - self.edge_deemphasis_seconds)
            )
            adjusted_cost = local_cost + (0.10 if is_near_edge else 0.0)

            if not current:
                if adjusted_cost <= self.max_local_cost:
                    current.append(point)
                continue

            if self._point_can_extend_region(current, point, adjusted_cost, step):
                current.append(point)
                bad_run = 0 if adjusted_cost <= self.max_local_cost else bad_run + 1
                continue

            if (
                adjusted_cost <= self.bad_cost_tolerance
                and bad_run < self.max_short_bad_run
                and self._point_has_reasonable_geometry(current[-1], point, step)
            ):
                current.append(point)
                bad_run += 1
                continue

            if (
                len(current) >= 20
                and bad_run < max(1, self.max_short_bad_run // 2)
                and self._point_fits_strong_geometry(current, point, step)
            ):
                current.append(point)
                bad_run += 1
                continue

            region = self._fit_region(current, cost, step)
            if region is not None:
                regions.append(region)

            current = [point] if adjusted_cost <= self.max_local_cost else []
            bad_run = 0

        region = self._fit_region(current, cost, step)
        if region is not None:
            regions.append(region)
        return regions

    def _collect_forward_anchors(self, path: List[_PathPoint]) -> List[_PathPoint]:
        """Keep only monotonic forward-moving path points."""
        if not path:
            return []
        anchors: List[_PathPoint] = [path[0]]
        prev = path[0]
        for point in path[1:]:
            if point.dvd_idx > prev.dvd_idx and point.tv_idx > prev.tv_idx:
                anchors.append(point)
                prev = point
        return anchors

    def _point_has_reasonable_geometry(self, prev: _PathPoint, point: _PathPoint, step: float) -> bool:
        dvd_gap = (point.dvd_idx - prev.dvd_idx) * step
        tv_gap = (point.tv_idx - prev.tv_idx) * step
        if dvd_gap <= 0.0 or tv_gap <= 0.0:
            return False
        slope = tv_gap / max(dvd_gap, 1e-6)
        if not (self.config.min_speed_ratio * 0.85 <= slope <= self.config.max_speed_ratio * 1.15):
            return False
        if dvd_gap > self.max_balanced_gap_seconds or tv_gap > self.max_balanced_gap_seconds:
            return False
        return abs(tv_gap - dvd_gap) <= self.max_gap_mismatch_seconds

    def _point_fits_strong_geometry(
        self,
        current: List[_PathPoint],
        point: _PathPoint,
        step: float,
    ) -> bool:
        """Check if point fits strong geometric consistency even with high cost."""
        if len(current) < 8:
            return False

        recent = current[-8:]
        dvd_times = np.array([p.dvd_idx * step for p in recent], dtype=np.float64)
        tv_times = np.array([p.tv_idx * step for p in recent], dtype=np.float64)
        try:
            slope, intercept = np.polyfit(dvd_times, tv_times, 1)
            point_dvd = point.dvd_idx * step
            predicted_tv = slope * point_dvd + intercept
            actual_tv = point.tv_idx * step
            prediction_error = abs(actual_tv - predicted_tv)

            last = current[-1]
            dvd_step = (point.dvd_idx - last.dvd_idx) * step
            tv_step = (point.tv_idx - last.tv_idx) * step
            if dvd_step <= 0 or tv_step <= 0:
                return False
            step_slope = tv_step / dvd_step

            return (
                prediction_error <= 2.0
                and abs(step_slope - slope) <= 0.03
                and 0.8 <= step_slope <= 1.25
            )
        except Exception:
            return False

    def _point_can_extend_region(
        self,
        current: List[_PathPoint],
        point: _PathPoint,
        local_cost: float,
        step: float,
    ) -> bool:
        """Decide whether a point belongs to the current region."""
        last = current[-1]
        dvd_gap = (point.dvd_idx - last.dvd_idx) * step
        tv_gap = (point.tv_idx - last.tv_idx) * step
        if dvd_gap <= 0.0 or tv_gap <= 0.0:
            return False
        if dvd_gap > self.max_anchor_gap_seconds or tv_gap > self.max_balanced_gap_seconds:
            return False

        slope = tv_gap / max(dvd_gap, 1e-6)
        if not (self.config.min_speed_ratio * 0.8 <= slope <= self.config.max_speed_ratio * 1.2):
            return False
        if abs(tv_gap - dvd_gap) > self.max_gap_mismatch_seconds:
            return False

        if local_cost <= self.max_local_cost:
            return True
        if len(current) < 6:
            return False

        pred_tv, err = self._predict_tv_time(current, point.dvd_idx * step, step)
        if pred_tv is None:
            return False
        actual_tv = point.tv_idx * step
        return abs(actual_tv - pred_tv) <= max(2.0, err + 1.0)

    def _predict_tv_time(
        self,
        points: List[_PathPoint],
        dvd_time: float,
        step: float,
    ) -> Tuple[Optional[float], float]:
        """Predict TV time from recent region points."""
        if len(points) < 4:
            return None, 0.0
        recent = points[-min(12, len(points)):]
        x = np.array([p.dvd_idx * step for p in recent], dtype=np.float64)
        y = np.array([p.tv_idx * step for p in recent], dtype=np.float64)
        try:
            slope, intercept = np.polyfit(x, y, 1)
            pred = slope * dvd_time + intercept
            residuals = np.abs(y - (slope * x + intercept))
            err = float(np.median(residuals)) if residuals.size else 0.0
            return float(pred), err
        except Exception:
            return None, 0.0

    def _fit_region(
        self,
        points: List[_PathPoint],
        cost: np.ndarray,
        step: float,
    ) -> Optional[AlignmentRegion]:
        """Fit a linear region from contiguous DTW support and refine its edges."""
        if len(points) < self.min_region_points:
            return None

        points = sorted(points, key=lambda p: p.dvd_idx)
        x = np.array([p.dvd_idx * step for p in points], dtype=np.float64)
        y = np.array([p.tv_idx * step for p in points], dtype=np.float64)

        try:
            slope, intercept = np.polyfit(x, y, 1)
        except Exception:
            return None

        pred = slope * x + intercept
        residuals = np.abs(y - pred)
        inlier_mask = residuals <= max(self.max_fit_residual_seconds, 2.5 * float(np.median(residuals) + 1e-6))
        inliers = [p for p, ok in zip(points, inlier_mask) if ok]

        if len(inliers) < self.min_region_points:
            return None

        xi = np.array([p.dvd_idx * step for p in inliers], dtype=np.float64)
        yi = np.array([p.tv_idx * step for p in inliers], dtype=np.float64)
        try:
            slope, intercept = np.polyfit(xi, yi, 1)
        except Exception:
            return None

        # Use contiguous evidence span rather than a synthetic duration.
        start_dvd = float(xi[0])
        end_dvd = float(xi[-1])
        if end_dvd - start_dvd < self.min_region_duration_seconds:
            return None

        # Trim a small amount of edge support to reduce DTW wobble.
        trim = self.refine_edge_trim_seconds
        if end_dvd - start_dvd > 2 * trim:
            start_dvd += trim
            end_dvd -= trim

        refined = self._refine_region_edges(
            start_dvd=start_dvd,
            end_dvd=end_dvd,
            slope=slope,
            intercept=intercept,
            cost=cost,
            step=step,
        )
        if refined is None:
            return None
        start_dvd, end_dvd = refined
        if end_dvd - start_dvd < self.min_region_duration_seconds:
            return None

        start_tv = slope * start_dvd + intercept
        end_tv = slope * end_dvd + intercept
        if end_tv <= start_tv:
            return None

        offset = start_tv - slope * start_dvd
        confidence = self._score_region(inliers, slope, intercept, step)

        return AlignmentRegion(
            dvd_start=float(start_dvd),
            dvd_end=float(end_dvd),
            tv_start=float(start_tv),
            tv_end=float(end_tv),
            offset_seconds=float(offset),
            speed_ratio=float(slope),
            confidence=float(confidence),
            region_type=RegionType.MATCHED,
        )

    def _refine_region_edges(
        self,
        start_dvd: float,
        end_dvd: float,
        slope: float,
        intercept: float,
        cost: np.ndarray,
        step: float,
    ) -> Optional[Tuple[float, float]]:
        """Refine coarse region edges with a local diagonal cost search."""
        start_idx = int(round(start_dvd / step))
        end_idx = int(round(end_dvd / step))
        if end_idx <= start_idx:
            return None

        intercept_frames = intercept / step
        window_frames = max(1, int(round(self.refine_window_seconds / step)))
        probe_frames = max(4, int(round(self.refine_probe_seconds / step)))

        refined_start_idx = self._refine_single_edge(
            edge_idx=start_idx,
            slope=slope,
            intercept_frames=intercept_frames,
            cost=cost,
            search_radius=window_frames,
            probe_frames=probe_frames,
            is_start=True,
        )
        refined_end_idx = self._refine_single_edge(
            edge_idx=end_idx,
            slope=slope,
            intercept_frames=intercept_frames,
            cost=cost,
            search_radius=window_frames,
            probe_frames=probe_frames,
            is_start=False,
        )

        if refined_end_idx <= refined_start_idx:
            return None
        return refined_start_idx * step, refined_end_idx * step

    def _refine_single_edge(
        self,
        edge_idx: int,
        slope: float,
        intercept_frames: float,
        cost: np.ndarray,
        search_radius: int,
        probe_frames: int,
        is_start: bool,
    ) -> int:
        """Find the locally best edge index along the fitted diagonal."""
        best_idx = edge_idx
        best_score = float("inf")
        max_dvd = cost.shape[0] - 1
        max_tv = cost.shape[1] - 1

        for delta in range(-search_radius, search_radius + 1):
            cand_dvd = edge_idx + delta
            if cand_dvd < 0 or cand_dvd > max_dvd:
                continue

            if is_start:
                dvd_indices = np.arange(cand_dvd, min(cand_dvd + probe_frames, max_dvd + 1))
            else:
                lo = max(0, cand_dvd - probe_frames + 1)
                dvd_indices = np.arange(lo, cand_dvd + 1)
            if dvd_indices.size < max(3, probe_frames // 2):
                continue

            tv_indices = np.rint(slope * dvd_indices + intercept_frames).astype(int)
            valid = (tv_indices >= 0) & (tv_indices <= max_tv)
            if np.count_nonzero(valid) < max(3, probe_frames // 2):
                continue

            local_costs = cost[dvd_indices[valid], tv_indices[valid]]
            score = float(np.mean(local_costs))
            # Light prior to avoid large unnecessary edge motion.
            score += 0.01 * abs(delta)
            if score < best_score:
                best_score = score
                best_idx = cand_dvd

        return best_idx

    def _score_region(
        self,
        inliers: List[_PathPoint],
        slope: float,
        intercept: float,
        step: float,
    ) -> float:
        """Score a region by fit quality and support density."""
        if not inliers:
            return 0.0
        x = np.array([p.dvd_idx * step for p in inliers], dtype=np.float64)
        y = np.array([p.tv_idx * step for p in inliers], dtype=np.float64)
        pred = slope * x + intercept
        residuals = np.abs(y - pred)
        median_residual = float(np.median(residuals)) if residuals.size else 10.0
        mean_cost = float(np.mean([p.cost for p in inliers]))
        duration = max(0.0, x[-1] - x[0])

        fit_score = max(0.0, 1.0 - (median_residual / 3.0))
        cost_score = max(0.0, 1.0 - (mean_cost / 0.9))
        support_score = min(1.0, duration / 180.0)
        confidence = 0.45 * fit_score + 0.35 * cost_score + 0.20 * support_score
        return float(np.clip(confidence, 0.0, 1.0))

    def _merge_regions(self, regions: List[AlignmentRegion]) -> List[AlignmentRegion]:
        """Merge neighboring matched regions with compatible transforms."""
        if not regions:
            return []
        regions = sorted(regions, key=lambda r: r.dvd_start)
        merged: List[AlignmentRegion] = [regions[0]]

        for region in regions[1:]:
            prev = merged[-1]
            if self._regions_can_merge(prev, region):
                merged[-1] = self._merge_pair(prev, region)
            else:
                merged.append(region)
        return merged

    def _regions_can_merge(self, left: AlignmentRegion, right: AlignmentRegion) -> bool:
        if left.region_type != RegionType.MATCHED or right.region_type != RegionType.MATCHED:
            return False
        if right.dvd_start <= left.dvd_start:
            return False

        if abs(left.speed_ratio - right.speed_ratio) > self.merge_speed_tolerance:
            return False

        # Predict right start from left transform and require reasonable agreement.
        predicted_tv = left.speed_ratio * right.dvd_start + left.offset_seconds
        edge_error = abs(predicted_tv - right.tv_start)
        if edge_error > self.max_edge_prediction_error_seconds:
            return False

        dvd_gap = right.dvd_start - left.dvd_end
        tv_gap = right.tv_start - left.tv_end
        if dvd_gap < 0 or tv_gap < 0:
            return False
        if dvd_gap > self.merge_gap_seconds or tv_gap > self.merge_gap_seconds:
            return False

        # Allow asymmetric gaps if they still imply a compatible continuation.
        if abs((tv_gap / max(dvd_gap, 1e-6)) - left.speed_ratio) > 0.4 and min(dvd_gap, tv_gap) > 10.0:
            return False

        return True

    def _merge_pair(self, left: AlignmentRegion, right: AlignmentRegion) -> AlignmentRegion:
        dvd_start = left.dvd_start
        dvd_end = right.dvd_end
        tv_start = left.tv_start
        tv_end = right.tv_end

        # Refit a single transform from the merged endpoints.
        dvd_duration = dvd_end - dvd_start
        tv_duration = tv_end - tv_start
        speed_ratio = tv_duration / max(dvd_duration, 1e-6)
        offset_seconds = tv_start - speed_ratio * dvd_start
        confidence = min(1.0, max(left.confidence, right.confidence) * 0.98 + 0.02)

        return AlignmentRegion(
            dvd_start=dvd_start,
            dvd_end=dvd_end,
            tv_start=tv_start,
            tv_end=tv_end,
            offset_seconds=offset_seconds,
            speed_ratio=speed_ratio,
            confidence=confidence,
            region_type=RegionType.MATCHED,
        )
