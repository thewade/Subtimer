"""Robust audio matcher using full-sequence feature alignment.

This matcher keeps the current DTW-backed approach, but makes region extraction
much less brittle:
- finer feature hop for better local continuity
- more tolerant anchor chaining within acts
- stronger local linear fit / inlier filtering
- smarter region merging based on transform compatibility rather than only
  absolute gap size

It is still a coarse matcher. The goal is to recover long act-sized matched
regions, not exact cut boundaries.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple

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
        # A smaller hop gives a denser path and reduces artificial fragmentation.
        self.feature_hop_seconds = 0.5
        self.feature_window_seconds = 2.0

        # Region extraction tuning.
        self.max_local_cost = 0.50
        self.max_anchor_gap_seconds = 20.0  # More tolerant of gaps
        self.max_balanced_gap_seconds = 40.0  # Allow larger gaps within acts
        self.min_region_duration_seconds = 15.0  # Slightly lower minimum
        self.min_region_points = 12  # More forgiving
        self.max_short_bad_run = 8  # Tolerate longer noisy runs
        self.bad_cost_tolerance = 0.65  # Accept slightly worse costs in bad runs
        self.edge_deemphasis_seconds = 30.0  # De-emphasize episode edges

        # Region merge tuning - more aggressive merging.
        self.merge_gap_seconds = 120.0  # Allow much larger merge gaps
        self.max_offset_jump_seconds = 5.0  # More tolerant of offset differences
        self.max_gap_mismatch_seconds = 8.0  # More tolerant of gap mismatches
        self.max_edge_prediction_error_seconds = 6.0  # More tolerant of prediction errors
        self.merge_speed_tolerance = 0.05  # Allow more speed difference
        self.prefer_long_spans_bonus = 0.15  # Bonus for longer regions in scoring

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
        raw_regions = self._path_to_regions(path, step)
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

    def _path_to_regions(self, path: List[_PathPoint], step: float) -> List[AlignmentRegion]:
        """Convert the DTW path into matched regions."""
        if not path:
            return []

        anchors = self._collect_forward_anchors(path)
        if len(anchors) < self.min_region_points:
            return []

        costs = np.array([p.cost for p in anchors], dtype=np.float32)
        if len(costs) >= 7:
            kernel = np.ones(7, dtype=np.float32) / 7.0
            smooth_costs = np.convolve(costs, kernel, mode="same")
        else:
            smooth_costs = costs

        regions: List[AlignmentRegion] = []
        current: List[_PathPoint] = []
        bad_run = 0
        total_duration = len(anchors) * step if anchors else 0.0
        
        for idx, point in enumerate(anchors):
            local_cost = float(smooth_costs[idx])
            point_time = point.dvd_idx * step
            
            # De-emphasize points near episode edges
            is_near_edge = (point_time < self.edge_deemphasis_seconds or 
                           point_time > total_duration - self.edge_deemphasis_seconds)
            edge_cost_bonus = 0.1 if is_near_edge else 0.0
            adjusted_cost = local_cost + edge_cost_bonus
            
            if not current:
                if adjusted_cost <= self.max_local_cost:
                    current.append(point)
                continue

            can_extend = self._point_can_extend_region(
                current=current,
                point=point,
                local_cost=adjusted_cost,
                step=step,
            )

            if can_extend:
                current.append(point)
                if adjusted_cost <= self.max_local_cost:
                    bad_run = 0
                else:
                    bad_run += 1
                continue

            # Much more tolerant of noisy runs - prefer continuity over purity.
            # Allow bridging through moderate cost spikes if geometry is reasonable.
            if (
                adjusted_cost <= self.bad_cost_tolerance
                and bad_run < self.max_short_bad_run
                and self._point_has_reasonable_geometry(current[-1], point, step)
            ):
                current.append(point)
                bad_run += 1
                continue

            # Even if cost is high, continue if we have strong geometric consistency
            # and this would extend a long region.
            if (
                len(current) >= 20  # Already have substantial region
                and bad_run < self.max_short_bad_run // 2
                and self._point_fits_strong_geometry(current, point, step)
            ):
                current.append(point)
                bad_run += 1
                continue

            # Finalize current region and start fresh.
            region = self._fit_region(current, step)
            if region is not None:
                regions.append(region)

            current = [point] if adjusted_cost <= self.max_local_cost else []
            bad_run = 0

        region = self._fit_region(current, step)
        if region is not None:
            regions.append(region)

        return regions

    def _collect_forward_anchors(self, path: List[_PathPoint]) -> List[_PathPoint]:
        """Keep only monotonic forward-moving path points."""
        anchors: List[_PathPoint] = []
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

        # Within one act, the DVD and TV gap should be broadly similar.
        if abs(tv_gap - dvd_gap) > self.max_gap_mismatch_seconds:
            return False

        return True

    def _point_fits_strong_geometry(
        self,
        current: List[_PathPoint],
        point: _PathPoint,
        step: float,
    ) -> bool:
        """Check if point fits strong geometric consistency even with high cost."""
        if len(current) < 8:
            return False
            
        # Use recent points to establish local trend
        recent = current[-8:]
        dvd_times = [p.dvd_idx * step for p in recent]
        tv_times = [p.tv_idx * step for p in recent]
        
        try:
            # Fit local trend
            slope, intercept = np.polyfit(dvd_times, tv_times, 1)
            
            # Predict where this point should be
            predicted_tv = slope * (point.dvd_idx * step) + intercept
            actual_tv = point.tv_idx * step
            
            # Check if geometric consistency is very strong
            prediction_error = abs(actual_tv - predicted_tv)
            
            # Also check if the step from last point is reasonable
            last = current[-1]
            dvd_step = (point.dvd_idx - last.dvd_idx) * step
            tv_step = (point.tv_idx - last.tv_idx) * step
            
            if dvd_step <= 0 or tv_step <= 0:
                return False
                
            step_slope = tv_step / dvd_step
            
            return (prediction_error <= 2.0 and  # Very tight geometric fit
                   abs(step_slope - slope) <= 0.03 and  # Consistent local slope
                   0.8 <= step_slope <= 1.25)  # Reasonable range
        except:
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

        # More tolerant of gap mismatches - might be within-act variation.
        if abs(tv_gap - dvd_gap) > self.max_gap_mismatch_seconds:
            return False

        if local_cost <= self.max_local_cost:
            return True

        # More aggressive model-based extension for longer regions.
        if len(current) < 6:
            return False

        pred_tv, err = self._predict_tv_time(current, point.dvd_idx * step, step)
        if pred_tv is None:
            return False

        # More tolerant of prediction errors, especially for longer regions.
        error_tolerance = self.max_edge_prediction_error_seconds
        if len(current) >= 15:
            error_tolerance *= 1.5  # More tolerance for established regions
            
        fit_tolerance = 3.5 if len(current) >= 12 else 2.0
        
        return (abs((point.tv_idx * step) - pred_tv) <= error_tolerance and 
                err <= fit_tolerance)

    def _predict_tv_time(
        self,
        current: List[_PathPoint],
        dvd_time: float,
        step: float,
    ) -> Tuple[float | None, float]:
        """Predict TV time from a recent local fit."""
        if len(current) < 8:
            return None, 999.0

        sample = current[-min(len(current), 40) :]
        dvd_times = np.array([p.dvd_idx * step for p in sample], dtype=np.float64)
        tv_times = np.array([p.tv_idx * step for p in sample], dtype=np.float64)
        try:
            a, b = np.polyfit(dvd_times, tv_times, 1)
        except Exception:
            return None, 999.0

        pred = a * dvd_times + b
        rms_err = float(np.sqrt(np.mean((tv_times - pred) ** 2))) if pred.size else 999.0
        return float(a * dvd_time + b), rms_err

    def _fit_region(
        self,
        points: List[_PathPoint],
        step: float,
    ) -> AlignmentRegion | None:
        """Fit a local linear map tv = a*dvd + b for one region."""
        if len(points) < self.min_region_points:
            return None

        dvd_times = np.array([p.dvd_idx * step for p in points], dtype=np.float64)
        tv_times = np.array([p.tv_idx * step for p in points], dtype=np.float64)
        duration = dvd_times[-1] - dvd_times[0]
        if duration < self.min_region_duration_seconds:
            return None

        try:
            a, b = np.polyfit(dvd_times, tv_times, 1)
        except Exception:
            return None

        pred = a * dvd_times + b
        residuals = tv_times - pred

        mad = np.median(np.abs(residuals - np.median(residuals)))
        tol = max(1.5, 3.0 * 1.4826 * mad)
        inliers = np.abs(residuals) <= tol
        if np.count_nonzero(inliers) < self.min_region_points:
            return None

        dvd_in = dvd_times[inliers]
        tv_in = tv_times[inliers]
        duration_in = float(dvd_in[-1] - dvd_in[0])
        if duration_in < self.min_region_duration_seconds:
            return None

        try:
            a, b = np.polyfit(dvd_in, tv_in, 1)
        except Exception:
            return None

        if not (self.config.min_speed_ratio * 0.85 <= a <= self.config.max_speed_ratio * 1.15):
            return None

        pred = a * dvd_in + b
        residuals = tv_in - pred
        rms_err = float(np.sqrt(np.mean(residuals**2))) if residuals.size else 999.0

        raw_costs = np.array([p.cost for p in points], dtype=np.float32)
        mean_cost = float(np.mean(raw_costs))
        cost_score = float(np.clip(1.0 - mean_cost / max(self.max_local_cost * 1.2, 1e-6), 0.0, 1.0))
        fit_score = float(np.clip(1.0 - rms_err / 3.0, 0.0, 1.0))
        coverage_score = float(np.clip(duration_in / 240.0, 0.0, 1.0))
        inlier_score = float(np.clip(np.count_nonzero(inliers) / max(len(points), 1), 0.0, 1.0))
        
        # Add bonus for long spans to prefer continuity over fragmentation
        length_bonus = min(self.prefer_long_spans_bonus, duration_in / 300.0 * self.prefer_long_spans_bonus)
        
        confidence = (0.35 * cost_score + 0.20 * fit_score + 0.25 * coverage_score + 
                     0.10 * inlier_score + 0.10 * length_bonus)
        if confidence < max(0.18, self.config.min_correlation * 0.75):
            return None

        dvd_start = float(dvd_in[0])
        dvd_end = float(dvd_in[-1] + step)
        tv_start = float(tv_in[0])
        tv_end = float(tv_in[-1] + step)
        if dvd_end <= dvd_start or tv_end <= tv_start:
            return None

        return AlignmentRegion(
            dvd_start=dvd_start,
            dvd_end=dvd_end,
            tv_start=tv_start,
            tv_end=tv_end,
            offset_seconds=float(b),
            speed_ratio=float(a),
            confidence=float(min(0.99, confidence)),
            region_type=RegionType.MATCHED,
        )

    def _merge_regions(self, regions: List[AlignmentRegion]) -> List[AlignmentRegion]:
        """Merge adjacent regions with compatible local transforms."""
        if not regions:
            return []

        regions = sorted(regions, key=lambda r: r.dvd_start)
        
        # Multi-pass merging: first pass is conservative, second is more aggressive
        merged = self._merge_regions_pass(regions, conservative=True)
        merged = self._merge_regions_pass(merged, conservative=False)
        
        return merged
    
    def _merge_regions_pass(self, regions: List[AlignmentRegion], conservative: bool) -> List[AlignmentRegion]:
        """Single pass of region merging."""
        if not regions:
            return []
            
        merged: List[AlignmentRegion] = [regions[0]]
        for region in regions[1:]:
            prev = merged[-1]
            if self._regions_are_mergeable(prev, region, conservative):
                # Weight parameters by region duration for better blending
                prev_duration = prev.dvd_end - prev.dvd_start
                curr_duration = region.dvd_end - region.dvd_start
                total_duration = prev_duration + curr_duration
                
                prev_weight = prev_duration / total_duration if total_duration > 0 else 0.5
                curr_weight = curr_duration / total_duration if total_duration > 0 else 0.5
                
                merged_offset = (prev.offset_seconds * prev_weight + 
                               region.offset_seconds * curr_weight)
                merged_speed = (prev.speed_ratio * prev_weight + 
                              region.speed_ratio * curr_weight)
                
                # Boost confidence for merged longer regions
                base_confidence = max(prev.confidence, region.confidence)
                total_duration = prev_duration + curr_duration
                
                # Significant boost for longer merged regions
                if total_duration > 60.0:  # More than 1 minute
                    duration_bonus = min(0.25, total_duration / 240.0 * 0.25)
                    merged_confidence = min(0.99, base_confidence + duration_bonus)
                else:
                    merged_confidence = base_confidence
                
                merged[-1] = AlignmentRegion(
                    dvd_start=prev.dvd_start,
                    dvd_end=max(prev.dvd_end, region.dvd_end),
                    tv_start=prev.tv_start,
                    tv_end=max(prev.tv_end, region.tv_end),
                    offset_seconds=merged_offset,
                    speed_ratio=merged_speed,
                    confidence=merged_confidence,
                    region_type=RegionType.MATCHED,
                )
            else:
                merged.append(region)
        return merged

    def _regions_are_mergeable(self, prev: AlignmentRegion, region: AlignmentRegion, conservative: bool = True) -> bool:
        """Check whether two adjacent regions likely belong to one act."""
        dvd_gap = region.dvd_start - prev.dvd_end
        tv_gap = region.tv_start - prev.tv_end

        # Allow small overlaps
        if dvd_gap < -5.0 or tv_gap < -5.0:
            return False
            
        # More generous gap tolerance, especially in second pass
        gap_limit = self.merge_gap_seconds if conservative else self.merge_gap_seconds * 1.8
        if dvd_gap > gap_limit or tv_gap > gap_limit:
            return False

        # More tolerant speed and offset criteria
        speed_tolerance = self.merge_speed_tolerance if conservative else self.merge_speed_tolerance * 2.0
        offset_tolerance = self.max_offset_jump_seconds if conservative else self.max_offset_jump_seconds * 1.5
        
        speed_close = abs(region.speed_ratio - prev.speed_ratio) <= speed_tolerance
        offset_close = abs(region.offset_seconds - prev.offset_seconds) <= offset_tolerance
        
        # In non-conservative mode, be more flexible if one criterion is very good
        if not conservative:
            very_close_speed = abs(region.speed_ratio - prev.speed_ratio) <= 0.01
            very_close_offset = abs(region.offset_seconds - prev.offset_seconds) <= 1.0
            if very_close_speed or very_close_offset:
                speed_close = abs(region.speed_ratio - prev.speed_ratio) <= 0.08
                offset_close = abs(region.offset_seconds - prev.offset_seconds) <= 8.0
        
        if not (speed_close and offset_close):
            return False

        # More tolerant gap prediction
        expected_tv_gap_prev = prev.speed_ratio * max(dvd_gap, 0.0)
        expected_tv_gap_region = region.speed_ratio * max(dvd_gap, 0.0) 
        gap_mismatch = min(
            abs(tv_gap - expected_tv_gap_prev),
            abs(tv_gap - expected_tv_gap_region),
        )
        
        gap_mismatch_limit = self.max_gap_mismatch_seconds if conservative else self.max_gap_mismatch_seconds * 1.5
        if gap_mismatch > gap_mismatch_limit:
            return False

        # More tolerant edge prediction
        pred_region_tv_start = prev.speed_ratio * region.dvd_start + prev.offset_seconds
        pred_prev_tv_end = region.speed_ratio * prev.dvd_end + region.offset_seconds
        edge_error = min(
            abs(region.tv_start - pred_region_tv_start),
            abs(prev.tv_end - pred_prev_tv_end),
        )
        
        edge_error_limit = self.max_edge_prediction_error_seconds if conservative else self.max_edge_prediction_error_seconds * 1.8
        if edge_error > edge_error_limit:
            return False

        return True
