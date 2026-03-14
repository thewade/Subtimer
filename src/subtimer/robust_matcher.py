
"""Robust audio matcher using full-sequence feature alignment.

This implementation treats the problem as piecewise monotonic audio alignment,
rather than a single global offset vote.  It extracts low-rate robust features,
runs DTW over the full episode, then converts the warping path into long matched
regions with local linear time maps.

It is designed for:
- same episode with commercials inserted in the TV recording
- small speed differences / drift
- encoder / loudness differences
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple

import librosa
import numpy as np
from scipy.spatial.distance import cdist

from .matcher import MatchConfig
from .alignment_map import AlignmentRegion, RegionType

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

        # Feature / time resolution.  Lower rate keeps DTW tractable over full episodes.
        self.feature_hop_seconds = 1.0
        self.feature_window_seconds = 2.0

        # Region extraction tuning.
        self.max_local_cost = 0.42
        self.max_anchor_gap_seconds = 6.0
        self.min_region_duration_seconds = 20.0
        self.min_region_points = 12
        self.merge_gap_seconds = 8.0
        self.max_offset_jump_seconds = 3.0

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

        # Normalize amplitude; avoid division by zero.
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 1e-9:
            audio = audio / peak

        hop_length = max(256, int(sample_rate * self.feature_hop_seconds))
        win_length = max(hop_length * 2, int(sample_rate * self.feature_window_seconds))
        n_fft = 1
        while n_fft < win_length:
            n_fft *= 2

        # MFCC captures speech/dialogue well; chroma + contrast add robustness.
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

        # Robust per-dimension normalization.
        mean = np.mean(features, axis=1, keepdims=True)
        std = np.std(features, axis=1, keepdims=True)
        std[std < 1e-6] = 1.0
        features = (features - mean) / std

        return features, hop_length / sample_rate

    def _compute_cost_matrix(self, dvd_feat: np.ndarray, tv_feat: np.ndarray) -> np.ndarray:
        """Cosine distance over frame embeddings."""
        # cdist expects observations as rows.
        cost = cdist(dvd_feat.T, tv_feat.T, metric="cosine").astype(np.float32)

        # Clean up any NaN / inf that can come from pathological constant frames.
        cost[~np.isfinite(cost)] = 1.0

        # Small smoothing to reduce isolated bad frames.
        if cost.shape[0] > 2 and cost.shape[1] > 2:
            kernel = (
                cost[:-2, :-2] + cost[:-2, 1:-1] + cost[:-2, 2:]
                + cost[1:-1, :-2] + cost[1:-1, 1:-1] + cost[1:-1, 2:]
                + cost[2:, :-2] + cost[2:, 1:-1] + cost[2:, 2:]
            ) / 9.0
            smooth = cost.copy()
            smooth[1:-1, 1:-1] = kernel
            cost = smooth

        return cost

    def _compute_warping_path(self, cost: np.ndarray) -> List[_PathPoint]:
        """Run DTW and return forward-ordered path points."""
        # Global constraints are intentionally disabled; commercials create large jumps.
        _, wp = librosa.sequence.dtw(C=cost, subseq=False, backtrack=True)

        if wp is None or len(wp) == 0:
            return []

        # librosa returns end->start; reverse to start->end.
        path = [
            _PathPoint(dvd_idx=int(i), tv_idx=int(j), cost=float(cost[int(i), int(j)]))
            for i, j in reversed(wp.tolist())
        ]
        return path

    def _path_to_regions(self, path: List[_PathPoint], step: float) -> List[AlignmentRegion]:
        """Convert the DTW path into matched regions."""
        if not path:
            return []

        # Keep only points that represent genuine forward progress on both axes.
        anchors: List[_PathPoint] = []
        prev = path[0]
        for point in path[1:]:
            if point.dvd_idx > prev.dvd_idx and point.tv_idx > prev.tv_idx:
                anchors.append(point)
            prev = point

        if len(anchors) < self.min_region_points:
            return []

        # Smooth local cost over a short window.
        costs = np.array([p.cost for p in anchors], dtype=np.float32)
        if len(costs) >= 5:
            kernel = np.ones(5, dtype=np.float32) / 5.0
            smooth_costs = np.convolve(costs, kernel, mode="same")
        else:
            smooth_costs = costs

        regions: List[AlignmentRegion] = []
        current: List[_PathPoint] = []

        max_frame_gap = max(1, int(round(self.max_anchor_gap_seconds / step)))

        for idx, point in enumerate(anchors):
            local_cost = float(smooth_costs[idx])

            if not current:
                if local_cost <= self.max_local_cost:
                    current.append(point)
                continue

            last = current[-1]
            dvd_gap = point.dvd_idx - last.dvd_idx
            tv_gap = point.tv_idx - last.tv_idx

            slope = tv_gap / max(dvd_gap, 1)
            allowed_slope = (
                self.config.min_speed_ratio * 0.9
                <= slope
                <= self.config.max_speed_ratio * 1.1
            )
            contiguous = dvd_gap <= max_frame_gap and tv_gap <= max_frame_gap

            if local_cost <= self.max_local_cost and contiguous and allowed_slope:
                current.append(point)
            else:
                region = self._fit_region(current, step, smooth_costs, anchors)
                if region is not None:
                    regions.append(region)
                current = [point] if local_cost <= self.max_local_cost else []

        region = self._fit_region(current, step, smooth_costs, anchors)
        if region is not None:
            regions.append(region)

        return regions

    def _fit_region(
        self,
        points: List[_PathPoint],
        step: float,
        all_costs: np.ndarray,
        all_points: List[_PathPoint],
    ) -> AlignmentRegion | None:
        """Fit a local linear map tv = a*dvd + b for one region."""
        if len(points) < self.min_region_points:
            return None

        dvd_times = np.array([p.dvd_idx * step for p in points], dtype=np.float64)
        tv_times = np.array([p.tv_idx * step for p in points], dtype=np.float64)

        duration = dvd_times[-1] - dvd_times[0]
        if duration < self.min_region_duration_seconds:
            return None

        # First-pass fit.
        a, b = np.polyfit(dvd_times, tv_times, 1)
        predicted = a * dvd_times + b
        residuals = tv_times - predicted

        # Robust inlier pass.
        mad = np.median(np.abs(residuals - np.median(residuals)))
        tol = max(1.0, 3.5 * 1.4826 * mad)
        inliers = np.abs(residuals) <= tol

        if np.count_nonzero(inliers) < self.min_region_points:
            return None

        dvd_in = dvd_times[inliers]
        tv_in = tv_times[inliers]

        duration_in = dvd_in[-1] - dvd_in[0]
        if duration_in < self.min_region_duration_seconds:
            return None

        a, b = np.polyfit(dvd_in, tv_in, 1)
        if not (self.config.min_speed_ratio * 0.9 <= a <= self.config.max_speed_ratio * 1.1):
            return None

        predicted = a * dvd_in + b
        residuals = tv_in - predicted
        rms_err = float(np.sqrt(np.mean(residuals ** 2))) if residuals.size else 999.0

        # Confidence combines path quality, fit quality, and coverage.
        raw_costs = np.array([p.cost for p in points], dtype=np.float32)
        mean_cost = float(np.mean(raw_costs))
        cost_score = float(np.clip(1.0 - mean_cost / max(self.max_local_cost, 1e-6), 0.0, 1.0))
        fit_score = float(np.clip(1.0 - rms_err / 2.5, 0.0, 1.0))
        coverage_score = float(np.clip(duration_in / 180.0, 0.0, 1.0))
        confidence = 0.5 * cost_score + 0.35 * fit_score + 0.15 * coverage_score

        if confidence < max(0.20, self.config.min_correlation * 0.8):
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
        merged: List[AlignmentRegion] = [regions[0]]

        for region in regions[1:]:
            prev = merged[-1]

            dvd_gap = region.dvd_start - prev.dvd_end
            tv_gap = region.tv_start - prev.tv_end
            speed_close = abs(region.speed_ratio - prev.speed_ratio) <= 0.02
            offset_close = abs(region.offset_seconds - prev.offset_seconds) <= self.max_offset_jump_seconds
            gaps_small = dvd_gap <= self.merge_gap_seconds and tv_gap <= self.merge_gap_seconds

            if speed_close and offset_close and gaps_small:
                merged[-1] = AlignmentRegion(
                    dvd_start=prev.dvd_start,
                    dvd_end=max(prev.dvd_end, region.dvd_end),
                    tv_start=prev.tv_start,
                    tv_end=max(prev.tv_end, region.tv_end),
                    offset_seconds=(prev.offset_seconds + region.offset_seconds) / 2.0,
                    speed_ratio=(prev.speed_ratio + region.speed_ratio) / 2.0,
                    confidence=max(prev.confidence, region.confidence),
                    region_type=RegionType.MATCHED,
                )
            else:
                merged.append(region)

        return merged
