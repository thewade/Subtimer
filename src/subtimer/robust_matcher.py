"""Robust audio fingerprinting matcher.

This matcher uses landmark hashing to discover *anchor correspondences* between DVD
and TV audio, then converts those anchors into piecewise alignment regions.

Unlike a simple global-offset fingerprint matcher, this implementation keeps the
2D structure of the matches (DVD time vs TV time) so it can tolerate:
- inserted commercials / TV-only material
- deleted segments
- slight speed differences / drift
- encoding and noise differences
"""

from __future__ import annotations

import hashlib
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple

import librosa
import numpy as np
from scipy.signal import find_peaks

from .alignment_map import AlignmentRegion, RegionType
from .matcher import MatchConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AudioLandmark:
    """A spectral landmark used for fingerprinting."""

    time_frame: int
    frequency_bin: int
    magnitude: float


@dataclass(frozen=True)
class FingerprintHash:
    """Compact hash representing a landmark pair."""

    hash_value: int
    time_offset: int


@dataclass(frozen=True)
class AnchorMatch:
    """Single DVD<->TV anchor correspondence in spectrogram-frame units."""

    dvd_frame: int
    tv_frame: int
    hash_value: int

    @property
    def time_diff(self) -> int:
        return self.tv_frame - self.dvd_frame


class RobustFingerprintMatcher:
    """Fingerprint matcher that builds piecewise alignment regions.

    The matcher works in four phases:
    1. Extract landmarks.
    2. Generate landmark-pair hashes.
    3. Find matching anchor pairs between DVD and TV.
    4. Cluster anchors into monotonic, locally linear alignment regions.
    """

    def __init__(self, config: MatchConfig = MatchConfig()):
        self.config = config

        # Spectrogram / landmark parameters.
        self.hop_length = 512
        self.n_fft = 2048
        self.frame_skip = 3
        self.landmark_density = 4
        self.hash_fan_value = 8
        self.hash_time_delta = 150

        # Matching / clustering parameters.
        self.max_tv_occurrences_per_hash = 50
        self.max_total_anchor_matches = 250_000
        self.offset_bin_frames = 12  # ~0.28 s at 44.1kHz / hop=512
        self.offset_tolerance_frames = 18
        self.max_anchor_gap_sec = max(20.0, self.config.chunk_duration * 0.75)
        self.min_anchor_count = 12
        self.min_region_duration_sec = 8.0
        self.max_regions_from_bins = max(8, min(24, self.config.max_candidates))
        self.max_residual_sec = 1.25
        self.region_padding_sec = 0.35

    def find_matches(
        self,
        dvd_audio: np.ndarray,
        tv_audio: np.ndarray,
        sample_rate: int,
    ) -> List[AlignmentRegion]:
        """Find matched alignment regions between DVD and TV audio."""
        logger.info(
            "Robust fingerprint matching: DVD %s, TV %s, SR=%s",
            dvd_audio.shape,
            tv_audio.shape,
            sample_rate,
        )

        logger.info("Extracting DVD landmarks...")
        dvd_landmarks = self._extract_landmarks(dvd_audio, sample_rate)
        logger.info("Found %d DVD landmarks", len(dvd_landmarks))

        logger.info("Extracting TV landmarks...")
        tv_landmarks = self._extract_landmarks(tv_audio, sample_rate)
        logger.info("Found %d TV landmarks", len(tv_landmarks))

        if len(dvd_landmarks) < self.min_anchor_count or len(tv_landmarks) < self.min_anchor_count:
            logger.warning("Insufficient landmarks for robust fingerprint matching")
            return []

        logger.info("Computing fingerprint hashes...")
        dvd_hashes = self._generate_hashes(dvd_landmarks)
        tv_hashes = self._generate_hashes(tv_landmarks)
        logger.info(
            "Generated %d DVD hashes and %d TV hashes",
            len(dvd_hashes),
            len(tv_hashes),
        )

        logger.info("Finding anchor matches...")
        anchors = self._find_anchor_matches(dvd_hashes, tv_hashes)
        logger.info("Found %d raw anchor matches", len(anchors))
        if len(anchors) < self.min_anchor_count:
            logger.warning("Not enough anchor matches to build alignment regions")
            return []

        logger.info("Clustering anchor matches into piecewise regions...")
        regions = self._anchors_to_regions(anchors, sample_rate)
        logger.info("Robust fingerprint matcher produced %d regions", len(regions))
        return regions

    def _extract_landmarks(self, audio: np.ndarray, sample_rate: int) -> List[AudioLandmark]:
        """Extract strong spectral landmarks from audio."""
        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(stft)
        if magnitude.size == 0:
            return []

        magnitude_db = librosa.amplitude_to_db(magnitude, ref=np.max)
        landmarks: List[AudioLandmark] = []

        for t_frame in range(0, magnitude_db.shape[1], self.frame_skip):
            spectrum = magnitude_db[:, t_frame]
            peaks, properties = find_peaks(
                spectrum,
                height=-45,
                distance=15,
                prominence=6,
            )
            if len(peaks) == 0:
                continue

            peak_strengths = spectrum[peaks]
            strongest = np.argsort(peak_strengths)[-self.landmark_density :]
            for idx in strongest:
                strength = float(peak_strengths[idx])
                if strength <= -40:
                    continue
                landmarks.append(
                    AudioLandmark(
                        time_frame=t_frame,
                        frequency_bin=int(peaks[idx]),
                        magnitude=strength,
                    )
                )

        return landmarks

    def _generate_hashes(self, landmarks: Sequence[AudioLandmark]) -> List[FingerprintHash]:
        """Generate landmark-pair hashes."""
        hashes: List[FingerprintHash] = []
        landmarks_sorted = sorted(landmarks, key=lambda l: l.time_frame)

        for i, anchor in enumerate(landmarks_sorted):
            limit = min(i + self.hash_fan_value + 1, len(landmarks_sorted))
            for j in range(i + 1, limit):
                target = landmarks_sorted[j]
                time_delta = target.time_frame - anchor.time_frame
                if time_delta > self.hash_time_delta:
                    break

                hash_input = f"{anchor.frequency_bin}|{target.frequency_bin}|{time_delta}"
                hash_value = int(hashlib.sha1(hash_input.encode()).hexdigest()[:10], 16)
                hashes.append(FingerprintHash(hash_value=hash_value, time_offset=anchor.time_frame))

        return hashes

    def _find_anchor_matches(
        self,
        dvd_hashes: Sequence[FingerprintHash],
        tv_hashes: Sequence[FingerprintHash],
    ) -> List[AnchorMatch]:
        """Find raw DVD<->TV anchor correspondences from matching hashes."""
        tv_hash_map: DefaultDict[int, List[int]] = defaultdict(list)
        for tv_hash in tv_hashes:
            tv_hash_map[tv_hash.hash_value].append(tv_hash.time_offset)

        anchors: List[AnchorMatch] = []
        seen_pairs: set[Tuple[int, int]] = set()

        for idx, dvd_hash in enumerate(dvd_hashes):
            if idx % 50_000 == 0 and idx > 0:
                logger.info("Processed %d/%d DVD hashes", idx, len(dvd_hashes))

            tv_positions = tv_hash_map.get(dvd_hash.hash_value)
            if not tv_positions:
                continue
            if len(tv_positions) > self.max_tv_occurrences_per_hash:
                continue

            for tv_frame in tv_positions:
                pair = (dvd_hash.time_offset, tv_frame)
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                anchors.append(
                    AnchorMatch(
                        dvd_frame=dvd_hash.time_offset,
                        tv_frame=tv_frame,
                        hash_value=dvd_hash.hash_value,
                    )
                )
                if len(anchors) >= self.max_total_anchor_matches:
                    logger.warning(
                        "Stopping anchor collection after %d matches",
                        self.max_total_anchor_matches,
                    )
                    return anchors

        return anchors

    def _anchors_to_regions(
        self,
        anchors: Sequence[AnchorMatch],
        sample_rate: int,
    ) -> List[AlignmentRegion]:
        """Cluster anchors into monotonic, piecewise linear alignment regions."""
        if not anchors:
            return []

        # Keep the 2D structure by first grouping anchors into coarse offset bands.
        bins: DefaultDict[int, List[AnchorMatch]] = defaultdict(list)
        for anchor in anchors:
            bucket = round(anchor.time_diff / self.offset_bin_frames)
            bins[bucket].append(anchor)

        candidate_regions: List[AlignmentRegion] = []
        sorted_bins = sorted(bins.items(), key=lambda item: len(item[1]), reverse=True)

        for bucket, bucket_anchors in sorted_bins[: self.max_regions_from_bins]:
            # Pull in neighboring anchors to tolerate slight drift instead of forcing
            # a perfectly constant offset.
            expanded = [
                anchor
                for anchor in anchors
                if abs(anchor.time_diff - bucket * self.offset_bin_frames) <= self.offset_tolerance_frames
            ]
            if len(expanded) < self.min_anchor_count:
                continue

            for segment in self._split_into_monotonic_segments(expanded, sample_rate):
                region = self._fit_region(segment, sample_rate)
                if region is None:
                    continue
                if region.confidence < self.config.min_correlation:
                    continue
                candidate_regions.append(region)

        return self._deduplicate_regions(candidate_regions)

    def _split_into_monotonic_segments(
        self,
        anchors: Sequence[AnchorMatch],
        sample_rate: int,
    ) -> List[List[AnchorMatch]]:
        """Split anchors into locally monotonic segments.

        Commercial breaks and other edits usually appear as jumps in the DVD-vs-TV time
        plane. This step preserves local runs of supporting anchors instead of collapsing
        everything into one global offset.
        """
        if not anchors:
            return []

        max_gap_frames = int(self.max_anchor_gap_sec * sample_rate / self.hop_length)
        sorted_anchors = sorted(anchors, key=lambda a: (a.dvd_frame, a.tv_frame))

        segments: List[List[AnchorMatch]] = []
        current: List[AnchorMatch] = [sorted_anchors[0]]
        last = sorted_anchors[0]

        for anchor in sorted_anchors[1:]:
            dvd_gap = anchor.dvd_frame - last.dvd_frame
            tv_gap = anchor.tv_frame - last.tv_frame
            diff_jump = abs(anchor.time_diff - last.time_diff)

            monotonic = anchor.dvd_frame > last.dvd_frame and anchor.tv_frame > last.tv_frame
            locally_consistent = diff_jump <= self.offset_tolerance_frames * 2

            if monotonic and dvd_gap <= max_gap_frames and tv_gap <= max_gap_frames and locally_consistent:
                current.append(anchor)
            else:
                if len(current) >= self.min_anchor_count:
                    segments.append(current)
                current = [anchor]
            last = anchor

        if len(current) >= self.min_anchor_count:
            segments.append(current)

        return segments

    def _fit_region(
        self,
        anchors: Sequence[AnchorMatch],
        sample_rate: int,
    ) -> Optional[AlignmentRegion]:
        """Fit a local linear mapping to a segment of anchors."""
        if len(anchors) < self.min_anchor_count:
            return None

        dvd_frames = np.array([a.dvd_frame for a in anchors], dtype=np.float64)
        tv_frames = np.array([a.tv_frame for a in anchors], dtype=np.float64)

        if np.unique(dvd_frames).size < 3:
            return None

        # Fit tv_frame ~= a * dvd_frame + b.
        A = np.column_stack([dvd_frames, np.ones_like(dvd_frames)])
        try:
            slope, intercept = np.linalg.lstsq(A, tv_frames, rcond=None)[0]
        except np.linalg.LinAlgError:
            return None

        if slope <= 0:
            return None

        residuals = np.abs(tv_frames - (slope * dvd_frames + intercept))
        max_residual_frames = self.max_residual_sec * sample_rate / self.hop_length
        inlier_mask = residuals <= max_residual_frames

        if int(np.sum(inlier_mask)) < self.min_anchor_count:
            return None

        dvd_inliers = dvd_frames[inlier_mask]
        tv_inliers = tv_frames[inlier_mask]

        # Refit using only inliers.
        A_inliers = np.column_stack([dvd_inliers, np.ones_like(dvd_inliers)])
        try:
            slope, intercept = np.linalg.lstsq(A_inliers, tv_inliers, rcond=None)[0]
        except np.linalg.LinAlgError:
            return None

        if slope <= 0:
            return None
        if not (self.config.min_speed_ratio <= slope <= self.config.max_speed_ratio):
            return None

        residuals = np.abs(tv_inliers - (slope * dvd_inliers + intercept))
        residual_sec = float(np.median(residuals) * self.hop_length / sample_rate)

        dvd_start = float(np.min(dvd_inliers) * self.hop_length / sample_rate) - self.region_padding_sec
        dvd_end = float(np.max(dvd_inliers) * self.hop_length / sample_rate) + self.region_padding_sec
        tv_start = float(np.min(tv_inliers) * self.hop_length / sample_rate) - self.region_padding_sec
        tv_end = float(np.max(tv_inliers) * self.hop_length / sample_rate) + self.region_padding_sec

        dvd_start = max(0.0, dvd_start)
        tv_start = max(0.0, tv_start)

        if dvd_end - dvd_start < self.min_region_duration_sec:
            return None
        if tv_end - tv_start < self.min_region_duration_sec:
            return None

        anchor_count = int(dvd_inliers.size)
        density_score = min(1.0, anchor_count / 60.0)
        residual_score = max(0.0, 1.0 - (residual_sec / self.max_residual_sec))
        span_score = min(1.0, (dvd_end - dvd_start) / max(self.config.chunk_duration, 1.0))
        confidence = float(0.45 * density_score + 0.4 * residual_score + 0.15 * span_score)

        offset_seconds = float(intercept * self.hop_length / sample_rate)
        region_type = RegionType.MATCHED if confidence >= max(self.config.min_correlation, 0.5) else RegionType.LOW_CONFIDENCE

        return AlignmentRegion(
            dvd_start=dvd_start,
            dvd_end=dvd_end,
            tv_start=tv_start,
            tv_end=tv_end,
            offset_seconds=offset_seconds,
            speed_ratio=float(slope),
            confidence=min(1.0, max(0.0, confidence)),
            region_type=region_type,
        )

    def _deduplicate_regions(self, regions: Sequence[AlignmentRegion]) -> List[AlignmentRegion]:
        """Remove heavily overlapping duplicate regions, keeping the strongest ones."""
        if not regions:
            return []

        sorted_regions = sorted(
            regions,
            key=lambda r: (r.confidence, r.dvd_duration, r.tv_duration),
            reverse=True,
        )
        kept: List[AlignmentRegion] = []
        for region in sorted_regions:
            if any(self._regions_overlap(region, other) for other in kept):
                continue
            kept.append(region)

        kept.sort(key=lambda r: r.dvd_start)
        return kept

    @staticmethod
    def _regions_overlap(region1: AlignmentRegion, region2: AlignmentRegion) -> bool:
        """Return True when two regions substantially overlap on either timeline."""
        dvd_overlap = min(region1.dvd_end, region2.dvd_end) - max(region1.dvd_start, region2.dvd_start)
        tv_overlap = min(region1.tv_end, region2.tv_end) - max(region1.tv_start, region2.tv_start)

        dvd_ratio = dvd_overlap / max(min(region1.dvd_duration, region2.dvd_duration), 1e-6)
        tv_ratio = tv_overlap / max(min(region1.tv_duration, region2.tv_duration), 1e-6)
        return dvd_ratio > 0.5 or tv_ratio > 0.5
