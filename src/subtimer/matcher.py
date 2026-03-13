"""Audio matching to discover coarse alignment regions.

Finds matched regions between DVD and TV audio using fingerprint-based matching.
"""

import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional
from pathlib import Path

import librosa
import numpy as np
from scipy.signal import correlate

from .alignment_map import AlignmentRegion, RegionType

logger = logging.getLogger(__name__)


@dataclass
class MatchConfig:
    """Configuration for audio matching."""
    chunk_duration: float = 30.0  # seconds per chunk
    hop_duration: float = 15.0    # hop between chunks
    min_correlation: float = 0.3   # minimum correlation threshold
    max_speed_ratio: float = 1.1   # maximum speed difference
    min_speed_ratio: float = 0.9   # minimum speed difference


@dataclass
class MatchCandidate:
    """Candidate match between DVD and TV audio segments."""
    dvd_start: float
    dvd_end: float  
    tv_start: float
    tv_end: float
    correlation: float
    speed_ratio: float
    
    
class AudioMatcher:
    """Discovers coarse matching regions between DVD and TV audio."""
    
    def __init__(self, config: MatchConfig = MatchConfig()):
        """Initialize audio matcher.
        
        Args:
            config: Matching configuration.
        """
        self.config = config
        
    def find_matches(
        self,
        dvd_audio: np.ndarray,
        tv_audio: np.ndarray, 
        sample_rate: int
    ) -> List[AlignmentRegion]:
        """Find matching regions between DVD and TV audio.
        
        Args:
            dvd_audio: DVD audio data.
            tv_audio: TV audio data.
            sample_rate: Audio sample rate.
            
        Returns:
            List of matched alignment regions.
        """
        logger.info("Finding audio matches")
        
        # Extract features for matching
        dvd_features = self._extract_features(dvd_audio, sample_rate)
        tv_features = self._extract_features(tv_audio, sample_rate)
        
        # Find candidate matches
        candidates = self._find_match_candidates(
            dvd_features, tv_features, sample_rate
        )
        
        # Filter and score candidates
        valid_candidates = self._filter_candidates(candidates)
        
        # Convert to alignment regions
        regions = self._candidates_to_regions(valid_candidates)
        
        logger.info(f"Found {len(regions)} matched regions")
        return regions
        
    def _extract_features(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Extract audio features for matching.
        
        Args:
            audio: Audio data.
            sample_rate: Sample rate.
            
        Returns:
            Feature matrix (n_features, n_frames).
        """
        # Use Mel-frequency cepstral coefficients (MFCC) as features
        # These are robust to encoding differences and noise
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sample_rate,
            n_mfcc=13,
            hop_length=512,
            n_fft=2048
        )
        
        # Add spectral features for additional robustness
        spectral_centroids = librosa.feature.spectral_centroid(
            y=audio, sr=sample_rate, hop_length=512
        )
        
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=audio, sr=sample_rate, hop_length=512
        )
        
        # Combine features
        features = np.vstack([mfcc, spectral_centroids, spectral_rolloff])
        
        logger.debug(f"Extracted features shape: {features.shape}")
        return features
        
    def _find_match_candidates(
        self,
        dvd_features: np.ndarray,
        tv_features: np.ndarray,
        sample_rate: int
    ) -> List[MatchCandidate]:
        """Find candidate matches using sliding window correlation.
        
        Args:
            dvd_features: DVD feature matrix.
            tv_features: TV feature matrix.
            sample_rate: Audio sample rate.
            
        Returns:
            List of match candidates.
        """
        candidates = []
        
        # Convert durations to frame counts
        hop_length = 512  # Should match feature extraction
        chunk_frames = int(self.config.chunk_duration * sample_rate / hop_length)
        hop_frames = int(self.config.hop_duration * sample_rate / hop_length)
        
        # Slide DVD chunks across TV features
        for dvd_start_frame in range(0, dvd_features.shape[1] - chunk_frames, hop_frames):
            dvd_end_frame = dvd_start_frame + chunk_frames
            dvd_chunk = dvd_features[:, dvd_start_frame:dvd_end_frame]
            
            # Test different speed ratios
            for speed_ratio in np.linspace(
                self.config.min_speed_ratio, 
                self.config.max_speed_ratio, 
                11  # Test 11 speed values
            ):
                scaled_chunk_frames = int(chunk_frames / speed_ratio)
                if scaled_chunk_frames < 10:  # Skip very small chunks
                    continue
                    
                # Find best TV match for this DVD chunk at this speed
                best_match = self._find_best_tv_match(
                    dvd_chunk, tv_features, scaled_chunk_frames
                )
                
                if best_match:
                    # Convert frame indices back to time
                    dvd_start_time = dvd_start_frame * hop_length / sample_rate
                    dvd_end_time = dvd_end_frame * hop_length / sample_rate
                    tv_start_time = best_match[0] * hop_length / sample_rate
                    tv_end_time = best_match[1] * hop_length / sample_rate
                    correlation = best_match[2]
                    
                    candidate = MatchCandidate(
                        dvd_start=dvd_start_time,
                        dvd_end=dvd_end_time,
                        tv_start=tv_start_time,
                        tv_end=tv_end_time,
                        correlation=correlation,
                        speed_ratio=speed_ratio
                    )
                    candidates.append(candidate)
                    
        logger.debug(f"Found {len(candidates)} raw candidates")
        return candidates
        
    def _find_best_tv_match(
        self,
        dvd_chunk: np.ndarray,
        tv_features: np.ndarray,
        scaled_chunk_frames: int
    ) -> Optional[Tuple[int, int, float]]:
        """Find best matching TV segment for DVD chunk.
        
        Args:
            dvd_chunk: DVD feature chunk.
            tv_features: Full TV feature matrix.
            scaled_chunk_frames: Size of TV segment to match.
            
        Returns:
            Tuple of (tv_start_frame, tv_end_frame, correlation) or None.
        """
        best_correlation = 0.0
        best_match = None
        
        # Slide TV window to find best correlation
        for tv_start in range(tv_features.shape[1] - scaled_chunk_frames):
            tv_end = tv_start + scaled_chunk_frames
            tv_chunk = tv_features[:, tv_start:tv_end]
            
            # Resize chunks to same size for correlation
            if tv_chunk.shape[1] != dvd_chunk.shape[1]:
                # Simple linear interpolation for resizing
                from scipy.interpolate import interp1d
                x_old = np.linspace(0, 1, tv_chunk.shape[1])
                x_new = np.linspace(0, 1, dvd_chunk.shape[1])
                
                tv_resized = np.zeros((tv_chunk.shape[0], dvd_chunk.shape[1]))
                for i in range(tv_chunk.shape[0]):
                    f = interp1d(x_old, tv_chunk[i], kind='linear')
                    tv_resized[i] = f(x_new)
                tv_chunk = tv_resized
                
            # Compute correlation between feature vectors
            correlation = self._compute_feature_correlation(dvd_chunk, tv_chunk)
            
            if correlation > best_correlation:
                best_correlation = correlation
                best_match = (tv_start, tv_end, correlation)
                
        if best_correlation >= self.config.min_correlation:
            return best_match
        return None
        
    def _compute_feature_correlation(
        self, 
        features1: np.ndarray, 
        features2: np.ndarray
    ) -> float:
        """Compute correlation between two feature matrices.
        
        Args:
            features1: First feature matrix.
            features2: Second feature matrix.
            
        Returns:
            Correlation coefficient (0-1).
        """
        # Flatten features and compute normalized correlation
        f1_flat = features1.flatten()
        f2_flat = features2.flatten()
        
        # Normalize
        f1_norm = (f1_flat - np.mean(f1_flat)) / (np.std(f1_flat) + 1e-8)
        f2_norm = (f2_flat - np.mean(f2_flat)) / (np.std(f2_flat) + 1e-8)
        
        # Correlation coefficient
        correlation = np.corrcoef(f1_norm, f2_norm)[0, 1]
        
        # Handle NaN (constant signals)
        if np.isnan(correlation):
            correlation = 0.0
            
        return max(0.0, correlation)  # Ensure non-negative
        
    def _filter_candidates(self, candidates: List[MatchCandidate]) -> List[MatchCandidate]:
        """Filter and score match candidates.
        
        Args:
            candidates: List of raw candidates.
            
        Returns:
            Filtered list of valid candidates.
        """
        # Sort by correlation score
        candidates.sort(key=lambda c: c.correlation, reverse=True)
        
        # Remove overlapping candidates (keep highest scoring)
        filtered = []
        for candidate in candidates:
            # Check overlap with existing candidates
            overlaps = False
            for existing in filtered:
                if (self._regions_overlap(
                    (candidate.dvd_start, candidate.dvd_end),
                    (existing.dvd_start, existing.dvd_end)
                ) or self._regions_overlap(
                    (candidate.tv_start, candidate.tv_end),
                    (existing.tv_start, existing.tv_end)
                )):
                    overlaps = True
                    break
                    
            if not overlaps:
                filtered.append(candidate)
                
        logger.debug(f"Filtered to {len(filtered)} non-overlapping candidates")
        return filtered
        
    def _regions_overlap(
        self, 
        region1: Tuple[float, float], 
        region2: Tuple[float, float]
    ) -> bool:
        """Check if two time regions overlap.
        
        Args:
            region1: (start1, end1)
            region2: (start2, end2)
            
        Returns:
            True if regions overlap.
        """
        return not (region1[1] <= region2[0] or region2[1] <= region1[0])
        
    def _candidates_to_regions(
        self, 
        candidates: List[MatchCandidate]
    ) -> List[AlignmentRegion]:
        """Convert match candidates to alignment regions.
        
        Args:
            candidates: List of match candidates.
            
        Returns:
            List of alignment regions.
        """
        regions = []
        
        for candidate in candidates:
            # Calculate offset: tv_time = dvd_time * speed_ratio + offset
            offset = candidate.tv_start - candidate.dvd_start * candidate.speed_ratio
            
            region = AlignmentRegion(
                dvd_start=candidate.dvd_start,
                dvd_end=candidate.dvd_end,
                tv_start=candidate.tv_start,
                tv_end=candidate.tv_end,
                offset_seconds=offset,
                speed_ratio=candidate.speed_ratio,
                confidence=candidate.correlation,
                region_type=RegionType.MATCHED
            )
            regions.append(region)
            
        return regions