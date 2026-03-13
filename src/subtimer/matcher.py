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
    max_speed_ratio: float = 1.05  # maximum speed difference (reduced)
    min_speed_ratio: float = 0.95  # minimum speed difference (reduced)
    max_candidates: int = 50       # limit candidates to prevent runaway processing


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
        logger.info(f"Finding audio matches: DVD {dvd_audio.shape}, TV {tv_audio.shape}, SR {sample_rate}")
        
        # Extract features for matching
        logger.info("Extracting DVD features...")
        dvd_features = self._extract_features(dvd_audio, sample_rate)
        logger.info("Extracting TV features...")
        tv_features = self._extract_features(tv_audio, sample_rate)
        
        logger.info(f"Feature shapes: DVD {dvd_features.shape}, TV {tv_features.shape}")
        
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
        
        # Reduced speed ratio testing for performance
        speed_ratios = np.linspace(
            self.config.min_speed_ratio, 
            self.config.max_speed_ratio, 
            3  # Test only 3 speed values instead of 11
        )
        
        total_chunks = (dvd_features.shape[1] - chunk_frames) // hop_frames + 1
        logger.info(f"Processing {total_chunks} DVD chunks with {len(speed_ratios)} speed ratios")
        
        processed = 0
        
        # Slide DVD chunks across TV features
        for dvd_start_frame in range(0, dvd_features.shape[1] - chunk_frames, hop_frames):
            processed += 1
            if processed % 10 == 0:
                logger.info(f"Processing chunk {processed}/{total_chunks}")
                
            # Early termination if we have enough candidates
            if len(candidates) >= self.config.max_candidates:
                logger.info(f"Reached max candidates limit ({self.config.max_candidates})")
                break
                
            dvd_end_frame = dvd_start_frame + chunk_frames
            dvd_chunk = dvd_features[:, dvd_start_frame:dvd_end_frame]
            
            # Test different speed ratios
            for speed_ratio in speed_ratios:
                scaled_chunk_frames = int(chunk_frames / speed_ratio)
                if scaled_chunk_frames < 10:  # Skip very small chunks
                    continue
                    
                # Find best TV match for this DVD chunk at this speed
                best_match = self._find_best_tv_match(
                    dvd_chunk, tv_features, scaled_chunk_frames, hop_length, sample_rate
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
                    
        logger.info(f"Found {len(candidates)} raw candidates from {processed} chunks")
        return candidates
        
    def _find_best_tv_match(
        self,
        dvd_chunk: np.ndarray,
        tv_features: np.ndarray,
        scaled_chunk_frames: int,
        hop_length: int,
        sample_rate: int
    ) -> Optional[Tuple[int, int, float]]:
        """Find best matching TV segment for DVD chunk.
        
        Args:
            dvd_chunk: DVD feature chunk.
            tv_features: Full TV feature matrix.
            scaled_chunk_frames: Size of TV segment to match.
            hop_length: Hop length for frame conversion.
            sample_rate: Audio sample rate.
            
        Returns:
            Tuple of (tv_start_frame, tv_end_frame, correlation) or None.
        """
        best_correlation = 0.0
        best_match = None
        
        # Use coarser search for performance (every 5th position instead of every 1)
        stride = max(1, scaled_chunk_frames // 20)  # Search every ~5% of chunk size
        
        # Slide TV window to find best correlation
        for tv_start in range(0, tv_features.shape[1] - scaled_chunk_frames, stride):
            tv_end = tv_start + scaled_chunk_frames
            tv_chunk = tv_features[:, tv_start:tv_end]
            
            # Resize chunks to same size for correlation
            if tv_chunk.shape[1] != dvd_chunk.shape[1]:
                # Use proper but efficient interpolation
                from scipy.interpolate import interp1d
                
                # Only interpolate if size difference is significant
                size_ratio = tv_chunk.shape[1] / dvd_chunk.shape[1]
                if abs(size_ratio - 1.0) > 0.1:  # Only interpolate if >10% size difference
                    x_old = np.linspace(0, 1, tv_chunk.shape[1])
                    x_new = np.linspace(0, 1, dvd_chunk.shape[1])
                    
                    tv_resized = np.zeros((tv_chunk.shape[0], dvd_chunk.shape[1]))
                    for i in range(tv_chunk.shape[0]):
                        if np.std(tv_chunk[i]) > 1e-6:  # Skip constant channels
                            f = interp1d(x_old, tv_chunk[i], kind='linear', bounds_error=False, fill_value='extrapolate')
                            tv_resized[i] = f(x_new)
                        else:
                            tv_resized[i] = tv_chunk[i, 0]  # Constant value
                    tv_chunk = tv_resized
                else:
                    # For small differences, just truncate or pad
                    min_size = min(tv_chunk.shape[1], dvd_chunk.shape[1])
                    if tv_chunk.shape[1] > dvd_chunk.shape[1]:
                        tv_chunk = tv_chunk[:, :min_size]
                        dvd_chunk = dvd_chunk[:, :min_size]
                    else:
                        tv_chunk = np.pad(tv_chunk, ((0, 0), (0, dvd_chunk.shape[1] - tv_chunk.shape[1])), mode='edge')
                        
            # Compute correlation between feature vectors
            correlation = self._compute_feature_correlation(dvd_chunk, tv_chunk)
            
            if correlation > best_correlation:
                best_correlation = correlation
                best_match = (tv_start, tv_end, correlation)
                
            # Early exit if we find a very good match
            if correlation > 0.8:
                break
                
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
        if features1.shape != features2.shape:
            logger.warning(f"Feature shape mismatch: {features1.shape} vs {features2.shape}")
            return 0.0
            
        # Flatten features and compute normalized correlation
        f1_flat = features1.flatten()
        f2_flat = features2.flatten()
        
        # Check for constant signals
        if np.std(f1_flat) < 1e-8 or np.std(f2_flat) < 1e-8:
            logger.debug("Constant signal detected, returning 0 correlation")
            return 0.0
        
        # Normalize
        f1_norm = (f1_flat - np.mean(f1_flat)) / np.std(f1_flat)
        f2_norm = (f2_flat - np.mean(f2_flat)) / np.std(f2_flat)
        
        # Correlation coefficient
        corr_matrix = np.corrcoef(f1_norm, f2_norm)
        if corr_matrix.shape == (2, 2):
            correlation = corr_matrix[0, 1]
        else:
            logger.warning(f"Unexpected correlation matrix shape: {corr_matrix.shape}")
            return 0.0
        
        # Handle NaN (shouldn't happen now that we check for constant signals)
        if np.isnan(correlation):
            logger.warning("NaN correlation detected")
            correlation = 0.0
        
        # Return absolute correlation (similarity regardless of sign)
        abs_correlation = abs(correlation)
        logger.debug(f"Raw correlation: {correlation:.4f}, absolute: {abs_correlation:.4f}")
        
        return abs_correlation
        
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