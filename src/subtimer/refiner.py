"""Refinement of alignment region boundaries.

Refines local region boundaries and estimates precise offset and speed ratio.
"""

import logging
from typing import List, Tuple

import numpy as np
from scipy import optimize
from scipy.signal import find_peaks, correlate

from .alignment_map import AlignmentRegion, RegionType

logger = logging.getLogger(__name__)


class AlignmentRefiner:
    """Refines coarse alignment regions to improve boundary precision."""
    
    def __init__(self, refinement_window: float = 5.0, min_confidence: float = 0.4):
        """Initialize alignment refiner.
        
        Args:
            refinement_window: Window size in seconds for boundary refinement.
            min_confidence: Minimum confidence threshold for refined regions.
        """
        self.refinement_window = refinement_window
        self.min_confidence = min_confidence
        
    def refine_regions(
        self,
        regions: List[AlignmentRegion],
        dvd_audio: np.ndarray,
        tv_audio: np.ndarray,
        sample_rate: int
    ) -> List[AlignmentRegion]:
        """Refine alignment regions for better precision.
        
        Args:
            regions: Coarse alignment regions from matcher.
            dvd_audio: DVD audio data.
            tv_audio: TV audio data.
            sample_rate: Audio sample rate.
            
        Returns:
            List of refined alignment regions.
        """
        logger.info(f"Refining {len(regions)} alignment regions")
        
        refined_regions = []
        
        for region in regions:
            if region.region_type != RegionType.MATCHED:
                # Don't refine non-matched regions
                refined_regions.append(region)
                continue
                
            try:
                refined_region = self._refine_single_region(
                    region, dvd_audio, tv_audio, sample_rate
                )
                
                # Check if refinement improved confidence
                if refined_region.confidence >= self.min_confidence:
                    refined_regions.append(refined_region)
                else:
                    # Mark as low confidence instead of discarding
                    refined_region.region_type = RegionType.LOW_CONFIDENCE
                    refined_regions.append(refined_region)
                    logger.warning(f"Region marked low confidence: {refined_region.confidence:.3f}")
                    
            except Exception as e:
                logger.error(f"Failed to refine region {region.dvd_start}-{region.dvd_end}: {e}")
                # Keep original region but mark as low confidence
                region.region_type = RegionType.LOW_CONFIDENCE
                refined_regions.append(region)
                
        logger.info(f"Refined to {len(refined_regions)} regions")
        return refined_regions
        
    def _refine_single_region(
        self,
        region: AlignmentRegion,
        dvd_audio: np.ndarray,
        tv_audio: np.ndarray,
        sample_rate: int
    ) -> AlignmentRegion:
        """Refine a single alignment region.
        
        Args:
            region: Original coarse region.
            dvd_audio: DVD audio data.
            tv_audio: TV audio data.
            sample_rate: Audio sample rate.
            
        Returns:
            Refined alignment region.
        """
        # Extract audio segments with expanded window for refinement
        window_samples = int(self.refinement_window * sample_rate)
        
        dvd_start_sample = max(0, int(region.dvd_start * sample_rate) - window_samples)
        dvd_end_sample = min(len(dvd_audio), int(region.dvd_end * sample_rate) + window_samples)
        
        tv_start_sample = max(0, int(region.tv_start * sample_rate) - window_samples)  
        tv_end_sample = min(len(tv_audio), int(region.tv_end * sample_rate) + window_samples)
        
        dvd_segment = dvd_audio[dvd_start_sample:dvd_end_sample]
        tv_segment = tv_audio[tv_start_sample:tv_end_sample]
        
        # Refine boundaries using cross-correlation
        refined_bounds = self._refine_boundaries(
            dvd_segment, tv_segment, sample_rate,
            dvd_start_sample / sample_rate, tv_start_sample / sample_rate
        )
        
        # Estimate refined speed ratio and offset
        speed_ratio, offset, confidence = self._estimate_alignment_params(
            dvd_segment, tv_segment, sample_rate, refined_bounds
        )
        
        return AlignmentRegion(
            dvd_start=refined_bounds[0],
            dvd_end=refined_bounds[1], 
            tv_start=refined_bounds[2],
            tv_end=refined_bounds[3],
            offset_seconds=offset,
            speed_ratio=speed_ratio,
            confidence=confidence,
            region_type=RegionType.MATCHED
        )
        
    def _refine_boundaries(
        self,
        dvd_segment: np.ndarray,
        tv_segment: np.ndarray,
        sample_rate: int,
        dvd_offset: float,
        tv_offset: float
    ) -> Tuple[float, float, float, float]:
        """Refine region boundaries using cross-correlation.
        
        Args:
            dvd_segment: DVD audio segment (with extra context).
            tv_segment: TV audio segment (with extra context).
            sample_rate: Audio sample rate.
            dvd_offset: Time offset of DVD segment start.
            tv_offset: Time offset of TV segment start.
            
        Returns:
            Tuple of (dvd_start, dvd_end, tv_start, tv_end) refined times.
        """
        # Use onset detection to find good boundary candidates
        dvd_onsets = self._detect_onsets(dvd_segment, sample_rate)
        tv_onsets = self._detect_onsets(tv_segment, sample_rate)
        
        if len(dvd_onsets) == 0 or len(tv_onsets) == 0:
            # Fallback: use original boundaries with small refinement
            dvd_duration = len(dvd_segment) / sample_rate
            tv_duration = len(tv_segment) / sample_rate
            
            return (
                dvd_offset + dvd_duration * 0.1,
                dvd_offset + dvd_duration * 0.9,
                tv_offset + tv_duration * 0.1, 
                tv_offset + tv_duration * 0.9
            )
            
        # Find best matching onset pairs for boundaries
        dvd_start_idx = min(len(dvd_onsets) // 4, 2) if len(dvd_onsets) > 4 else 0
        dvd_end_idx = max(len(dvd_onsets) * 3 // 4, len(dvd_onsets) - 1) if len(dvd_onsets) > 4 else -1
        
        tv_start_idx = min(len(tv_onsets) // 4, 2) if len(tv_onsets) > 4 else 0
        tv_end_idx = max(len(tv_onsets) * 3 // 4, len(tv_onsets) - 1) if len(tv_onsets) > 4 else -1
        
        dvd_start = dvd_offset + dvd_onsets[dvd_start_idx] / sample_rate
        dvd_end = dvd_offset + dvd_onsets[dvd_end_idx] / sample_rate
        tv_start = tv_offset + tv_onsets[tv_start_idx] / sample_rate
        tv_end = tv_offset + tv_onsets[tv_end_idx] / sample_rate
        
        return (dvd_start, dvd_end, tv_start, tv_end)
        
    def _detect_onsets(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Detect onset times in audio segment.
        
        Args:
            audio: Audio data.
            sample_rate: Sample rate.
            
        Returns:
            Array of onset sample indices.
        """
        try:
            import librosa
            # Use librosa for robust onset detection
            onsets = librosa.onset.onset_detect(
                y=audio,
                sr=sample_rate,
                units='samples',
                hop_length=512,
                backtrack=True
            )
            return onsets
        except ImportError:
            # Fallback: simple energy-based onset detection
            return self._simple_onset_detection(audio, sample_rate)
            
    def _simple_onset_detection(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Simple energy-based onset detection fallback.
        
        Args:
            audio: Audio data.
            sample_rate: Sample rate.
            
        Returns:
            Array of onset sample indices.
        """
        # Compute energy envelope
        hop_length = 512
        frame_length = 2048
        
        energy = []
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i:i + frame_length]
            energy.append(np.sum(frame ** 2))
            
        energy = np.array(energy)
        
        # Find peaks in energy (potential onsets)
        if len(energy) > 10:
            threshold = np.mean(energy) + 2 * np.std(energy)
            peaks, _ = find_peaks(energy, height=threshold, distance=5)
            
            # Convert back to sample indices
            onset_samples = peaks * hop_length
            return onset_samples
        else:
            return np.array([])
            
    def _estimate_alignment_params(
        self,
        dvd_segment: np.ndarray,
        tv_segment: np.ndarray,
        sample_rate: int,
        bounds: Tuple[float, float, float, float]
    ) -> Tuple[float, float, float]:
        """Estimate refined speed ratio, offset, and confidence.
        
        Args:
            dvd_segment: DVD audio segment.
            tv_segment: TV audio segment.
            sample_rate: Sample rate.
            bounds: Region boundaries (dvd_start, dvd_end, tv_start, tv_end).
            
        Returns:
            Tuple of (speed_ratio, offset_seconds, confidence).
        """
        dvd_start, dvd_end, tv_start, tv_end = bounds
        
        # Calculate duration-based speed ratio
        dvd_duration = dvd_end - dvd_start
        tv_duration = tv_end - tv_start
        
        if dvd_duration <= 0 or tv_duration <= 0:
            return 1.0, 0.0, 0.0
            
        speed_ratio = tv_duration / dvd_duration
        
        # Calculate offset: tv_time = speed_ratio * dvd_time + offset
        offset = tv_start - speed_ratio * dvd_start
        
        # Estimate confidence using cross-correlation
        confidence = self._compute_segment_correlation(
            dvd_segment, tv_segment, speed_ratio
        )
        
        return speed_ratio, offset, confidence
        
    def _compute_segment_correlation(
        self,
        dvd_segment: np.ndarray,
        tv_segment: np.ndarray,
        speed_ratio: float
    ) -> float:
        """Compute correlation between segments accounting for speed difference.
        
        Args:
            dvd_segment: DVD audio segment.
            tv_segment: TV audio segment.  
            speed_ratio: Estimated speed ratio.
            
        Returns:
            Correlation coefficient (0-1).
        """
        try:
            # Align segments to same length first
            min_length = min(len(dvd_segment), len(tv_segment))
            if min_length < 1000:  # Need at least 1000 samples for reliable correlation
                return 0.1
            
            dvd_aligned = dvd_segment[:min_length]
            tv_aligned = tv_segment[:min_length]
            
            # Check for constant signals
            if np.std(dvd_aligned) < 1e-6 or np.std(tv_aligned) < 1e-6:
                return 0.1  # Low but non-zero correlation for constant signals
            
            # For small speed differences (< 10%), skip resampling
            if abs(speed_ratio - 1.0) < 0.1:
                # Compute direct correlation
                correlation = np.corrcoef(dvd_aligned, tv_aligned)[0, 1]
            else:
                # For larger speed differences, apply simple time stretching
                if speed_ratio < 1.0:
                    # TV is slower, compress it
                    step = 1.0 / speed_ratio
                    indices = np.arange(0, len(tv_aligned), step)[:len(dvd_aligned)]
                    tv_resampled = tv_aligned[indices.astype(int)]
                else:
                    # TV is faster, expand it by decimation
                    step = speed_ratio
                    indices = np.arange(0, len(dvd_aligned), step)[:len(tv_aligned)]
                    dvd_resampled = dvd_aligned[indices.astype(int)]
                    tv_resampled = tv_aligned
                    dvd_aligned = dvd_resampled
                
                # Align to shortest after resampling
                min_len = min(len(dvd_aligned), len(tv_resampled))
                if min_len < 100:
                    return 0.1
                    
                correlation = np.corrcoef(
                    dvd_aligned[:min_len], 
                    tv_resampled[:min_len]
                )[0, 1]
            
            # Handle NaN
            if np.isnan(correlation):
                return 0.1
                
            # Return absolute correlation (similarity regardless of phase)
            abs_correlation = abs(correlation)
            
            # Boost correlation if it looks reasonable
            if abs_correlation > 0.1:
                # Apply a curve to boost moderate correlations
                abs_correlation = min(1.0, abs_correlation * 1.2)
            
            return abs_correlation
            
        except Exception as e:
            logger.warning(f"Correlation computation failed: {e}")
            return 0.1  # Return low but non-zero correlation