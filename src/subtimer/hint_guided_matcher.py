"""Hint-guided audio matching for improved alignment accuracy.

Uses timing hints to focus search regions and validate matches.
"""

import logging
from typing import List, Optional
import numpy as np

from .matcher import AudioMatcher, MatchConfig, MatchCandidate
from .alignment_map import AlignmentRegion, RegionType  
from .hint_loader import EpisodeHints

logger = logging.getLogger(__name__)


class HintGuidedMatcher(AudioMatcher):
    """Audio matcher that uses timing hints to improve accuracy."""
    
    def __init__(self, config: MatchConfig = MatchConfig(), hints: Optional[EpisodeHints] = None):
        """Initialize hint-guided matcher.
        
        Args:
            config: Matching configuration.
            hints: Episode timing hints.
        """
        super().__init__(config) 
        self.hints = hints
        
    def find_matches(
        self,
        dvd_audio: np.ndarray,
        tv_audio: np.ndarray,
        sample_rate: int
    ) -> List[AlignmentRegion]:
        """Find matching regions using hint guidance.
        
        Args:
            dvd_audio: DVD audio data.
            tv_audio: TV audio data.
            sample_rate: Audio sample rate.
            
        Returns:
            List of matched alignment regions.
        """
        logger.info(f"Finding hint-guided matches: DVD {dvd_audio.shape}, TV {tv_audio.shape}")
        
        if not self.hints:
            # Fall back to normal matching
            logger.info("No hints available, using standard matching")
            return super().find_matches(dvd_audio, tv_audio, sample_rate)
            
        # Get expected regions from hints
        expected_regions = self.hints.get_expected_regions()
        logger.info(f"Using {len(expected_regions)} hint regions to guide matching")
        
        # Extract features for matching
        dvd_features = self._extract_features(dvd_audio, sample_rate)
        tv_features = self._extract_features(tv_audio, sample_rate)
        
        logger.info(f"Feature shapes: DVD {dvd_features.shape}, TV {tv_features.shape}")
        
        all_candidates = []
        
        # Search in each expected region
        for label, dvd_start, dvd_end, tv_start, tv_end in expected_regions:
            logger.info(f"Searching {label}: DVD {dvd_start:.1f}-{dvd_end:.1f}s, TV {tv_start:.1f}-{tv_end:.1f}s")
            
            candidates = self._search_hint_region(
                dvd_features, tv_features, sample_rate,
                dvd_start, dvd_end, tv_start, tv_end, label
            )
            
            if candidates:
                logger.info(f"Found {len(candidates)} candidates in {label}")
                all_candidates.extend(candidates)
            else:
                logger.warning(f"No matches found in {label} region")
        
        # If no hint-guided matches, fall back to global search
        if not all_candidates:
            logger.warning("No hint-guided matches found, falling back to global search")
            return super().find_matches(dvd_audio, tv_audio, sample_rate)
            
        # Filter and convert candidates
        valid_candidates = self._filter_candidates(all_candidates)
        regions = self._candidates_to_regions(valid_candidates)
        
        logger.info(f"Found {len(regions)} hint-guided regions")
        return regions
        
    def _search_hint_region(
        self,
        dvd_features: np.ndarray,
        tv_features: np.ndarray,
        sample_rate: int,
        dvd_start: float,
        dvd_end: float,
        tv_start: float,
        tv_end: float,
        label: str
    ) -> List[MatchCandidate]:
        """Search for matches within a specific hint region.
        
        Args:
            dvd_features: DVD feature matrix.
            tv_features: TV feature matrix.
            sample_rate: Audio sample rate.
            dvd_start, dvd_end: DVD region bounds in seconds.
            tv_start, tv_end: TV region bounds in seconds.
            label: Region label for logging.
            
        Returns:
            List of match candidates in this region.
        """
        candidates = []
        hop_length = 512
        
        # Convert time bounds to feature frame indices
        dvd_start_frame = max(0, int(dvd_start * sample_rate / hop_length))
        dvd_end_frame = min(dvd_features.shape[1], int(dvd_end * sample_rate / hop_length))
        tv_start_frame = max(0, int(tv_start * sample_rate / hop_length))  
        tv_end_frame = min(tv_features.shape[1], int(tv_end * sample_rate / hop_length))
        
        # Extract region features
        dvd_region = dvd_features[:, dvd_start_frame:dvd_end_frame]
        tv_region = tv_features[:, tv_start_frame:tv_end_frame]
        
        if dvd_region.shape[1] < 10 or tv_region.shape[1] < 10:
            logger.warning(f"Region {label} too small to match")
            return []
            
        # Use larger chunks for scene-level matching
        region_chunk_frames = min(
            int(self.config.chunk_duration * 2 * sample_rate / hop_length),  # Larger chunks
            dvd_region.shape[1] // 2,
            tv_region.shape[1] // 2
        )
        
        if region_chunk_frames < 10:
            return []
            
        # Search with coarser hops for efficiency
        hop_frames = max(region_chunk_frames // 4, 10)
        
        # Try different positions within the hint region
        for dvd_offset in range(0, dvd_region.shape[1] - region_chunk_frames, hop_frames):
            dvd_chunk = dvd_region[:, dvd_offset:dvd_offset + region_chunk_frames]
            
            # Search across TV region with some tolerance
            tv_search_start = max(0, -tv_start_frame if tv_start_frame < 0 else 0)
            tv_search_end = min(tv_region.shape[1] - region_chunk_frames, tv_region.shape[1])
            
            if tv_search_end <= tv_search_start:
                continue
                
            best_correlation = 0.0
            best_tv_offset = None
            
            # Search TV positions within the hint region
            for tv_offset in range(tv_search_start, tv_search_end, hop_frames):
                tv_chunk = tv_region[:, tv_offset:tv_offset + region_chunk_frames]
                
                if tv_chunk.shape[1] != dvd_chunk.shape[1]:
                    continue
                    
                correlation = self._compute_feature_correlation(dvd_chunk, tv_chunk)
                
                if correlation > best_correlation:
                    best_correlation = correlation
                    best_tv_offset = tv_offset
                    
            # Accept matches above threshold
            if best_correlation >= self.config.min_correlation and best_tv_offset is not None:
                # Convert back to absolute time coordinates
                abs_dvd_start = (dvd_start_frame + dvd_offset) * hop_length / sample_rate
                abs_dvd_end = (dvd_start_frame + dvd_offset + region_chunk_frames) * hop_length / sample_rate
                abs_tv_start = (tv_start_frame + best_tv_offset) * hop_length / sample_rate  
                abs_tv_end = (tv_start_frame + best_tv_offset + region_chunk_frames) * hop_length / sample_rate
                
                # Calculate speed ratio
                dvd_duration = abs_dvd_end - abs_dvd_start
                tv_duration = abs_tv_end - abs_tv_start
                speed_ratio = tv_duration / dvd_duration if dvd_duration > 0 else 1.0
                
                candidate = MatchCandidate(
                    dvd_start=abs_dvd_start,
                    dvd_end=abs_dvd_end,
                    tv_start=abs_tv_start,
                    tv_end=abs_tv_end,
                    correlation=best_correlation,
                    speed_ratio=speed_ratio
                )
                candidates.append(candidate)
                
                logger.debug(f"{label}: Found match at DVD {abs_dvd_start:.1f}s, TV {abs_tv_start:.1f}s (corr={best_correlation:.3f})")
                
        return candidates