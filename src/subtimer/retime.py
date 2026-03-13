"""Subtitle retiming using alignment map.

Applies time mapping to subtitle cues and handles special cases.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

from .alignment_map import AlignmentMap, RegionType
from .subtitle_io import SubtitleCue

logger = logging.getLogger(__name__)


class CueAction(Enum):
    """Action taken for a subtitle cue during retiming."""
    RETIMED = "retimed"                # Successfully retimed
    DROPPED_TV_ONLY = "dropped_tv_only"         # Dropped (TV-only region)
    DROPPED_UNMATCHED = "dropped_unmatched"     # Dropped (unmatched region)
    FLAGGED_BOUNDARY = "flagged_boundary"       # Flagged (crosses boundary)
    FLAGGED_LOW_CONFIDENCE = "flagged_low_confidence"  # Flagged (low confidence)
    SPLIT_BOUNDARY = "split_boundary"           # Split across boundary


@dataclass
class RetimeResult:
    """Result of retiming a single subtitle cue."""
    original_cue: SubtitleCue
    action: CueAction
    retimed_cue: Optional[SubtitleCue] = None
    split_cues: Optional[List[SubtitleCue]] = None
    reason: str = ""
    confidence: Optional[float] = None


@dataclass
class RetimingSummary:
    """Summary of retiming operation."""
    total_cues: int
    retimed_count: int
    dropped_count: int
    flagged_count: int
    split_count: int
    results: List[RetimeResult]


class SubtitleRetimer:
    """Retimes subtitles using alignment map."""
    
    def __init__(
        self,
        drop_tv_only: bool = True,
        drop_unmatched: bool = True,
        flag_boundary_crossings: bool = True,
        flag_low_confidence: bool = True,
        min_confidence_threshold: float = 0.4,
        split_boundary_crossings: bool = False
    ):
        """Initialize subtitle retimer.
        
        Args:
            drop_tv_only: Drop cues in TV-only regions.
            drop_unmatched: Drop cues in unmatched regions.
            flag_boundary_crossings: Flag cues crossing region boundaries.
            flag_low_confidence: Flag cues in low-confidence regions.
            min_confidence_threshold: Minimum confidence for retiming.
            split_boundary_crossings: Split cues at region boundaries.
        """
        self.drop_tv_only = drop_tv_only
        self.drop_unmatched = drop_unmatched
        self.flag_boundary_crossings = flag_boundary_crossings
        self.flag_low_confidence = flag_low_confidence
        self.min_confidence_threshold = min_confidence_threshold
        self.split_boundary_crossings = split_boundary_crossings
        
    def retime_subtitles(
        self,
        subtitles: List[SubtitleCue],
        alignment_map: AlignmentMap
    ) -> RetimingSummary:
        """Retime subtitle cues using alignment map.
        
        Args:
            subtitles: Original TV-timed subtitle cues.
            alignment_map: DVD-TV time alignment map.
            
        Returns:
            RetimingSummary with retiming results.
        """
        logger.info(f"Retiming {len(subtitles)} subtitle cues")
        
        results = []
        
        for cue in subtitles:
            result = self._retime_single_cue(cue, alignment_map)
            results.append(result)
            
        # Generate summary
        summary = self._generate_summary(results)
        
        logger.info(
            f"Retiming complete: {summary.retimed_count} retimed, "
            f"{summary.dropped_count} dropped, {summary.flagged_count} flagged, "
            f"{summary.split_count} split"
        )
        
        return summary
        
    def _retime_single_cue(
        self,
        cue: SubtitleCue,
        alignment_map: AlignmentMap
    ) -> RetimeResult:
        """Retime a single subtitle cue.
        
        Args:
            cue: Original subtitle cue (TV timeline).
            alignment_map: Alignment map for conversion.
            
        Returns:
            RetimeResult with action taken and result.
        """
        # Find regions for cue start and end times
        start_region = alignment_map.find_region_for_tv_time(cue.start_time)
        end_region = alignment_map.find_region_for_tv_time(cue.end_time)
        
        # Handle different region scenarios
        if start_region is None and end_region is None:
            # Cue completely outside mapped regions
            if self.drop_unmatched:
                return RetimeResult(
                    original_cue=cue,
                    action=CueAction.DROPPED_UNMATCHED,
                    reason="Cue outside any mapped region"
                )
            else:
                return RetimeResult(
                    original_cue=cue,
                    action=CueAction.FLAGGED_LOW_CONFIDENCE,
                    reason="Cue outside any mapped region"
                )
                
        elif start_region == end_region and start_region is not None:
            # Cue entirely within one region
            return self._retime_within_region(cue, start_region)
            
        else:
            # Cue crosses region boundary
            return self._handle_boundary_crossing(cue, start_region, end_region, alignment_map)
            
    def _retime_within_region(
        self,
        cue: SubtitleCue,
        region
    ) -> RetimeResult:
        """Retime cue that falls entirely within one region.
        
        Args:
            cue: Subtitle cue.
            region: Alignment region containing the cue.
            
        Returns:
            RetimeResult for the cue.
        """
        # Handle different region types
        if region.region_type == RegionType.TV_ONLY:
            if self.drop_tv_only:
                return RetimeResult(
                    original_cue=cue,
                    action=CueAction.DROPPED_TV_ONLY,
                    reason="Cue in TV-only region (commercial/insertion)"
                )
            else:
                return RetimeResult(
                    original_cue=cue,
                    action=CueAction.FLAGGED_LOW_CONFIDENCE,
                    reason="Cue in TV-only region"
                )
                
        elif region.region_type == RegionType.UNMATCHED:
            if self.drop_unmatched:
                return RetimeResult(
                    original_cue=cue,
                    action=CueAction.DROPPED_UNMATCHED,
                    reason="Cue in unmatched region"
                )
            else:
                return RetimeResult(
                    original_cue=cue,
                    action=CueAction.FLAGGED_LOW_CONFIDENCE,
                    reason="Cue in unmatched region"
                )
                
        elif region.region_type == RegionType.LOW_CONFIDENCE:
            if self.flag_low_confidence:
                return RetimeResult(
                    original_cue=cue,
                    action=CueAction.FLAGGED_LOW_CONFIDENCE,
                    reason=f"Low confidence region (confidence: {region.confidence:.3f})",
                    confidence=region.confidence
                )
                
        # Region is matched - perform retiming
        try:
            dvd_start = region.tv_to_dvd(cue.start_time)
            dvd_end = region.tv_to_dvd(cue.end_time)
            
            retimed_cue = SubtitleCue(
                index=cue.index,
                start_time=dvd_start,
                end_time=dvd_end,
                text=cue.text
            )
            
            return RetimeResult(
                original_cue=cue,
                action=CueAction.RETIMED,
                retimed_cue=retimed_cue,
                reason="Successfully retimed",
                confidence=region.confidence
            )
            
        except ValueError as e:
            return RetimeResult(
                original_cue=cue,
                action=CueAction.FLAGGED_LOW_CONFIDENCE,
                reason=f"Retiming failed: {e}",
                confidence=region.confidence if region else None
            )
            
    def _handle_boundary_crossing(
        self,
        cue: SubtitleCue,
        start_region,
        end_region,
        alignment_map: AlignmentMap
    ) -> RetimeResult:
        """Handle cue that crosses region boundaries.
        
        Args:
            cue: Subtitle cue crossing boundaries.
            start_region: Region containing cue start (or None).
            end_region: Region containing cue end (or None).
            alignment_map: Full alignment map.
            
        Returns:
            RetimeResult for boundary crossing handling.
        """
        if self.split_boundary_crossings:
            return self._split_cue_at_boundaries(cue, alignment_map)
        elif self.flag_boundary_crossings:
            return RetimeResult(
                original_cue=cue,
                action=CueAction.FLAGGED_BOUNDARY,
                reason="Cue crosses region boundaries",
                confidence=min(
                    start_region.confidence if start_region else 0.0,
                    end_region.confidence if end_region else 0.0
                )
            )
        else:
            # Try to retime using majority region
            if start_region and end_region:
                # Use region with higher confidence
                primary_region = (start_region if start_region.confidence >= end_region.confidence
                                else end_region)
            elif start_region:
                primary_region = start_region
            elif end_region:
                primary_region = end_region
            else:
                return RetimeResult(
                    original_cue=cue,
                    action=CueAction.DROPPED_UNMATCHED,
                    reason="No valid region for boundary crossing"
                )
                
            return self._retime_within_region(cue, primary_region)
            
    def _split_cue_at_boundaries(
        self,
        cue: SubtitleCue,
        alignment_map: AlignmentMap
    ) -> RetimeResult:
        """Split cue at region boundaries.
        
        Args:
            cue: Subtitle cue to split.
            alignment_map: Alignment map for finding boundaries.
            
        Returns:
            RetimeResult with split cues.
        """
        # Find all region transitions within cue timespan
        split_points = []
        
        for region in alignment_map.regions:
            if cue.start_time < region.tv_start < cue.end_time:
                split_points.append(region.tv_start)
            if cue.start_time < region.tv_end < cue.end_time:
                split_points.append(region.tv_end)
                
        split_points.sort()
        
        if not split_points:
            # No actual splits needed - shouldn't happen but handle gracefully
            return self._retime_within_region(
                cue, alignment_map.find_region_for_tv_time(cue.start_time)
            )
            
        # Create split cues
        split_cues = []
        current_start = cue.start_time
        
        for split_time in split_points:
            if current_start < split_time:
                split_cue = SubtitleCue(
                    index=cue.index,  # Will be updated later
                    start_time=current_start,
                    end_time=split_time,
                    text=cue.text
                )
                
                # Attempt to retime this segment
                region = alignment_map.find_region_for_tv_time(current_start)
                if region and region.region_type == RegionType.MATCHED:
                    try:
                        dvd_start = region.tv_to_dvd(split_cue.start_time)
                        dvd_end = region.tv_to_dvd(split_cue.end_time)
                        
                        retimed_split = SubtitleCue(
                            index=split_cue.index,
                            start_time=dvd_start,
                            end_time=dvd_end,
                            text=split_cue.text
                        )
                        split_cues.append(retimed_split)
                    except ValueError:
                        # Skip invalid segments
                        pass
                        
                current_start = split_time
                
        # Handle final segment
        if current_start < cue.end_time:
            final_cue = SubtitleCue(
                index=cue.index,
                start_time=current_start,
                end_time=cue.end_time,
                text=cue.text
            )
            
            region = alignment_map.find_region_for_tv_time(current_start)
            if region and region.region_type == RegionType.MATCHED:
                try:
                    dvd_start = region.tv_to_dvd(final_cue.start_time)
                    dvd_end = region.tv_to_dvd(final_cue.end_time)
                    
                    retimed_final = SubtitleCue(
                        index=final_cue.index,
                        start_time=dvd_start,
                        end_time=dvd_end,
                        text=final_cue.text
                    )
                    split_cues.append(retimed_final)
                except ValueError:
                    pass
                    
        if split_cues:
            return RetimeResult(
                original_cue=cue,
                action=CueAction.SPLIT_BOUNDARY,
                split_cues=split_cues,
                reason=f"Split into {len(split_cues)} segments"
            )
        else:
            return RetimeResult(
                original_cue=cue,
                action=CueAction.DROPPED_UNMATCHED,
                reason="All split segments were invalid"
            )
            
    def _generate_summary(self, results: List[RetimeResult]) -> RetimingSummary:
        """Generate summary from retiming results.
        
        Args:
            results: List of retiming results.
            
        Returns:
            RetimingSummary with counts and statistics.
        """
        retimed_count = sum(1 for r in results if r.action == CueAction.RETIMED)
        dropped_count = sum(
            1 for r in results 
            if r.action in [CueAction.DROPPED_TV_ONLY, CueAction.DROPPED_UNMATCHED]
        )
        flagged_count = sum(
            1 for r in results
            if r.action in [CueAction.FLAGGED_BOUNDARY, CueAction.FLAGGED_LOW_CONFIDENCE]
        )
        split_count = sum(1 for r in results if r.action == CueAction.SPLIT_BOUNDARY)
        
        return RetimingSummary(
            total_cues=len(results),
            retimed_count=retimed_count,
            dropped_count=dropped_count,
            flagged_count=flagged_count,
            split_count=split_count,
            results=results
        )
        
    def get_retimed_cues(self, summary: RetimingSummary) -> List[SubtitleCue]:
        """Extract successfully retimed cues from summary.
        
        Args:
            summary: Retiming summary.
            
        Returns:
            List of retimed subtitle cues for output.
        """
        retimed_cues = []
        index = 1
        
        for result in summary.results:
            if result.action == CueAction.RETIMED and result.retimed_cue:
                result.retimed_cue.index = index
                retimed_cues.append(result.retimed_cue)
                index += 1
            elif result.action == CueAction.SPLIT_BOUNDARY and result.split_cues:
                for split_cue in result.split_cues:
                    split_cue.index = index
                    retimed_cues.append(split_cue)
                    index += 1
                    
        return retimed_cues