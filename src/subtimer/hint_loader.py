"""Hint loading and validation for guided alignment.

Loads timing hints from YAML files to guide and validate audio alignment.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import yaml

logger = logging.getLogger(__name__)


@dataclass
class TimeEvent:
    """A timed event marker."""
    label: str
    time_seconds: float


@dataclass
class EpisodeHints:
    """Timing hints for DVD and TV versions of an episode."""
    episode_id: str
    dvd_events: List[TimeEvent]
    tv_events: List[TimeEvent]
    
    def get_expected_regions(self) -> List[Tuple[str, float, float, float, float]]:
        """Get expected alignment regions based on hints.
        
        Returns:
            List of (label, dvd_start, dvd_end, tv_start, tv_end) tuples.
        """
        regions = []
        
        # Match acts between DVD and TV
        dvd_acts = {e.label: e.time_seconds for e in self.dvd_events if 'act' in e.label or 'intro' in e.label or 'credits' in e.label}
        tv_acts = {e.label: e.time_seconds for e in self.tv_events if 'act' in e.label or 'intro' in e.label or 'credits' in e.label}
        
        # Create regions for matching segments
        for label in dvd_acts:
            if label in tv_acts:
                # Find the end time for this segment
                dvd_start = dvd_acts[label]
                tv_start = tv_acts[label]
                
                # Find next segment to determine end time
                dvd_end = None
                tv_end = None
                
                for next_event in self.dvd_events:
                    if next_event.time_seconds > dvd_start:
                        dvd_end = next_event.time_seconds
                        break
                        
                for next_event in self.tv_events:
                    if next_event.time_seconds > tv_start and 'commercial' not in next_event.label:
                        tv_end = next_event.time_seconds
                        break
                
                if dvd_end and tv_end:
                    regions.append((label, dvd_start, dvd_end, tv_start, tv_end))
                    
        return regions


def load_hints_file(hints_path: Path) -> Optional[EpisodeHints]:
    """Load timing hints from YAML file.
    
    Args:
        hints_path: Path to hints.yaml file.
        
    Returns:
        EpisodeHints object or None if loading failed.
    """
    try:
        with open(hints_path, 'r') as f:
            data = yaml.safe_load(f)
            
        dvd_events = []
        for event_data in data.get('dvd_events', []):
            time_str = event_data['time']
            time_seconds = _parse_time_string(time_str)
            dvd_events.append(TimeEvent(event_data['label'], time_seconds))
            
        tv_events = []
        for event_data in data.get('tv_events', []):
            time_str = event_data['time'] 
            time_seconds = _parse_time_string(time_str)
            tv_events.append(TimeEvent(event_data['label'], time_seconds))
            
        return EpisodeHints(
            episode_id=data.get('episode_id', 'unknown'),
            dvd_events=dvd_events,
            tv_events=tv_events
        )
        
    except Exception as e:
        logger.warning(f"Failed to load hints from {hints_path}: {e}")
        return None


def _parse_time_string(time_str: str) -> float:
    """Parse time string like '00:07:07' to seconds.
    
    Args:
        time_str: Time in format HH:MM:SS or MM:SS.
        
    Returns:
        Time in seconds.
    """
    parts = time_str.strip().split(':')
    
    if len(parts) == 3:
        hours, minutes, seconds = map(int, parts)
        return hours * 3600 + minutes * 60 + seconds
    elif len(parts) == 2:
        minutes, seconds = map(int, parts)
        return minutes * 60 + seconds
    else:
        raise ValueError(f"Invalid time format: {time_str}")


def validate_alignment_against_hints(
    regions: List,
    hints: EpisodeHints,
    tolerance_seconds: float = 5.0
) -> List[str]:
    """Validate alignment regions against timing hints.
    
    Args:
        regions: List of alignment regions.
        hints: Episode timing hints.
        tolerance_seconds: Tolerance for timing validation.
        
    Returns:
        List of validation warnings.
    """
    warnings = []
    expected_regions = hints.get_expected_regions()
    
    logger.info(f"Validating {len(regions)} regions against {len(expected_regions)} expected regions")
    
    for expected_label, exp_dvd_start, exp_dvd_end, exp_tv_start, exp_tv_end in expected_regions:
        # Find matching region
        found_match = False
        
        for region in regions:
            dvd_overlap = (
                region.dvd_start < exp_dvd_end and region.dvd_end > exp_dvd_start
            )
            tv_overlap = (
                region.tv_start < exp_tv_end and region.tv_end > exp_tv_start  
            )
            
            if dvd_overlap and tv_overlap:
                found_match = True
                
                # Check alignment accuracy
                dvd_offset_error = abs(region.dvd_start - exp_dvd_start)
                tv_offset_error = abs(region.tv_start - exp_tv_start)
                
                if dvd_offset_error > tolerance_seconds:
                    warnings.append(
                        f"{expected_label}: DVD start {region.dvd_start:.1f}s "
                        f"differs from expected {exp_dvd_start:.1f}s by {dvd_offset_error:.1f}s"
                    )
                    
                if tv_offset_error > tolerance_seconds:
                    warnings.append(
                        f"{expected_label}: TV start {region.tv_start:.1f}s "
                        f"differs from expected {exp_tv_start:.1f}s by {tv_offset_error:.1f}s"
                    )
                break
                
        if not found_match:
            warnings.append(f"Missing alignment for {expected_label} segment")
            
    return warnings