"""Alignment map data structures.

Stores ordered mapping regions and provides time conversion helpers.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class RegionType(Enum):
    """Type of alignment region."""
    MATCHED = "matched"          # DVD and TV content matched
    TV_ONLY = "tv_only"         # TV-only insertion (commercial, etc.)
    UNMATCHED = "unmatched"     # Gap in alignment
    LOW_CONFIDENCE = "low_confidence"  # Matched but uncertain


@dataclass  
class AlignmentRegion:
    """Single region in the alignment map.
    
    Represents linear mapping: tv_time = speed_ratio * dvd_time + offset_seconds
    """
    dvd_start: float
    dvd_end: float
    tv_start: float
    tv_end: float
    offset_seconds: float
    speed_ratio: float
    confidence: float
    region_type: RegionType
    
    def __post_init__(self) -> None:
        """Validate region consistency."""
        if self.dvd_start >= self.dvd_end:
            raise ValueError("DVD start must be before DVD end")
        if self.tv_start >= self.tv_end:
            raise ValueError("TV start must be before TV end") 
        if self.confidence < 0 or self.confidence > 1:
            raise ValueError("Confidence must be between 0 and 1")
            
    @property 
    def dvd_duration(self) -> float:
        """Duration on DVD timeline."""
        return self.dvd_end - self.dvd_start
        
    @property
    def tv_duration(self) -> float:
        """Duration on TV timeline."""
        return self.tv_end - self.tv_start
        
    def dvd_to_tv(self, dvd_time: float) -> float:
        """Convert DVD time to TV time within this region.
        
        Args:
            dvd_time: Time on DVD timeline.
            
        Returns:
            Corresponding time on TV timeline.
            
        Raises:
            ValueError: If dvd_time is outside this region.
        """
        if not (self.dvd_start <= dvd_time <= self.dvd_end):
            raise ValueError(f"DVD time {dvd_time} outside region [{self.dvd_start}, {self.dvd_end}]")
            
        # Linear mapping: tv_time = speed_ratio * dvd_time + offset
        return self.speed_ratio * dvd_time + self.offset_seconds
        
    def tv_to_dvd(self, tv_time: float) -> float:
        """Convert TV time to DVD time within this region.
        
        Args:
            tv_time: Time on TV timeline.
            
        Returns:
            Corresponding time on DVD timeline.
            
        Raises:
            ValueError: If tv_time is outside this region.
        """
        if not (self.tv_start <= tv_time <= self.tv_end):
            raise ValueError(f"TV time {tv_time} outside region [{self.tv_start}, {self.tv_end}]")
            
        # Inverse mapping: dvd_time = (tv_time - offset) / speed_ratio  
        return (tv_time - self.offset_seconds) / self.speed_ratio
        
    def contains_dvd_time(self, dvd_time: float) -> bool:
        """Check if DVD time falls within this region."""
        return self.dvd_start <= dvd_time <= self.dvd_end
        
    def contains_tv_time(self, tv_time: float) -> bool:
        """Check if TV time falls within this region."""
        return self.tv_start <= tv_time <= self.tv_end
        

class AlignmentMap:
    """Ordered collection of alignment regions with time conversion."""
    
    def __init__(self, regions: Optional[List[AlignmentRegion]] = None):
        """Initialize alignment map.
        
        Args:
            regions: List of alignment regions. Will be sorted by DVD time.
        """
        self.regions = regions or []
        self.sort_regions()
        
    def add_region(self, region: AlignmentRegion) -> None:
        """Add region to map and maintain sorting."""
        self.regions.append(region)
        self.sort_regions()
        
    def sort_regions(self) -> None:
        """Sort regions by DVD start time."""
        self.regions.sort(key=lambda r: r.dvd_start)
        
    def find_region_for_dvd_time(self, dvd_time: float) -> Optional[AlignmentRegion]:
        """Find region containing the given DVD time.
        
        Args:
            dvd_time: Time on DVD timeline.
            
        Returns:
            AlignmentRegion containing the time, or None if no region found.
        """
        for region in self.regions:
            if region.contains_dvd_time(dvd_time):
                return region
        return None
        
    def find_region_for_tv_time(self, tv_time: float) -> Optional[AlignmentRegion]:
        """Find region containing the given TV time.
        
        Args:
            tv_time: Time on TV timeline.
            
        Returns:
            AlignmentRegion containing the time, or None if no region found.
        """
        for region in self.regions:
            if region.contains_tv_time(tv_time):  
                return region
        return None
        
    def dvd_to_tv(self, dvd_time: float) -> Optional[float]:
        """Convert DVD time to TV time.
        
        Args:
            dvd_time: Time on DVD timeline.
            
        Returns:
            Converted time on TV timeline, or None if no mapping exists.
        """
        region = self.find_region_for_dvd_time(dvd_time)
        if region and region.region_type == RegionType.MATCHED:
            return region.dvd_to_tv(dvd_time)
        return None
        
    def tv_to_dvd(self, tv_time: float) -> Optional[float]:
        """Convert TV time to DVD time.
        
        Args:
            tv_time: Time on TV timeline.
            
        Returns:
            Converted time on DVD timeline, or None if no mapping exists.
        """
        region = self.find_region_for_tv_time(tv_time)
        if region and region.region_type == RegionType.MATCHED:
            return region.tv_to_dvd(tv_time)
        return None
        
    def get_tv_only_regions(self) -> List[AlignmentRegion]:
        """Get regions that are TV-only (commercials, insertions)."""
        return [r for r in self.regions if r.region_type == RegionType.TV_ONLY]
        
    def get_matched_regions(self) -> List[AlignmentRegion]:
        """Get regions with matched content.""" 
        return [r for r in self.regions if r.region_type == RegionType.MATCHED]
        
    def get_low_confidence_regions(self) -> List[AlignmentRegion]:
        """Get regions with low confidence matches."""
        return [r for r in self.regions if r.region_type == RegionType.LOW_CONFIDENCE]
        
    def merge_compatible_regions(self, tolerance: float = 0.1) -> None:
        """Merge adjacent regions with similar speed ratios and offsets.
        
        Args:
            tolerance: Maximum difference in speed ratio to consider compatible.
        """
        if len(self.regions) < 2:
            return
            
        merged_regions = [self.regions[0]]
        
        for current in self.regions[1:]:
            previous = merged_regions[-1]
            
            # Check if regions are adjacent and compatible
            if (abs(previous.dvd_end - current.dvd_start) < 0.1 and
                abs(previous.speed_ratio - current.speed_ratio) < tolerance and
                previous.region_type == current.region_type == RegionType.MATCHED):
                
                # Merge regions
                merged_region = AlignmentRegion(
                    dvd_start=previous.dvd_start,
                    dvd_end=current.dvd_end,
                    tv_start=previous.tv_start,
                    tv_end=current.tv_end, 
                    offset_seconds=(previous.offset_seconds + current.offset_seconds) / 2,
                    speed_ratio=(previous.speed_ratio + current.speed_ratio) / 2,
                    confidence=min(previous.confidence, current.confidence),
                    region_type=RegionType.MATCHED
                )
                merged_regions[-1] = merged_region
                logger.debug(f"Merged regions: {previous.dvd_start}-{current.dvd_end}")
            else:
                merged_regions.append(current)
                
        self.regions = merged_regions
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "regions": [
                {
                    "dvd_start": r.dvd_start,
                    "dvd_end": r.dvd_end,
                    "tv_start": r.tv_start,
                    "tv_end": r.tv_end,
                    "offset_seconds": r.offset_seconds,
                    "speed_ratio": r.speed_ratio,
                    "confidence": r.confidence,
                    "region_type": r.region_type.value
                }
                for r in self.regions
            ]
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AlignmentMap":
        """Create from dictionary (JSON deserialization)."""
        regions = []
        for r_data in data.get("regions", []):
            region = AlignmentRegion(
                dvd_start=r_data["dvd_start"],
                dvd_end=r_data["dvd_end"],
                tv_start=r_data["tv_start"],
                tv_end=r_data["tv_end"],
                offset_seconds=r_data["offset_seconds"],
                speed_ratio=r_data["speed_ratio"],
                confidence=r_data["confidence"],
                region_type=RegionType(r_data["region_type"])
            )
            regions.append(region)
        return cls(regions)
        
    def save_to_file(self, file_path: Path) -> None:
        """Save alignment map to JSON file."""
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved alignment map: {file_path}")
        
    @classmethod
    def load_from_file(cls, file_path: Path) -> "AlignmentMap":
        """Load alignment map from JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)