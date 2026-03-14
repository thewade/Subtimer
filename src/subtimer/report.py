"""Report generation for alignment and retiming results.

Generates alignment JSON and human-readable summary reports.
"""

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from .alignment_map import AlignmentMap, RegionType
from .retime import RetimingSummary, CueAction

logger = logging.getLogger(__name__)


@dataclass
class ProcessingMetadata:
    """Metadata about processing inputs and settings."""
    dvd_file: str
    tv_file: str
    subtitle_file: str
    output_srt: str
    processing_time: str
    tool_version: str
    settings: Dict[str, Any]


@dataclass
class AlignmentReport:
    """Complete alignment and retiming report."""
    metadata: ProcessingMetadata
    alignment_summary: Dict[str, Any]
    retiming_summary: Dict[str, Any]
    regions: List[Dict[str, Any]]
    warnings: List[str]
    dropped_subtitles: List[Dict[str, Any]]
    flagged_subtitles: List[Dict[str, Any]]


class ReportGenerator:
    """Generates alignment and retiming reports."""
    
    def __init__(self, tool_version: str = "0.1.0"):
        """Initialize report generator.
        
        Args:
            tool_version: Version string for the tool.
        """
        self.tool_version = tool_version
        
    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds as hh:mm:ss.sss string.
        
        Args:
            seconds: Time in seconds.
            
        Returns:
            Formatted time string.
        """
        # Handle string inputs that might come from JSON deserialization
        if isinstance(seconds, str):
            try:
                seconds = float(seconds)
            except (ValueError, TypeError):
                return "00:00:00.000"  # Default for invalid input
        
        # Handle None or other invalid inputs
        if seconds is None or not isinstance(seconds, (int, float)):
            return "00:00:00.000"
            
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
        
    def _format_offset_seconds(self, offset: float) -> str:
        """Format offset seconds with proper sign handling.
        
        Args:
            offset: Offset in seconds (can be positive or negative).
            
        Returns:
            Formatted time string with appropriate sign.
        """
        # Handle string inputs that might come from JSON deserialization
        if isinstance(offset, str):
            try:
                offset = float(offset)
            except (ValueError, TypeError):
                return "00:00:00.000"
        
        # Handle None or other invalid inputs
        if offset is None or not isinstance(offset, (int, float)):
            return "00:00:00.000"
            
        if offset >= 0:
            return self._format_time(offset)
        else:
            return f"-{self._format_time(-offset)}"
        
    def generate_alignment_json(
        self,
        alignment_map: AlignmentMap,
        retiming_summary: RetimingSummary,
        metadata: ProcessingMetadata,
        output_path: Path,
        warnings: Optional[List[str]] = None
    ) -> None:
        """Generate detailed JSON alignment report.
        
        Args:
            alignment_map: Audio alignment results.
            retiming_summary: Subtitle retiming results.
            metadata: Processing metadata.
            output_path: Output JSON file path.
            warnings: Optional warnings to include.
        """
        logger.info("Generating alignment JSON report")
        
        warnings = warnings or []
        
        # Build report structure
        report = AlignmentReport(
            metadata=metadata,
            alignment_summary=self._create_alignment_summary(alignment_map),
            retiming_summary=self._create_retiming_summary(retiming_summary),
            regions=self._create_regions_list(alignment_map),
            warnings=warnings,
            dropped_subtitles=self._extract_dropped_subtitles(retiming_summary),
            flagged_subtitles=self._extract_flagged_subtitles(retiming_summary)
        )
        
        # Convert to dictionary and save
        report_dict = asdict(report)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Saved alignment report: {output_path}")
        
    def generate_summary_report(
        self,
        alignment_map: AlignmentMap,
        retiming_summary: RetimingSummary,
        metadata: ProcessingMetadata,
        output_path: Path,
        warnings: Optional[List[str]] = None
    ) -> None:
        """Generate human-readable summary report.
        
        Args:
            alignment_map: Audio alignment results.
            retiming_summary: Subtitle retiming results.
            metadata: Processing metadata.
            output_path: Output text file path.
            warnings: Optional warnings to include.
        """
        logger.info("Generating summary report")
        
        warnings = warnings or []
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            self._write_summary_content(
                f, alignment_map, retiming_summary, metadata, warnings
            )
            
        logger.info(f"Saved summary report: {output_path}")
        
    def _create_alignment_summary(self, alignment_map: AlignmentMap) -> Dict[str, Any]:
        """Create alignment summary section.
        
        Args:
            alignment_map: Alignment map to summarize.
            
        Returns:
            Dictionary with alignment statistics.
        """
        matched_regions = alignment_map.get_matched_regions()
        tv_only_regions = alignment_map.get_tv_only_regions()
        low_confidence_regions = alignment_map.get_low_confidence_regions()
        
        total_dvd_time = sum(r.dvd_duration for r in matched_regions)
        total_tv_time = sum(r.tv_duration for r in matched_regions)
        
        avg_confidence = (
            sum(r.confidence for r in matched_regions) / len(matched_regions)
            if matched_regions else 0.0
        )
        
        speed_ratios = [r.speed_ratio for r in matched_regions]
        avg_speed_ratio = sum(speed_ratios) / len(speed_ratios) if speed_ratios else 1.0
        
        tv_only_duration = sum(r.tv_duration for r in tv_only_regions)
        
        return {
            "total_regions": len(alignment_map.regions),
            "matched_regions": len(matched_regions),
            "tv_only_regions": len(tv_only_regions),
            "low_confidence_regions": len(low_confidence_regions),
            "total_matched_dvd_time": self._format_time(total_dvd_time),
            "total_matched_tv_time": self._format_time(total_tv_time),
            "tv_only_duration": self._format_time(tv_only_duration),
            "average_confidence": round(avg_confidence, 3),
            "average_speed_ratio": round(avg_speed_ratio, 4),
            "speed_difference_percent": round((avg_speed_ratio - 1.0) * 100, 2)
        }
        
    def _create_retiming_summary(self, retiming_summary: RetimingSummary) -> Dict[str, Any]:
        """Create retiming summary section.
        
        Args:
            retiming_summary: Retiming results to summarize.
            
        Returns:
            Dictionary with retiming statistics.
        """
        return {
            "total_input_cues": retiming_summary.total_cues,
            "retimed_cues": retiming_summary.retimed_count,
            "dropped_cues": retiming_summary.dropped_count,
            "flagged_cues": retiming_summary.flagged_count,
            "split_cues": retiming_summary.split_count,
            "success_rate_percent": round(
                (retiming_summary.retimed_count / retiming_summary.total_cues) * 100, 1
            ) if retiming_summary.total_cues > 0 else 0.0
        }
        
    def _create_regions_list(self, alignment_map: AlignmentMap) -> List[Dict[str, Any]]:
        """Create detailed regions list.
        
        Args:
            alignment_map: Alignment map.
            
        Returns:
            List of region dictionaries.
        """
        regions = []
        
        for region in alignment_map.regions:
            region_dict = {
                "dvd_start": self._format_time(region.dvd_start),
                "dvd_end": self._format_time(region.dvd_end),
                "dvd_duration": self._format_time(region.dvd_duration),
                "tv_start": self._format_time(region.tv_start),
                "tv_end": self._format_time(region.tv_end),
                "tv_duration": self._format_time(region.tv_duration),
                "offset_seconds": self._format_offset_seconds(region.offset_seconds),
                "speed_ratio": round(region.speed_ratio, 4),
                "confidence": round(region.confidence, 3),
                "region_type": region.region_type.value
            }
            
            # Add interpretation
            if region.region_type == RegionType.MATCHED:
                if region.speed_ratio > 1.05:
                    region_dict["interpretation"] = "TV content runs faster than DVD"
                elif region.speed_ratio < 0.95:
                    region_dict["interpretation"] = "TV content runs slower than DVD"
                else:
                    region_dict["interpretation"] = "Normal speed match"
            elif region.region_type == RegionType.TV_ONLY:
                region_dict["interpretation"] = "TV-only content (likely commercial)"
            elif region.region_type == RegionType.LOW_CONFIDENCE:
                region_dict["interpretation"] = "Uncertain match quality"
                
            regions.append(region_dict)
            
        return regions
        
    def _extract_dropped_subtitles(self, retiming_summary: RetimingSummary) -> List[Dict[str, Any]]:
        """Extract dropped subtitle information.
        
        Args:
            retiming_summary: Retiming results.
            
        Returns:
            List of dropped subtitle dictionaries.
        """
        dropped = []
        
        for result in retiming_summary.results:
            if result.action in [CueAction.DROPPED_TV_ONLY, CueAction.DROPPED_UNMATCHED]:
                dropped.append({
                    "original_index": result.original_cue.index,
                    "start_time": self._format_time(result.original_cue.start_time),
                    "end_time": self._format_time(result.original_cue.end_time),
                    "text_preview": result.original_cue.text[:100] + "..." if len(result.original_cue.text) > 100 else result.original_cue.text,
                    "reason": result.reason,
                    "action": result.action.value
                })
                
        return dropped
        
    def _extract_flagged_subtitles(self, retiming_summary: RetimingSummary) -> List[Dict[str, Any]]:
        """Extract flagged subtitle information.
        
        Args:
            retiming_summary: Retiming results.
            
        Returns:
            List of flagged subtitle dictionaries.
        """
        flagged = []
        
        for result in retiming_summary.results:
            if result.action in [CueAction.FLAGGED_BOUNDARY, CueAction.FLAGGED_LOW_CONFIDENCE]:
                flagged.append({
                    "original_index": result.original_cue.index,
                    "start_time": self._format_time(result.original_cue.start_time),
                    "end_time": self._format_time(result.original_cue.end_time),
                    "text_preview": result.original_cue.text[:100] + "..." if len(result.original_cue.text) > 100 else result.original_cue.text,
                    "reason": result.reason,
                    "confidence": result.confidence,
                    "action": result.action.value
                })
                
        return flagged
        
    def _write_summary_content(
        self,
        f,
        alignment_map: AlignmentMap,
        retiming_summary: RetimingSummary,
        metadata: ProcessingMetadata,
        warnings: List[str]
    ) -> None:
        """Write human-readable summary content.
        
        Args:
            f: Output file handle.
            alignment_map: Alignment results.
            retiming_summary: Retiming results.
            metadata: Processing metadata.
            warnings: Warning messages.
        """
        # Header
        f.write("DVD SUBTITLE RETIMING REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        # Processing info
        f.write("PROCESSING INFORMATION\n")
        f.write("-" * 25 + "\n")
        f.write(f"DVD File: {metadata.dvd_file}\n")
        f.write(f"TV File: {metadata.tv_file}\n")
        f.write(f"Subtitle File: {metadata.subtitle_file}\n")
        f.write(f"Output SRT: {metadata.output_srt}\n")
        f.write(f"Processing Time: {metadata.processing_time}\n")
        f.write(f"Tool Version: {metadata.tool_version}\n\n")
        
        # Alignment summary
        alignment_summary = self._create_alignment_summary(alignment_map)
        f.write("AUDIO ALIGNMENT SUMMARY\n")
        f.write("-" * 25 + "\n")
        f.write(f"Total Regions Found: {alignment_summary['total_regions']}\n")
        f.write(f"Matched Regions: {alignment_summary['matched_regions']}\n")
        f.write(f"TV-Only Regions: {alignment_summary['tv_only_regions']}\n")
        f.write(f"Low Confidence Regions: {alignment_summary['low_confidence_regions']}\n")
        f.write(f"Total Matched DVD Time: {alignment_summary['total_matched_dvd_time']}\n")
        f.write(f"Total TV-Only Time: {alignment_summary['tv_only_duration']}\n")
        f.write(f"Average Confidence: {alignment_summary['average_confidence']:.3f}\n")
        f.write(f"Average Speed Ratio: {alignment_summary['average_speed_ratio']:.4f}\n")
        f.write(f"Speed Difference: {alignment_summary['speed_difference_percent']:+.2f}%\n\n")
        
        # Retiming summary  
        retiming_sum = self._create_retiming_summary(retiming_summary)
        f.write("SUBTITLE RETIMING SUMMARY\n")
        f.write("-" * 25 + "\n")
        f.write(f"Input Subtitles: {retiming_sum['total_input_cues']}\n")
        f.write(f"Successfully Retimed: {retiming_sum['retimed_cues']}\n")
        f.write(f"Dropped (TV-only/Unmatched): {retiming_sum['dropped_cues']}\n")
        f.write(f"Flagged for Review: {retiming_sum['flagged_cues']}\n")
        f.write(f"Split at Boundaries: {retiming_sum['split_cues']}\n")
        f.write(f"Success Rate: {retiming_sum['success_rate_percent']:.1f}%\n\n")
        
        # Warnings
        if warnings:
            f.write("WARNINGS\n")
            f.write("-" * 10 + "\n")
            for warning in warnings:
                f.write(f"• {warning}\n")
            f.write("\n")
            
        # Detailed regions (first 10)
        f.write("ALIGNMENT REGIONS (first 10)\n")
        f.write("-" * 30 + "\n")
        regions = self._create_regions_list(alignment_map)
        for i, region in enumerate(regions[:10]):
            f.write(f"Region {i+1}: {region['region_type']}\n")
            f.write(f"  DVD: {region['dvd_start']} - {region['dvd_end']} ({region['dvd_duration']})\n")
            f.write(f"  TV:  {region['tv_start']} - {region['tv_end']} ({region['tv_duration']})\n")
            f.write(f"  Speed: {region['speed_ratio']:.4f}x, Confidence: {region['confidence']:.3f}\n")
            if 'interpretation' in region:
                f.write(f"  {region['interpretation']}\n")
            f.write("\n")
            
        if len(regions) > 10:
            f.write(f"... and {len(regions) - 10} more regions (see alignment.json for full details)\n\n")
            
        # Sample of dropped/flagged subtitles
        dropped = self._extract_dropped_subtitles(retiming_summary)
        if dropped:
            f.write("DROPPED SUBTITLES (first 5)\n")
            f.write("-" * 25 + "\n")
            for subtitle in dropped[:5]:
                f.write(f"Cue {subtitle['original_index']}: {subtitle['start_time']}-{subtitle['end_time']}\n")
                f.write(f"  Reason: {subtitle['reason']}\n")
                f.write(f"  Text: {subtitle['text_preview']}\n\n")
            if len(dropped) > 5:
                f.write(f"... and {len(dropped) - 5} more dropped subtitles\n\n")
                
        flagged = self._extract_flagged_subtitles(retiming_summary)
        if flagged:
            f.write("FLAGGED SUBTITLES (first 5)\n")
            f.write("-" * 25 + "\n")
            for subtitle in flagged[:5]:
                f.write(f"Cue {subtitle['original_index']}: {subtitle['start_time']}-{subtitle['end_time']}\n")
                f.write(f"  Reason: {subtitle['reason']}\n")
                if subtitle['confidence']:
                    f.write(f"  Confidence: {subtitle['confidence']:.3f}\n")
                f.write(f"  Text: {subtitle['text_preview']}\n\n")
            if len(flagged) > 5:
                f.write(f"... and {len(flagged) - 5} more flagged subtitles\n\n")