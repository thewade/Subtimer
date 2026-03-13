"""Tests for subtitle retiming functionality."""

import pytest

from subtimer.alignment_map import AlignmentRegion, RegionType, AlignmentMap
from subtimer.subtitle_io import SubtitleCue
from subtimer.retime import SubtitleRetimer, CueAction, RetimingSummary


class TestSubtitleRetimer:
    """Test subtitle retiming with various scenarios."""
    
    def test_simple_constant_offset(self):
        """Test retiming with simple constant offset."""
        # Create alignment with constant offset
        region = AlignmentRegion(
            dvd_start=0.0,
            dvd_end=100.0,
            tv_start=5.0,     # TV is 5 seconds ahead
            tv_end=105.0,
            offset_seconds=5.0,
            speed_ratio=1.0,  # Same speed
            confidence=0.9,
            region_type=RegionType.MATCHED
        )
        
        alignment_map = AlignmentMap([region])
        
        # Create test subtitles on TV timeline
        subtitles = [
            SubtitleCue(1, 10.0, 12.0, "First subtitle"),   # TV: 10-12s
            SubtitleCue(2, 15.0, 18.0, "Second subtitle"),  # TV: 15-18s
        ]
        
        retimer = SubtitleRetimer()
        summary = retimer.retime_subtitles(subtitles, alignment_map)
        
        # Should successfully retime both
        assert summary.retimed_count == 2
        assert summary.dropped_count == 0
        assert summary.flagged_count == 0
        
        retimed_cues = retimer.get_retimed_cues(summary)
        assert len(retimed_cues) == 2
        
        # Check converted times (TV time - 5s = DVD time)
        assert retimed_cues[0].start_time == pytest.approx(5.0)  # 10 - 5
        assert retimed_cues[0].end_time == pytest.approx(7.0)    # 12 - 5
        assert retimed_cues[1].start_time == pytest.approx(10.0) # 15 - 5
        assert retimed_cues[1].end_time == pytest.approx(13.0)   # 18 - 5
        
    def test_speed_difference(self):
        """Test retiming with speed difference."""
        # TV content runs 20% faster than DVD
        region = AlignmentRegion(
            dvd_start=0.0,
            dvd_end=100.0,
            tv_start=0.0,
            tv_end=120.0,     # TV is 20% longer
            offset_seconds=0.0,
            speed_ratio=1.2,  # TV runs faster
            confidence=0.8,
            region_type=RegionType.MATCHED
        )
        
        alignment_map = AlignmentMap([region])
        
        subtitles = [
            SubtitleCue(1, 12.0, 24.0, "Speed test"),  # TV: 12-24s
        ]
        
        retimer = SubtitleRetimer()
        summary = retimer.retime_subtitles(subtitles, alignment_map)
        
        assert summary.retimed_count == 1
        
        retimed_cues = retimer.get_retimed_cues(summary)
        
        # DVD time = TV time / speed_ratio
        assert retimed_cues[0].start_time == pytest.approx(10.0)  # 12 / 1.2
        assert retimed_cues[0].end_time == pytest.approx(20.0)    # 24 / 1.2
        
    def test_tv_only_region_dropping(self):
        """Test dropping subtitles in TV-only regions (commercials)."""
        # Create matched region and TV-only region
        matched_region = AlignmentRegion(
            dvd_start=0.0, dvd_end=50.0, tv_start=0.0, tv_end=50.0,
            offset_seconds=0.0, speed_ratio=1.0, confidence=0.9,
            region_type=RegionType.MATCHED
        )
        tv_only_region = AlignmentRegion(
            dvd_start=0.0, dvd_end=0.0,    # No DVD equivalent
            tv_start=50.0, tv_end=80.0,    # TV commercial break
            offset_seconds=0.0, speed_ratio=1.0, confidence=0.0,
            region_type=RegionType.TV_ONLY
        )
        
        alignment_map = AlignmentMap([matched_region, tv_only_region])
        
        subtitles = [
            SubtitleCue(1, 25.0, 27.0, "Before commercial"),     # In matched
            SubtitleCue(2, 60.0, 65.0, "During commercial"),     # In TV-only
            SubtitleCue(3, 85.0, 87.0, "After commercial"),      # Outside regions
        ]
        
        retimer = SubtitleRetimer(drop_tv_only=True, drop_unmatched=True)
        summary = retimer.retime_subtitles(subtitles, alignment_map)
        
        assert summary.retimed_count == 1
        assert summary.dropped_count == 2
        
        # Check which cues were processed how
        results = {r.original_cue.index: r.action for r in summary.results}
        assert results[1] == CueAction.RETIMED
        assert results[2] == CueAction.DROPPED_TV_ONLY
        assert results[3] == CueAction.DROPPED_UNMATCHED
        
    def test_boundary_crossing_flagging(self):
        """Test flagging subtitles that cross region boundaries."""
        # Two separate matched regions with gap
        region1 = AlignmentRegion(
            dvd_start=0.0, dvd_end=30.0, tv_start=0.0, tv_end=30.0,
            offset_seconds=0.0, speed_ratio=1.0, confidence=0.9,
            region_type=RegionType.MATCHED
        )
        region2 = AlignmentRegion(
            dvd_start=40.0, dvd_end=70.0, tv_start=50.0, tv_end=80.0,
            offset_seconds=10.0, speed_ratio=1.0, confidence=0.8,
            region_type=RegionType.MATCHED
        )
        
        alignment_map = AlignmentMap([region1, region2])
        
        subtitles = [
            SubtitleCue(1, 25.0, 35.0, "Crosses boundary"),  # Starts in region1, ends in gap
            SubtitleCue(2, 45.0, 55.0, "Crosses regions"),   # Spans gap between regions
        ]
        
        retimer = SubtitleRetimer(flag_boundary_crossings=True)
        summary = retimer.retime_subtitles(subtitles, alignment_map)
        
        assert summary.flagged_count == 2 
        
        results = {r.original_cue.index: r.action for r in summary.results}
        assert results[1] == CueAction.FLAGGED_BOUNDARY
        assert results[2] == CueAction.FLAGGED_BOUNDARY
        
    def test_low_confidence_flagging(self):
        """Test flagging subtitles in low-confidence regions."""
        low_conf_region = AlignmentRegion(
            dvd_start=0.0, dvd_end=30.0, tv_start=0.0, tv_end=30.0,
            offset_seconds=0.0, speed_ratio=1.0, confidence=0.2,  # Low confidence
            region_type=RegionType.LOW_CONFIDENCE
        )
        
        alignment_map = AlignmentMap([low_conf_region])
        
        subtitles = [
            SubtitleCue(1, 10.0, 15.0, "Low confidence subtitle"),
        ]
        
        retimer = SubtitleRetimer(flag_low_confidence=True, min_confidence_threshold=0.4)
        summary = retimer.retime_subtitles(subtitles, alignment_map)
        
        assert summary.flagged_count == 1
        assert summary.results[0].action == CueAction.FLAGGED_LOW_CONFIDENCE
        
    def test_boundary_splitting(self):
        """Test splitting subtitles at region boundaries."""
        # Two regions with different offsets
        region1 = AlignmentRegion(
            dvd_start=0.0, dvd_end=30.0, tv_start=0.0, tv_end=30.0,
            offset_seconds=0.0, speed_ratio=1.0, confidence=0.9,
            region_type=RegionType.MATCHED
        )
        region2 = AlignmentRegion(
            dvd_start=30.0, dvd_end=60.0, tv_start=40.0, tv_end=70.0,
            offset_seconds=10.0, speed_ratio=1.0, confidence=0.8,
            region_type=RegionType.MATCHED
        )
        
        alignment_map = AlignmentMap([region1, region2])
        
        subtitles = [
            # Subtitle crosses from region1 to region2
            SubtitleCue(1, 25.0, 45.0, "Split me at boundary"),
        ]
        
        retimer = SubtitleRetimer(split_boundary_crossings=True)
        summary = retimer.retime_subtitles(subtitles, alignment_map)
        
        assert summary.split_count == 1
        assert summary.results[0].action == CueAction.SPLIT_BOUNDARY
        assert summary.results[0].split_cues is not None
        # Should have segments before and after boundary
        
    def test_keep_vs_drop_settings(self):
        """Test different keep/drop setting combinations."""
        # TV-only region
        tv_only = AlignmentRegion(
            dvd_start=0.0, dvd_end=0.0, tv_start=0.0, tv_end=30.0,
            offset_seconds=0.0, speed_ratio=1.0, confidence=0.0,
            region_type=RegionType.TV_ONLY
        )
        
        alignment_map = AlignmentMap([tv_only])
        
        subtitles = [
            SubtitleCue(1, 10.0, 15.0, "TV-only subtitle"),
        ]
        
        # Test drop_tv_only=True
        retimer_drop = SubtitleRetimer(drop_tv_only=True)
        summary_drop = retimer_drop.retime_subtitles(subtitles, alignment_map)
        assert summary_drop.dropped_count == 1
        
        # Test drop_tv_only=False (keep as flagged)
        retimer_keep = SubtitleRetimer(drop_tv_only=False)
        summary_keep = retimer_keep.retime_subtitles(subtitles, alignment_map)
        assert summary_keep.flagged_count == 1
        assert summary_keep.dropped_count == 0
        
    def test_multiple_regions_complex_case(self):
        """Test complex case with multiple region types."""
        # Create complex alignment: matched, TV-only, matched
        region1 = AlignmentRegion(
            dvd_start=0.0, dvd_end=30.0, tv_start=0.0, tv_end=30.0,
            offset_seconds=0.0, speed_ratio=1.0, confidence=0.9,
            region_type=RegionType.MATCHED
        )
        commercial = AlignmentRegion(
            dvd_start=0.0, dvd_end=0.0, tv_start=30.0, tv_end=50.0,
            offset_seconds=0.0, speed_ratio=1.0, confidence=0.0,
            region_type=RegionType.TV_ONLY  
        )
        region2 = AlignmentRegion(
            dvd_start=30.0, dvd_end=60.0, tv_start=50.0, tv_end=80.0,
            offset_seconds=20.0, speed_ratio=1.0, confidence=0.8,
            region_type=RegionType.MATCHED
        )
        
        alignment_map = AlignmentMap([region1, commercial, region2])
        
        subtitles = [
            SubtitleCue(1, 10.0, 15.0, "Before commercial"),   # region1
            SubtitleCue(2, 35.0, 40.0, "During commercial"),   # commercial
            SubtitleCue(3, 55.0, 65.0, "After commercial"),    # region2  
            SubtitleCue(4, 25.0, 35.0, "Crosses boundary"),    # spans region1 -> commercial
        ]
        
        retimer = SubtitleRetimer(
            drop_tv_only=True,
            flag_boundary_crossings=True
        )
        summary = retimer.retime_subtitles(subtitles, alignment_map)
        
        # Check results
        assert summary.retimed_count == 2    # Cues 1 and 3
        assert summary.dropped_count == 1    # Cue 2 (commercial)
        assert summary.flagged_count == 1    # Cue 4 (boundary crossing)
        
        retimed_cues = retimer.get_retimed_cues(summary)
        assert len(retimed_cues) == 2
        
        # Check retimed values
        # Cue 1: no offset (region1)
        assert retimed_cues[0].start_time == pytest.approx(10.0)
        assert retimed_cues[0].end_time == pytest.approx(15.0)
        
        # Cue 3: offset by 20s (region2)  
        assert retimed_cues[1].start_time == pytest.approx(35.0)  # 55 - 20
        assert retimed_cues[1].end_time == pytest.approx(45.0)    # 65 - 20
        
    def test_empty_alignment_map(self):
        """Test retiming with no alignment regions."""
        alignment_map = AlignmentMap()  # Empty
        
        subtitles = [
            SubtitleCue(1, 10.0, 15.0, "No alignment"),
        ]
        
        retimer = SubtitleRetimer(drop_unmatched=True)
        summary = retimer.retime_subtitles(subtitles, alignment_map)
        
        # Should drop all subtitles
        assert summary.dropped_count == 1
        assert summary.retimed_count == 0
        
        retimed_cues = retimer.get_retimed_cues(summary)
        assert len(retimed_cues) == 0


class TestRetimingSummary:
    """Test retiming summary functionality."""
    
    def test_summary_statistics(self):
        """Test summary statistics calculation."""
        # Create dummy results 
        from subtimer.retime import RetimeResult
        
        results = [
            RetimeResult(
                SubtitleCue(1, 0.0, 1.0, "1"), CueAction.RETIMED,
                SubtitleCue(1, 0.0, 1.0, "1")
            ),
            RetimeResult(
                SubtitleCue(2, 1.0, 2.0, "2"), CueAction.DROPPED_TV_ONLY
            ),
            RetimeResult(
                SubtitleCue(3, 2.0, 3.0, "3"), CueAction.FLAGGED_BOUNDARY
            ),
        ]
        
        summary = RetimingSummary(
            total_cues=3,
            retimed_count=1,
            dropped_count=1,
            flagged_count=1,
            split_count=0,
            results=results
        )
        
        assert summary.total_cues == 3
        assert summary.retimed_count == 1
        assert summary.dropped_count == 1
        assert summary.flagged_count == 1