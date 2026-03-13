"""Tests for subtitle I/O operations."""

import pytest
from pathlib import Path
from textwrap import dedent

from subtimer.subtitle_io import SRTParser, SRTWriter, SubtitleCue, SubtitleProcessor


class TestSubtitleCue:
    """Test SubtitleCue data class."""
    
    def test_valid_cue_creation(self):
        """Test creating a valid subtitle cue."""
        cue = SubtitleCue(
            index=1,
            start_time=0.0,
            end_time=5.0,
            text="Hello world"
        )
        assert cue.index == 1
        assert cue.start_time == 0.0
        assert cue.end_time == 5.0
        assert cue.text == "Hello world"
        assert cue.duration == 5.0
        
    def test_invalid_timing(self):
        """Test cue with invalid timing."""
        with pytest.raises(ValueError, match="Start time must be before end time"):
            SubtitleCue(
                index=1,
                start_time=5.0,
                end_time=2.0,  # Invalid: end before start
                text="Invalid"
            )
            
    def test_invalid_index(self):
        """Test cue with invalid index."""
        with pytest.raises(ValueError, match="Index must be positive"):
            SubtitleCue(
                index=0,  # Invalid: must be >= 1
                start_time=0.0,
                end_time=5.0,
                text="Invalid"
            )


class TestSRTParser:
    """Test SRT file parsing."""
    
    def test_parse_basic_srt(self, tmp_path):
        """Test parsing basic SRT format."""
        srt_content = dedent("""
            1
            00:00:01,000 --> 00:00:03,500
            Hello world

            2
            00:00:04,000 --> 00:00:06,000
            This is a test
            """).strip()
            
        srt_file = tmp_path / "test.srt"
        srt_file.write_text(srt_content)
        
        parser = SRTParser()
        cues = parser.parse_file(srt_file)
        
        assert len(cues) == 2
        
        assert cues[0].index == 1
        assert cues[0].start_time == 1.0
        assert cues[0].end_time == 3.5
        assert cues[0].text == "Hello world"
        
        assert cues[1].index == 2
        assert cues[1].start_time == 4.0
        assert cues[1].end_time == 6.0
        assert cues[1].text == "This is a test"
        
    def test_parse_multiline_text(self, tmp_path):
        """Test parsing SRT with multiline text."""
        srt_content = dedent("""
            1
            00:00:01,000 --> 00:00:03,000
            First line
            Second line
            
            2
            00:00:04,000 --> 00:00:05,000
            Single line
            """).strip()
            
        srt_file = tmp_path / "multiline.srt"
        srt_file.write_text(srt_content)
        
        parser = SRTParser()
        cues = parser.parse_file(srt_file)
        
        assert len(cues) == 2
        assert cues[0].text == "First line\nSecond line"
        assert cues[1].text == "Single line"
        
    def test_parse_milliseconds(self, tmp_path):
        """Test parsing precise millisecond timestamps."""
        srt_content = dedent("""
            1
            00:01:23,456 --> 00:02:34,789
            Precise timing
            """).strip()
            
        srt_file = tmp_path / "precise.srt"
        srt_file.write_text(srt_content)
        
        parser = SRTParser()
        cues = parser.parse_file(srt_file)
        
        assert len(cues) == 1
        assert cues[0].start_time == 83.456  # 1*60 + 23 + 0.456
        assert cues[0].end_time == 154.789   # 2*60 + 34 + 0.789
        
    def test_parse_missing_file(self):
        """Test parsing non-existent file."""
        parser = SRTParser()
        with pytest.raises(FileNotFoundError):
            parser.parse_file(Path("nonexistent.srt"))
            
    def test_parse_invalid_format(self, tmp_path):
        """Test parsing invalid SRT format."""
        srt_content = "This is not a valid SRT file"
        srt_file = tmp_path / "invalid.srt"
        srt_file.write_text(srt_content)
        
        parser = SRTParser()
        cues = parser.parse_file(srt_file)
        
        # Should handle gracefully and return empty list
        assert len(cues) == 0


class TestSRTWriter:
    """Test SRT file writing."""
    
    def test_write_basic_srt(self, tmp_path):
        """Test writing basic SRT format."""
        cues = [
            SubtitleCue(1, 1.0, 3.5, "Hello world"),
            SubtitleCue(2, 4.0, 6.0, "This is a test")
        ]
        
        output_file = tmp_path / "output.srt"
        writer = SRTWriter()
        writer.write_file(cues, output_file)
        
        # Read back and verify
        content = output_file.read_text()
        
        assert "1" in content
        assert "00:00:01,000 --> 00:00:03,500" in content
        assert "Hello world" in content
        assert "2" in content
        assert "00:00:04,000 --> 00:00:06,000" in content
        assert "This is a test" in content
        
    def test_write_multiline_text(self, tmp_path):
        """Test writing multiline subtitle text."""
        cues = [
            SubtitleCue(1, 0.0, 2.0, "First line\nSecond line")
        ]
        
        output_file = tmp_path / "multiline.srt" 
        writer = SRTWriter()
        writer.write_file(cues, output_file)
        
        content = output_file.read_text()
        assert "First line\nSecond line" in content
        
    def test_write_precise_timing(self, tmp_path):
        """Test writing precise millisecond timing."""
        cues = [
            SubtitleCue(1, 83.456, 154.789, "Precise timing")
        ]
        
        output_file = tmp_path / "precise.srt"
        writer = SRTWriter()
        writer.write_file(cues, output_file)
        
        content = output_file.read_text()
        assert "00:01:23,456 --> 00:02:34,789" in content
        
    def test_reindex_cues(self, tmp_path):
        """Test that writer creates sequential indices."""
        cues = [
            SubtitleCue(5, 1.0, 3.0, "First"),    # Original index 5
            SubtitleCue(10, 4.0, 6.0, "Second")   # Original index 10
        ]
        
        output_file = tmp_path / "reindexed.srt"
        writer = SRTWriter()
        writer.write_file(cues, output_file)
        
        # Read back and check indices are sequential
        parser = SRTParser()
        parsed_cues = parser.parse_file(output_file)
        
        assert parsed_cues[0].index == 1
        assert parsed_cues[1].index == 2


class TestSubtitleProcessor:
    """Test high-level subtitle processing."""
    
    def test_roundtrip_load_save(self, tmp_path):
        """Test loading and saving subtitles."""
        # Create test SRT
        srt_content = dedent("""
            1
            00:00:01,000 --> 00:00:03,000
            Hello world

            2
            00:00:04,000 --> 00:00:06,000
            Test subtitle
            """).strip()
            
        input_file = tmp_path / "input.srt"
        input_file.write_text(srt_content)
        
        output_file = tmp_path / "output.srt"
        
        processor = SubtitleProcessor()
        
        # Load
        cues = processor.load_subtitles(input_file)
        assert len(cues) == 2
        
        # Save
        processor.save_subtitles(cues, output_file)
        
        # Load again and verify
        loaded_cues = processor.load_subtitles(output_file)
        assert len(loaded_cues) == 2
        assert loaded_cues[0].text == "Hello world"
        assert loaded_cues[1].text == "Test subtitle"
        
    def test_validate_good_subtitles(self):
        """Test validation of good subtitles."""
        cues = [
            SubtitleCue(1, 0.0, 2.0, "First"),
            SubtitleCue(2, 3.0, 5.0, "Second"),  # Non-overlapping
            SubtitleCue(3, 6.0, 8.0, "Third")
        ]
        
        processor = SubtitleProcessor()
        issues = processor.validate_subtitles(cues)
        
        assert len(issues) == 0
        
    def test_validate_overlapping_subtitles(self):
        """Test validation catches overlapping subtitles."""
        cues = [
            SubtitleCue(1, 0.0, 3.0, "First"),
            SubtitleCue(2, 2.0, 5.0, "Second"),  # Overlaps with first
        ]
        
        processor = SubtitleProcessor()
        issues = processor.validate_subtitles(cues)
        
        assert len(issues) == 1
        assert "Overlaps" in issues[0]
        
    def test_validate_negative_timing(self):
        """Test validation catches negative timing."""
        cues = [
            SubtitleCue(1, -1.0, 2.0, "Negative start")  # Invalid
        ]
        
        processor = SubtitleProcessor()
        issues = processor.validate_subtitles(cues)
        
        assert len(issues) == 1
        assert "Negative start time" in issues[0]
        
    def test_validate_empty_text(self):
        """Test validation catches empty text."""
        cues = [
            SubtitleCue(1, 0.0, 2.0, "   ")  # Whitespace only
        ]
        
        processor = SubtitleProcessor()
        issues = processor.validate_subtitles(cues)
        
        assert len(issues) == 1
        assert "Empty text" in issues[0]