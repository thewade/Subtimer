"""Test fixtures and utilities for subtitle retimer tests."""

import pytest
from pathlib import Path
from textwrap import dedent

from subtimer.subtitle_io import SubtitleCue
from subtimer.alignment_map import AlignmentRegion, RegionType


# Test data fixtures

@pytest.fixture
def sample_srt_content():
    """Sample SRT file content for testing."""
    return dedent("""
        1
        00:00:01,000 --> 00:00:03,500
        Hello world

        2
        00:00:04,000 --> 00:00:06,000
        This is a test
        with multiple lines

        3
        00:01:23,456 --> 00:02:34,789
        Precise timing test
        """).strip()


@pytest.fixture 
def sample_subtitles():
    """Sample subtitle cues for testing."""
    return [
        SubtitleCue(1, 1.0, 3.5, "Hello world"),
        SubtitleCue(2, 4.0, 6.0, "This is a test\nwith multiple lines"),
        SubtitleCue(3, 83.456, 154.789, "Precise timing test")
    ]


@pytest.fixture
def simple_matched_region():
    """Simple matched alignment region."""
    return AlignmentRegion(
        dvd_start=0.0,
        dvd_end=100.0,
        tv_start=5.0,
        tv_end=105.0,
        offset_seconds=5.0,
        speed_ratio=1.0,
        confidence=0.9,
        region_type=RegionType.MATCHED
    )


@pytest.fixture
def tv_only_region():
    """TV-only region (commercial).""" 
    return AlignmentRegion(
        dvd_start=0.0,
        dvd_end=0.0,
        tv_start=100.0,
        tv_end=130.0,
        offset_seconds=0.0,
        speed_ratio=1.0,
        confidence=0.0,
        region_type=RegionType.TV_ONLY
    )


@pytest.fixture
def speed_difference_region():
    """Region with speed difference."""
    return AlignmentRegion(
        dvd_start=0.0,
        dvd_end=100.0,
        tv_start=0.0,
        tv_end=120.0,  # TV 20% faster
        offset_seconds=0.0,
        speed_ratio=1.2,
        confidence=0.8,
        region_type=RegionType.MATCHED
    )


@pytest.fixture
def low_confidence_region():
    """Low confidence region."""
    return AlignmentRegion(
        dvd_start=0.0,
        dvd_end=50.0,
        tv_start=0.0,
        tv_end=50.0,
        offset_seconds=0.0,
        speed_ratio=1.0,
        confidence=0.2,  # Low confidence
        region_type=RegionType.LOW_CONFIDENCE
    )


# Test utility functions

def create_test_srt_file(content: str, path: Path) -> Path:
    """Create a test SRT file with given content."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')
    return path


def assert_subtitle_equal(cue1: SubtitleCue, cue2: SubtitleCue, tolerance: float = 0.001) -> None:
    """Assert two subtitle cues are equal within tolerance.""" 
    assert cue1.index == cue2.index
    assert abs(cue1.start_time - cue2.start_time) < tolerance
    assert abs(cue1.end_time - cue2.end_time) < tolerance
    assert cue1.text == cue2.text


def create_test_audio_data(duration: float, sample_rate: int = 22050) -> bytes:
    """Create dummy audio data for testing."""
    import numpy as np
    
    # Generate simple sine wave
    t = np.linspace(0, duration, int(duration * sample_rate), False)
    audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
    
    # Convert to int16
    audio_int16 = (audio * 32767).astype(np.int16)
    return audio_int16.tobytes()


class MockMediaProcessor:
    """Mock media processor for testing without FFmpeg dependency."""
    
    def __init__(self):
        self.extracted_files = {}
        
    def probe_media(self, media_path: Path):
        """Mock media probing."""
        from subtimer.media_io import MediaInfo
        return MediaInfo(
            path=media_path,
            duration=120.0,  # 2 minutes
            sample_rate=48000,
            channels=2,
            audio_codec="aac",
            format_name="mp4",
            has_audio=True
        )
        
    def extract_audio(self, media_path: Path, output_path: Path = None, **kwargs):
        """Mock audio extraction."""
        if output_path is None:
            output_path = Path(f"/tmp/{media_path.stem}_audio.wav")
            
        # Create dummy audio file
        audio_data = create_test_audio_data(120.0)  # 2 minutes
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(audio_data)
        
        self.extracted_files[media_path] = output_path
        return output_path