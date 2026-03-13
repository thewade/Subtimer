"""SRT subtitle file input/output operations.

Handles parsing and writing SRT subtitle files.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
from datetime import timedelta

logger = logging.getLogger(__name__)


@dataclass
class SubtitleCue:
    """Single subtitle cue with timing and text."""
    index: int
    start_time: float  # seconds
    end_time: float    # seconds
    text: str
    
    def __post_init__(self) -> None:
        """Validate cue data."""
        if self.start_time >= self.end_time:
            raise ValueError("Start time must be before end time")
        if self.index < 1:
            raise ValueError("Index must be positive")
            
    @property
    def duration(self) -> float:
        """Duration of subtitle cue in seconds."""
        return self.end_time - self.start_time
        
    def __str__(self) -> str:
        """String representation for debugging."""
        return f"Cue {self.index}: {self.start_time:.3f}-{self.end_time:.3f} '{self.text[:30]}...'"


class SRTParser:
    """Parser for SRT subtitle files."""
    
    # SRT timestamp pattern: 00:01:23,456 --> 00:02:34,789
    TIMESTAMP_PATTERN = re.compile(
        r'(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})'
    )
    
    def parse_file(self, srt_path: Path) -> List[SubtitleCue]: 
        """Parse SRT file and return list of subtitle cues.
        
        Args:
            srt_path: Path to SRT file.
            
        Returns:
            List of parsed subtitle cues.
            
        Raises:
            FileNotFoundError: If SRT file doesn't exist.
            ValueError: If SRT format is invalid.
        """
        if not srt_path.exists():
            raise FileNotFoundError(f"SRT file not found: {srt_path}")
            
        try:
            with open(srt_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            # Try with different encodings
            for encoding in ['latin1', 'cp1252', 'iso-8859-1']:
                try:
                    with open(srt_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    logger.warning(f"Used {encoding} encoding for {srt_path}")
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError(f"Could not decode {srt_path} with any encoding")
                
        return self._parse_content(content)
        
    def _parse_content(self, content: str) -> List[SubtitleCue]: 
        """Parse SRT content string.
        
        Args:
            content: SRT file content.
            
        Returns:
            List of parsed subtitle cues.
        """
        cues = []
        
        # Split into subtitle blocks (separated by blank lines)
        blocks = re.split(r'\n\s*\n', content.strip())
        
        for block in blocks:
            if not block.strip():
                continue
                
            try:
                cue = self._parse_block(block)
                if cue:
                    cues.append(cue)
            except Exception as e:
                logger.warning(f"Failed to parse subtitle block: {e}")
                logger.debug(f"Block content: {repr(block)}")
                
        logger.info(f"Parsed {len(cues)} subtitle cues")
        return cues
        
    def _parse_block(self, block: str) -> Optional[SubtitleCue]:
        """Parse single subtitle block.
        
        Args:
            block: Single subtitle block text.
            
        Returns:
            SubtitleCue or None if parsing failed.
        """
        lines = block.strip().split('\n')
        if len(lines) < 3:
            return None
            
        # First line: index number
        try:
            index = int(lines[0].strip())
        except ValueError:
            logger.warning(f"Invalid subtitle index: {lines[0]}")
            return None
            
        # Second line: timestamp
        timestamp_match = self.TIMESTAMP_PATTERN.match(lines[1])
        if not timestamp_match:
            logger.warning(f"Invalid timestamp format: {lines[1]}")
            return None
            
        # Parse timestamps
        start_time = self._parse_timestamp(timestamp_match.groups()[:4])
        end_time = self._parse_timestamp(timestamp_match.groups()[4:])
        
        # Remaining lines: subtitle text
        text = '\n'.join(lines[2:]).strip()
        
        return SubtitleCue(
            index=index,
            start_time=start_time,
            end_time=end_time,
            text=text
        )
        
    def _parse_timestamp(self, timestamp_parts: tuple) -> float:
        """Parse timestamp parts to seconds.
        
        Args:
            timestamp_parts: Tuple of (hours, minutes, seconds, milliseconds).
            
        Returns:
            Time in seconds as float.
        """
        hours, minutes, seconds, milliseconds = map(int, timestamp_parts)
        total_seconds = hours * 3600 + minutes * 60 + seconds + milliseconds / 1000.0
        return total_seconds
        

class SRTWriter:
    """Writer for SRT subtitle files."""
    
    def write_file(self, cues: List[SubtitleCue], output_path: Path) -> None:
        """Write subtitle cues to SRT file.
        
        Args:
            cues: List of subtitle cues to write.
            output_path: Output SRT file path.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, cue in enumerate(cues, 1):
                # Update index to be sequential
                cue_copy = SubtitleCue(
                    index=i,
                    start_time=cue.start_time,
                    end_time=cue.end_time,
                    text=cue.text
                )
                f.write(self._format_cue(cue_copy))
                f.write('\n\n')
                
        logger.info(f"Wrote {len(cues)} subtitle cues to {output_path}")
        
    def _format_cue(self, cue: SubtitleCue) -> str:
        """Format subtitle cue as SRT block.
        
        Args:
            cue: Subtitle cue to format.
            
        Returns:
            Formatted SRT block string.
        """
        start_timestamp = self._format_timestamp(cue.start_time)
        end_timestamp = self._format_timestamp(cue.end_time)
        
        return f"{cue.index}\n{start_timestamp} --> {end_timestamp}\n{cue.text}"
        
    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds as SRT timestamp.
        
        Args:
            seconds: Time in seconds.
            
        Returns:
            SRT timestamp string (HH:MM:SS,mmm).
        """
        # Handle negative times (shouldn't happen but be safe)
        seconds = max(0, seconds)
        
        td = timedelta(seconds=seconds)
        total_seconds = int(td.total_seconds())
        
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        milliseconds = int((seconds - total_seconds) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


class SubtitleProcessor:
    """High-level subtitle processing operations."""
    
    def __init__(self):
        """Initialize subtitle processor."""
        self.parser = SRTParser()
        self.writer = SRTWriter()
        
    def load_subtitles(self, srt_path: Path) -> List[SubtitleCue]:
        """Load subtitles from SRT file.
        
        Args:
            srt_path: Path to SRT file.
            
        Returns:
            List of subtitle cues.
        """
        return self.parser.parse_file(srt_path)
        
    def save_subtitles(self, cues: List[SubtitleCue], output_path: Path) -> None:
        """Save subtitles to SRT file.
        
        Args:
            cues: Subtitle cues to save.
            output_path: Output SRT file path.
        """
        self.writer.write_file(cues, output_path)
        
    def validate_subtitles(self, cues: List[SubtitleCue]) -> List[str]:
        """Validate subtitle cues and return list of issues.
        
        Args:
            cues: Subtitle cues to validate.
            
        Returns:
            List of validation error messages.
        """
        issues = []
        
        for i, cue in enumerate(cues):
            # Check timing
            if cue.start_time >= cue.end_time:
                issues.append(f"Cue {cue.index}: Invalid timing {cue.start_time} >= {cue.end_time}")
                
            if cue.start_time < 0:
                issues.append(f"Cue {cue.index}: Negative start time {cue.start_time}")
                
            # Check for overlaps with next cue
            if i < len(cues) - 1:
                next_cue = cues[i + 1]
                if cue.end_time > next_cue.start_time:
                    issues.append(
                        f"Cue {cue.index}: Overlaps with next cue "
                        f"({cue.end_time} > {next_cue.start_time})"
                    )
                    
            # Check text content
            if not cue.text.strip():
                issues.append(f"Cue {cue.index}: Empty text")
                
        return issues