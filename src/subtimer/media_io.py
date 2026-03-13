"""Media input/output operations.

Handles probing inputs, extracting audio with FFmpeg, and managing temp/cache files.
"""

import json
import logging
import subprocess
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Union

logger = logging.getLogger(__name__)


def find_ffmpeg_executable(name: str) -> str:
    """Find FFmpeg executable (ffprobe or ffmpeg) in system PATH or common locations.
    
    Args:
        name: Executable name ('ffprobe' or 'ffmpeg').
        
    Returns:
        Full path to executable.
        
    Raises:
        FileNotFoundError: If executable not found.
    """
    # First try system PATH
    exe_path = shutil.which(name)
    if exe_path:
        return exe_path
        
    # Common Windows installation paths
    common_paths = [
        # WinGet installation
        os.path.expandvars(r"$LOCALAPPDATA\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin"),
        # Other common paths
        r"C:\Program Files\ffmpeg\bin",
        r"C:\Program Files (x86)\ffmpeg\bin", 
        r"C:\ffmpeg\bin",
        os.path.expanduser("~/ffmpeg/bin"),
    ]
    
    # Check common paths
    for path_dir in common_paths:
        if os.path.exists(path_dir):
            exe_full_path = os.path.join(path_dir, f"{name}.exe")
            if os.path.exists(exe_full_path):
                logger.info(f"Found {name} at: {exe_full_path}")
                return exe_full_path
                
    # If not found, return original name and let subprocess handle the error
    raise FileNotFoundError(f"Could not find {name} executable. Please ensure FFmpeg is installed.")


@dataclass
class MediaInfo:
    """Information about a media file."""
    path: Path
    duration: float
    sample_rate: int
    channels: int
    audio_codec: str
    format_name: str
    has_audio: bool


class MediaProcessor:
    """Handles media file operations using FFmpeg."""
    
    def __init__(self, temp_dir: Optional[Path] = None, cache_enabled: bool = True):
        """Initialize media processor.
        
        Args:
            temp_dir: Directory for temporary files. Uses system temp if None.
            cache_enabled: Whether to cache extracted audio files.
        """
        self.temp_dir = temp_dir or Path.cwd() / "temp"
        self.cache_enabled = cache_enabled
        self.temp_dir.mkdir(exist_ok=True)
        
    def probe_media(self, media_path: Path) -> MediaInfo:
        """Probe media file to get metadata.
        
        Args:
            media_path: Path to media file.
            
        Returns:
            MediaInfo with file metadata.
            
        Raises:
            subprocess.CalledProcessError: If ffprobe fails.
            FileNotFoundError: If media file doesn't exist.
        """
        if not media_path.exists():
            raise FileNotFoundError(f"Media file not found: {media_path}")
            
        # Find ffprobe executable
        try:
            ffprobe_path = find_ffmpeg_executable("ffprobe")
        except FileNotFoundError as e:
            logger.error(str(e))
            raise
            
        # Use ffprobe to get media information
        cmd = [
            ffprobe_path,
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(media_path)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)  # Safely parse JSON
            
            # Extract audio stream info
            audio_stream = None
            for stream in data.get("streams", []):
                if stream.get("codec_type") == "audio":
                    audio_stream = stream
                    break
                    
            if not audio_stream:
                raise ValueError(f"No audio stream found in {media_path}")
                
            return MediaInfo(
                path=media_path,
                duration=float(data["format"]["duration"]),
                sample_rate=int(audio_stream["sample_rate"]),
                channels=int(audio_stream["channels"]),
                audio_codec=audio_stream["codec_name"],
                format_name=data["format"]["format_name"],
                has_audio=True
            )
            
        except subprocess.CalledProcessError as e:
            logger.error(f"FFprobe failed for {media_path}: {e}")
            raise
            
    def extract_audio(
        self, 
        media_path: Path, 
        output_path: Optional[Path] = None,
        sample_rate: int = 22050,
        channels: int = 1,
        format_ext: str = "wav"
    ) -> Path:
        """Extract audio from media file.
        
        Args:
            media_path: Input media file path.
            output_path: Output audio file path. Auto-generated if None.
            sample_rate: Target sample rate for extracted audio.
            channels: Target channel count (1 for mono, 2 for stereo).  
            format_ext: Output format extension.
            
        Returns:
            Path to extracted audio file.
            
        Raises:
            subprocess.CalledProcessError: If ffmpeg fails.
        """
        if output_path is None:
            filename = f"{media_path.stem}_audio_{sample_rate}hz.{format_ext}"
            output_path = self.temp_dir / filename
            
        # Check cache
        if self.cache_enabled and output_path.exists():
            logger.info(f"Using cached audio: {output_path}")
            return output_path
            
        # Find ffmpeg executable
        try:
            ffmpeg_path = find_ffmpeg_executable("ffmpeg")
        except FileNotFoundError as e:
            logger.error(str(e))
            raise
            
        # Extract audio with ffmpeg
        cmd = [
            ffmpeg_path,
            "-i", str(media_path),
            "-ac", str(channels),  # audio channels
            "-ar", str(sample_rate),  # audio rate
            "-y",  # overwrite output
            str(output_path)
        ]
        
        try:
            logger.info(f"Extracting audio: {media_path} -> {output_path}")
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                check=True
            )
            logger.debug(f"FFmpeg output: {result.stderr}")
            return output_path
            
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg failed: {e}")
            logger.error(f"FFmpeg stderr: {e.stderr}")
            raise
            
    def cleanup_temp_files(self) -> None:
        """Remove temporary files created during processing."""
        if not self.temp_dir.exists():
            return
            
        for temp_file in self.temp_dir.glob("*"):
            if temp_file.is_file():
                temp_file.unlink()
                logger.debug(f"Removed temp file: {temp_file}")