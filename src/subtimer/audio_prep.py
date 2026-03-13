"""Audio preprocessing operations.

Handles downmix, resample, and normalization to prepare audio for alignment.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import librosa
import numpy as np
from scipy.io import wavfile

logger = logging.getLogger(__name__)


@dataclass
class AudioConfig:
    """Configuration for audio processing."""
    sample_rate: int = 22050
    channels: int = 1  # mono
    dtype: str = "float32"
    normalize: bool = True


class AudioPreprocessor:
    """Preprocesses audio files for alignment."""
    
    def __init__(self, config: AudioConfig = AudioConfig()):
        """Initialize audio preprocessor.
        
        Args:
            config: Audio processing configuration.
        """
        self.config = config
        
    def load_audio(self, audio_path: Path) -> Tuple[np.ndarray, int]:
        """Load audio file.
        
        Args:
            audio_path: Path to audio file.
            
        Returns:
            Tuple of (audio_data, sample_rate).
            
        Raises:
            FileNotFoundError: If audio file doesn't exist.
            ValueError: If audio format is not supported.
        """
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
        try:
            # Use librosa for robust audio loading
            audio, sr = librosa.load(
                str(audio_path),
                sr=None,  # Keep original sample rate initially
                mono=False,  # Keep all channels initially
                dtype=np.float32
            )
            logger.info(f"Loaded audio: {audio_path} ({sr} Hz, {audio.shape})")
            return audio, sr
            
        except Exception as e:
            logger.error(f"Failed to load audio {audio_path}: {e}")
            raise ValueError(f"Unsupported audio format") from e
            
    def normalize_audio(
        self, 
        dvd_audio: Tuple[np.ndarray, int],
        tv_audio: Tuple[np.ndarray, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Normalize both audio sources to working format.
        
        Args:
            dvd_audio: DVD audio (data, sample_rate).
            tv_audio: TV audio (data, sample_rate).
            
        Returns:
            Tuple of normalized (dvd_data, tv_data) both at target config.
        """
        dvd_data, dvd_sr = dvd_audio
        tv_data, tv_sr = tv_audio
        
        logger.info("Normalizing audio to working format")
        
        # Process DVD audio
        dvd_normalized = self._process_single_audio(dvd_data, dvd_sr)
        
        # Process TV audio  
        tv_normalized = self._process_single_audio(tv_data, tv_sr)
        
        logger.info(
            f"Normalized audio shapes: DVD={dvd_normalized.shape}, "
            f"TV={tv_normalized.shape}"
        )
        
        return dvd_normalized, tv_normalized
        
    def _process_single_audio(
        self, 
        audio_data: np.ndarray, 
        sample_rate: int
    ) -> np.ndarray:
        """Process single audio stream to working format.
        
        Args:
            audio_data: Input audio array.
            sample_rate: Input sample rate.
            
        Returns:
            Processed audio array matching config.
        """
        # Handle multi-dimensional arrays
        if audio_data.ndim > 1:
            # Convert to mono by averaging channels
            if self.config.channels == 1:
                audio_data = np.mean(audio_data, axis=0)
            elif audio_data.shape[0] > self.config.channels:
                # Downmix by taking first N channels
                audio_data = audio_data[:self.config.channels]
            
        # Ensure it's 1D for mono
        if self.config.channels == 1 and audio_data.ndim > 1:
            audio_data = audio_data.flatten()
            
        # Resample if needed
        if sample_rate != self.config.sample_rate:
            logger.debug(f"Resampling from {sample_rate} to {self.config.sample_rate}")
            audio_data = librosa.resample(
                audio_data, 
                orig_sr=sample_rate, 
                target_sr=self.config.sample_rate
            )
            
        # Normalize amplitude if requested
        if self.config.normalize:
            # RMS normalization to prevent clipping
            rms = np.sqrt(np.mean(audio_data ** 2))
            if rms > 0:
                audio_data = audio_data / (rms * 3)  # Conservative scaling
                audio_data = np.clip(audio_data, -1.0, 1.0)
                
        return audio_data.astype(self.config.dtype)
        
    def save_working_audio(
        self, 
        audio_data: np.ndarray, 
        output_path: Path
    ) -> None:
        """Save processed audio to working file.
        
        Args:
            audio_data: Processed audio data.
            output_path: Output file path.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to int16 for WAV output
        if audio_data.dtype == np.float32:
            audio_int16 = (audio_data * 32767).astype(np.int16)
        else:
            audio_int16 = audio_data.astype(np.int16)
            
        wavfile.write(
            str(output_path),
            self.config.sample_rate,
            audio_int16
        )
        
        logger.info(f"Saved working audio: {output_path}")
        
    def prepare_audio_pair(
        self,
        dvd_path: Path,
        tv_path: Path,
        output_dir: Path
    ) -> Tuple[Path, Path]:
        """Prepare both audio files for alignment.
        
        Args:
            dvd_path: DVD audio file path.
            tv_path: TV audio file path.
            output_dir: Directory for working audio files.
            
        Returns:
            Tuple of (dvd_working_path, tv_working_path).
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load both audio files
        dvd_audio = self.load_audio(dvd_path)
        tv_audio = self.load_audio(tv_path)
        
        # Normalize to working format
        dvd_normalized, tv_normalized = self.normalize_audio(dvd_audio, tv_audio)
        
        # Save working files
        dvd_working_path = output_dir / "dvd_working.wav"
        tv_working_path = output_dir / "tv_working.wav"
        
        self.save_working_audio(dvd_normalized, dvd_working_path)
        self.save_working_audio(tv_normalized, tv_working_path)
        
        return dvd_working_path, tv_working_path