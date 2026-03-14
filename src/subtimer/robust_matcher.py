"""Robust audio fingerprinting matcher using landmark-based hashing.

Implements a Shazam-like audio fingerprinting approach for reliable matching
even with encoding differences and noise.
"""

import logging
import hashlib
from dataclasses import dataclass
from typing import List, Tuple, Dict, Set
from collections import defaultdict

import numpy as np
import librosa
from scipy.signal import find_peaks

from .matcher import MatchConfig, MatchCandidate
from .alignment_map import AlignmentRegion, RegionType

logger = logging.getLogger(__name__)


@dataclass 
class AudioLandmark:
    """A spectral landmark for fingerprinting."""
    time_frame: int
    frequency_bin: int
    magnitude: float


@dataclass
class FingerprintHash:
    """A fingerprint hash pair."""
    hash_value: int
    time_offset: int
    source: str  # 'dvd' or 'tv'


class RobustFingerprintMatcher:
    """Robust audio matcher using spectral landmark fingerprinting."""
    
    def __init__(self, config: MatchConfig = MatchConfig()):
        """Initialize fingerprint matcher.
        
        Args:
            config: Matching configuration.
        """
        self.config = config
        self.landmark_density = 5  # landmarks per second
        self.hash_fan_value = 15   # number of points to pair with each anchor
        self.hash_time_delta = 200 # max time delta for hash pairs (frames)
        
    def find_matches(
        self,
        dvd_audio: np.ndarray,
        tv_audio: np.ndarray,
        sample_rate: int
    ) -> List[AlignmentRegion]:
        """Find matching regions using robust fingerprinting.
        
        Args:
            dvd_audio: DVD audio data.
            tv_audio: TV audio data.
            sample_rate: Audio sample rate.
            
        Returns:
            List of matched alignment regions.
        """
        logger.info(f"Robust fingerprint matching: DVD {dvd_audio.shape}, TV {tv_audio.shape}")
        
        # Extract landmarks from both audio tracks
        logger.info("Extracting DVD landmarks...")
        dvd_landmarks = self._extract_landmarks(dvd_audio, sample_rate)
        logger.info(f"Found {len(dvd_landmarks)} DVD landmarks")
        
        logger.info("Extracting TV landmarks...")  
        tv_landmarks = self._extract_landmarks(tv_audio, sample_rate)
        logger.info(f"Found {len(tv_landmarks)} TV landmarks")
        
        if len(dvd_landmarks) < 10 or len(tv_landmarks) < 10:
            logger.warning("Insufficient landmarks for matching")
            return []
            
        # Generate fingerprint hashes
        logger.info("Computing fingerprint hashes...")
        dvd_hashes = self._generate_hashes(dvd_landmarks, 'dvd')
        tv_hashes = self._generate_hashes(tv_landmarks, 'tv')
        
        logger.info(f"Generated {len(dvd_hashes)} DVD hashes, {len(tv_hashes)} TV hashes")
        
        # Find matching hashes and align
        matches = self._find_hash_matches(dvd_hashes, tv_hashes, sample_rate)
        
        # Convert to alignment regions
        regions = self._matches_to_regions(matches, sample_rate)
        
        logger.info(f"Found {len(regions)} robust fingerprint regions")
        return regions
        
    def _extract_landmarks(self, audio: np.ndarray, sample_rate: int) -> List[AudioLandmark]:
        """Extract spectral landmarks from audio.
        
        Args:
            audio: Audio data.
            sample_rate: Sample rate.
            
        Returns:
            List of spectral landmarks.
        """
        # Compute spectrogram
        hop_length = 512
        n_fft = 2048
        
        stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        magnitude = np.abs(stft)
        
        # Convert to dB and apply nonlinear scaling for landmark detection
        magnitude_db = librosa.amplitude_to_db(magnitude, ref=np.max)
        
        landmarks = []
        
        # Extract landmarks by finding local maxima in frequency-time
        # Use a more sophisticated approach than simple peak detection
        for t_frame in range(magnitude_db.shape[1]):
            # Get spectrum at this time frame
            spectrum = magnitude_db[:, t_frame]
            
            # Find peaks in frequency domain 
            peaks, properties = find_peaks(
                spectrum,
                height=-60,  # Minimum dB threshold
                distance=10, # Minimum frequency separation
                prominence=3 # Minimum prominence above neighbors
            )
            
            # Take strongest peaks as landmarks
            if len(peaks) > 0:
                peak_strengths = spectrum[peaks]
                # Sort by strength and take top landmarks
                top_indices = np.argsort(peak_strengths)[-self.landmark_density:]
                
                for idx in top_indices:
                    if peak_strengths[idx] > -50:  # Quality threshold
                        landmark = AudioLandmark(
                            time_frame=t_frame,
                            frequency_bin=peaks[idx], 
                            magnitude=peak_strengths[idx]
                        )
                        landmarks.append(landmark)
                        
        return landmarks
        
    def _generate_hashes(self, landmarks: List[AudioLandmark], source: str) -> List[FingerprintHash]:
        """Generate fingerprint hashes from landmarks.
        
        Args:
            landmarks: List of landmarks.
            source: Source identifier ('dvd' or 'tv').
            
        Returns:
            List of fingerprint hashes.
        """
        hashes = []
        
        # Sort landmarks by time for efficient pairing
        landmarks_sorted = sorted(landmarks, key=lambda l: l.time_frame)
        
        for i, anchor in enumerate(landmarks_sorted):
            # Find landmarks within hash fan range
            for j in range(i + 1, min(i + self.hash_fan_value + 1, len(landmarks_sorted))):
                target = landmarks_sorted[j]
                
                # Check time delta constraint
                time_delta = target.time_frame - anchor.time_frame
                if time_delta > self.hash_time_delta:
                    break
                    
                # Create hash from landmark pair
                hash_input = f"{anchor.frequency_bin}|{target.frequency_bin}|{time_delta}"
                hash_value = int(hashlib.sha1(hash_input.encode()).hexdigest()[:8], 16)
                
                fingerprint_hash = FingerprintHash(
                    hash_value=hash_value,
                    time_offset=anchor.time_frame,
                    source=source
                )
                hashes.append(fingerprint_hash)
                
        return hashes
        
    def _find_hash_matches(
        self,
        dvd_hashes: List[FingerprintHash],
        tv_hashes: List[FingerprintHash], 
        sample_rate: int
    ) -> List[Tuple[int, int, int]]:
        """Find matching hash pairs between DVD and TV.
        
        Args:
            dvd_hashes: DVD fingerprint hashes.
            tv_hashes: TV fingerprint hashes.
            sample_rate: Audio sample rate.
            
        Returns:
            List of (dvd_time, tv_time, match_count) tuples.
        """
        # Build hash lookup table for TV
        tv_hash_map = defaultdict(list)
        for tv_hash in tv_hashes:
            tv_hash_map[tv_hash.hash_value].append(tv_hash.time_offset)
            
        # Find matching hashes and compute time alignments
        time_alignments = defaultdict(int)
        
        for dvd_hash in dvd_hashes:
            if dvd_hash.hash_value in tv_hash_map:
                for tv_time in tv_hash_map[dvd_hash.hash_value]:
                    # Compute time difference (potential alignment)
                    time_diff = tv_time - dvd_hash.time_offset
                    time_alignments[time_diff] += 1
                    
        # Find most common time alignments (indicating good matches)
        if not time_alignments:
            return []
            
        # Sort by match count and filter strong matches
        sorted_alignments = sorted(
            time_alignments.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        matches = []
        min_matches = max(10, len(dvd_hashes) * 0.01)  # At least 1% hash matches
        
        for time_diff, match_count in sorted_alignments:
            if match_count >= min_matches:
                # Convert frame difference to time alignment
                hop_length = 512
                dvd_time = 0  # Start of potential region
                tv_time = time_diff * hop_length / sample_rate
                
                matches.append((dvd_time, tv_time, match_count))
                
        return matches
        
    def _matches_to_regions(
        self,
        matches: List[Tuple[int, int, int]], 
        sample_rate: int
    ) -> List[AlignmentRegion]:
        """Convert hash matches to alignment regions.
        
        Args:
            matches: List of time alignment matches.
            sample_rate: Audio sample rate.
            
        Returns:
            List of alignment regions.
        """
        regions = []
        
        for dvd_time, tv_time, match_count in matches:
            # Estimate region duration from match density
            duration = min(60.0, max(10.0, match_count * 2))  # 10-60 second regions
            
            # Confidence based on match count (normalized)
            max_possible_matches = 100  # Rough estimate
            confidence = min(1.0, match_count / max_possible_matches)
            
            # Skip very low confidence matches
            if confidence < self.config.min_correlation:
                continue
                
            # Calculate speed ratio (assume 1.0 for fingerprint matches)
            speed_ratio = 1.0
            
            region = AlignmentRegion(
                dvd_start=max(0, dvd_time),
                dvd_end=dvd_time + duration,
                tv_start=max(0, tv_time), 
                tv_end=tv_time + duration,
                offset_seconds=tv_time - dvd_time,
                speed_ratio=speed_ratio,
                confidence=confidence,
                region_type=RegionType.MATCHED
            )
            regions.append(region)
            
        return regions