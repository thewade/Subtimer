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
        self.landmark_density = 3      # balanced selection
        self.hash_fan_value = 8        # moderate hash combinations
        self.hash_time_delta = 150     # balanced time window
        self.chunk_size_sec = 120      # process audio in 2-minute chunks
        
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
        logger.info("Finding hash matches...")
        matches = self._find_hash_matches(dvd_hashes, tv_hashes, sample_rate)
        logger.info(f"Hash matching complete, found {len(matches)} potential alignments")
        
        # Convert to alignment regions
        logger.info("Converting matches to alignment regions...")
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
        # Process every 3rd frame for efficiency while maintaining coverage
        frame_skip = 3
        for t_frame in range(0, magnitude_db.shape[1], frame_skip):
            # Get spectrum at this time frame
            spectrum = magnitude_db[:, t_frame]
            
            # Find peaks in frequency domain with balanced parameters
            peaks, properties = find_peaks(
                spectrum,
                height=-45,  # Balanced threshold - not too restrictive
                distance=15, # Moderate separation
                prominence=6 # Reasonable prominence for quality
            )
            
            # Take strongest peaks as landmarks
            if len(peaks) > 0:
                peak_strengths = spectrum[peaks]
                # Sort by strength and take top landmarks
                top_indices = np.argsort(peak_strengths)[-self.landmark_density:]
                
                for idx in top_indices:
                    if peak_strengths[idx] > -40:  # Reasonable quality threshold
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
                    
                # Create hash from landmark pair with some discrimination
                hash_input = f"{anchor.frequency_bin}|{target.frequency_bin}|{time_delta}"
                hash_value = int(hashlib.sha1(hash_input.encode()).hexdigest()[:10], 16)
                
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
        logger.info("Building TV hash lookup table...")
        tv_hash_map = defaultdict(list)
        for tv_hash in tv_hashes:
            tv_hash_map[tv_hash.hash_value].append(tv_hash.time_offset)
        logger.info(f"Built lookup table with {len(tv_hash_map)} unique hash values")
            
        # Find matching hashes and compute time alignments
        logger.info("Computing time alignments from hash matches...")
        time_alignments = defaultdict(int)
        matches_found = 0
        max_matches_limit = 1000000  # Limit to 1M matches to prevent runaway processing
        
        for i, dvd_hash in enumerate(dvd_hashes):
            if i % 50000 == 0 and i > 0:
                logger.info(f"Processed {i}/{len(dvd_hashes)} DVD hashes, found {matches_found} matches so far")
                
            # Early termination if too many matches found
            if matches_found > max_matches_limit:
                logger.warning(f"Stopping hash matching: exceeded {max_matches_limit} matches limit at {i}/{len(dvd_hashes)} hashes")
                break
                
            if dvd_hash.hash_value in tv_hash_map:
                tv_matches = tv_hash_map[dvd_hash.hash_value]
                # Skip hashes that match too many TV positions (likely noise)
                if len(tv_matches) > 75:  # Balanced threshold
                    continue
                    
                for tv_time in tv_matches:
                    # Compute time difference (potential alignment)
                    time_diff = tv_time - dvd_hash.time_offset
                    time_alignments[time_diff] += 1
                    matches_found += 1
                    
        # Find most common time alignments (indicating good matches)
        logger.info(f"Hash matching complete: found {matches_found} total hash matches")
        logger.info(f"Identified {len(time_alignments)} unique time alignment candidates")
        
        if not time_alignments:
            logger.warning("No time alignments found - no matching hashes")
            return []
            
        # Sort by match count and filter strong matches
        logger.info("Sorting and filtering alignment candidates...")
        sorted_alignments = sorted(
            time_alignments.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Log top candidates for debugging
        logger.info(f"Top 10 alignment candidates:")
        for i, (time_diff, count) in enumerate(sorted_alignments[:10]):
            hop_length = 512
            time_offset_sec = time_diff * hop_length / sample_rate
            logger.info(f"  {i+1}. Time offset: {time_offset_sec:.2f}s, Match count: {count}")
        
        matches = []
        min_matches = max(10, len(dvd_hashes) * 0.01)  # At least 1% hash matches
        logger.info(f"Filtering alignments with minimum {min_matches} matches...")
        
        accepted_count = 0
        for time_diff, match_count in sorted_alignments:
            if match_count >= min_matches:
                # Convert frame difference to time alignment
                hop_length = 512
                # Use actual timing information from the hash matches
                # Find representative DVD and TV times for this alignment
                dvd_times = []
                tv_times = []
                
                for dvd_hash in dvd_hashes:
                    if dvd_hash.hash_value in tv_hash_map:
                        for tv_time_offset in tv_hash_map[dvd_hash.hash_value]:
                            computed_diff = tv_time_offset - dvd_hash.time_offset
                            if abs(computed_diff - time_diff) < 5:  # Allow small tolerance
                                dvd_times.append(dvd_hash.time_offset * hop_length / sample_rate)
                                tv_times.append(tv_time_offset * hop_length / sample_rate)
                
                if dvd_times and tv_times:
                    # Use median times for robustness
                    dvd_time = np.median(dvd_times)
                    tv_time = np.median(tv_times)
                    
                    matches.append((dvd_time, tv_time, match_count))
                    accepted_count += 1
                    
                    time_offset_sec = time_diff * hop_length / sample_rate
                    logger.info(f"  Accepted alignment #{accepted_count}: DVD={dvd_time:.1f}s, TV={tv_time:.1f}s, offset={time_offset_sec:.1f}s, {match_count} matches")
                else:
                    logger.debug(f"  Skipped alignment with {match_count} matches (no valid timepoints found)")
            else:
                break  # Stop at first alignment below threshold
                
        logger.info(f"Accepted {accepted_count} alignments out of {len(sorted_alignments)} candidates")
                
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
        logger.info(f"Converting {len(matches)} time alignments to alignment regions...")
        regions = []
        
        for i, (dvd_time, tv_time, match_count) in enumerate(matches):
            # Estimate region duration from match density
            duration = min(60.0, max(10.0, match_count * 2))  # 10-60 second regions
            
            # Confidence based on match count (normalized)
            max_possible_matches = 100  # Rough estimate
            confidence = min(1.0, match_count / max_possible_matches)
            
            # Skip very low confidence matches
            if confidence < self.config.min_correlation:
                logger.debug(f"  Skipping region {i+1}: confidence {confidence:.3f} below threshold {self.config.min_correlation}")
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
            logger.info(f"  Created region {len(regions)}: DVD {dvd_time:.1f}-{dvd_time + duration:.1f}s, TV {tv_time:.1f}-{tv_time + duration:.1f}s, confidence {confidence:.3f}")
            
        logger.info(f"Region creation complete: {len(regions)} regions created from {len(matches)} alignments")
            
        return regions