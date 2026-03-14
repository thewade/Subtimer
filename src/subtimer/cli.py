"""Command-line interface for the subtitle retiming tool."""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import click

from . import __version__
from .media_io import MediaProcessor 
from .audio_prep import AudioPreprocessor, AudioConfig
from .matcher import AudioMatcher, MatchConfig
from .robust_matcher import RobustFingerprintMatcher
from .hint_guided_matcher import HintGuidedMatcher 
from .refiner import AlignmentRefiner
from .alignment_map import AlignmentMap
from .subtitle_io import SubtitleProcessor
from .retime import SubtitleRetimer
from .report import ReportGenerator, ProcessingMetadata
from .hint_loader import load_hints_file, validate_alignment_against_hints

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.command()
@click.argument('dvd_file', type=click.Path(exists=True, path_type=Path))
@click.argument('tv_file', type=click.Path(exists=True, path_type=Path)) 
@click.argument('subtitle_file', type=click.Path(exists=True, path_type=Path))
@click.option('-o', '--output-dir', type=click.Path(path_type=Path), default=Path.cwd(),
              help='Output directory for results')
@click.option('--output-srt', type=click.Path(path_type=Path), 
              help='Output SRT filename (default: input_retimed.srt)')
@click.option('--temp-dir', type=click.Path(path_type=Path),
              help='Temporary directory for working files')
@click.option('--cache/--no-cache', default=True,
              help='Enable/disable audio caching')
@click.option('--sample-rate', type=int, default=22050,
              help='Audio sample rate for processing')
@click.option('--chunk-duration', type=float, default=30.0,
              help='Chunk duration for matching (seconds)')
@click.option('--min-correlation', type=float, default=0.3,
              help='Minimum correlation threshold for matches')
@click.option('--min-confidence', type=float, default=0.4,
              help='Minimum confidence for retiming')
@click.option('--drop-tv-only/--keep-tv-only', default=True,
              help='Drop subtitles in TV-only regions')
@click.option('--drop-unmatched/--keep-unmatched', default=True, 
              help='Drop subtitles in unmatched regions')
@click.option('--flag-boundaries/--no-flag-boundaries', default=True,
              help='Flag subtitles crossing region boundaries')
@click.option('--split-boundaries/--no-split-boundaries', default=False,
              help='Split subtitles at region boundaries')
@click.option('--matcher', type=click.Choice(['basic', 'hints', 'robust'], case_sensitive=False), 
              default='hints', help='Audio matching algorithm to use')
@click.option('--hints', type=click.Path(exists=True, path_type=Path), 
              help='Path to hints YAML file (optional, auto-detects hints.yaml in subtitle dir if not provided)')
@click.option('--verbose', '-v', is_flag=True,
              help='Enable verbose logging')
@click.option('--debug', is_flag=True,
              help='Enable debug logging')
@click.version_option(version=__version__)
def main(
    dvd_file: Path,
    tv_file: Path,
    subtitle_file: Path,
    output_dir: Path,
    output_srt: Optional[Path],
    temp_dir: Optional[Path],
    cache: bool,
    sample_rate: int,
    chunk_duration: float,
    min_correlation: float,
    min_confidence: float,
    drop_tv_only: bool,
    drop_unmatched: bool,
    flag_boundaries: bool,
    split_boundaries: bool,
    matcher: str,
    hints: Optional[Path],
    verbose: bool,
    debug: bool
) -> None:
    """DVD Subtitle Retimer - Align subtitles using audio matching.
    
    Retimes SRT subtitles from TV recording timeline to DVD timeline using
    audio alignment between DVD and TV sources.
    
    Arguments:
        DVD_FILE: DVD media file or extracted audio
        TV_FILE: TV media file or extracted audio  
        SUBTITLE_FILE: TV-timed SRT subtitle file
    """
    # Configure logging level
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
    elif verbose:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.WARNING)
        
    start_time = datetime.now()
    
    try:
        # Setup directories and paths
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if output_srt is None:
            output_srt = output_dir / f"{subtitle_file.stem}_retimed.srt"
        elif not output_srt.is_absolute():
            output_srt = output_dir / output_srt
            
        alignment_json = output_dir / "alignment.json"
        summary_txt = output_dir / "summary.txt"
        
        if temp_dir:
            temp_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"DVD Subtitle Retimer v{__version__}")
        logger.info(f"DVD: {dvd_file}")
        logger.info(f"TV: {tv_file}")
        logger.info(f"Subtitles: {subtitle_file}")
        logger.info(f"Output: {output_srt}")
        
        warnings = []
        
        # Load hints file - explicit parameter or auto-detect for hints matcher
        hints_obj = None
        if hints:
            # Explicit hints file provided
            hints_obj = load_hints_file(hints)
            logger.info(f"Loaded timing hints from {hints} for {hints_obj.episode_id}")
        elif matcher == 'hints':
            # Auto-detect hints.yaml only when using hints matcher
            auto_hints_path = Path(subtitle_file).parent / 'hints.yaml'
            if auto_hints_path.exists():
                hints_obj = load_hints_file(auto_hints_path)
                logger.info(f"Auto-detected timing hints from {auto_hints_path} for {hints_obj.episode_id}")
            else:
                logger.info("No hints file found for hints matcher, will use basic matching")
        else:
            logger.info(f"Using {matcher} matcher - no hints file loading")
        
        # Step 1: Media I/O and Audio Extraction
        logger.info("Step 1: Processing media files")
        media_processor = MediaProcessor(temp_dir=temp_dir, cache_enabled=cache)
        
        try:
            dvd_info = media_processor.probe_media(dvd_file)
            tv_info = media_processor.probe_media(tv_file)
        except Exception as e:
            logger.error(f"Failed to probe media files: {e}")
            sys.exit(1)
            
        logger.info(f"DVD: {dvd_info.duration:.1f}s, {dvd_info.sample_rate}Hz")
        logger.info(f"TV: {tv_info.duration:.1f}s, {tv_info.sample_rate}Hz")
        
        # Extract audio if needed (detect if inputs are already audio)
        if dvd_info.format_name.lower() in ['wav', 'mp3', 'flac', 'aac']:
            dvd_audio_path = dvd_file
        else:
            dvd_audio_path = media_processor.extract_audio(
                dvd_file, sample_rate=sample_rate, channels=1
            )
            
        if tv_info.format_name.lower() in ['wav', 'mp3', 'flac', 'aac']:
            tv_audio_path = tv_file
        else:
            tv_audio_path = media_processor.extract_audio(
                tv_file, sample_rate=sample_rate, channels=1
            )
            
        # Step 2: Audio Preprocessing
        logger.info("Step 2: Preprocessing audio")
        audio_config = AudioConfig(sample_rate=sample_rate, channels=1)
        audio_processor = AudioPreprocessor(config=audio_config)
        
        working_dir = temp_dir or output_dir / "temp"
        working_dir.mkdir(exist_ok=True)
        
        try:
            dvd_working, tv_working = audio_processor.prepare_audio_pair(
                dvd_audio_path, tv_audio_path, working_dir
            )
        except Exception as e:
            logger.error(f"Audio preprocessing failed: {e}")
            sys.exit(1)
            
        # Step 3: Audio Matching
        logger.info("Step 3: Finding audio matches")
        match_config = MatchConfig(
            chunk_duration=chunk_duration,
            min_correlation=min_correlation
        )
        
        # Select matcher based on user preference
        if matcher == 'robust':
            matcher_obj = RobustFingerprintMatcher(config=match_config)
            logger.info("Using robust fingerprint matching")
        elif matcher == 'hints' and hints_obj:
            matcher_obj = HintGuidedMatcher(config=match_config, hints=hints_obj)
            logger.info("Using hint-guided matching")
        elif matcher == 'hints' and not hints_obj:
            logger.warning("Hints matcher requested but no hints file found, falling back to basic matcher")
            matcher_obj = AudioMatcher(config=match_config)
            logger.info("Using standard audio matching")
        else:  # matcher == 'basic' or fallback
            matcher_obj = AudioMatcher(config=match_config)
            logger.info("Using standard audio matching")
        
        try:
            # Load working audio for matching
            dvd_audio_data, dvd_sr = audio_processor.load_audio(dvd_working)
            tv_audio_data, tv_sr = audio_processor.load_audio(tv_working)
            
            # Normalize audio for matching
            dvd_normalized, tv_normalized = audio_processor.normalize_audio(
                (dvd_audio_data, dvd_sr), (tv_audio_data, tv_sr)
            )
            
            coarse_regions = matcher_obj.find_matches(
                dvd_normalized, tv_normalized, sample_rate
            )
            
            logger.info(f"Found {len(coarse_regions)} coarse matches")
            
        except Exception as e:
            logger.error(f"Audio matching failed: {e}")
            sys.exit(1)
            
        if not coarse_regions:
            logger.error("No audio matches found - files may be too different")
            warnings.append("No audio matches found")
            # Create empty alignment map and continue
            alignment_map = AlignmentMap()
        else:
            # Step 4: Refinement
            logger.info("Step 4: Refining alignment")
            
            # TEMPORARY: Skip refiner since it's breaking good correlations
            if matcher == 'hints':
                logger.info("Bypassing refiner to preserve hint-guided matches")
            else:
                logger.info("Bypassing refiner to preserve matcher results")
            alignment_map = AlignmentMap(coarse_regions)
            
            # Validate against hints only if using hint-guided matcher
            if hints_obj and matcher == 'hints':
                hint_warnings = validate_alignment_against_hints(
                    alignment_map.regions, hints_obj, tolerance_seconds=5.0
                )
                if hint_warnings:
                    logger.warning(f"Hint validation found {len(hint_warnings)} issues")
                    warnings.extend([f"Hint validation: {w}" for w in hint_warnings])
                else:
                    logger.info("Alignment matches timing hints well")
        # Step 5: Load Subtitles
        logger.info("Step 5: Loading subtitles")
        subtitle_processor = SubtitleProcessor()
        
        try:
            subtitles = subtitle_processor.load_subtitles(subtitle_file)
            logger.info(f"Loaded {len(subtitles)} subtitle cues")
            
            # Validate subtitles
            issues = subtitle_processor.validate_subtitles(subtitles)
            if issues:
                for issue in issues[:5]:  # Show first 5 issues
                    warnings.append(f"Subtitle issue: {issue}")
                if len(issues) > 5:
                    warnings.append(f"... and {len(issues) - 5} more subtitle issues")
                    
        except Exception as e:
            logger.error(f"Failed to load subtitles: {e}")
            sys.exit(1)
            
        # Step 6: Subtitle Retiming
        logger.info("Step 6: Retiming subtitles")
        retimer = SubtitleRetimer(
            drop_tv_only=drop_tv_only,
            drop_unmatched=drop_unmatched,
            flag_boundary_crossings=flag_boundaries,
            split_boundary_crossings=split_boundaries,
            min_confidence_threshold=min_confidence
        )
        
        try:
            retiming_summary = retimer.retime_subtitles(subtitles, alignment_map)
            retimed_cues = retimer.get_retimed_cues(retiming_summary)
            
            logger.info(f"Retimed {len(retimed_cues)} subtitle cues")
            
        except Exception as e:
            logger.error(f"Subtitle retiming failed: {e}")
            sys.exit(1)
            
        # Step 7: Save Results
        logger.info("Step 7: Saving results")
        
        # Save retimed SRT
        if retimed_cues:
            subtitle_processor.save_subtitles(retimed_cues, output_srt)
        else:
            logger.warning("No subtitles were successfully retimed")
            warnings.append("No subtitles were successfully retimed")
            # Create empty SRT file
            output_srt.write_text("")
            
        # Generate reports
        processing_time = (datetime.now() - start_time).total_seconds()
        metadata = ProcessingMetadata(
            dvd_file=str(dvd_file),
            tv_file=str(tv_file),
            subtitle_file=str(subtitle_file),
            output_srt=str(output_srt),
            processing_time=f"{processing_time:.1f} seconds",
            tool_version=__version__,
            settings={
                'sample_rate': sample_rate,
                'chunk_duration': chunk_duration,
                'min_correlation': min_correlation,
                'min_confidence': min_confidence,
                'drop_tv_only': drop_tv_only,
                'drop_unmatched': drop_unmatched,
                'flag_boundaries': flag_boundaries,
                'split_boundaries': split_boundaries
            }
        )
        
        report_generator = ReportGenerator(__version__)
        
        try:
            report_generator.generate_alignment_json(
                alignment_map, retiming_summary, metadata, alignment_json, warnings
            )
            
            report_generator.generate_summary_report(
                alignment_map, retiming_summary, metadata, summary_txt, warnings
            )
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            warnings.append("Report generation failed")
            
        # Cleanup temporary files if not debugging
        if not debug:
            try:
                media_processor.cleanup_temp_files()
            except Exception as e:
                logger.warning(f"Cleanup failed: {e}")
                
        # Final summary
        logger.info(f"Processing complete in {processing_time:.1f}s")
        logger.info(f"Results saved to: {output_dir}")
        
        if warnings:
            logger.warning("Processing completed with warnings:")
            for warning in warnings:
                logger.warning(f"  - {warning}")
                
        if len(retimed_cues) == 0:
            logger.error("No subtitles were successfully retimed - check alignment quality")
            sys.exit(1)
        elif retiming_summary.retimed_count < retiming_summary.total_cues * 0.5:
            logger.warning(
                f"Only {retiming_summary.retimed_count}/{retiming_summary.total_cues} "
                f"subtitles were retimed - check alignment quality"
            )
            
        logger.info("Success!")
        
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()