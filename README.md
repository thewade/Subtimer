# DVD Subtitle Retimer

A Python CLI tool that uses audio alignment to retime SRT subtitles from TV recordings onto DVD timelines.

## Overview

This tool solves the problem of subtitle timing mismatches when you have:
- DVD media files with accurate timing
- TV recording with subtitles but different timing (due to commercials, speed differences, etc.)

The tool analyzes the audio from both sources to create a piecewise time mapping, then retimes the TV subtitles to match the DVD timeline.

## Features

- **Audio-based alignment**: Uses robust audio fingerprinting to match DVD and TV content
- **Handles complex timing differences**: 
  - Commercial insertions
  - Speed differences (PAL speedup, etc.)
  - Missing or censored sections
  - Drift and timing variations
- **Intelligent subtitle processing**:
  - Retimes matched content accurately  
  - Drops subtitles in TV-only regions (commercials)
  - Flags uncertain matches for review
  - Optionally splits subtitles at region boundaries
- **Comprehensive reporting**: 
  - Detailed JSON alignment report
  - Human-readable summary
  - Lists dropped and flagged subtitles

## Installation

### Requirements
- Python 3.12 or higher
- FFmpeg (for audio extraction)

### Install from source
```bash
git clone <repository-url>
cd subtimer
pip install -e .
```

### Install dependencies
```bash
pip install -e .[dev]  # Include development dependencies
```

## Usage

### Basic usage
```bash
subtimer dvd_file.mkv tv_file.mkv subtitles.srt
```

### With custom output location
```bash
subtimer dvd_file.mkv tv_file.mkv subtitles.srt -o output_directory/
```

### Advanced options
```bash
subtimer dvd_file.mkv tv_file.mkv subtitles.srt \
  --output-srt retimed_subtitles.srt \
  --sample-rate 22050 \
  --chunk-duration 30.0 \
  --min-correlation 0.3 \
  --min-confidence 0.4 \
  --drop-tv-only \
  --flag-boundaries \
  --verbose
```

## Input Formats

### Media Files
Supports any format that FFmpeg can decode:
- Video: MKV, MP4, AVI, MOV, etc.
- Audio: WAV, MP3, FLAC, AAC, etc.

### Subtitles  
- SRT format only (initial version)
- UTF-8 encoding recommended

## Output Files

The tool generates three output files:

1. **Retimed SRT** (`*_retimed.srt`): The main output with DVD-timed subtitles
2. **Alignment JSON** (`alignment.json`): Detailed alignment data and processing results
3. **Summary Report** (`summary.txt`): Human-readable processing summary

## How It Works

1. **Audio Extraction**: Extracts and normalizes audio from both DVD and TV sources
2. **Feature Matching**: Uses MFCC and spectral features to find matching audio segments  
3. **Coarse Alignment**: Identifies candidate matching regions using cross-correlation
4. **Refinement**: Improves boundary precision using onset detection and optimization
5. **Time Mapping**: Creates piecewise linear mapping between DVD and TV timelines
6. **Subtitle Retiming**: Applies time mapping to convert subtitle timestamps
7. **Quality Control**: Flags uncertain matches and handles special cases

## Command Line Options

### Input/Output
- `DVD_FILE`: DVD media file or extracted audio
- `TV_FILE`: TV media file or extracted audio  
- `SUBTITLE_FILE`: TV-timed SRT subtitle file
- `-o, --output-dir`: Output directory (default: current directory)
- `--output-srt`: Custom output SRT filename
- `--temp-dir`: Temporary directory for working files

### Audio Processing
- `--sample-rate`: Audio sample rate for processing (default: 22050)
- `--cache/--no-cache`: Enable/disable audio caching (default: enabled)

### Matching Parameters
- `--chunk-duration`: Chunk size for matching in seconds (default: 30.0)
- `--min-correlation`: Minimum correlation threshold (default: 0.3)
- `--min-confidence`: Minimum confidence for retiming (default: 0.4)

### Subtitle Handling
- `--drop-tv-only/--keep-tv-only`: Drop subtitles in TV-only regions (default: drop)
- `--drop-unmatched/--keep-unmatched`: Drop subtitles in unmatched regions (default: drop)  
- `--flag-boundaries/--no-flag-boundaries`: Flag boundary-crossing subtitles (default: flag)
- `--split-boundaries/--no-split-boundaries`: Split subtitles at boundaries (default: no split)

### Logging
- `-v, --verbose`: Enable verbose logging
- `--debug`: Enable debug logging and preserve temp files

## Examples

### DVD and TV recordings with commercials
```bash
subtimer movie_dvd.mkv movie_tv_recording.mkv tv_subtitles.srt
```

### Different video files, same content with speed difference
```bash  
subtimer pal_version.mkv ntsc_version.mkv pal_subs.srt --verbose
```

### Already extracted audio files
```bash
subtimer dvd_audio.wav tv_audio.wav subtitles.srt
```

## Troubleshooting

### No matches found
- Check that the files contain the same content
- Try lowering `--min-correlation` threshold
- Ensure audio quality is reasonable
- Check that files aren't heavily compressed or corrupted

### Poor alignment quality
- Increase `--chunk-duration` for longer content
- Adjust `--min-confidence` threshold
- Try different `--sample-rate` (22050 or 16000)

### Many dropped subtitles
- Review alignment quality in `alignment.json`
- Use `--keep-tv-only` and `--keep-unmatched` to preserve more subtitles
- Check `summary.txt` for specific issues

### Performance issues
- Use `--cache` to avoid re-extracting audio
- Lower `--sample-rate` for faster processing
- Process shorter segments if memory is limited

## Development

### Running tests
```bash
pytest
```

### Code formatting
```bash
black src/ tests/
ruff check src/ tests/
```

### Type checking
```bash
mypy src/
```

## Architecture

The codebase is organized into specialized modules:

- `media_io.py`: FFmpeg integration and media file handling
- `audio_prep.py`: Audio preprocessing and normalization  
- `matcher.py`: Coarse audio matching using fingerprints
- `refiner.py`: Boundary refinement and precision improvement
- `alignment_map.py`: Time mapping data structures and conversion
- `subtitle_io.py`: SRT parsing and writing
- `retime.py`: Subtitle retiming logic and special case handling
- `report.py`: JSON and text report generation
- `cli.py`: Command-line interface

## Limitations (v1)

- SRT subtitles only (no ASS/SSA support)
- No GUI (command-line only)
- No subtitle OCR (requires text subtitles)
- No automatic scene reordering detection
- Single-threaded processing

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]