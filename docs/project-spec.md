# DVD Subtitle Retiming Project Spec

## Goal

Build a Python CLI tool that uses **DVD audio as the reference timeline** to retime an **SRT subtitle file from a TV recording** onto the **DVD version**.

The TV recording may contain:
- commercials or other inserted material
- missing or censored sections
- slight speed differences or drift
- different audio encoding, gain, or noise

The tool must produce a **piecewise time map** between DVD and TV audio, then use that map to convert the TV-timed SRT into a DVD-timed SRT.

---

## Primary Deliverable

A Python command-line tool that takes:
- a DVD media file or extracted DVD audio
- a TV media file or extracted TV audio
- a TV-timed `.srt`

and outputs:
- a DVD-timed `.srt`
- an `alignment.json` report
- a human-readable summary report

---

## Functional Requirements

### 1. Inputs
Support these inputs:
- DVD media file or extracted audio
- TV media file or extracted audio
- TV subtitle file in `.srt`

Initial formats:
- media: anything FFmpeg can decode
- working audio: mono WAV
- subtitles: SRT only

### 2. Audio Preparation
The tool must:
- extract audio with FFmpeg when given media files
- normalize both sources to the same working format
- downmix to mono
- resample to a consistent sample rate
- save or cache working audio files

### 3. Alignment
The tool must align **DVD time -> TV time** using audio.

It must support:
- multiple matched regions
- unmatched gaps
- commercials or inserted TV-only spans
- slight speed differences
- slight timing drift

It must **not** assume a single global offset.

### 4. Matching Strategy
Implementation may use:
- fingerprint-based matching
- feature-based matching
- DTW-based refinement
- a hybrid approach

Requirements for the strategy:
- tolerate encoding differences and noise
- tolerate inserted commercials
- tolerate moderate speed changes
- produce ordered matched regions
- refine boundaries beyond crude fixed chunk edges

### 5. Time Map
Represent alignment as ordered regions.
Each region must include:
- `dvd_start`
- `dvd_end`
- `tv_start`
- `tv_end`
- `offset_seconds`
- `speed_ratio`
- `confidence`
- `region_type`

Preferred internal model:
- linear mapping per region
- `tv_time = a * dvd_time + b`

### 6. Commercial / Gap Detection
The tool must identify likely TV-only inserted spans by detecting unmatched gaps between matched regions.

These gaps must be marked in the JSON output.

### 7. Speed Difference Detection
The tool must detect whether the TV source:
- has only a constant offset
- has linear drift suggesting speed difference
- has abrupt jumps caused by edits
- has a mix of drift and jumps

### 8. Subtitle Retiming
The tool must parse the TV-timed SRT and retime cues onto the DVD timeline.

Handling rules:
- cue fully inside one matched region -> retime normally
- cue fully inside a TV-only unmatched region -> drop by default and report it
- cue crossing a region boundary -> flag for review; optional split later
- cue in low-confidence region -> flag for review

### 9. Outputs
The tool must output:
- retimed SRT
- alignment JSON
- summary report

`alignment.json` must contain:
- input file metadata
- processing settings
- matched regions
- unmatched regions
- inferred commercials / inserted spans
- warnings
- dropped or flagged subtitle entries

---

## Non-Goals for Initial Version

Do not implement yet:
- GUI
- OCR of DVD subtitle images
- ASS/SSA subtitle support
- subtitle text rewriting
- scene reordering support
- batch processing across many episodes

---

## Suggested Project Structure

```text
project/
  src/
    media_io.py
    audio_prep.py
    matcher.py
    refiner.py
    alignment_map.py
    subtitle_io.py
    retime.py
    report.py
    cli.py
  tests/
    fixtures/
    test_subtitle_io.py
    test_alignment_map.py
    test_retime.py
  docs/
    project_spec.md
  pyproject.toml
  README.md
```

The agent may change names, but responsibilities should stay similar.

---

## Module Responsibilities

### `media_io.py`
- probe inputs
- extract audio with FFmpeg
- manage temp/cache files

### `audio_prep.py`
- downmix
- resample
- normalize working audio format

### `matcher.py`
- discover coarse matching regions
- score candidate matches
- reject low-quality matches

### `refiner.py`
- refine local region boundaries
- estimate local offset and speed ratio

### `alignment_map.py`
- store ordered mapping regions
- merge compatible adjacent regions
- label unmatched gaps
- expose time conversion helpers

### `subtitle_io.py`
- parse SRT
- write SRT
- represent subtitle cues internally

### `retime.py`
- transform subtitle times from TV -> DVD
- drop or flag unsupported cues

### `report.py`
- write JSON report
- write summary text/markdown

### `cli.py`
- parse arguments
- orchestrate pipeline
- return clear exit codes

---

## CLI Requirements

The CLI must support a command conceptually like:

```bash
python -m app \
  --dvd path/to/dvd.mkv \
  --tv path/to/tv.mkv \
  --subs path/to/tv.srt \
  --out-srt path/to/dvd.srt \
  --out-json path/to/alignment.json
```

Exact syntax may vary.

Required options:
- DVD input
- TV input
- subtitle input
- output SRT path
- output JSON path

Optional options:
- reuse pre-extracted audio
- debug logging
- sample rate override
- confidence threshold
- unmatched cue policy

---

## Milestones

### Milestone 1: Skeleton
Implement:
- CLI scaffolding
- SRT parser/writer
- FFmpeg audio extraction
- normalized working audio generation
- JSON/report scaffolding

### Milestone 2: Coarse Matching
Implement:
- first-pass audio matching
- candidate matched region discovery
- initial confidence scoring

### Milestone 3: Refinement
Implement:
- boundary refinement
- local offset estimation
- local speed ratio estimation
- ordered region assembly
- unmatched gap labeling

### Milestone 4: Subtitle Retiming
Implement:
- cue retiming using the region map
- dropped cue reporting
- flagged cue reporting

### Milestone 5: Validation
Implement tests and sample reports for:
- constant offset only
- commercials inserted in TV
- speed difference
- cuts plus drift

---

## Acceptance Criteria

The project is successful when it can:

1. Read DVD input, TV input, and TV SRT.
2. Build a non-empty piecewise alignment map.
3. Detect major matched program regions.
4. Detect major TV-only inserted spans.
5. Produce a usable DVD-timed SRT.
6. Drop or flag subtitles that belong only to inserted TV material.
7. Report low-confidence regions for manual review.

Failure cases:
- forcing the entire episode into one constant offset despite obvious cuts
- keeping commercial-only subtitles without warning
- large drift inside clearly matched regions with no report

---

## Minimum Test Scenarios

Implement tests or fixtures for:

1. same content, constant offset only
2. TV version with inserted commercials
3. same content with slight speed difference
4. cuts plus speed difference
5. noisy or weak audio
6. repeated music causing possible false matches
7. subtitle cue crossing a region boundary

---

## Coding Notes for the Agent

- Prioritize correctness and inspectability over premature optimization.
- Keep the alignment backend replaceable.
- Keep region mapping explicit and serializable.
- Do not hardcode a single matching method into unrelated modules.
- Make subtitle retiming logic independent from the audio matcher.
- Preserve intermediate artifacts when debug mode is enabled.

---

## First Implementation Expectation

The first implementation does **not** need to be perfect.
It should be able to:
- discover coarse shared regions
- infer major inserted TV gaps
- retime most subtitle cues correctly
- clearly report anything uncertain

That is sufficient for an initial usable version.
