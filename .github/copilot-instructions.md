# Copilot Instructions

Read `docs/project-spec.md` before making changes.

## Project Goals
- Build a Python CLI tool.
- DVD audio is the reference timeline.
- Output: retimed SRT, `alignment.json`, and a summary report.

## Engineering Rules
- Keep audio alignment, region mapping, and subtitle retiming in separate modules.
- Prefer inspectable intermediate artifacts over opaque heuristics.
- Keep the alignment backend replaceable.
- Add or update tests for every behavior change.
- Use clear logging and deterministic outputs where possible.
- Do not add GUI code in v1.
- Do not add subtitle OCR in v1.
- Do not add ASS/SSA support in v1.

## Coding Style
- Use Python 3.12+ style unless the repo specifies otherwise.
- Prefer `pathlib`, `dataclasses`, and type hints.
- Keep functions small and single-purpose.
- Raise clear exceptions with stage-specific context.
- Keep pure mapping logic separate from FFmpeg/media side effects.

## Testing Expectations
- Add pytest coverage for parsing, mapping, and retiming behavior.
- Cover at least: constant offset, inserted commercials, speed drift, and boundary-crossing cues.
- Prefer small deterministic fixtures.

## Workflow
- When starting a task, identify which milestone in `docs/project-spec.md` it belongs to.
- Make the smallest reasonable change that satisfies the task.
- Update docs when behavior or CLI contracts change.
