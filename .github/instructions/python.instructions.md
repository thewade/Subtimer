---
applyTo: "**/*.py"
---

# Python Instructions

## General
- Use Python 3.12+ idioms.
- Prefer standard library solutions first.
- Use `pathlib.Path` instead of raw string paths where practical.
- Use `dataclasses` for structured domain objects.
- Add type hints for public functions and key internal data structures.
- Keep side-effect-heavy code isolated from pure logic.

## Project Conventions
- Keep media extraction separate from audio matching logic.
- Keep subtitle parsing/writing separate from retiming logic.
- Model alignment as explicit ordered regions, not ad hoc tuples.
- Make time-conversion helpers easy to unit test.

## Error Handling
- Fail fast on invalid inputs.
- Include stage context in exceptions and logs.
- Do not silently drop subtitle cues without reporting them.

## Logging
- Use structured, readable logging.
- Log major stages: extraction, normalization, matching, refinement, retiming, reporting.
- In debug mode, preserve useful intermediate values and artifacts.

## Dependencies
- Minimize heavy dependencies.
- If adding a non-trivial dependency, justify it in code comments or docs.
- Do not tightly couple unrelated modules to a specific alignment backend.
