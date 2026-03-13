# AGENTS.md

This repository is intended to work well with coding agents.

## Read First
1. Read `docs/project-spec.md`.
2. Read `.github/copilot-instructions.md`.
3. Follow any scoped instructions in `.github/instructions/` that apply to the files being edited.

## Agent Expectations
- Prefer small, reviewable changes.
- Keep architecture modular.
- Preserve a clear separation between:
  - media/audio preparation
  - matching/refinement
  - alignment map construction
  - subtitle retiming
  - reporting
- Do not implement out-of-scope features from v1 unless explicitly requested.

## Definition of Done
A task is not complete unless:
- code changes are coherent and minimal
- tests were added or updated where relevant
- docs were updated if behavior changed
- outputs remain inspectable and debuggable

## Preferred Task Order
- implement data models first
- implement pure logic second
- implement side-effecting media/CLI integration last

## Avoid
- hiding mapping behavior inside one large function
- hardcoding one global offset model
- silently discarding uncertain subtitle cues
