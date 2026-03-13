---
applyTo: "tests/**/*,**/test_*.py"
---

# Testing Instructions

## Test Framework
- Use `pytest`.
- Prefer small deterministic tests over broad integration-only coverage.

## Required Coverage Areas
Add or maintain tests for:
- SRT parsing and writing
- piecewise alignment region behavior
- time conversion helpers
- subtitle retiming inside one mapped region
- subtitle dropping/reporting inside TV-only unmatched regions
- subtitle handling for region-boundary crossings
- constant-offset alignment cases
- speed-drift cases

## Fixtures
- Keep fixtures minimal.
- Prefer synthetic fixtures for mapping and subtitle logic.
- Use larger media-derived fixtures only when needed for alignment behavior.

## Assertions
- Assert both transformed timestamps and classification behavior.
- Verify dropped and flagged cues are reported, not just omitted.
- Verify low-confidence behavior explicitly.

## Maintenance
- Update tests whenever CLI behavior or mapping rules change.
- Avoid brittle assertions tied to irrelevant formatting.
