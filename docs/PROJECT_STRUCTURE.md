# Project Structure (v1 / v2)

## Overview
- `verifiquant_v1/`: legacy v1 logic (kept stable as baseline)
- `verifiquant_v2/`: new v2 logic (FIC v2 pipeline, diagnostics, expansion)
- `preprocessing/`: compatibility wrappers for easier CLI execution
- `archive/`: deprecated/old scripts kept for traceability

## Recommended Entry Points

### v2: dataset -> FIC (two-stage)
- Preferred:
  - `python preprocessing/dataset_case_to_fic_v2.py ...`
- Also supported:
  - `python verifiquant_v2/preprocessing/dataset_case_to_fic.py ...`

### v2: expand clean/M/F/E cases (investment_analysis)
- Preferred:
  - `python preprocessing/expand_cases_v2.py ...`
- Also supported:
  - `python verifiquant_v2/pipeline/expand_cases.py ...`

## Why wrappers exist
Running scripts inside nested folders can change `sys.path` and break package imports.
Wrappers and root-path bootstrapping are added so both direct and wrapper execution work.

## Notes
- Keep v1 untouched as baseline for comparisons.
- Put experimental or deprecated scripts into `archive/` instead of deleting.
