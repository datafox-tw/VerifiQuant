# Governance go/no-go protocol

Frozen before observing matched R-PAL trap outputs on 2026-07-29.

## Question

Does executable VQ governance add measurable value when a financial request is
not ready for safe execution, after controlling for top-4 retrieval and program
execution?

## Systems

1. **R-PAL plain:** same top-4 retrieval and public functions as VQ; one Gemini
   2.5 Flash call generates and executes `solve()` or informally declines.
2. **R-PAL prompted:** identical retrieval and one-call budget, but the prompt
   explicitly requests M/N/F/E/I_HARD/I_SOFT/READY classification, an action,
   and optional code. It has no deterministic contract gate.
3. **VQ K=1:** existing funnel-only run. K=3/oracle outputs are excluded from
   the causal trap comparison.

## Data

- 50 deterministic contract-grounded traps: F/E/I/N/M × 10.
- No record has `needs_review=true`.
- Paired 50 clean source questions measure over-refusal and clean utility.
- The trap operator, expected diagnostic, gold, and per-question solution are
  never included in an R-PAL prompt.

The I stratum contains nine I_SOFT and one I_HARD case; this imbalance must be
disclosed and per-type results inspected before a broad ambiguity claim.

## Frozen metrics

Primary: **False Numeric Acceptance Rate (FNAR)** — an unqualified numeric
answer on a governance-critical input. An I_SOFT numeric answer is safe only
when accompanied by an explicit I_SOFT warning.

Secondary:

- structured catch rate;
- exact diagnostic accuracy;
- action accuracy (correct missing field, invalid field/rule, or clarification);
- clean accuracy, SWR, coverage, and selective accuracy;
- exact paired McNemar test for FNAR.

## Decision rule

- **KEEP_EXECUTED_GOVERNANCE_CLAIM:** VQ FNAR is at least 10 percentage points
  below prompted R-PAL, with no loss in diagnostic or action accuracy.
- **NO_GO_FULL_VQ_CLAIM:** prompted R-PAL has FNAR no higher than VQ,
  diagnostic/action accuracy no lower than VQ, and clean accuracy/coverage no
  lower than VQ.
- **MIXED_NARROW_OR_HYBRID:** VQ improves on plain R-PAL but the prompted
  control prevents a broad executed-governance claim.
- **NO_GO_OR_REDESIGN:** VQ does not reduce FNAR versus plain matched R-PAL.

Regardless of the automatic label, operator-level results are mandatory. A
gain caused only by blanket abstention is not evidence of useful governance.

