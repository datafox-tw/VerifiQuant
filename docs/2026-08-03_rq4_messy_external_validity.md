# RQ4 external-validity pipeline — messy-query human panel (2026-08-03)

Auditable record of how `tab:v2ext` / RQ4 in `paper/main_v2.tex` was produced: annotation
collection → aggregation → system runs → scoring → numbers locked into the paper.

## 1. Annotation

- **Set**: 82 "messy" queries (`verifiquant/data/runs/messy_query_pilot/messy_pilot.jsonl`),
  natural-language rewrites of FinanceReasoning test questions spanning 4 latent design
  conditions (`intended` field): clean/well-posed (27), missing (18), ambiguous (18),
  judgment/out-of-scope (17).
- **Annotators**: 5 domain experts, finance graduates of the authors' institution, each an
  independent CSV in `verifiquant/data/runs/messy_query_pilot/labels/A1.csv`–`A5.csv`.
  De-identified filenames (`A1`–`A5`); no annotator PII in the repo or the paper (referenced
  only as *Annotator 1–5*). Labels: A/R/C/O (ANSWER/REQUEST-INFO/CLARIFY/OUT-OF-SCOPE),
  optional `?` for genuine uncertainty, one-line rationale. Guideline:
  `verifiquant/data/runs/messy_query_pilot/LABELING_GUIDELINE(_zh).md`.
- **Ethics**: no funding/grant involved; non-personal, publicly-derived benchmark items;
  no PII collected about annotators; advisor-confirmed no REC review required. See
  `project_icaif_ethics_annotation.md` (session memory) and `main_v2.tex` §Ethical
  considerations (`sec:v2ext`).
- **Author's own labels are excluded from ground truth** by design, to preserve panel
  independence from the system authors (the paper claims annotators are "blind to every
  system"; the authors are not).

## 2. Aggregation (`scripts/aggregate_messy_labels.py`)

Ran once, offline, before any system was scored (`ground_truth.jsonl` frozen prior to step 3):

```
n_annotators: 5, n_items: 82, majority_resolved: 82 (100%), disagreement_pool: 0
fleiss_kappa: 0.838  (almost perfect agreement)
majority_label_distribution: {A: 36, R: 20, C: 10, O: 16}
```

Quality checks performed (not just the headline kappa):
- Per-rater coverage 81–82/82; the one short rater (A3) left a genuine `?` on `mq-58`
  (correctly excluded from that item's vote, not a data defect) — that item still reached
  a 4/5 majority.
- Reason-field fill rate 98–100% across all 5 raters.
- Majority label vs. original design intent (`intended` field) is near-perfect for
  clean→A (26/27), missing→R (18/18), judgment→O (16/17); `ambiguous`→C is looser
  (10/18, rest mostly A on close 3:2/4:1 votes). This divergence is evidence the panel is
  exercising independent judgment rather than echoing the trap-design labels — it
  strengthens, not weakens, the non-circularity argument RQ4 exists to make.

Outputs: `ground_truth.jsonl`, `disagreement_pool.csv` (empty), `agreement_report.json`.

## 3. System runs (`verifiquant/data/runs/messy_query_pilot/agreement/runs/`)

Gemini 2.5 Flash, 250-card DB, top-4 retrieval, run over the 82 messy queries:

| File | System | Answered (raw) |
|---|---|---|
| `cot_single_shot.jsonl` | CoT single-shot | 46/82 |
| `rpal_plain.jsonl` | R-PAL (plain) | 62/82 |
| `rpal_plus.jsonl` | R-PAL+ (taxonomy prompt) | 29/82 |
| `vq_k1.jsonl` | VerifiQuant K=1 | 11/82 (7 of these carry an I_SOFT warning) |

**K=3 was not run for RQ4** (deliberate): RQ4 mirrors Panel B's funnel-only, autonomous
setting; the K=3 oracle protocol needs a per-question intent specification that messy,
paraphrased queries don't have a clean operational definition for; and K-ablation (oracle
recoverability) is already answered elsewhere (`tab:v2kabl`). Introducing K=3 here would
blur RQ4's actual question (does an independent panel corroborate the boundary?).

## 4. Scoring (`scripts/score_messy_external.py`, spec in `score_messy_external_SPEC.md`)

Per-system adapter → `(emit, value, action∈{A,R,C,O})`:
- **CoT / R-PAL (plain)**: no structured refusal reason exposed → `action` is `A` or
  unresolved (`None`); **Match falls back to binary answer-vs-abstain agreement** with the
  human majority action (not real 4-way, since these baselines can't express *why* they
  didn't answer).
- **R-PAL+**: exposes `assessment.decision`/`assessment.diagnostic` (READY/N/F/I_SOFT/
  I_HARD/M) → real 4-way action. 2/82 items have an unparseable assessment (`exec_fail` /
  `assessment_parse_fail`); scored as `action=None`, counted as a non-match, not silently
  imputed.
- **VerifiQuant**: `status`/`diagnostic_type` → action, with **`diagnostic_type=F` mapped to
  `R` even when `status=error`** (F wins over the generic error status, per the locked
  scorer-adapter rule the user specified). I_SOFT rows are `status=success`, so they count
  as `emit=True` (a warned number is still a committed one) — this is D1, locked.
- **M-refusal convention (locked, "M→O")**: VerifiQuant's M (intent-level ambiguity) is
  scored as `O` per the paper's stated `\textsc{out-of-scope}$\leftrightarrow$N/M` mapping.
  All 6 M-refusals in this run in fact carry an explicit clarification question (verified:
  100% of M-refusals have a non-empty `clarification_request.questions`), so this is a
  **conservative** convention that slightly understates VerifiQuant's Match; the paper's
  Table~\ref{tab:v2ext} caption discloses this. An `M→C` alternative was scored for
  comparison (Match 34.1% vs. 31.7%) but not used, to keep the paper's stated funnel-outcome
  mapping internally consistent rather than special-casing after seeing the numbers.
- Numeric matching for Sel.Acc reuses the pipeline's existing tolerance function
  (`run_error_classification_pipeline._answer_match`: `rel_tol=1e-6, abs_tol=1e-2`, plus
  the `output_scale_x100` decimal-vs-percentage-point handling) — not a new tolerance.
- Gold values are joined from `messy_pilot_provenance.json` (`messy_pilot.jsonl` itself
  carries no gold field).

Two adapter bugs were caught and fixed during scoring (see script history / this doc):
1. A naive "any structured action" check misclassified CoT/R-PAL(plain) as having real
   taxonomy (because `action='A'` on emit was populated), which silently dropped credit for
   correct abstention. Fixed to require the system to expose an actual `R`/`C`/`O` value on
   *some* item before it gets 4-way scoring; otherwise it falls back to binary.
2. `R-PAL+`'s two unparseable-assessment items initially degraded the *entire* system to
   binary Match (an all-or-nothing bug). Fixed so unresolved items count as individual
   non-matches without discarding the other 80 items' structured scoring.

## 5. Locked numbers (`tab:v2ext`, `paper/main_v2.tex`)

| System | Human-FNAR ↓ | Match | Sel.Acc |
|---|---|---|---|
| CoT (single-shot) | 21.7% | 87.8% (binary) | 15.2% |
| R-PAL (top-4 retrieval) | 56.5% | 68.3% (binary) | 11.3% |
| R-PAL+ (taxonomy prompt) | 13.0% | 65.9% (4-way) | 17.2% |
| **VerifiQuant (K=1)** | **8.7%** | 31.7% (4-way) | 9.1% |

Human-FNAR ordering (VQ < R-PAL+ < CoT < R-PAL) matches the contract-derived Panel B
ordering among the three shared systems, corroborating that the governance boundary is not
an artifact of the paper's own trap construction.

**VerifiQuant's low Match/answered-count is diagnosed, not just reported.** Of the 28 items
where VQ abstained but the human panel said "answer," 26 are N-refusals where **top-4
retrieval never surfaced an applicable contract** for the paraphrased query (verified by
inspecting `rpal_plain.jsonl`'s candidate list for the same items, e.g. a LIFO-costing
question retrieves only a FIFO-costing card — not a substitutable formula). This is the
paper's own pre-declared *selection frontier* (§`sec:v2boundary`), observed here under
naturalistic paraphrase rather than contract-derived perturbation, not a new or unexplained
failure mode. It also preempts the "low FNAR via indiscriminate refusal" critique: VQ's
FNAR is low because it declines exactly where retrieval can't ground an answer, not because
it refuses everything.

## Reproduce

```bash
PYTHONPATH=. python3 scripts/aggregate_messy_labels.py   # -> ground_truth.jsonl (idempotent, already frozen)
PYTHONPATH=. python3 scripts/score_messy_external.py     # -> agreement/score_report.json + LaTeX rows
```

Full per-system score detail (raw and the M→C comparison variant):
`verifiquant/data/runs/messy_query_pilot/agreement/score_report.json`.
