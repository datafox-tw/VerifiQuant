# VerifiQuant — CLAUDE.md

Project: Financial diagnostic pipeline using a 6-layer **M/N/F/E/I/C** funnel and Financial Inference Contracts (FICs).
Model: Gemini 2.5 Flash (via `google.genai`). Environment: Python 3.11+, SQLite, Flask, SQLAlchemy.

The goal is **not** higher LLM arithmetic accuracy — it is *predictable, verifiable failure behavior*: intercept a broken inference at the right funnel layer and either refuse, slot-fill, alert, or apply a **mathematically verified** repair.

---

## Authoritative vs. Historical

- **Authoritative code:** the `verifiquant/` package. Trust it over any doc.
- **CLI shims:** root `preprocessing/*.py` are mostly 12-line re-exports of `verifiquant/*`. Editing them does nothing — edit the real file in `verifiquant/`.
- **Authoritative docs:** this file + `README.md`.
- **Design lineage (not current state):** `docs/archive/*` (date-prefixed). Read for *intent*; never assume a feature exists because an archive doc describes it. The old NSFCA vision describes a SymPy math engine and PDF/RAG that were **never built**.
- **Experiment logs:** `docs/results/*` (date-prefixed). The README results table is derived from these.
- **Scratch:** `scratch/*.py` is throwaway debug code, not part of any pipeline.

---

## Project Structure

```
verifiquant/                         # THE PACKAGE — authoritative
  contracts.py                       # DiagnosticReport, DiagnosticFinding; RefusalCategory = M/N/F/E/I/C
  taxonomy.py                        # financial domain taxonomy (card tagging)
  card_store.py                      # SQLAlchemy ORM: fic_core / fic_retrieval / fic_repair_rules + FTS
  preprocessing/                     # build-time: turn formulas into FIC cards
    common.py                        #   load/dump records, gemini client, normalizers
    stage_core.py                    #   LLM → core (formula, invariants, scale_checks, semantic_hints)
    stage_retrieval.py               #   LLM → retrieval metadata
    stage_repair.py                  #   diagnostic_checks + semantic_hints → repair rules
    stage_transform.py               #   ★ verifiable transforms (result_postprocess / code_patch)
    fic_generation_pipeline.py       #   orchestrates the 3 stages + dedup/merge
    seed_builder.py                  #   build seed rows from config YAML / JSONL
    validate_relations.py            #   cross-artifact integrity (core ↔ retrieval ↔ repair)
    dataset_case_to_fic.py           #   thin per-case wrapper
  pipeline/                          # run-time: diagnose a question
    run_error_classification_pipeline.py   # MAIN engine (~2100 lines); ErrorClassificationAPI
    run_framework_guided_self_improve_pipeline.py  # multi-turn VerifiQuant + recovery
    run_cot_self_improve_pipeline.py       # CoT baseline
    run_iterative_agents_pipeline.py       # exploratory
    expand_cases.py                        # trap-variant generator (F/E/I)
  data/                              # datasets + data/runs/<run>/cards.db + outputs
app.py                               # Flask UI over ErrorClassificationAPI
test_transform_poc.py                # runnable, no-API PoC for stage_transform.py
preprocessing/                       # CLI shims (12-line) + 3 standalone tools:
                                     #   extract_config_questions_to_jsonl.py, visualize_*_eval.py
docs/archive/  docs/results/  docs/environment-setup.md  scratch/
```

---

## Key Data Types

### DiagnosticReport (`contracts.py`) — output of `run_case()`
- `status`: `success | refusal | error | alert | needs_clarification`
- `diagnostic_type`: `M | N | F | E | I | C | None | Unknown`
- `funnel_layer`, `gate_action`, `reason_code`
- `findings`: `DiagnosticFinding[]` (`id, category, severity, message, rule, evidence, suggested_fix, ui_action`)
- `requested_fields`: missing required fields (F-class)
- `clarification_request`: structured I-class question

### FIC artifacts (SQLite `cards.db`) — keyed by `fic_id`
- **core**: `inputs (JSON), python_code, invariants, scale_checks, semantic_hints`
- **retrieval**: `selection_hints, keywords, negative_keywords`
- **repair**: `diagnostic_check_id, suggested_fix, ui_action`; I-class rules also carry `repair_action.transform_map`

### Transform specs (`stage_transform.py`)
- `TransformSpec` (`patch_type="result_postprocess"`): `result_expr`, `max_expr_nodes`, `invariant`, `affected_inputs`
- `CodePatchSpec` (`patch_type="code_patch"`): `target_pattern`, `replacement`, `max_changed_nodes`, `cross_verify_result_expr`, `cross_verify_max_nodes`, `invariant`, `affected_inputs`
- `invariant` is a plain Python `lhs == rhs` equation (over `result_old`/`result_new`/input names), safety-audited then verified numerically. No SymPy.
- `patch_spec_from_dict()` routes by `patch_type`. `get_transform_spec_for_choice(repair_rule, value)` pulls the spec for a user's clarification choice.

---

## The Verifiable-Transform Flow (I-class repair)

```
stage_core    : semantic_hints[].options = [{label,value,is_default, transform_spec?}]
                (default option has no transform_spec; alternatives do)
   ↓ _normalize_semantic_hint()  — MUST preserve dict options (uses _normalize_hint_options,
                                    NOT _norm_str_list; the latter dropped transform_spec — fixed)
stage_repair  : _global_i_rules() → _parse_hint_options() → repair_action.transform_map
   ↓ get_transform_spec_for_choice(repair_rule, chosen_value)
stage_transform: verify_transform() | verify_code_patch()
```

**Why `code_patch` is safe:** acceptance requires the patched code's numerical
output to equal `eval(cross_verify_result_expr, {result: original_output, **inputs})`.
This binds a free-form code edit to a declared algebraic identity. Buggy math,
oversized AST diffs, sneaked imports/`__dunder__`, multi-occurrence patterns → rejected.

**Verification layers:**
- `result_postprocess`: AST node-count ≤ bound · safe-name whitelist · numerical determinism/finiteness · numerical invariant check
- `code_patch`: occurrence==1 · shallow AST-diff ≤ `max_changed_nodes` · no new dangerous calls/imports · numerical cross-verify · numerical invariant check

`_shallow_node_key()` hashes only a node's own primitive fields (not child subtrees), so `start=1`→`start=0` counts as **1** changed node, not the whole parent chain.

---

## Running

```bash
source .venv/bin/activate && export GEMINI_API_KEY=...

python3 test_transform_poc.py          # offline; verifies stage_transform end-to-end

python3 preprocessing/build_card_store.py \
  --input verifiquant/data/medium_config_50_0408.jsonl \
  --db-url sqlite:///verifiquant/data/runs/R/cards.db

python3 preprocessing/run_error_classification_pipeline.py \
  --input verifiquant/data/testing_5Q.jsonl \
  --db-url sqlite:///verifiquant/data/runs/R/cards.db \
  --output R.jsonl --debug-sanity

python3 preprocessing/run_framework_guided_self_improve_pipeline.py \
  --input verifiquant/data/medium_config_50_0408.jsonl \
  --db-url sqlite:///verifiquant/data/runs/R/cards.db \
  --output fg.jsonl --summary-output s.json --max-turns 3

python3 app.py
```

`ErrorClassificationAPI.from_db(db_url, client)` then `.run(question, context)` → DiagnosticReport dict.

---

## Baselines (2026-04-15, 50Q medium, Gemini 2.5 Flash)

| System | Accuracy |
|---|---|
| JPMorgan Multi-Agent (paper) | 0.46 Pass@1 |
| CoT self-improve (no GT) | 0.90 (45/50) |
| VerifiQuant framework-guided | 0.88 (44/50), recovery 28% |

VerifiQuant advantage is in *failure-mode behavior on trap/incomplete questions*, not clean-question accuracy. Trap-set evaluation is the main open item.

---

## Common Pitfalls

- **Editing a `preprocessing/` shim has no effect** — change the real file under `verifiquant/`.
- `_normalize_semantic_hint` once flattened structured options via `_norm_str_list`, dropping `transform_spec`. It now uses `_normalize_hint_options` (backward-compatible with legacy string options). Don't reintroduce `_norm_str_list` there.
- `duplicate core fic_id` / `repair rule has no matching core diagnostic_check`: LLM emitted clashing `fic_id`s. `fic_generation_pipeline.py` dedups (last-write wins); see `docs/archive/2026-04-07_diary.md` for the incident history.
- CoT accuracy varies ±5% run-to-run (LLM non-determinism). Deterministic E/F layers are the stable part.
- Missing `GEMINI_API_KEY` → immediate `RuntimeError` (no silent fallback).
- Archive docs may claim features (SymPy math engine, PDF/RAG) that **do not exist**. Verify against code.

---

## Open Work (see README "What Is Not Done Yet")

Trap-dataset eval + Reliability Calibration · N-class OOD test set · true HITL for I-class · Silent Wrong Rate · broader baselines · validate `transform_spec` auto-generation on a full Gemini card-build run.
