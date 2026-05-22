# VerifiQuant

**"We verify before we compute. We compute only what we can verify."**

A neuro-symbolic diagnostic framework for reliable financial reasoning. Instead of asking "did the LLM get the right number?", VerifiQuant asks **"at which layer did the reasoning break — and can we intercept it before it does damage?"**

---

## The Problem

Standard LLM benchmarks measure final-answer accuracy. In finance, this misses three failure modes that are far more dangerous in practice:

- **Silent Failures** — the model answers with confidence when it shouldn't answer at all (missing inputs, ambiguous intent)
- **Reasoning Crumble** — correct formula, correct variables, wrong arithmetic (CoT errors)
- **Semantic Drift** — formula chosen correctly, but a hidden ambiguity (e.g. FX direction, beginning-of-period vs. end-of-period) causes a systematically wrong result

VerifiQuant's thesis: **intercept failures layer-by-layer before they reach the user**, using a Financial Inference Contract (FIC) as the control structure, and apply only **mathematically verifiable** repairs.

---

## Architecture: The M/N/F/E/I/C Diagnostic Funnel

Each incoming question passes through six checkpoints. A question can exit at any layer with an actionable signal — not a hallucinated answer.

| Layer | Code | Name | What it catches | System response |
|---|---|---|---|---|
| 1. Intent | **M** | Misunderstanding | Semantic ambiguity — can't map to any formula family | **Refusal** — ask user to clarify intent |
| 2. Scope | **N** | Not Supported | Intent is clear, but no FIC card exists for it | **Graceful exit** — declares knowledge boundary |
| 3. Schema | **F** | Formula Spec | FIC selected, but required inputs are missing | **Slot-filling** — lists exactly which fields are absent |
| 4. Boundary | **E** | Extraction/Value | Inputs provided but numerically implausible (Rf > Rm, negative std, etc.) | **Deterministic alert** — rule-based, not LLM judgment |
| 5. Critic | **I** | Interception | Hidden semantic ambiguity (FX direction, time basis, annual vs monthly) | **Critic intervention** — clarification + verifiable transform |
| 6. Logic | **C** | Calculation | Execution errors in deterministic Python | **Audit log** — traceable, reproducible |

**Outcome taxonomy:** `success` · `refusal` · `error` · `alert` · `needs_clarification`

---

## Repository Map

```
verifiquant-update/
├── README.md                    ← you are here (project overview)
├── .claude/CLAUDE.md            ← AI-agent working guide (structure, commands, pitfalls)
├── app.py                       ← Flask web UI wrapping ErrorClassificationAPI
├── test_transform_poc.py        ← runnable PoC for the verifiable-transform feature
├── requirements.txt
│
├── verifiquant/                 ← THE PACKAGE (current, authoritative code)
│   ├── contracts.py             ← DiagnosticReport / DiagnosticFinding dataclasses
│   ├── taxonomy.py              ← financial domain taxonomy
│   ├── card_store.py            ← SQLAlchemy FIC artifact store + FTS retrieval
│   ├── preprocessing/           ← FIC card generation (build-time)
│   │   ├── stage_core.py        ← LLM → FIC core (formula, invariants, semantic_hints)
│   │   ├── stage_retrieval.py   ← LLM → retrieval metadata
│   │   ├── stage_repair.py      ← diagnostic_checks + semantic_hints → repair rules
│   │   ├── stage_transform.py   ← ★ verifiable atomic transforms (result_postprocess / code_patch)
│   │   ├── fic_generation_pipeline.py  ← orchestrates the 3 stages
│   │   ├── seed_builder.py / validate_relations.py / common.py
│   │   └── dataset_case_to_fic.py
│   ├── pipeline/                ← inference (run-time)
│   │   ├── run_error_classification_pipeline.py  ← main engine; ErrorClassificationAPI
│   │   ├── run_framework_guided_self_improve_pipeline.py  ← multi-turn VerifiQuant
│   │   ├── run_cot_self_improve_pipeline.py      ← CoT baseline
│   │   ├── run_iterative_agents_pipeline.py      ← exploratory variant
│   │   └── expand_cases.py      ← trap-variant question generator
│   └── data/                    ← datasets, FIC artifacts, run outputs (data/runs/…)
│
├── preprocessing/               ← thin CLI shims → re-export verifiquant/* (12-line files)
│   │                              (keeps `python3 preprocessing/X.py` working post-refactor)
│   ├── extract_config_questions_to_jsonl.py  ← standalone (not a shim)
│   ├── visualize_config_eval.py              ← standalone viz tool
│   └── visualize_expand_eval.py              ← standalone viz tool
│
├── static/ · templates/         ← Flask web assets
│
├── docs/
│   ├── environment-setup.md     ← setup reference (kept accessible)
│   ├── archive/                 ← historical planning/vision docs (date-prefixed)
│   │   ├── 2026-03-15_執行攻略.md
│   │   ├── 2026-04-05_Readme-old_NSFCA-vision.md
│   │   ├── 2026-04-05_V2_Status_and_Roadmap.md
│   │   ├── 2026-04-05_paper_chinese.md
│   │   ├── 2026-04-06_dataset_exploration.md
│   │   ├── 2026-04-07_diary.md
│   │   └── 2026-04-07_bug_to_solve.md
│   └── results/                 ← experiment outputs (date-prefixed)
│       ├── 2026-04-14_inference.json / _evaluation.json
│       └── 2026-04-15_verifiquant_no_gt.txt / _cot_no_gt.txt
│
└── scratch/                     ← debug/scratch scripts (not part of pipeline)
    ├── test.py · test_diagnose.py · scratch.py
```

> **Refactor note:** the root `preprocessing/` directory predates a package refactor.
> Most of its files are now 12-line shims that re-export from `verifiquant/`.
> The authoritative code lives in `verifiquant/`. CLI commands using
> `python3 preprocessing/X.py` still work via the shims.

---

## Code ↔ Document Relationship

Which document explains which part of the code, and how current each doc is:

| Document | Describes | Status vs. code |
|---|---|---|
| **README.md** (this) | Whole project, architecture, results | Current |
| **.claude/CLAUDE.md** | File structure, run commands, data types, pitfalls | Current |
| `docs/archive/2026-04-05_Readme-old_NSFCA-vision.md` | Original NSFCA vision (SymPy engine, PDF/RAG) | **Aspirational** — describes features never built (SymPy math engine, PDF parser). Kept for design lineage only. |
| `docs/archive/2026-04-05_V2_Status_and_Roadmap.md` | The M/N/F/E/I/C funnel design rationale | Conceptually current — this is the spec the funnel implements. Milestones/roadmap dates are stale. |
| `docs/archive/2026-04-05_paper_chinese.md` | Paper draft, error taxonomy definitions | The M/F/E taxonomy here maps directly to `contracts.py:RefusalCategory`. Numbers are pre-2026-04-15. |
| `docs/archive/2026-04-06_dataset_exploration.md` | FinChain dataset structure analysis | Background for `verifiquant/data/` and `seed_builder.py`. |
| `docs/archive/2026-04-07_diary.md` | Dev diary; FIC dedup/duplicate-id incidents | Explains *why* `fic_generation_pipeline.py` has merge/dedup logic. Historical narrative. |
| `docs/archive/2026-04-07_bug_to_solve.md` | Known-issue scratchpad | Historical; may be partially resolved. Verify against code before acting. |
| `docs/archive/2026-03-15_執行攻略.md` | Early run playbook | Superseded by Quick Start below. |
| `docs/results/2026-04-15_*.txt` | Raw run logs (50Q) | The numbers in the Results table below are extracted from these. |

**Rule of thumb:** README + CLAUDE.md are authoritative. `docs/archive/*` is design lineage — read it to understand *intent*, not *current state*. Always verify archive claims against the code.

---

## Core Components (Implemented)

### Financial Inference Contract (FIC)

A FIC is a machine-readable "computation contract" attached to each financial formula, generated in three stages:

- **Core** (`stage_core.py`) — `fic_id`, input schema, deterministic Python execution code, `invariants` (E-class), `scale_checks` (E-class), `semantic_hints` (I-class trap descriptions, now carrying structured `transform_spec` options)
- **Retrieval** (`stage_retrieval.py`) — `selection_hints`, `keywords`, `negative_keywords` for BM25 FIC selection
- **Repair** (`stage_repair.py`) — maps each diagnostic finding → `suggested_fix` + `ui_action`; for I-class, threads `transform_map` from semantic-hint options into the repair rule

All three artifacts are stored in **SQLite** via `card_store.py` (SQLAlchemy ORM + FTS + integrity validation).

### Verifiable Atomic Transforms (`stage_transform.py`) ★ new

When an I-class clarification resolves a hidden ambiguity (e.g. "these are beginning-of-period payments"), the system must adjust the computation **without** letting the LLM freely rewrite code. Two patch types, each verifiable:

| Patch type | What it does | How it is verified |
|---|---|---|
| `result_postprocess` | Applies a pure arithmetic expression to the final result (`result * (1+r)`, `1/result`, …) | AST node-count bound + safe-name whitelist + numerical determinism + numerical invariant check |
| `code_patch` | Surgical text replacement in the FIC code (`start=1` → `start=0`) | (1) pattern occurs exactly once, (2) shallow AST-diff ≤ declared blast-radius bound, (3) **cross-verification**: patched code's output must numerically match an equivalent `result_postprocess` expression, (4) numerical invariant check |

The `invariant` field is a plain Python `lhs == rhs` equation over `result_old` / `result_new` / input names. It is safety-audited (same whitelist as `result_expr`) then verified numerically across sample inputs — no symbolic-math dependency, and it works for iterative (loop-based) formulas.

The **cross-verification** is the mathematical guarantee: a code change is only accepted if its numerical effect provably equals a declared algebraic relationship. A malicious or buggy patch (wrong math, sneaked import, oversized diff) is rejected. See `test_transform_poc.py` for the end-to-end NPV annuity-due demonstration.

**Scope of Gemini-generated `transform_spec` (production cards).** The prompt in
`stage_core.py` includes a domain → typical-ambiguity table (TVM, interest rates, FX,
returns, volatility, bonds, statistics) and four worked few-shot examples: NPV
annuity-due, monthly↔annual rate conversion, percent↔decimal output, and one
**anti-pattern** (benchmark choice → `transform_spec=null`). Gemini occasionally still
emits malformed specs (subscript syntax, function calls, identity transforms on
default options); `_normalize_hint_option()` in `stage_core.py` is the hard gate that
strips these post-LLM so bad specs never reach the verifier. `code_patch` (the more
powerful patch type used in the PoC) is intentionally NOT exposed to Gemini yet — only
`result_postprocess` is generated automatically; `code_patch` requires hand-authoring.

### Pipeline (`run_error_classification_pipeline.py`, ~2100 lines)

BM25 retrieval → LLM selector (M/N) → LLM extractor (F) → deterministic E-checks → Critic agent (I) → deterministic Python execution (C). Exposed as `ErrorClassificationAPI` (used by `app.py` and eval scripts).

### Evaluation Pipelines

- `run_framework_guided_self_improve_pipeline.py` — multi-turn VerifiQuant loop with recovery tracking
- `run_cot_self_improve_pipeline.py` — plain CoT baseline with self-reflection
- `expand_cases.py` — generates trap variants (Incomplete-F, Ambiguous-M, Trap-E/I)

### Web App (`app.py`)

Flask app wrapping `ErrorClassificationAPI`. Returns the full diagnostic report with funnel layer, findings, and suggested repair actions.

---

## Experimental Results

50 medium-difficulty FinanceReasoning questions. Model: **Gemini 2.5 Flash**. Run date: 2026-04-15. Raw logs: `docs/results/2026-04-15_*.txt`.

| System | Accuracy | Notes |
|---|---|---|
| JPMorgan Multi-Agent (2025, paper) | 0.46 (Pass@1) | multi-agent CoT baseline |
| CoT self-improve (no GT) | **0.90** (45/50) | Gemini 2.5 Flash, 3-turn self-reflection |
| VerifiQuant framework-guided | **0.88** (44/50) | FIC funnel + multi-turn recovery |

**VerifiQuant 50Q breakdown:** success 47/50 (94%), correct 44/50 (88%), multi-turn recoveries 14 (28%). Blocked: E=1, F=1, C=1.

**Diagnostic distribution (testing_50Q_result-final):** clean success 33 · I (clarification) 14 · E 1 · F 1 · C 1.

**Variance note:** CoT accuracy is non-deterministic (32/40, 30/44, 29/45 across three runs, ±5%). VerifiQuant's E/F intercepts are deterministic, giving a more predictable failure distribution even when headline accuracy is comparable. The differentiating advantage is expected on **trap datasets** (not yet systematically run).

---

## What Is Not Done Yet

| Item | Status |
|---|---|
| Trap dataset evaluation (M/N/F/E/I categories) | `expand_cases.py` exists for F/E/I; M/N variants need design + a run |
| Reliability Calibration metric (correctly_refused / total_unsolvable) | Not computed; needs labeled unsolvable set |
| N-class graceful-exit testing | `GLOBAL_N_NOT_SUPPORTED_RULE` exists; no OOD test set |
| True HITL for I-class | Current loop is LLM↔LLM, not human confirmation |
| Silent Wrong Rate measurement | Needs ground truth on trap cases |
| Broader baselines (GPT-4 / DeepSeek direct CoT) | Only JPMorgan paper number + Gemini CoT compared |
| SymPy symbolic *math engine* | Not built, and SymPy was removed entirely. The old vision assumed prompt→SymPy formalization; the current design verifies transforms numerically instead (no symbolic dependency) |
| PDF/RAG over financial documents | Not implemented; system uses structured JSONL inputs |
| `transform_spec` auto-generation (full card-build run) | **Validated 2026-05-20 on testing_5Q.jsonl with Gemini 2.5 Flash.** Generates correct specs for NPV annuity-due, percent↔decimal scaling, sign convention. Anti-patterns (benchmark choice, input rescaling) correctly emit `transform_spec=null`. A normalization safety net (`_normalize_hint_option` in `stage_core.py`) catches Gemini's residual mistakes: subscript syntax (`inputs['x']`), function calls (`round`, `exp`), identity transforms on default options, missing `result` reference, and unparseable expressions. **Not yet validated on the full 50Q card build.** |
| `code_patch` auto-generation by Gemini | Only `result_postprocess` is in the `stage_core` schema enum. `code_patch` is supported by `stage_transform` and demonstrated in `test_transform_poc.py`, but Gemini does not yet generate `code_patch` specs in production cards — they must be hand-authored. |

---

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export GEMINI_API_KEY=your_key

# 0. (offline) verify the transform feature without any API calls
python3 test_transform_poc.py

# 1. Build FIC card store from formula source
python3 preprocessing/build_card_store.py \
  --input verifiquant/data/medium_config_50_0408.jsonl \
  --db-url sqlite:///verifiquant/data/runs/my_run/cards.db

# 2. Run the diagnostic pipeline
python3 preprocessing/run_error_classification_pipeline.py \
  --input verifiquant/data/medium_config_50_0408.jsonl \
  --db-url sqlite:///verifiquant/data/runs/my_run/cards.db \
  --output verifiquant/data/runs/my_run/results.jsonl --max-records 10

# 3. Framework-guided multi-turn (with recovery)
python3 preprocessing/run_framework_guided_self_improve_pipeline.py \
  --input verifiquant/data/medium_config_50_0408.jsonl \
  --db-url sqlite:///verifiquant/data/runs/my_run/cards.db \
  --output verifiquant/data/runs/my_run/fg.jsonl \
  --summary-output verifiquant/data/runs/my_run/summary.json --max-turns 3

# 4. Web demo
python3 app.py
```

---

## Tech Stack

Google Gemini 2.5 Flash · SQLite/SQLAlchemy · Flask · FinChain/FinanceReasoning dataset. No symbolic-math dependency (transforms are verified numerically).
