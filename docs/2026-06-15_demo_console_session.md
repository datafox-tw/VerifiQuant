# 2026-06-15 — Demo Console session changelog

Work to make the Flask demo (`/demo`) into a recordable ~5-minute walkthrough on the **fair
`paper_v1`** run. Covers the conversational console, the runtime verifiable atomic transform,
the raw-LLM control group, the funnel-class showcase, and a batch of UX/selector fixes.

> Demo-DB note: several scripts edit the **demo** card store (`paper_v1/fic/cards.db`) for
> presentation. They are idempotent and must be **re-run after any card-store rebuild**. They do
> NOT change the paper's reported results.

---

## 1. Dataset / demo-run plumbing
- **Config-driven demo run.** `app.py` resolves the cards DB + question bank from one run via
  `VERIFIQUANT_DEMO_RUN` (default `paper_v1`). Helpers: `_resolve_run_db_path`,
  `_resolve_run_question_bank`, `_resolve_run_showcase`. Swapping datasets = set the env var +
  allowlist the run in `.dockerignore`.
- **`.dockerignore`** now ships `paper_v1` (`questions_50.jsonl`, `demo_showcase.jsonl`,
  `trap/trap_set.jsonl`, `fic/cards.db`) + the legacy `demo_50q_0415` fallback. Root cause of the
  original "questions_50.jsonl not found" container error.
- `requirements.txt` += `openai>=2.0` (for the GPT control group).

## 2. Conversational console (chat UI)
- New **Conversation tab** (default) in `templates/demo.html`; `static/conversation.js` holds the
  client-side turn state (original question/context kept **immutable**; each turn re-diagnoses
  with original + accumulated answers). `static/diagnostic_render.js` = shared `VQRender` renderers
  (extracted from `demo.js`); `demo.js` delegates to it.
- **`_build_repaired_context` fix** (`app.py`): non-JSON context is preserved verbatim and answers
  appended as a `[clarified_fields]` block — fixes the destructive-repair F-error.
- Per-answer **"How this was derived & verified"** disclosure: funnel stepper (M→N→F→E→I→C) +
  chosen contract + Python execution.
- **Funnel stepper bug fix**: lit the wrong layer (`funnel_layer` words like "Intent" → charAt(0)
  = "I"); now driven by `diagnostic_type`/`status`.

## 3. Runtime Verifiable Atomic Transform
- **`POST /api/transform/apply`** (`app.py`): wires `stage_transform.verify_transform` into the
  live flow (was build-time/PoC only). Re-runs the FIC, verifies (AST + numerical invariant),
  returns `result_old→result_new` + proof + FIC code; rejects unverified transforms.
- Soft/hard clarification options are **structured** `{label,value,has_transform}` via
  `_normalize_warning_options` (pipeline) so the UI can offer a verified-transform button.
- **I_HARD heroes** (scripts; demo-DB only):
  - `scripts/promote_loan_timing_hard.py` — loan payment `2164` timing soft→hard
    (579.98 → 577.10 verified).
  - `scripts/promote_npv_timing_transform.py` — injects the timing transform into NPV `2940`
    (end −2103.68 → beginning +7685.95 verified). Before this, NPV "Beginning" had no transform
    and silently recomputed the same −2103.68.

## 4. Real M-class (which method?)
- `scripts/import_m_cards.py` imports **NPV / NPVGO / Discounted-Payback** from `0415` into
  `paper_v1` (via `ingest_artifacts`, keeps FTS in sync) so a vague "is this project worth it?"
  retrieves competing methods.
- **Selector prompt** (`run_error_classification_pipeline.py`) rewritten: judge **intent only**.
  Named metric (in question OR context/clarification) → `select_card` even if values look bad
  (value checks belong to the E-gate); vague goal with ≥2 fitting methods → `abstain_m`; truly
  unsupported/invented → `abstain_n`. (Earlier the strict version was commented out; a first pass
  over-abstained E cases — now scoped.)

## 5. Control group (raw AI, no funnel)
- **`POST /api/baseline`**: raw LLM, generic prompt, provider `gemini|gpt` chosen in the UI
  ("Compare with raw AI" toggle + model select). Fires on **every** turn when checked.

## 6. Showcase + traps + verdicts
- `verifiquant/data/runs/paper_v1/demo_showcase.jsonl`: one curated case per class
  (C/M/N/F/E/I). Plus the existing `trap/trap_set.jsonl` surfaced in a grouped picker
  (`/api/demo/questions?showcase=1` and `?path=…trap_set.jsonl`).
- **Behaviour-aware verdicts**: flag-classes (M/N/F/E/I) → raw AI answering = ✗, VQ raising the
  exact gate = ✓; numeric "matches gold" only for clean C cases. Trap `ground_truth` is treated
  as the NAIVE/wrong number.
- **Gold badge**: showcase/sample cases with a single numeric `ground_truth` show
  "✦ 此題有正確答案 (correct answer): X" under the question, before answering
  (`renderGoldBadge`).

## 7. UX / multi-turn fixes
- **Dropdown = stage only** (no disruption); the **next Send** starts it as a fresh question
  (`sendAsNewQuestion`). Removed the "Load Sample" button.
- **Clarification transcript**: answers accumulate into `cvNotes` (never overwritten) and are
  composed into context each turn — fixes "forgot my earlier answer" (e.g. NPV then timing).
- **Multi-question clarification**: pipeline returns per-question `groups`; UI renders 1 group as
  transform-aware buttons, 2+ groups as one dropdown each + a single **Submit answers**.
  `cvAnsweredHints` suppresses already-answered hints so nothing is re-asked.

---

## Files touched
- Code: `app.py`, `verifiquant/pipeline/run_error_classification_pipeline.py`,
  `static/{conversation,diagnostic_render,demo}.js`, `static/styles.css`, `templates/demo.html`,
  `requirements.txt`, `.dockerignore`.
- Scripts (demo-DB, idempotent): `scripts/promote_loan_timing_hard.py`,
  `scripts/promote_npv_timing_transform.py`, `scripts/import_m_cards.py`.
- Data: `verifiquant/data/runs/paper_v1/demo_showcase.jsonl`, `…/fic/cards.db` (mutated by scripts).

## Re-run after any card-store rebuild (PYTHONPATH=.)
```bash
PYTHONPATH=. python3 scripts/import_m_cards.py
PYTHONPATH=. python3 scripts/promote_loan_timing_hard.py
PYTHONPATH=. python3 scripts/promote_npv_timing_transform.py
```

## Run the demo
```bash
GEMINI_API_KEY=… PORT=6222 python3 app.py   # OPENAI_API_KEY too for the GPT control group
# open http://127.0.0.1:6222/demo  (hard-refresh after JS/CSS changes)
```

## Verified live (Gemini 2.5 Flash)
C clean → 2.0 · M vague → abstain_m (NPV vs payback) · N invent-formula → N · F missing rate → F ·
E working-ratio negative expense → alert/E · I_HARD loan timing → 577.10 verified / end 579.98 ·
NPV path → beginning +7685.95 verified.

## Known limitations
- Gate triggers (M/N/F/E/I) are LLM-driven → verify live; the deterministic F/E checks + transform
  math are stable.
- CAGR E-case dropped: its `cagr_output_lower_bound` check calls `compute()`, which crashes on a
  negative base (complex) and surfaces as a C-error instead of an E-alert. Working-ratio is the
  clean E showcase.
- `store.load_repair_rules` / `load_core_by_id` query the DB fresh (no cache) → DB script changes
  are picked up live; only static JS/CSS need a browser hard-refresh.
