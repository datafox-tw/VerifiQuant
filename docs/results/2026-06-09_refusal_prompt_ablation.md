# Refusal Prompt Ablation — clean 50Q (2026-06-09)

**Question (teacher feedback):** CoT's "safe refusal @ k=1 = 0%" — is that an
architectural limit, or just because our prompt never *let* CoT refuse? If we
vary how strongly the prompt encourages refusal, can a pure-prompt CoT produce
safe refusals, and at what cost?

**Method.** Single-shot (K=1), clean canonical 50Q (`paper_v1/questions_50.jsonl`).
Four refusal-encouragement levels with an explicit `decision={answer,refuse}`
channel:
- **L0 Forced** — must answer (reproduces canonical baseline; no refusal channel)
- **L1 Permitted** — may refuse if info missing / genuinely ambiguous
- **L2 Encouraged** — should refuse under 5 explicit criteria (missing field,
  ambiguous interpretation, percent/decimal, unspecified convention, out-of-scope)
- **L3 Strict** — pre-answer M/F/E/I self-check; stop at first failing layer

Grid: `gpt-5.2` × `reasoning_effort {low, medium}` × {L0..L3} (8 cells) +
`gemini-2.5-flash` × {L0..L3} (4 cells). 12 cells × 50Q = **600 questions**
(GPT 8 cells + Gemini L0/L3 in the first 500-budget run; Gemini L1/L2 added on
explicit follow-up). Code: [run_refusal_ablation.py](../../verifiquant/pipeline/run_refusal_ablation.py),
[run_refusal_ablation_grid.py](../../scripts/run_refusal_ablation_grid.py).

## Results

| cell | model | effort | L | Correct | SW | Safe-Refusal | Acc | SWR | good_ref | over_ref |
|---|---|---|---|---|---|---|---|---|---|---|
| gpt_low_L0 | gpt-5.2 | low | 0 | 44 | 6 | 0 | 88% | 12% | – | – |
| gpt_low_L1 | gpt-5.2 | low | 1 | 42 | 6 | 2 | 84% | 12% | 0 | 2 |
| gpt_low_L2 | gpt-5.2 | low | 2 | 40 | 4 | 6 | 80% | 9% | 1 | 5 |
| gpt_low_L3 | gpt-5.2 | low | 3 | 38 | 3 | 9 | 76% | 7% | 2 | 7 |
| gpt_medium_L0 | gpt-5.2 | medium | 0 | 45 | 5 | 0 | 90% | 10% | – | – |
| gpt_medium_L1 | gpt-5.2 | medium | 1 | 45 | 4 | 1 | 90% | 8% | 0 | 1 |
| gpt_medium_L2 | gpt-5.2 | medium | 2 | 39 | 3 | 8 | 78% | 7% | 3 | 5 |
| gpt_medium_L3 | gpt-5.2 | medium | 3 | 38 | 4 | 8 | 76% | 10% | 2 | 6 |
| gem_L0 | flash 2.5 | – | 0 | 42 | 8 | 0 | 84% | 16% | – | – |
| gem_L1 | flash 2.5 | – | 1 | 39 | 9 | 2 | 78% | 19% | 2 | 0 |
| gem_L2 | flash 2.5 | – | 2 | 41 | 6 | 3 | 82% | 13% | 2 | 1 |
| gem_L3 | flash 2.5 | – | 3 | 41 | 8 | 1 | 82% | 16% | 1 | 0 |

`good_ref` = L0 (same model/effort) answered **wrong** & this cell refused (an SWR rescued).
`over_ref` = L0 answered **correct** & this cell refused (accuracy lost).

## Findings

1. **The 0% was a prompt artifact, not an architectural limit — for GPT.**
   With an explicit refusal channel, gpt-5.2 produces 6–9 safe refusals at L2/L3.
   So the teacher is right: forcing the answer is what zeroed safe-refusal. The
   honest baseline must give CoT a refusal channel.

2. **But CoT's refusals are badly targeted — it abstains on the wrong questions.**
   Across every GPT cell the over/good ratio is poor (L3 low: 2 good / 7 over;
   medium L2: 3 good / 5 over). SWR falls only modestly (12%→7%) while accuracy
   craters (88%→76%). CoT cannot reach SWR=0 without abandoning ~1/4 of the set,
   and even then SWR stays 7–10%. It refuses many questions it *would have gotten
   right* and keeps answering ones it gets *wrong*.

3. **Steerability is model-dependent, and Gemini is nearly inert.** gemini-2.5-flash
   refuses only 1–3 times across L1–L3; its refusals are actually *better* targeted
   than GPT's (gem L1: 2 good/0 over; L2: 2 good/1 over) but far too few to move the
   needle — SWR wanders 13–19% (it even *rises* to 19% at L1) with no monotonic
   trend. The same four prompts that make GPT abstain 6–9× barely register on
   Gemini. "Encourage refusal via prompt" is not a portable lever.

4. **Refusal reasons look principled but are mostly false alarms.** GPT's L3
   refusals carry M/F/E/I tags (e.g. `[M] working ratio has multiple definitions`,
   `[I] annualization 365 vs 360`), yet 7/9 (low) are over-refusals on questions
   it had answered correctly at L0. The self-check inflates plausible ambiguity
   rather than detecting real blockers.

## Implication for the paper claim

This *strengthens* the core claim while answering the fairness objection. VQ
Flash V3: **90% acc, SWR 0%, 5 abstain** — it abstains *precisely* (deterministic
M/N/F/E/I gates), reaching SWR=0 with almost no accuracy cost. Prompt-induced CoT
refusal is the opposite: poorly targeted, high over-refusal, SWR floor ~7%, and
not even reliably triggerable (Gemini). The advantage is not "CoT can't refuse"
(it can) but **"CoT can't refuse *selectively*"** — the risk–coverage tradeoff is
strictly worse than a contract-grounded gate. Recommend reframing the §5.5
sentence from "CoT architecturally cannot abstain" to "CoT, even when prompted to
abstain at four escalating strengths, abstains non-selectively (over/good ≫ 1) and
cannot push SWR below ~7% without sacrificing ~12pp accuracy."

## Multi-K blind self-correction (K=6, added 2026-06-10)

**Method.** 6 cells — `gpt-5.2` medium {L0, L1, L3} and `gemini-2.5-flash` {L0, L1, L3}
— each run ONCE at K=6 with the blind oracle loop (oracle reviews every turn, no
GT gating, may use ground-truth *code* logic but not the final numeric value),
then truncated to recover the K=1..6 curve (paper §5.9 trick). Code:
[run_multik_refusal.py](../../scripts/run_multik_refusal.py),
`refusal_ablation_multik/multik_curves.json`.

> ⚠️ These are **independent runs** from the K=1 grid above: the multi-K K=1 column
> is this run's own first turn and differs from the standalone grid by LLM
> nondeterminism (e.g. gpt_medium_L0 K=1 here = 43 correct vs 45 in the grid).
> Read recovery *within* a row (vs its own K=1), not across the two tables.

| cell | avg_rounds | K=1 acc / SWR / refuse | K=3 | K=6 | recovered@K6 | broken@K6 |
|---|---|---|---|---|---|---|
| gpt-med L0 | 6.0 | 86% / 14% / 0 | 96% / 4% / 0 | **100% / 0% / 0** | 7 | 0 |
| gpt-med L1 | 5.96 | 86% / 14% / 0 | 96% / 4% / 0 | **100% / 0% / 0** | 7 | 0 |
| gpt-med L3 | 5.98 | 78% / 11% / 6 | 96% / 4% / 0 | **100% / 0% / 0** | 11 | 0 |
| gem L0 | 1.62 | 80% / 20% / 0 | 92% / 8% / 0 | 92% / 8% / 0 | 6 | 0 |
| gem L1 | 1.60 | 86% / 10% / 2 | 94% / 6% / 0 | 94% / 6% / 0 | 5 | 1 |
| gem L3 | 1.48 | 82% / 16% / 1 | 92% / 8% / 0 | 92% / 8% / 0 | 5 | 0 |

### Findings (K=6)

1. **The refusal-level distinction washes out under multi-round oracle (GPT).** All
   three GPT-medium cells converge to **100% acc / SWR 0% / 0 refusals by K=4–6**,
   regardless of starting refusal level. The L3 cell's 6 K=1 refusals are *transient*:
   the oracle clarifies the flagged ambiguity and they convert to correct (refuse
   6→0 by K=2). So the refusal channel only matters in the **low-K / no-oracle**
   regime; with enough blind-oracle rounds, refusals are clarified away.

2. **Blind self-correction is safe (broken ≈ 0).** Across all GPT cells the oracle
   recovers 7–11 questions and breaks 0–1 — consistent with the paper's
   `broken_count=0` claim. Iteration converts wrong/refused → correct without
   manufacturing new silent-wrongs.

3. **Gemini saturates early and never reaches SWR=0.** avg_rounds ≈1.5 (its oracle
   returns the question unchanged after 1–2 turns → natural termination), so its
   curve plateaus by K=2–3 at **SWR 6–8%**. More K does not help. Combined with its
   near-inert refusal behaviour, Gemini both refuses weakly *and* self-corrects
   weakly.

4. **Reconciles with the K=1 story and the VQ claim.** At K=1 (no oracle), refusal
   prompts trade accuracy for poorly-targeted abstention (§Findings above). At high
   K, GPT+oracle reaches the same ~100%/SWR-0 ceiling the paper already reports for
   CoT+oracle — but this ceiling is bought by the **oracle reading ground-truth
   code each round**, i.e. the recovery mechanism, not by abstention. VQ's
   distinction is unchanged: it reaches SWR=0 by *deterministic execution +
   precise abstention* at K=1, without an oracle that consults ground truth.

### Placebo control (added 2026-06-10): is K=6→100% just re-rolling?

Concern: with the oracle rewriting the question every turn (GPT avg_rounds=6.0),
does K=6→100% come from genuine clarification or just from re-sampling a stochastic
solver 6× (cf. test-1443 CAGR jitter 19.11–19.16)? Control: a **placebo oracle** —
no API call, no solution-derived info, just a content-free "re-check and recompute"
nudge each turn (`--oracle-mode placebo`), so the only thing that changes is the
solver gets another attempt.

| cell (L0, K=6) | K=1 | K=6 blind | K=6 **placebo** |
|---|---|---|---|
| gpt-med | 86% / SWR 14% | **100% / 0%** (+14pp) | **92% / 8% (+2pp)** |
| gem | 80% / SWR 20% | 92% / 8% (+12pp) | **82% / 18% (−2pp)** |

**Re-sampling alone gains ≈0–2pp and never reaches 100%** (Gemini even regresses —
it second-guesses correct answers when re-asked with no new info). The blind oracle's
solution-derived rewrites drive the +12–14pp and the SWR→0. So the K=6 recovery is
**clarification, not re-rolls** — but note that clarification uses `python_solution`
(privileged intent spec), so K=6→100% is an **oracle-assisted recovery ceiling**, not
deployment performance. VQ's SWR=0 at K=1 (deterministic execution + abstention, no
solution-reading oracle) is unaffected.

## Open / not done
- 12-cell K=1 grid complete (600Q). Multi-K: 6 cells at K=6 + 2 placebo controls complete.
- Could compute a proper risk–coverage AUC and overlay VQ's single operating point.
- L2 not covered in multi-K (only L0/L1/L3); GPT-low multi-K not run.
