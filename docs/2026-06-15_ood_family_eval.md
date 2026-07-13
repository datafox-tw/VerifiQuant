# OOD formula-family evaluation (reviewer attack #1, setting c)

> **FULL-RUN UPDATE (overnight, 199 OOD + 51 ID, with CoT baseline on identical questions):**
> run `ood_summary_20260713_005129.json`, 67 min, seed 0, K=1. Headline numbers below;
> the original 60-sample analysis is kept underneath for provenance.

## Full-run result (199 OOD + 51 ID, same questions to both systems)

| System | Split | Abstain | Correct | Silent-wrong (raw) | Silent-wrong (unit-normalized*) |
|---|---|---|---|---|---|
| **VQ** | OOD (199) | **177 (88.9%)** | 14 (7.0%) | 8 (4.0%) | **5 (2.5%)** |
| **VQ** | ID (51) | 10 (19.6%) | 34 (66.7%) | 7 (13.7%) | **1 (2.0%)** |
| CoT (refusal-permitted) | OOD (199) | 120 (60.3%) | 71 (35.7%) | 8 (4.0%) | — |
| CoT (refusal-permitted) | ID (51) | 25 (49.0%) | 23 (45.1%) | 3 (5.9%) | — |

\* Unit-normalized: excluding decimal-vs-percent near-misses (output = gold ×100 or ÷100 —
same value, wrong scale; e.g. 0.1158 vs 11.58). 3/8 OOD and 6/7 ID silent-wrongs are such
near-misses. The paper pipeline already handles output-scale normalization (the
`prescalarfix` rerun); this harness scores strictly. With normalization, VQ ID effective
accuracy is 40/51 ≈ 78%, consistent with the 250Q K=1 autonomous 81.2%.

**The headline: abstention selectivity.**
- **VQ: OOD abstain 88.9% vs ID abstain 19.6% → 69.3-pt selectivity gap.** Its abstention
  *tracks library coverage* — it refuses because the contract is missing, not because the
  question looks hard.
- **CoT (explicitly permitted to say CANNOT SOLVE): 60.3% vs 49.0% → 11.3-pt gap.** Its
  refusal is generic hedging: it abstains on half the questions it could answer (ID accuracy
  collapses to 45.1%, vs 77.6% when forced to answer) while still answering 40% of
  out-of-scope families. This directly replicates the refusal-prompt ablation on the same
  questions and answers the reviewer's "let the baseline also refuse" demand: **prompted
  refusal is non-selective; contract-grounded refusal is.**

**Caveats (state in paper):** (1) library here is the 53-card demo DB (3 imported valuation
cards; one hint promoted to I_HARD for the demo — slightly inflates ID abstain); for
paper-grade numbers rerun against a pristine paper_v1 build. (2) The CoT baseline prompt
permits abstention by design — that is the point of the comparison. (3) OOD = out-of-library,
not out-of-LLM-knowledge; CoT answering 35.7% of OOD correctly from parametric memory is
expected and unverified.


**Question:** when the contract library LACKS the formula family, does VerifiQuant **abstain**
(honest "no verified contract") or **force-fit a wrong card** and emit a confident,
verified-looking number (silent-wrong)?

**Design (no card rebuild needed).** The 250Q paper experiment used a **250-card library — one
card per question family, ~100% coverage** (this is itself the reviewer's "task-aware library
construction" evidence). The paper_v1 demo library has only **53 cards**, so **199 of the 250Q
families are genuinely OOD** relative to it. We run OOD families against the 53-card library and
measure behaviour. ID (covered) families are a sanity control. Single-shot (K=1): abstention
happens on the first pass; a missing card cannot be conjured by multi-turn.

Harness: `scripts/run_ood_family_eval.py` (config recorded → K-variations are cheap re-runs).
Run: `ood_summary_20260713_003139.json` · seed 0 · top_k 3 · gemini-2.5-flash · K=1 · 60 OOD / 20 ID.

## Result

| Split (vs 53-card lib) | n | Abstain (good) | Force-fit **silent-wrong** | Force-fit correct | Accuracy |
|---|---|---|---|---|---|
| **OOD** (family absent) | 60 | **56 (93.3%)** | **3 (5.0%)** | 1 (1.7%) | — |
| **ID** (family covered) | 20 | 3 (15%) | 1 (5.0%) | — | **16/20 = 80%** |

Baseline contrast (from the 250Q summary): CoT **never abstains** (`abstain = 0` by construction),
so on these OOD families it would emit an unverified number for **100%** of cases.

**Reading:**
- **VerifiQuant abstains on 93.3% of out-of-library families** — it honestly refuses when it has no
  verified contract, instead of hallucinating. This is exactly the reviewer's setting (c) ("unseen
  formula family → must refuse or extend").
- **ID control = 80% accuracy at K=1** (with 15% calibrated abstention) — confirms VQ isn't just
  "abstaining on everything"; it still answers covered families. (80% ≈ the 81.2% autonomous K=1 on
  the full 250Q — consistent.)
- Together with the K-ablation: **high in-library accuracy + high off-library abstention + low
  silent-wrong in both** = the calibrated behaviour that is the actual contribution.

## Honest residual failure mode (the 3 OOD silent-wrongs)
VQ force-fit a **semantically adjacent** card and computed a wrong number:
- `Put` (option payoff) → forced onto **Bermuda Option** card → 497 vs gold 200.
- `Holding Period Return` → forced onto **CAGR** card → 0.1158 vs gold 11.58. *(Note: 0.1158 vs
  11.58 is the SAME value in decimal vs percent — a unit-scale near-miss, arguably not a true
  silent-wrong; our 1% tolerance flagged it.)*
- `Price-Weighted Index` → forced onto **Basket of Goods** card → 131.33 vs gold 118.57.

So the true "confidently wrong on OOD" rate is **2/60 ≈ 3.3%** (excluding the unit-scale near-miss),
vs a raw LLM that commits an unverified answer on 100%. Worth stating both numbers.

## How this answers reviewer attack #1
The reviewer's three settings, addressed:
- (a) known family, unknown instance → the 250Q main result (in-library routing+binding).
- (b) unseen phrasing, known family → covered by the trap-set paraphrases (existing).
- (c) **unseen formula family → this experiment**: 93.3% abstain, ≤5% silent-wrong.

Reframe for the paper: **do not** claim general financial-reasoning superiority. Claim: *given a
contract library, VerifiQuant routes+binds in-distribution AND abstains ~93% of the time when a
family is out-of-library* — turning the "you only test in-library routing" attack into a headline
calibration result. Report library coverage explicitly (250Q ran at ~100% coverage by construction;
the OOD test deliberately drops coverage to 0 for the tested families).

## Caveats / next steps
- **Sample = 60 OOD / 20 ID.** For the paper, run the full **199 OOD** (`--limit-ood 199
  --limit-id 51`) — the harness makes this one command; results append to `runs/ood_eval/`.
- **"OOD" = out-of-library, not out-of-LLM-knowledge.** These are still real financial families the
  LLM may know parametrically; the point is VQ won't emit a *verified-looking* answer from a wrong
  contract. State this explicitly.
- **Baseline on the same OOD set:** run CoT single-shot on the identical 60 to show its
  hallucination/silent-wrong directly (currently inferred from `abstain=0`). Quick follow-up.
- **K:** this is K=1 (abstention is a first-pass property). If you later test K>1, the harness's
  recorded config lets you re-run only the delta.
