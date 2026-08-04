# Review-response map (2026-08-04)

Maps each reviewer weakness to what the paper now says and where. Content was added first;
page budget is reconciled separately (the last two related-work entries currently spill onto
page 9).

## Addressed by NEW content in this pass

| # | Reviewer point | What was added | Where |
|---|---|---|---|
| 1 | No latency / token / compute cost analysis, yet cost matters in financial deployment | New "What the funnel costs" paragraph. Uses the gate ladder as a price list because every rung is the same 250 questions, one backbone, one environment: compute-only 6.6 s/q, +F/E **+0.2 s** (deterministic Python, essentially free), +I **+5.9 s** (introduces a third model call), full funnel 12.2 s/q vs plain R-PAL 10.5 and R-PAL+ 8.2, and SC@5 at 38.1. Also states the architectural bound (≤3 model calls vs 1) and that early exits reduce it. Explicitly caveats wall-clock variance and the absence of token accounting. | §5.3 |
| 2 | A per-class confusion report for Diag/Action on traps would help | Table 3 extended from FNAR-only to per-operator **FNAR + VQ Diag + VQ Act + R-PAL+ Diag**, plus a paragraph reading the confusion: VQ is strongest on F (Diag 90%) and weakest on I (60%); of the 20 I traps, **9 are downgraded to a soft warning and 4 pass as ready**, so the residual is a *severity* misjudgment inside the ambiguity gate, not a failure to notice ambiguity. R-PAL+ shows the mirror profile (F 100%, N 88%, I collapses to 10%). | Table 3 + §5.2 |
| 3 | Retrieval details sparse; no top-k sensitivity; alternative retrievers not evaluated | New sentences at the M/N gate stating what the retriever actually is: SQLite FTS5 BM25 over card title/summary/keywords/embedding text, merged with keyword-overlap ranking, filtered by declared selection hints. States that **nothing is trained, tuned, or fitted to the evaluation sets**, and that the same retriever and top-4 depth serve VQ, R-PAL, and R-PAL+, so retrieval is held fixed rather than optimized by one system. Depth sensitivity and dense/hybrid retrievers are declared future work. | §3.2 |
| 4 | Out-of-library comparison under-detailed (setup, baselines, context, fairness) | Setup expanded in Experimental Design (both systems: same backbone, same 199 questions, same question/context, top-4 from the **same** 50-family library; split fixed by formula family in advance, not by difficulty). Result paragraph expanded with the full picture rather than one number: baseline 95.0% correct at 0.5% abstention, VQ 88.9% abstention and 7.0% correct, VQ selective accuracy 63.6% vs 95.5%, and VQ's residual silent-wrong there **no better** than the baseline's (4.0% vs 4.5%). Framed as the result that most constrains the claim. | §4.1 + §5.3 |

## Already stated before this pass (verified present in rendered text)

| Reviewer point | Where it already lives |
|---|---|
| Clean set is in-library; traps inherit library semantics | §4.1 ("the evaluation is *in-library*"), §4.3 ("mechanism-targeted stress test, not an estimate of natural deployment incidence"), §4.5 (the skeptic framing motivating RQ4) |
| Bounded repair is sampled numerical validation, not formal equivalence; 2.3% false accepts | §3.3 states "bounded numerical validation, not formal equivalence", names the compensating-patch failure mode, and reports 20/879 = 2.3% residual false accepts |
| Residual frontiers LLM-mediated, no formal guarantees; input provenance unverified | §6 names the three predeclared LLM-mediated runtime frontiers and states plainly that input *provenance* is not verified, only schema and boundary |
| Single backbone family | Limitations ("a single formula-provenance benchmark and a single backbone family... extending the funnel to other backbones is future work") plus the RQ1 Flash-to-Pro scaling result |

## Still open (declared, not resolved)

- **Top-k sensitivity and alternative retrievers.** Declared future work in §3.2. A depth sweep
  would need a full rerun per k (~51 min per setting for VQ on 250Q), which does not fit before
  the deadline.
- **Token accounting.** The runs record wall-clock only; no token counters were instrumented.
  Stated as a caveat rather than estimated.
- **Cross-model triangulation for VerifiQuant itself.** See `2026-08-03_rq4_messy_external_validity.md`
  and the RQ1 note; the existing `vq50_pro` run is a binding-mitigation diagnostic, not a clean
  backbone-transfer test, and running one now creates a disclosure obligation with no time to
  diagnose an unfavorable result.
- **RQ4 statistical power.** Unchanged from `2026-08-03_rq4_messy_external_validity.md` §6: the
  RQ4 ordering is decisive only against plain R-PAL (4 vs 26 of 46). Ready-to-paste caveat text
  is in that file.

## Provenance for the new numbers

- Latency: `elapsed_s / n` from `verifiquant/data/runs/gate_ladder/r{1..5}_*_summary.json`
  (250 each), `retrieval_pal_baseline/rpal_top4_gemini_25_flash_summary.json` (2635.5 s / 250),
  `governance_eval/prompted_clean250_run1_summary.json` (2042.4 s / 250),
  `sc_baseline/sc5_gemini_25_flash_summary.json` (9520.2 s / 250).
- Per-operator Diag/Action: recomputed with `analyze_governance_eval.score_traps`
  (`per_operator` field) over `governance_confirmatory/{trap_set,vq_k1_traps,conf_plain_run1,
  conf_prompted_run{1,2,3}}.jsonl`. R-PAL+ columns are means of the three prompted runs.
- I-trap confusion (9 soft-warning, 4 ready of 20): the `details` field of the same call,
  cross-tabulating `operator` against `got_diagnostic`.
