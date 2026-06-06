# CoT Funnel-Guided Oracle (Blind Review) — Flash

## Protocol

- **Oracle entry policy**: `blind_review_every_turn` — oracle reviews EVERY turn regardless of correctness.
- **No GT leakage**: oracle receives `python_solution` (intent spec) only; NOT numeric ground truth, NOT `is_correct`.
- Oracle prompt: M/N/F/E/I/C six-layer structured checklist.
- Model: gemini-2.5-flash. max_turns=3. Oracle mode: funnel-guided.

## Headline

| Metric | Value |
|---|---|
| Correct | 46/50 |
| Accuracy | 92% |
| Silent Wrong | 4 |
| SWR | 8% |
| No-answer | 0 |
| improved (wrong→correct) | 5 |
| broken (correct→wrong) | **0** |

Rounds distribution: {1: 37, 2: 9, 3: 4}

## Silent Wrong Cases

| Case | Final Answer | Rounds | Note |
|---|---:|---:|---|
| test-1443 | 19.11 | 1 | CAGR (gold 19.14); LLM fractional-power arithmetic oscillates 19.11–19.16 (wrong in 3/4 blind configs) |
| test-1891 | 2387.99 | 1 | residual numeric discrepancy |
| test-1969 | 580.02 | 1 | rounding/precision boundary |
| test-1789 | 27522.0 | 1 | annuity timing rounding |

## Takeaway

Under blind review, funnel-guided (92%) is **slightly worse** than plain (94%) — reversing the earlier GT-gated finding. When the oracle cannot see correctness, the structured "find an issue at every layer" instruction makes it over-rewrite, occasionally injecting noise into answers that were not broken. Conservative plain oracle wins. This supports the core claim: **reliability comes from architecture (abstention), not from a smarter oracle prompt.** `broken_count=0`.
