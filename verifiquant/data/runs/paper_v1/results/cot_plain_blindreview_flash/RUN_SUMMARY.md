# CoT Plain Oracle (Blind Review) — Flash

## Protocol

- **Oracle entry policy**: `blind_review_every_turn` — oracle reviews EVERY turn regardless of correctness.
- **No GT leakage**: oracle receives `python_solution` (intent spec) only; NOT the numeric ground truth, NOT `is_correct`.
- Model: gemini-2.5-flash (CoT + oracle). max_turns=3. Oracle mode: plain.

## Headline

| Metric | Value |
|---|---|
| Correct | 47/50 |
| Accuracy | 94% |
| Silent Wrong | 3 |
| SWR | 6% |
| No-answer | 0 |
| improved (wrong→correct) | 6 |
| broken (correct→wrong) | **0** |

Rounds distribution: {1: 14, 2: 26, 3: 10}

## Silent Wrong Cases

| Case | Final Answer | Rounds | Note |
|---|---:|---:|---|
| test-1443 | 19.12 | 3 | CAGR (gold 19.14); LLM fractional-power arithmetic oscillates 19.11–19.16 (wrong in 3/4 blind configs) |
| test-1891 | 2388.05 | 2 | residual numeric discrepancy; oracle found no rewriteable ambiguity |
| test-1969 | 579.93 | 2 | rounding/precision boundary |

## Takeaway

With a GT-blind oracle reviewing every turn, accuracy 94% / SWR 6%. `broken_count=0`: blind review never destroyed a correct answer. SWR remains nonzero — the oracle "passes" wrong answers it sees no issue with. This is the realistic, leakage-free failure mode.
