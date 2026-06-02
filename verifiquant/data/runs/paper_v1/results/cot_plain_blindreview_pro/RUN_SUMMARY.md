# CoT Plain Oracle (Blind Review) — Pro

## Protocol

- **Oracle entry policy**: `blind_review_every_turn`.
- **No GT leakage**: oracle receives `python_solution` only; NOT numeric ground truth, NOT `is_correct`.
- Model: gemini-2.5-pro. max_turns=3. Oracle mode: plain.

## Headline

| Metric | Value |
|---|---|
| Correct | 49/50 |
| Accuracy | 98% |
| Silent Wrong | 1 |
| SWR | 2% |
| No-answer | 0 |
| improved (wrong→correct) | 6 |
| broken (correct→wrong) | **0** |

Rounds distribution: {1: 7, 2: 14, 3: 29}

## Silent Wrong Case

| Case | Final Answer | Rounds | Note |
|---|---:|---:|---|
| test-1443 | 19.13 | 3 | CAGR percent/decimal — persistent SW across ALL four configs |

## Takeaway

The strongest leakage-free configuration: 98% accuracy. **Yet SWR is still 2%** — one silent wrong (test-1443) survives even at this ceiling, persisting through all 3 oracle rounds. This is the headline comparison point: even a GT-blind oracle on the strongest model cannot reach SWR=0%, because the architecture forces a numeric output. VQ reaches SWR=0% via abstention. `broken_count=0`.
