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
| test-1443 | 19.13 | 3 | CAGR (gold 19.14); turn2 hit 19.14 then turn3 drifted back to 19.13 — LLM arithmetic non-determinism |

## Takeaway

The strongest leakage-free configuration: 98% accuracy. **Yet SWR is still 2%** — one silent wrong (test-1443, CAGR gold=19.14). Notably the model hit the correct 19.14 at turn 2 but drifted back to 19.13 at turn 3: the LLM cannot even reproduce its own correct answer, because it is doing fractional-power arithmetic (`2.4^(1/5)`) by token generation, oscillating across 19.11–19.16. This is the headline comparison point: even a GT-blind oracle on the strongest model cannot reach SWR=0%, because the architecture forces a (non-deterministic) numeric output. VQ executes this step in deterministic Python (exactly 0.19135…→19.14, every time) and reaches SWR=0% via abstention. `broken_count=0`.
