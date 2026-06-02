# CoT Funnel-Guided Oracle (Blind Review) — Pro

## Protocol

- **Oracle entry policy**: `blind_review_every_turn`.
- **No GT leakage**: oracle receives `python_solution` only; NOT numeric ground truth, NOT `is_correct`.
- Oracle prompt: M/N/F/E/I/C six-layer structured checklist.
- Model: gemini-2.5-pro. max_turns=3. Oracle mode: funnel-guided.

## Headline

| Metric | Value |
|---|---|
| Correct | 48/50 |
| Accuracy | 96% |
| Silent Wrong | 1 |
| SWR | 2% |
| No-answer | 1 |
| improved (wrong→correct) | 4 |
| broken (correct→wrong) | **0** |

Rounds distribution: {1: 23, 2: 17, 3: 10}

## Wrong Cases

| Case | Final Answer | Rounds | Verdict | Note |
|---|---:|---:|---|---|
| test-1789 | 27520.0 | 1 | silent_wrong | annuity timing rounding |
| test-1242 | None | 3 | no_answer | oracle over-rewrote; CoT could not produce a parseable number after 3 rounds |

## Takeaway

Funnel-guided Pro (96%) is again **slightly below** plain Pro (98%) under blind review, consistent with the Flash result. The funnel oracle's aggressive per-layer rewriting produced one `no_answer` case (test-1242) where it rewrote the question into something CoT could no longer answer numerically. Confirms: a more elaborate oracle prompt does not improve — and can mildly hurt — reliability when GT is not visible. `broken_count=0`.
