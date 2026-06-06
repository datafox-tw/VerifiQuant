# CoT Funnel-Guided Oracle Flash Run Summary

## Run Files

- Raw output: `cot_funnel_guided_50.jsonl`
- Summary: `cot_funnel_guided_50_summary.json`
- Standard aliases for downstream scripts:
  - `output.jsonl`
  - `summary.json`
- Terminal log: `stdout_50.log`

## Headline Result

| System | Correct | Accuracy | Silent Wrong | Abstain |
|---|---:|---:|---:|---:|
| CoT single-shot Flash | 41/50 | 82% | 9/50 | 0 |
| CoT + plain oracle Flash | 45/50 | 90% | 5/50 | 0 |
| CoT + funnel-guided oracle Flash | 47/50 | 94% | 3/50 | 0 |
| VQ Flash V3 | 45/50 | 90% | 0/50 | 5/50 |

## Key Takeaway

The funnel-guided oracle improves CoT accuracy from 90% to 94% relative to the plain oracle baseline. It reduces silent wrong cases from 5 to 3, but still has no abstention mechanism. VQ remains the only system here with SWR=0%, because its incorrect cases are routed to safe refusal/abstention rather than forced numeric output.

## Iteration Behavior

- 43 cases finished in 1 turn.
- 5 cases finished in 2 turns.
- 2 cases used all 3 turns.
- Summary `improved_count`: 4 cases where turn 1 was explicitly wrong and final answer became correct.
- Broader rescue count: 6 cases if including turn-1 parse failures (`None -> True`).

## Final Wrong Cases

| Case | Final Answer | Abs Error | Rounds | Note |
|---|---:|---:|---:|---|
| `test-1891` | 2388.05 | 0.97 | 1 | Oracle found no rewriteable ambiguity; residual numeric discrepancy. |
| `test-1969` | 580.0 | 0.02 | 1 | Plain oracle got this right; funnel run rounded differently. |
| `test-1789` | 27520.0 | 1.0 | 3 | Oracle clarified annuity timing, but CoT still missed exact rounded target. |

## Improved / Rescued Cases

| Case | First Answer | Final Answer | Rounds | Oracle Signal |
|---|---:|---:|---:|---|
| `test-1590` | 116000.0 | 121000.0 | 3 | `[I_HARD]`, then `[M]` principal protection / ELN interpretation. |
| `test-1710` | parse failure | 57.5 | 2 | `[M]` PMI formula ambiguity. |
| `test-1865` | -1.73 | -2.0 | 2 | `[I_HARD]` midpoint vs initial-value elasticity convention. |
| `test-1810` | 10120.0 | 10124.0 | 2 | `[M]` deferred interest / payment interaction. |
| `test-1890` | parse failure | 138000.0 | 2 | `[I_HARD]` cash-flow sign convention. |
| `test-1593` | 1000000.0 | 1150000.0 | 2 | `[M]` base contract value vs additional profit interpretation. |

## Difference From Plain Oracle

Cases where final correctness changed relative to `cot_basic_oracle_flash`:

| Case | Plain Oracle | Funnel-Guided Oracle |
|---|---|---|
| `test-1443` | wrong | correct |
| `test-1590` | wrong | correct |
| `test-1810` | wrong | correct |
| `test-1969` | correct | wrong |

Net effect: +2 correct cases overall (45/50 -> 47/50).

## Oracle Note Tags

Across 11 oracle rewrite calls:

- `[M]`: 5
- `[I_HARD]`: 5
- `[C]`: 1

The prompt was active and produced layer-labeled diagnostics, mostly at the intent/meaning and hard interpretation layers.
