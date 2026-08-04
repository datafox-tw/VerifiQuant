# R-PAL+ on the clean 250Q set (2026-08-04)

Replaces the 50-question pilot that previously supplied R-PAL$^{+}$'s clean coverage in
Figure 2, so all three autonomous systems now share one 250-question clean denominator.

## Why

Figure 2 plots clean coverage against trap FNAR. Plain R-PAL and VerifiQuant were measured on
the 250Q clean set, but R-PAL+ only had a 50Q pilot. That forced a paragraph of apparatus in
the paper (open diamond, 95% Wilson interval, "not a point on a fitted frontier", "filled marks
denote the common 250-question set") and left the frontier claim unearned.

## Runs

Same dataset, card DB, backbone, and retrieval depth as the plain R-PAL 250Q run, so the points
are directly comparable:

- dataset: `verifiquant/data/runs/paper_v2_250/questions_250.jsonl` (250 items)
- card DB: `verifiquant/data/runs/paper_v2_250/fic/cards.db`
- model: `gemini-2.5-flash`, `--top-k 4`, `--mode prompted`
- outputs: `verifiquant/data/runs/governance_eval/prompted_clean250_run{1,2,3}.jsonl`

```bash
set -a; . ./.env; set +a
PYTHONPATH=. python3 scripts/run_retrieval_pal_governance_eval.py \
  --mode prompted \
  --dataset verifiquant/data/runs/paper_v2_250/questions_250.jsonl \
  --db verifiquant/data/runs/paper_v2_250/fic/cards.db \
  --model gemini-2.5-flash --top-k 4 \
  --out-dir verifiquant/data/runs/governance_eval \
  --run-name prompted_clean250_runN --resume
```

Each run took ~34 min (250 items, ~8.1 s/item). All three completed with 250/250 unique
`case_id`s and no error signatures.

## Scoring

`scripts/score_rpal_plus_clean250.py` reuses `analyze_governance_eval.clean_labels_from_rpal`
and `clean_summary`, so coverage / accuracy / SWR / selective accuracy are computed by the
*same* functions used for plain R-PAL and VerifiQuant. No new metric definition was introduced.
Coverage = (correct + silent_wrong) / n.

## Results

| Run | coverage | accuracy | SWR | sel. acc | correct / SW / abstain |
|---|---|---|---|---|---|
| run1 | 93.2% | 91.6% | 1.6% | 98.3% | 229 / 4 / 17 |
| run2 | 93.6% | 93.2% | 0.4% | 99.6% | 233 / 1 / 16 |
| run3 | 92.0% | 91.2% | 0.8% | 99.1% | 228 / 2 / 20 |
| **mean** | **92.9%** (sd 0.83) | **92.0%** (sd 1.06) | **0.9%** (sd 0.61) | **99.0%** (sd 0.66) | 230 / 2.3 / 17.7 |

The 50Q pilot had reported 96.0% coverage. The 250Q value is **3.1 points lower**, i.e. the
pilot overstated coverage, and the three-run spread (0.83) shows the new figure is stable.

## Resulting Figure 2 frontier

All three points now sit on the same 250Q clean denominator and are monotone in both axes:

| System | clean coverage | trap FNAR |
|---|---|---|
| R-PAL (plain) | 99.2% | 48.7% |
| R-PAL+ (taxonomy prompt) | 92.9% | 27.5% |
| VerifiQuant (K=1) | 84.0% | 11.3% |

## Paper changes

- `\govPromptCovAll/AccAll/SWRAll/SelAccAll` macros added (single source of truth; the three
  raw run values are recorded in a comment beside them).
- Figure 2 prose: the pilot apparatus paragraph is deleted and replaced with a plain statement
  that all three systems share the 250 clean questions and trace a monotone frontier.
- `scripts/plot_safety_utility.py`: `comparable` now includes all three systems (the frontier
  line connects them), the open-diamond + Wilson errorbar branch is removed, the legend reads
  "taxonomy prompt", and the in-figure footnote no longer mentions a pilot.
- `mixed_risk_coverage.json`: `rpal_prompted` clean fields updated, `n_clean` 50 to 250.
- **Panel A now lists R-PAL+** (92.0% accuracy, 0.9% SWR, 99.0% selective accuracy).

## Honest consequence of adding R-PAL+ to Panel A

R-PAL+ beats autonomous VerifiQuant on the clean set on **both** accuracy (92.0% vs 81.2%) and
SWR (0.9% vs 2.8%). We report it rather than showing only its coverage in Figure 2, since
citing one clean-set number while withholding the rest would be selective. RQ1 now states this
outright and uses it: R-PAL+ leads on the clean set yet accepts more than twice the false
numbers on the traps (27.5% vs 11.3%), which is the cleanest available demonstration that
clean-set standing does not predict governed behavior.

## Note on the monitor bug found while running this

The first heartbeat monitor used `pgrep -fc` for liveness. `-c` is a Linux procps flag that
macOS `pgrep` does not have, so the call exited 2 and the `|| echo 0` fallback reported "no
process running" on every tick, producing a false crash alarm while run1 was healthy at
177/250. `pgrep -f` also matches the monitor's own shell. The replacement uses output-file
mtime staleness (`idle > 600s` while incomplete) instead of process counting, which also
catches a hung-but-alive process.
