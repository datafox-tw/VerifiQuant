#!/bin/bash
# Wave 1 remaining chain: VQ-Pro on/off ablation -> Pro CoT 250Q -> PAL -> SC@5.
# Each step continues on failure. ~5 h total.
# Usage: GEMINI_API_KEY=... bash scripts/run_wave1_chain.sh
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LOG=verifiquant/data/runs/wave1_logs
mkdir -p "$LOG"

step() {  # step <name> <cmd...>
  local name="$1"; shift
  echo "=== [$(date '+%F %T')] STEP $name start ==="
  PYTHONPATH=. "$@" > "$LOG/$name.log" 2>&1
  echo "=== [$(date '+%F %T')] STEP $name done (exit $?) ==="
}

step vqpro_off  $PY scripts/run_vq50_pro.py --tag pro_off --model gemini-2.5-pro
step vqpro_on   env VERIFIQUANT_STRICT_BINDING=1 $PY scripts/run_vq50_pro.py --tag pro_on --model gemini-2.5-pro
step procot250  $PY preprocessing/run_cot_self_improve_pipeline.py \
  --input verifiquant/data/runs/paper_v2_250/questions_250.jsonl \
  --output verifiquant/data/runs/paper_v2_250/results/cot_single_shot_pro/output.jsonl \
  --summary-output verifiquant/data/runs/paper_v2_250/results/cot_single_shot_pro/summary.json \
  --max-turns 1 --cot-model gemini-2.5-pro
step pal250     $PY scripts/run_pal_baseline.py --model gemini-2.5-flash
step sc5_250    $PY scripts/run_sc_baseline.py --k 5 --model gemini-2.5-flash
echo "=== WAVE1 CHAIN COMPLETE $(date '+%F %T') ==="
