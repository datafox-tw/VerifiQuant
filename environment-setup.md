# VerifiQuant V2 Environment Setup & Pipeline Runbook

本文件整理目前 VerifiQuant V2（M/N/F/E/I/C）可直接執行的後端流程，包含：
- 前處理（建卡、入庫、擴展 Trap）
- 主流程（診斷 + deterministic execution）
- 兩個 iterative 對照流程（framework-guided vs pure CoT）
- 視覺化輸出

## 0. Prerequisites

在專案根目錄執行：

```bash
cd /Users/blackwingedkite/Desktop/verifiquant-update
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

設定金鑰（建議放 `.env`）：

```bash
cat > .env <<'EOF'
GEMINI_API_KEY=YOUR_KEY_HERE
EOF
```

每次執行前載入環境變數：

```bash
set +a
source .env
set -a
```

## 1. Pipeline 1: Dataset -> FIC v2 Cards（core/retrieval/repair）

用途：把 `question + context + python_solution (+ answer)` 轉成三卡。

```bash
python3 preprocessing/dataset_case_to_fic.py \
  --input verifiquant/data/testing_10Q.jsonl \
  --core-output /tmp/v2/core.jsonl \
  --retrieval-output /tmp/v2/retrieval.jsonl \
  --repair-output /tmp/v2/repair.jsonl \
  --max-records 10
```

常用可選參數：
- `--model gemini-2.5-flash`：三階段共用模型。
- `--stage1-model/--stage2-model/--stage3-model`：覆寫各階段模型。
- `--db-url sqlite:////tmp/v2/cards.db`：生成後直接入庫。
- `--disallow-new-topic`：topic 必須已在既有 taxonomy 中。

Input 預覽（單筆）：
```json
{
  "question": "...",
  "context": "...",
  "python_solution": "...",
  "ground_truth": 10185,
  "question_id": "test-1045"
}
```

Output 預覽（core card 節錄）：
```json
{
  "fic_id": "fic_xxx",
  "domain": "corporate_finance",
  "topic": "npv",
  "inputs": [...],
  "diagnostics": {"invariants": [], "scale_checks": []},
  "semantic_hints": [...],
  "execution": {"language": "python", "entrypoint": "compute", "code": "..."}
}
```

## 2. Pipeline 2: Cards -> SQLAlchemy Store（RAG）

用途：將三卡寫入 DB，並可做檢索 smoke check。

```bash
python3 preprocessing/build_card_store.py \
  --db-url sqlite:////tmp/v2/cards.db \
  --core /tmp/v2/core.jsonl \
  --retrieval /tmp/v2/retrieval.jsonl \
  --repair /tmp/v2/repair.jsonl
```

可選：
- `--query "calculate NPV"`：即時測檢索。
- `--top-k 3`：檢索回傳數量。

## 3. Pipeline 3: Expansion / Trap Dataset

用途：從乾淨題生成 `M/N/F/E/I` 擾動版本。

```bash
python3 preprocessing/expand_cases.py \
  --input verifiquant/data/testing_10Q.jsonl \
  --output /tmp/v2/expanded.jsonl \
  --mode semi-llm \
  --max-records 10
```

可選：
- `--mode llm|semi-llm`（建議 `semi-llm`）。
- `--model gemini-2.5-flash`。

Output 預覽（單筆）：
```json
{
  "case_id": "test-1045_E_trap",
  "variant_type": "E_trap",
  "expected_status": "alert",
  "expected_diagnostic_type": "E",
  "question": "...",
  "context": "..."
}
```

## 4. Pipeline 4+5: 主流程（錯誤診斷 + 正常題 deterministic trace）

用途：同一支流程同時處理：
- 有問題：輸出 `M/N/F/E/I/C` 診斷與 repair hints
- 無問題：輸出 `success + output_value + execution_trace`

### 4A. 使用 DB 載卡（建議）
```bash
python3 preprocessing/run_error_classification_pipeline.py \
  --input /tmp/v2/expanded.jsonl \
  --db-url sqlite:////tmp/v2/cards.db \
  --output /tmp/v2/run_output.jsonl \
  --top-k 3 \
  --max-records 0
```

### 4B. 直接讀 JSON 卡
```bash
python3 preprocessing/run_error_classification_pipeline.py \
  --input /tmp/v2/expanded.jsonl \
  --core /tmp/v2/core.jsonl \
  --retrieval /tmp/v2/retrieval.jsonl \
  --repair /tmp/v2/repair.jsonl \
  --output /tmp/v2/run_output.jsonl
```

常用可選參數：
- `--selector-model / --extractor-model / --judge-model`
- `--m-min-top-score 0.05`：M/N 前置閾值
- `--top-k 3`：RAG 候選卡數

Output 預覽（診斷）：
```json
{
  "case_id": "test-1045_I_trap",
  "status": "needs_clarification",
  "diagnostic_type": "I",
  "funnel_layer": "Critic",
  "gate_action": "critic_intervention",
  "clarification_request": {
    "questions": ["..."],
    "options": ["A", "B"]
  }
}
```

Output 預覽（成功）：
```json
{
  "case_id": "test-1045_clean",
  "status": "success",
  "diagnostic_type": "None",
  "funnel_layer": "Logic",
  "gate_action": "audit_log",
  "output_value": 10185.0,
  "is_correct": true,
  "execution_trace": {"engine": "deterministic_python", "entrypoint": "compute"}
}
```

## 5. 新增 Iterative Pipelines（最多 3 輪）

## 5.1 Framework-guided self-improve（使用 VerifiQuant 輸出驅動修復）

```bash
python3 preprocessing/run_framework_guided_self_improve_pipeline.py \
  --input verifiquant/data/testing_10Q.jsonl \
  --db-url sqlite:////tmp/v2/cards.db \
  --output /tmp/v2/framework_guided.jsonl \
  --summary-output /tmp/v2/framework_guided_summary.json \
  --max-records 10 \
  --max-turns 3
```

可選：
- `--oracle-model`（Oracle-in-the-loop rewriter）
- `--selector-model/--extractor-model/--judge-model`

## 5.2 Pure CoT self-improve（不使用 VerifiQuant 診斷）

```bash
python3 preprocessing/run_cot_self_improve_pipeline.py \
  --input verifiquant/data/testing_10Q.jsonl \
  --output /tmp/v2/cot_self_improve.jsonl \
  --summary-output /tmp/v2/cot_self_improve_summary.json \
  --max-records 10 \
  --max-turns 3
```

可選：
- `--cot-model`
- `--oracle-model`

## 5.3 一次跑兩個比較流程（A/B）

```bash
python3 preprocessing/run_iterative_agents_pipeline.py \
  --input verifiquant/data/testing_10Q.jsonl \
  --db-url sqlite:////tmp/v2/cards.db \
  --output /tmp/v2/iterative_compare.jsonl \
  --summary-output /tmp/v2/iterative_summary.json \
  --max-records 10 \
  --max-turns 3
```

## 6. Pipeline 6: 視覺化與指標

用途：把 `expanded input` 與 `run output` 對齊，產出 summary + dashboard。

```bash
python3 preprocessing/visualize_expand_eval.py \
  --expanded-input /tmp/v2/expanded.jsonl \
  --run-output /tmp/v2/run_output.jsonl \
  --outdir /tmp/v2/viz
```

輸出：
- `/tmp/v2/viz/merged_eval.jsonl`
- `/tmp/v2/viz/summary.json`
- `/tmp/v2/viz/dashboard.html`

主要指標（summary 內）：
- `full_match_rate_pct`
- `e_trap_alert_rate_pct`
- `i_trap_clarification_rate_pct`
- `diag_confusion_expected_vs_actual`

## 7. 一鍵順序（建議最小可重現）

```bash
cd /Users/blackwingedkite/Desktop/verifiquant-update
set +a && source .env && set -a

python3 preprocessing/dataset_case_to_fic.py \
  --input verifiquant/data/testing_10Q.jsonl \
  --core-output /tmp/v2/core.jsonl \
  --retrieval-output /tmp/v2/retrieval.jsonl \
  --repair-output /tmp/v2/repair.jsonl \
  --max-records 10

python3 preprocessing/build_card_store.py \
  --db-url sqlite:////tmp/v2/cards.db \
  --core /tmp/v2/core.jsonl \
  --retrieval /tmp/v2/retrieval.jsonl \
  --repair /tmp/v2/repair.jsonl

python3 preprocessing/expand_cases.py \
  --input verifiquant/data/testing_10Q.jsonl \
  --output /tmp/v2/expanded.jsonl \
  --mode semi-llm \
  --max-records 10

python3 preprocessing/run_error_classification_pipeline.py \
  --input /tmp/v2/expanded.jsonl \
  --db-url sqlite:////tmp/v2/cards.db \
  --output /tmp/v2/run_output.jsonl

python3 preprocessing/visualize_expand_eval.py \
  --expanded-input /tmp/v2/expanded.jsonl \
  --run-output /tmp/v2/run_output.jsonl \
  --outdir /tmp/v2/viz
```

## 8. 常見問題

- `Missing GEMINI_API_KEY in environment.`  
  請確認 `source .env` 已執行。

- `google.genai import failed`  
  請重新安裝依賴：`.venv/bin/python -m pip install -r requirements.txt`

- `When --db-url is not provided, --core --retrieval --repair are required.`  
  主流程需擇一：`--db-url` 或三張卡檔案路徑。
