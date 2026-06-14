# Runbook：paper_v2_250（250Q 擴大實驗）本地執行指令

所有指令在專案根目錄 `/Users/blackwingedkite/Desktop/verifiquant-update` 執行。

⚠️ **關鍵雷點**：`preprocessing/*` 與 `verifiquant/pipeline/*` 不會自動讀 `.env`，必須先
`set -a; source .env; set +a`（否則 `Missing GEMINI_API_KEY`）。
唯一例外是 `scripts/run_paper_experiments.py`（它內部有 `load_dotenv()`）。

---

## Phase 0 — 抽樣（已完成）
產出 `verifiquant/data/runs/paper_v2_250/questions_250.jsonl`（180 medium + 70 hard，superset of 50）。
若要重抽：`python3 scripts/sample_dataset_250.py`

## Phase 1 — FIC 卡片生成（進行中，自癒守護）
進度查詢：
```bash
wc -l verifiquant/data/runs/paper_v2_250/fic/core.jsonl   # 目標 250
tail -5 verifiquant/data/runs/paper_v2_250/fic/supervisor.log
```
若守護程序掛了（完全關機後）重啟（會從 checkpoint 接續，不重跑已完成卡）：
```bash
set -a; source .env; set +a
pkill -9 -f "Python.*dataset_case_to_fic"; pkill -9 -f supervise_card_build
nohup ./scripts/supervise_card_build.sh \
  >> verifiquant/data/runs/paper_v2_250/fic/supervisor.log 2>&1 < /dev/null &
disown
```

## Phase 1b–3 — card store + CoT + VQ Flash（一條龍）
卡片跑滿 250 後，**一個指令**就跑完 card store 建置 + CoT 變體 + VQ Flash（config 已設定好
這三個 pipeline；已存在的產物會自動 skip）：
```bash
python3 scripts/run_paper_experiments.py \
  --config verifiquant/data/runs/paper_v2_250/experiment_config.yaml
```
輸出在 `verifiquant/data/runs/paper_v2_250/results/<baseline>/`：
- `vq_flash`（VQ K=3, Flash）
- `cot_single_shot_flash`
- `cot_basic_oracle_flash`（blind oracle K=3）

只跑某幾個：加 `--only vq_flash cot_single_shot_flash`
強制重跑：加 `--force`

> 注意：VQ 跑 250 題也很重、一樣可能因斷網/睡眠卡住。建議跑這步時別讓電腦睡眠。
> 若中途斷，重跑同指令即可（有 output.jsonl 的 baseline 會 skip；單一 baseline 內部
> 若沒 checkpoint 則該 baseline 會整個重跑）。

## Phase 1c — 卡片良率（go/no-go 關卡）
```bash
# 1) build 階段已產出的驗證報告
cat verifiquant/data/runs/paper_v2_250/fic/validation_report.json
# 2) cross-artifact 完整性
set -a; source .env; set +a
python3 verifiquant/preprocessing/validate_relations.py \
  --core verifiquant/data/runs/paper_v2_250/fic/core.jsonl \
  --retrieval verifiquant/data/runs/paper_v2_250/fic/retrieval.jsonl \
  --repair verifiquant/data/runs/paper_v2_250/fic/repair.jsonl
```
（這步我可以幫你把結果彙整成「250 張卡有幾張 smoke/relation 失敗」的良率數字。）

## Phase 4 — IMR（Input-Mutation Rate, Flash vs Pro）
**尚未實作**。需要在 binding 階段加 instrumentation（log 綁定值 vs context 字面 span）。
等卡片 + 主跑分穩定後再做；這是驗證「Pro over-normalization」的便宜決定性實驗。

## 已完成、不需重跑
- Refusal prompt ablation（K=1 12-cell grid + K=6 multi-K 6 cells + placebo 控制）
  → `docs/results/2026-06-09_refusal_prompt_ablation.md`
