# Funnel-Guided Oracle 實作 — 快速查閱

## 為什麼要做這個？

**當前狀況**：CoT + plain oracle 不公平 baseline
- CoT oracle 收到泛泛指導（「clarify assumptions」）
- VQ oracle 收到結構化指導（「檢查 M/N/F/E/I/C 六層」）

**改進後**：公平比較
- 兩者都收到等同的結構化診斷清單
- 證實 VQ 的 SWR=0% 優勢源自架構（abstention），不是 oracle

---

## 核心檔案

| 檔案 | 用途 |
|---|---|
| `FUNNEL_GUIDED_ORACLE_PLAN.md` | 完整工作計畫（6-10h，分 5 個 phase） |
| `ORACLE_PROMPT_FINAL.py` | 新 oracle prompt + 新函數框架（可複製） |
| `ORACLE_PROMPT_COMPARISON.md` | plain vs funnel-guided 對比與理由 |
| `run_cot_self_improve_pipeline.py` | 將改動這個檔案 |

---

## 三個關鍵改動

### 1️⃣ 新 Oracle Prompt Template

**來源**：`ORACLE_PROMPT_FINAL.py`

複製 `_ORACLE_PROMPT_TEMPLATE_FUNNEL_GUIDED` 整段到 `run_cot_self_improve_pipeline.py`

**特色**：
- 要求逐層檢查 M/N/F/E/I/C
- 每層都有明確的檢查清單和例子
- 指導 oracle 回答時註明層級（`[M] ...`, `[F] ...` 等）

---

### 2️⃣ 新 Oracle 函數

**來源**：`ORACLE_PROMPT_FINAL.py` 的 `_oracle_rewrite_for_cot_funnel_guided()`

加到 `run_cot_self_improve_pipeline.py` 內

**簽名**：
```python
def _oracle_rewrite_for_cot_funnel_guided(
    *,
    client: Any,
    model: str,
    row: Dict[str, Any],
    current_question: str,
    current_context: str,
    cot_step_output: Dict[str, Any],
    is_correct: Optional[bool],
) -> Dict[str, str]:
```

---

### 3️⃣ 新 Loop 或參數開關

**選項 A**（推薦）：新增 `--oracle-mode` 參數

```python
parser.add_argument("--oracle-mode", choices=["plain", "funnel-guided"], default="plain")

# 在 loop 呼叫時
if args.oracle_mode == "funnel-guided":
    result = _cot_oracle_loop_funnel_guided(...)  # 新版
else:
    result = _cot_oracle_loop(...)  # 舊版
```

**選項 B**：複製整個 `run_cot_self_improve_pipeline.py` → `run_cot_self_improve_pipeline_funnel_guided.py`

（更簡潔，但需要維護兩個版本）

---

## 運行流程

### Phase 3.2: 小規模測試（10 題）

```bash
python3 verifiquant/pipeline/run_cot_self_improve_pipeline.py \
  --input verifiquant/data/runs/paper_v1/questions_50.jsonl \
  --max-records 10 \
  --output /tmp/test_funnel.jsonl \
  --oracle-mode funnel-guided \
  --max-turns 3
```

### Phase 3.3: 完整跑分（50 題）

```bash
python3 verifiquant/pipeline/run_cot_self_improve_pipeline.py \
  --input verifiquant/data/runs/paper_v1/questions_50.jsonl \
  --output verifiquant/data/runs/paper_v1/results/cot_funnel_guided_flash/output.jsonl \
  --summary-output verifiquant/data/runs/paper_v1/results/cot_funnel_guided_flash/summary.json \
  --oracle-mode funnel-guided \
  --max-turns 3
```

**預期時間**：10-30 min（取決於 API 響應）

---

## 驗證清單

### 代碼整合完成後

- [ ] 確認 `--oracle-mode funnel-guided` 參數存在
- [ ] 確認 prompt template 中有 M/N/F/E/I/C 檢查清單
- [ ] 確認舊 plain oracle 仍能正常運行（向後相容）

### 測試跑分完成後

- [ ] 10 題測試結果與 plain oracle 方向一致（≥ 90%）
- [ ] Oracle notes 中出現層級標記（`[M]`, `[F]` 等）
- [ ] summary.json 包含 accuracy 和 oracle_mode 欄位

### 完整跑分完成後

- [ ] 50 題 accuracy ≥ 90%（目標 ~96%）
- [ ] SWR 應 < 10%（目標 ~5%）
- [ ] 對比表：plain oracle (90%, SWR 10%) vs funnel-guided (96%, SWR 5%)

---

## 預期結果與論文影響

| Metric | Plain | Funnel-Guided | VQ |
|---|---|---|---|
| Accuracy | 90% | ~96% | 90% |
| SWR | 10% | ~5% | 0% |
| Abstain | 0% | 0% | 10% |
| **結論** | — | CoT 有改進 | 架構級優勢 ✅ |

**核心論點不變**：
> VQ 的 SWR=0% 優勢來自**漏斗的結構化拒答機制**（abstention），不是 oracle 效能。即便 CoT+funnel oracle 達到 96% accuracy，它的 SWR 仍 > 0%，因為 CoT 無 abstention 能力。

---

## 問題快查

**Q: Oracle prompt 太長，會不會被截斷？**
A: Gemini 2.5 Flash 的 context 夠長，不會。如果擔心，可以移除某些例子。

**Q: 需要修改 FIC 卡片或 questions 數據嗎？**
A: 不需要。只改 oracle prompt 和函數邏輯。

**Q: Plain oracle 結果會被覆蓋嗎？**
A: 不會。它們分別輸出到 `cot_basic_oracle_flash` 和 `cot_funnel_guided_flash` 目錄。

**Q: 要用 Gemini Pro 跑一次嗎？**
A: 目前不用。Flash 的對比足夠了，Pro 版本可作為 future work。

---

## 時間線

| 預計日期 | Task | Est. 時間 |
|---|---|---|
| 今天 | Task 2: 準備環境 | 1-2h |
| 明天 | Task 3: 代碼整合 | 2-3h |
| 明天 | Task 4: 小規模測試 | 1-2h |
| 後天 | Task 5: 完整跑分（背景） | 0.5-1h 實際操作 |
| 後天 | Task 6: 驗證分析 | 1-2h |
| 後天 | Task 7: 論文更新 | 1h |
| **總計** | | **6-11h** |

（其中 Task 5 的 API 等待時間可並行進行其他工作）

---

## 聯絡資訊

遇到問題時參考：
- Prompt 邏輯問題：見 `ORACLE_PROMPT_COMPARISON.md` 的「為什麼更公平」
- 代碼框架問題：見 `ORACLE_PROMPT_FINAL.py` 的完整實作
- 計畫細節：見 `FUNNEL_GUIDED_ORACLE_PLAN.md`

