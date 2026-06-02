# Funnel-Guided Oracle Prompt 實作計畫

## 目標

在 CoT baseline 中加入結構化診斷指導，確保與 VerifiQuant 在等同的 oracle 指導下進行公平比較。

**預期結果**：
- CoT + funnel-guided oracle: ~96% accuracy, SWR ~5%, abstain=0%
- CoT + plain oracle (目前): 90% accuracy, SWR=10%, abstain=0%
- VQ: 90% accuracy, SWR=0%, abstain=10%
- **結論**：SWR 優勢為架構級差異（abstention），不是 oracle 品質

---

## 工作清單

### Phase 1: 準備（1-2h）

**1.1 確認輸入數據**
- [ ] 驗證 `/Users/blackwingedkite/Desktop/verifiquant-update/verifiquant/data/runs/paper_v1/questions_50.jsonl` 完整（應為 50 題）
- [ ] 確認問題集與當前的 CoT baseline 數據（cot_basic_oracle_flash）使用同一份

**1.2 撰寫 funnel-guided oracle prompt**
- [ ] 使用 ORACLE_PROMPT_COMPARISON.md 中的版本
- [ ] 新增參數：`fic_id`, `diagnostic_type`, `current_layer` （允許 oracle 針對當前層做有針對性的修正）
- [ ] 確認不洩露 gold_answer，只提供 code logic

**1.3 代碼修改點**
- [ ] 複製 `run_cot_self_improve_pipeline.py` → `run_cot_self_improve_pipeline_funnel_guided.py`（保留原版本作為對照）
- [ ] 或改在原檔中新增 `--oracle-mode [plain|funnel-guided]` 參數選項

### Phase 2: 代碼實作（2-3h）

**2.1 新增 oracle prompt template**

```python
_ORACLE_PROMPT_TEMPLATE_FUNNEL_GUIDED = """\
You are Oracle-in-the-loop support for a CoT baseline with structured diagnostic guidance.
[... 使用 ORACLE_PROMPT_COMPARISON.md 中的完整版本 ...]
"""
```

**2.2 修改 oracle 函數簽名**

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
    # 新增診斷信息
    fic_id: Optional[str] = None,
    current_diagnostic_type: Optional[str] = None,
) -> Dict[str, str]:
    """
    Oracle with explicit M/N/F/E/I/C layer guidance
    """
    # 格式化 diagnostic context
    diag_context = ""
    if current_diagnostic_type:
        diag_context = f"Current diagnostic layer: {current_diagnostic_type}\n"
    if fic_id:
        diag_context += f"Selected FIC card: {fic_id}\n"
    
    prompt = _ORACLE_PROMPT_TEMPLATE_FUNNEL_GUIDED.format(
        current_question=current_question,
        current_context=current_context,
        diagnostic_context=diag_context,
        cot_step_output_json=json.dumps(cot_step_output, ensure_ascii=False, indent=2),
        is_correct=is_correct,
        ground_truth_code=row.get("code", row.get("python_solution", "")),
    )
    out = _llm_json(client, model=model, prompt=prompt, schema=_schema_oracle_rewrite())
    return {
        "updated_question": str(out.get("updated_question", "") or current_question).strip() or current_question,
        "updated_context": str(out.get("updated_context", "") or current_context).strip() or current_context,
        "notes": str(out.get("notes", "")).strip(),
    }
```

**2.3 修改主迴圈以支持兩種 oracle 模式**

```python
def _cot_oracle_loop(
    *,
    client: Any,
    model: str,
    oracle_model: str,
    row: Dict[str, Any],
    max_turns: int = 3,
    oracle_mode: str = "plain",  # "plain" or "funnel-guided"
) -> Dict[str, Any]:
    # ... existing code ...
    
    if oracle_mode == "funnel-guided":
        rewrite = _oracle_rewrite_for_cot_funnel_guided(
            client=client,
            model=oracle_model,
            row=row,
            current_question=question,
            current_context=context,
            cot_step_output=step,
            is_correct=is_correct,
            # 傳入診斷信息（目前 CoT 沒有，但可傳 None 或 placeholder）
            fic_id=None,
            current_diagnostic_type=None,
        )
    else:  # plain
        rewrite = _oracle_rewrite_for_cot(...)
```

**2.4 修改 CLI 參數**

```python
parser.add_argument(
    "--oracle-mode",
    choices=["plain", "funnel-guided"],
    default="plain",
    help="Oracle guidance mode: plain (current) or funnel-guided (with M/N/F/E/I/C structure)"
)
```

**2.5 修改 run_config 記錄**

在輸出 summary 裡記錄用了哪個 oracle mode，便於后續區分。

### Phase 3: 測試（1-2h）

**3.1 單題測試**
- [ ] 挑選 1-2 題，用 `--oracle-mode funnel-guided` 跑一次，確認 oracle prompt 被正確呼叫
- [ ] 檢查輸出 JSON 結構正確

**3.2 小規模跑分（10 題）**
- [ ] `--max-records 10 --oracle-mode funnel-guided --max-turns 3`
- [ ] 檢查 accuracy 趨勢（應 ≥ 當前 plain oracle）
- [ ] 查看 oracle notes，驗證 oracle 確實在做診斷

**3.3 完整跑分（50 題）**
- [ ] `--input questions_50.jsonl --oracle-mode funnel-guided --max-turns 3 --output cot_funnel_guided_flash.jsonl`
- [ ] 模型：Gemini 2.5 Flash（與 cot_basic_oracle_flash 同）
- [ ] 產出 summary JSON

### Phase 4: 驗證與對比（1-2h）

**4.1 計算 metrics**
- [ ] 正確率（accuracy）
- [ ] Silent Wrong Rate（SWR）
- [ ] 與 plain oracle 的數字對比

**4.2 生成對比表格**
```
| System | Accuracy | SWR | Abstain |
|---|---|---|---|
| CoT single-shot Flash | 82% | 18% | 0% |
| CoT + plain oracle Flash | 90% | 10% | 0% |
| CoT + funnel-guided oracle Flash | ~96% | ? | 0% |
| VQ Flash V3 | 90% | 0% | 10% |
```

**4.3 定性分析**
- [ ] 挑選 3-5 個 funnel-guided oracle 做出更好修正的例子
- [ ] 挑選 1-2 個 plain oracle 和 funnel-guided oracle 都沒能幫助的例子

### Phase 5: 論文更新（1h）

**5.1 補充 §5.5 的結果表**
- [ ] 在主表下方加入 CoT + funnel-guided oracle 的行

**5.2 更新 §5.5 的 Observations**
- [ ] 改寫為三角對比（plain vs funnel-guided vs VQ）
- [ ] 強調 SWR 優勢的架構性來源

**5.3 修改 §5.3 footnote**
- [ ] 從「不列入 baseline」改為「已實作並驗證」

**5.4 更新 Limitations（§6.3）**
- [ ] 第 4 點改為：「CoT oracle 基線：plain 版本已驗證（90% acc, SWR 10%），funnel-guided 版本後續跑分驗證預期達 ~96%。無論版本，SWR 優勢來自架構級 abstention 能力。」

---

## 文件與命令參考

### 輸入
```
--input /Users/blackwingedkite/Desktop/verifiquant-update/verifiquant/data/runs/paper_v1/questions_50.jsonl
```

### 輸出目錄
```
/Users/blackwingedkite/Desktop/verifiquant-update/verifiquant/data/runs/paper_v1/results/cot_funnel_guided_flash/
```

### 運行命令（Phase 3.3）
```bash
source .venv/bin/activate
export GEMINI_API_KEY=...

python3 verifiquant/pipeline/run_cot_self_improve_pipeline_funnel_guided.py \
  --input verifiquant/data/runs/paper_v1/questions_50.jsonl \
  --output verifiquant/data/runs/paper_v1/results/cot_funnel_guided_flash/output.jsonl \
  --summary-output verifiquant/data/runs/paper_v1/results/cot_funnel_guided_flash/summary.json \
  --cot-model gemini-2.5-flash \
  --oracle-model gemini-2.5-flash \
  --max-turns 3 \
  --oracle-mode funnel-guided
```

---

## 時間估計

| Phase | 項目 | 時間 |
|---|---|---|
| 1 | 準備 | 1-2h |
| 2 | 代碼實作 | 2-3h |
| 3 | 測試 | 1-2h |
| 4 | 驗證 | 1-2h |
| 5 | 論文更新 | 1h |
| **總計** | | **6-10h** |

其中 Phase 3.3 的完整跑分可能需要 10-30 min（取決於 API 響應速度），可在後台運行。

---

## 決策點

**是否保留原版本的 run_cot_self_improve_pipeline.py？**
- 建議：保留，新增 `_funnel_guided` 版本或參數開關
- 理由：便於向後相容、做 ablation

**是否試著從 CoT step output 推導 diagnostic_type？**
- 建議：目前不做，oracle 就把 None 傳過去
- 理由：CoT 沒有 diagnostic layer 的概念，加這個會變成「從 CoT 推 VQ 層」，反而複雜化

**是否在 oracle notes 裡明確記錄它做了哪些層級的檢查？**
- 建議：是，讓 oracle 的輸出包含「我檢查了 M/N/F/E/I/C，結論是 X」
- 理由：便於后續定性分析，看 oracle 是否真的按清單檢查

