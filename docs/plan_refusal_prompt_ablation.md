# 草稿：CoT Refusal Prompt Ablation（clean 50Q）

狀態：**EXECUTED（2026-06-09）** — 結果見 [docs/results/2026-06-09_refusal_prompt_ablation.md](results/2026-06-09_refusal_prompt_ablation.md)。回應老師三點：(1) CoT safe refusal @ k=1 = 0% 是 prompt 造成的，需用不同強度 prompt 讓 CoT 自然拒答；(2) 用 GPT 做 ablation search；(3) 用 prompt 控制，`reasoning_effort` 分 low/medium。

---

## 1. 問題診斷（為什麼現在是 0%）

- `run_cot_self_improve_pipeline.py` 的 `_COT_PROMPT_TEMPLATE` 強制輸出數字；`needs_more_info` 欄位存在但 **loop 從不把它轉成棄答**，永遠回報 `final_answer`。
- `score_trap_set.py` 對 CoT 的棄答判定是「`answer is None`」，但 prompt 從沒給模型回空的許可 → safe refusal 結構性恆為 0。
- 因此 **0% 是設計造成，非 CoT 能力上限**。不補這個對照，論文「VQ 差異在失敗模式」的主張會被質疑為「沒給 baseline 拒答權利」。

## 2. clean set 上拒答的雙面性

clean 50Q 每題都有正解，拒答因此是 risk–coverage tradeoff：
- 對「L0 會答錯」的題拒答 = **good refusal**（SWR↓）
- 對「L0 會答對」的題拒答 = **over-refusal**（accuracy↓）

科學問題：**用 prompt 鼓勵拒答，CoT 能否選擇性地壓低 SWR，而非無差別亂拒？** VQ 的 abstention 若真是架構性的，CoT 即使被鼓勵也應在「該拒的題」上不如 deterministic gate 精準。

## 3. 實驗設計（定案）

### 軸 A — 拒答鼓勵強度（4 級）
| Level | 名稱 | prompt 內容要點 |
|---|---|---|
| L0 | Forced | 現況 baseline：強制數字，無拒答出口 |
| L1 | Permitted | 「資訊不足/有歧義時*可以* `decision=refuse`」 |
| L2 | Encouraged + criteria | 明列觸發條件：缺必要輸入、percent/decimal 歧義、period start/end、超出金融常規範圍… |
| L3 | Strict self-check | 先做 M/F/E/I 自檢，任一不過就 refuse（純 prompt 版 VQ funnel） |

### 軸 B — reasoning_effort：`low`, `medium`（GPT only）

### 軸 C — 模型（model-agnostic 對照）
- `gpt-5.2`（`GPT_FLASH`）：4 levels × {low, medium} = **8 cells**
- `gemini-2.5-flash`：4 levels × 1 = **4 cells**（Gemini 無 reasoning_effort；作 model-agnostic 對照，thinking 用預設）

總計 **12 cells**，全部 **K=1 single-shot**。

## 4. 度量（每 cell）

沿用論文 §5.4 selective-prediction 三元分類：
- `Correct / Silent-Wrong / Safe-Refusal`
- `Coverage = (Correct+SW)/N`、`Selective-Acc = Correct/(Correct+SW)`、`SWR = SW/(Correct+SW)`

**拒答品質（核心產出）**——對齊同模型同 effort 的 L0 逐題結果：
- `good_refusal`：L0 答錯且此 cell 拒答（救回的 SWR）
- `over_refusal`：L0 答對且此 cell 拒答（犧牲的 accuracy）

輸出一張 risk–coverage 表 + 每 level 的 good/over refusal 拆解。

## 5. 實作工作項

1. `pip install openai python-dotenv`；pipeline 入口加 `load_dotenv()`（目前需手動 export）。
2. **LLM backend 抽象**：新增 OpenAI 路徑（`chat.completions` + `response_format` json_schema + `reasoning_effort`），與 Gemini `_llm_json` 並存，`--provider {gemini,openai}` 與 `--reasoning-effort {low,medium}` 切換。
3. **讓棄答成真**：schema 加 `decision ∈ {answer, refuse}` + `refusal_reason`；loop 把 `refuse` → `final_answer=None`。
4. **prompt 模板族 L0–L3**：以 `--refusal-level {0,1,2,3}` 參數化。
5. **clean-set ternary scorer**（新檔，目前缺）：算每-cell 指標 + 對照 L0 的 good/over refusal。
6. **ablation 編排腳本**：跑 12 cells、彙整成表、輸出 risk–coverage 圖資料。

## 6. 預期結論（兩種都可發表）

- **若 CoT 即使 L3 仍 over-refusal 高 / SWR 壓不到 0**：佐證 VQ abstention 是架構性的（deterministic gate 比 prompt 精準）。
- **若 L2/L3 讓 CoT 接近 VQ 的 SWR**：誠實揭露，並把 VQ 的優勢收斂到「結構化、可審計、可路由的拒答」(structured-catch) 而非單純 SWR 數字。

## 7. 開放決定（待確認）
- 拒答訊號用獨立 `decision` 欄位 vs 沿用 `needs_more_info`+空 answer → 傾向獨立 `decision`（語意明確、好 parse）。
- Gemini 是否也掃 thinking budget 當作 reasoning_effort 類比 → 目前不做，保持單一配置。
