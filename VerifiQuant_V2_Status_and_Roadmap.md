這份更新後的 **VerifiQuant V2** 文件將我們討論的「多層防禦漏斗（Multi-layer Diagnostic Funnel）」完整整合，並深化了關於 **I-Class (Critic Agent)** 與 **N-Class (Scope Boundary)** 的定義。這不再只是個工具說明，而是一份定義「高可靠性金融 AI」的技術準則。

---

# VerifiQuant V2：合約化診斷與多層防禦漏斗框架
**Status:** Research & Development (V2 Evolution)  
**Core Vision:** 從「機率性答題」轉向「確定性審計」，建立金融領域的 **Trust-but-Verify** 範式。

---

## 1. 核心挑戰：超越「最終答案正確率」
傳統 LLM 評測（如 FinanceReasoning）僅關注 Outcome Accuracy，但在金融實務中，以下三種「隱性失敗」更為致命：
1. **Silent Failures (自信的錯誤)**：模型在資訊不足或理解偏差時，仍給出確定的數字。
2. **Reasoning Crumbling**：理解對了、公式選對了，但算術（CoT）算錯了。
3. **Semantic Drift**：忽略了金融常識（如匯率方向、年初年末定義），導致計算基礎錯誤。

**VerifiQuant V2 的目標：** 透過 **Financial Inference Contract (FIC)** 與 **多層診斷漏斗**，將風險攔截在計算執行之前。

---

## 2. VerifiQuant V2 多層防禦漏斗 (Taxonomy)
我們將推理過程拆解為六個可診斷的階段，每一層都有其專屬的攔截機制（Exit Valves）。

| 漏斗層級 | 代碼 | 名稱 | 診斷定義 | 系統行動 (System Action) |
| :--- | :--- | :--- | :--- | :--- |
| **1. Intent** | **M** | **M**isunderstanding | 語義模糊，無法對應到任何 FIC 家族。 | **Refusal**: 主動引導用戶縮小範圍。 |
| **2. Scope** | **N** | **N**ot Supported | 意圖明確，但系統庫內暫無對應公式。 | **Graceful Exit**: 宣告知識邊界，防止幻覺。 |
| **3. Schema** | **F** | **F**ormula Spec | FIC 已選定，但缺少必要輸入參數。 | **Slot-filling**: 精確要求用戶補足變數。 |
| **4. Boundary** | **E** | **E**xtraction/Value | 參數齊全但數值荒謬（如 $Rf < 0$）。 | **Deterministic Alert**: 基於規則要求確認。 |
| **5. Critic** | **I** | **I**nterception | **隱性歧義**（匯率方向、時間基準等）。 | **Critic Intervention**: 啟動 HITL 對齊。 |
| **6. Logic** | **C** | **C**alculation | 最終執行。錯誤多來自代碼或數據標註。 | **Audit Log**: 100% 確定性 Python 執行。 |



---

## 3. 核心組件：FIC v2 (Financial Inference Contract)
FIC v2 不再只是模板，而是一份**計算合約**。

### 3.1 結構化 Schema
* **Discovery Metadata**: `selection_hints` 用於 RAG 精準匹配。
* **Input Contract**: 嚴格定義類型、單位、必要性。
* **Execution Block**: **Deterministic Python Code**，禁止 LLM 在此處進行任何算術運算。
* **Static Guardrails (E-Class)**:
    * `invariants`: 邏輯不變量（例如：$Growth Rate < Discount Rate$）。
    * `scale_checks`: 數值區間合理性（例如：$0 < PE < 1000$）。
* **Semantic Hints (I-Class)**: 專供 **Critic Agent** 使用，列出該公式常見的「陷阱」語義（如：「請檢查現金流是期初還是期末發生」）。

---

## 4. 系統工作流 (The Antigravity Pipeline)

1. **Mapping (M/N)**:
   - 透過 RAG 檢索最相關的 3 張 FIC。
   - 若信心值低於閾值（M）或無相關卡片（N），直接中斷並回報。
2. **Binding & Extraction (F)**:
   - LLM 將 `question + context` 中的數據填入 FIC 的 inputs。
   - 自動檢測是否有 `required: true` 的變數缺失。
3. **Logic Validation (E)**:
   - 執行 FIC 內的 `diagnostics` 代碼，檢查數值是否有悖金融常理。
4. **Semantic Alignment (I)**:
   - **Critic Agent** 介入：讀取 FIC 的 `semantic_hints`，審視用戶原始問題是否存在「隱性歧義」。
   - 若有風險，產生 `ClarificationRequest` 給用戶。
5. **Deterministic Execution (C)**:
   - 只有通過上述所有 Gate，系統才會執行 Python 代碼並輸出結果。

---

## 5. 實驗設計：從「跑分」到「壓力測試」
為了證明 V2 的優越性，實驗將專注於 **"Safety-Critical Scenarios"**。

### 7.1 陷阱資料集 (Trap Datasets)
我們將基於標準金融題目（如 FinanceReasoning）改造出五種測試集：
- **Normal**: 標準題目，測試基本正確率。
- **Ambiguous-M**: 移除關鍵字，測試系統是否會亂猜。
- **OOD-N**: 放入庫存公式無法處理的問題，測試系統是否知難而退。
- **Incomplete-F**: 移除一個必要數值，測試動態補值能力。
- **Trap-E/I**: 給出荒謬數值或具備歧義的敘述（如沒說匯率方向），測試攔截率。

### 7.2 評估指標 (Key Metrics)
* **Reliability Calibration**: $\frac{\text{Correctly Refused}}{\text{Total Unsolvable}}$ (理想值 1.0)。
* **Silent Wrong Rate**: 流程顯示 Success 但答案錯誤的比率 (理想值 0%)。
* **Alignment Efficiency**: 達成正確對齊所需的人機對話輪次。

---

## 6. V2 階段性里程碑
1. **[Q2] 知識工程化**: 將 `python_solution` 批量轉化為帶有 `semantic_hints` 的 FIC v2。
2. **[Q2] Critic Agent 實作**: 開發專門負責「找麻煩」的 LLM 提示工程，對接 I-Class 攔截。
3. **[Q3] 閉環測試**: 在 M/N/F/E/I/C 全鏈路上運行測試，對比 GPT-4 直接 CoT 的失敗率。

---

## 7. 總結
VerifiQuant V2 追求的是**「可預測的安全性」**。我們承認 LLM 具有不可控性，因此我們不改善 LLM 的算術能力，而是建立一個強大的「體制（Funnel）」，讓 LLM 在規範好的「合約（FIC）」下工作，確保每一筆輸出的財金數據都能被審計、被追蹤、被信任。

---

**這份文件現在是否符合你對 V2 完整邏輯的期待？**
如果您準備好了，我們可以針對 **「第 4 步：Semantic Alignment (I-Class)」** 進行更深入的 Prompt 設計，或是開始定義 **FIC v2 的 JSON 範例**，以便自動化生成 pipeline。