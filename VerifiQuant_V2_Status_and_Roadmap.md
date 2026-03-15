# VerifiQuant V2：現況、目標與偵測方向

## 1. 問題定義
當前多數財金 LLM Benchmark 側重最終答案正確率（Outcome Accuracy），但這不足以回答以下關鍵問題：

1. 模型是「真的理解題意」還是剛好猜中？
2. 錯誤來自哪個層次：語意理解、公式規格、輸入綁定，還是算術誤差？
3. 系統在不確定時，是否會安全拒答與要求補充，而不是輸出看似合理但不可審計的數字？

VerifiQuant V2 的核心目標：  
把系統從「模板計算器」升級為「合約化診斷系統（Contract-Validated, Diagnostic-Driven Financial AI）」。

---

## 2. 目前系統狀況（V1 基線）
目前流程（已可運作）：

1. 檢索候選卡（RAG-like card retrieval）
2. LLM 選卡
3. LLM 抽取輸入值
4. 計算引擎執行（目前 v1 為符號/程式計算 + fallback）
5. 缺值或不匹配時直接拒答

### 2.1 已具備能力
- 卡片檢索與領域/主題過濾
- 選卡與輸入抽取
- 可重現計算步驟輸出
- 基礎拒答機制（例如缺必要輸入）

### 2.2 主要缺口
1. 缺少完整 FIC（Financial Inference Contract）結構化約束
2. 拒答邏輯偏粗（多為缺值即拒答，尚未提供診斷語義）
3. 缺少標準化 `DiagnosticReport` 與 taxonomy 映射
4. 尚未有完整的 M/F/E/C 量化評測框架

---

## 3. V2 核心主張（Main Claim）
**Contract 是模糊語義與確定性計算之間的唯一橋樑，也是責任分配協議。**

責任分工如下：

1. LLM：將自然語言問題整理成合格 Contract（FIC）或宣告失敗（Refusal）
2. Verification Layer：檢查不變量、尺度、規格與綁定一致性（Guardrails）
3. Execution Engine：基於固定公式卡與固定程式碼進行 deterministic 計算

---

## 4. 錯誤分類與偵測方向（M/F/E/C）

## 4.1 M — Misunderstanding（語義誤解）
- 定義：任務意圖不清或映射到錯誤公式家族
- 典型案例：NPV vs IRR vs ROI 混淆
- 系統策略：拒答 + 候選卡差異提示 + 反問澄清（Refusal with guided disambiguation）

## 4.2 F — Formula/Spec Mismatch（規格/公式錯配）
- 定義：必要規格缺失或不符題目條件
- 典型案例：缺 `discount_rate`、缺 benchmark、時間基準未定義
- 系統策略：Error + 動態補值欄位（請用戶補必要參數）

## 4.3 E — Input Binding Error（輸入綁定錯誤）
- 定義：公式正確但變數綁值、單位、尺度、方向錯誤
- 典型案例：`8` vs `0.08`、年/月頻率不一致、欄位誤綁
- 系統策略：Alert + 規則證據 + 建議修復（轉換、對調、人工確認）

## 4.4 C — Calculation Error（計算誤差）
- 定義：多步計算導致算術誤差
- V2 立場：透過 deterministic engine 盡可能壓低到接近 0

---

## 5. V2 最小 FIC 方向（已對齊）
FIC v2 最小核心欄位：

- `id`, `name`, `domain`, `topic`
- `inputs`
- `output_var`
- `execution`（Python deterministic code）
- `diagnostics.invariants`
- `diagnostics.scale_checks`
- `selection_hints`
- `refusal_hints`

關鍵原則：

1. 以 `python_solution` 為主要計算邏輯來源
2. 以 `function` 為概念泛化與語義輔助
3. 去除題目常數與敘事細節，保留可重用公式結構
4. 分類嚴格受限於既有 taxonomy（不自由擴張）

---

## 6. 系統流程骨架（每題一致）
1. 使用者提供 `question + context`
2. 檢索候選 FIC 並選卡
3. 若語義不清，進入 `M` 拒答分支
4. 抽取/綁定輸入值
5. 跑 `F`（規格缺失）檢查
6. 跑 `E`（不變量/尺度/綁定）檢查
7. 通過後執行 deterministic `execution.code`
8. 輸出 `Success` 或 `Refusal/Error/Alert` + HITL 修復建議

---

## 7. 實驗設計方向（重點）
## 7.1 實驗目標
衡量不同 LLM 在風險情境下的：

1. 攔截能力（是否抓到 M/F/E）
2. 誤報率（False Positive 是否過高）
3. 漏報率（False Negative 是否過高）
4. 最終正確率與安全性（是否出現 silent wrong）

## 7.2 資料設計
使用客製化資料集，基於原本可答對題目製造陷阱版本：

- `normal`
- `M-trap`
- `F-trap`
- `E-trap`
- `mixed-trap`

## 7.3 比較維度
- 各模型在同一操作者、同一提示框架、同一資料條件下表現
- 觀察是否「過度保守」或「過度放行」

## 7.4 指標建議
- Selection Accuracy
- True Refusal Rate
- False Refusal Rate
- Diagnostic Type Accuracy（M/F/E/C）
- Final Answer Accuracy
- Silent Wrong Rate（流程看似通過但最終數值錯）

---

## 8. 接下來的工程里程碑
1. 鎖定 FIC v2 schema 與 `DiagnosticReport` 格式
2. 完成 `dataset -> FIC v2` 生成 pipeline（python_solution 主導）
3. 插入最小 Diagnostic Gate（invariant + scale + missing checks）
4. 建立 M/F/E/C 測試集與評分腳本
5. 比較多模型結果並做錯誤案例分析

---

## 9. 總結
VerifiQuant V2 不追求「只看答對率」的表面進步，而是追求：

1. 在不確定時可拒絕
2. 在錯誤時可診斷
3. 在執行時可重跑
4. 在決策時可審計

這使它更接近金融實務真正需要的可靠性與責任可追溯性。
