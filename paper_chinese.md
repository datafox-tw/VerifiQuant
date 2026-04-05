# VerifiQuant：以合約化診斷漏斗實現可審計金融推理  
**VerifiQuant: Contract-Validated Financial Reasoning via Multi-layer Diagnostic Funnels and Oracle-in-the-Loop Alignment**

## 摘要（Abstract）
金融任務對錯誤容忍度極低，但主流 LLM 評測多聚焦於「最終答案準確率」，忽略了真實場景中的歧義、資料缺口與不可審計風險。本文提出 VerifiQuant，一個以 Financial Inference Contract（FIC）為核心的診斷式系統，將金融問答從單次生成轉為多層閘門流程。我們定義 M/N/F/E/I/C 六類錯誤分類學，並以 Retrieval/Core/Repair 三種卡片驅動執行：先判斷意圖與邊界，再做規格與數值檢核，通過後才進入 deterministic Python 計算。為了評估「可修復性」與「安全性」，本文引入 Oracle-in-the-Loop（O-ITL）自動迭代機制，並與純 CoT self-improving 流程比較。VerifiQuant 的目標不是讓 LLM 更會算術，而是建立一套可追蹤、可拒答、可修復、可審計的高可靠金融推理方法。

## 1. 研究動機與問題定義（Introduction）
### 1.1 現實金融問答的核心痛點
- 問題常欠定（under-defined）：使用者常只描述目標，不提供完整規格。
- 錯誤常隱性：模型可能在資訊不足時仍輸出高自信數值（silent wrong）。
- 流程不可審計：傳統 CoT 每次軌跡不同，難以做一致性稽核。

### 1.2 本研究要解決的問題
我們關注三個系統層問題，而非僅最終分數：
1. 如何把「語義不確定」轉成「可診斷狀態」。
2. 如何把「計算」外部化到 deterministic 引擎。
3. 如何在錯誤發生時，用結構化修復流程提升可恢復性（recovery）。

## 2. 相關工作 (Related Works)

- LLM in Finance：現有財金模型的演進與評測限制。
- Reasoning and Tool-use：CoT 推理、ReAct 框架及其在數學穩定性上的缺陷。
- Symbolic AI & Formal Verification：符號邏輯與神經網絡的結合，以及合約化編程（Design by Contract）在 AI 中的應用。
- Human-in-the-loop (HITL)：對齊技術與主動詢問（Proactive Clarification）的研究。

## 3. VerifiQuant 系統總覽（System Overview）
VerifiQuant 採用「漏斗式執行」：
`Question + Context -> Retrieval Mapping -> Schema/Boundary/Critic Gates -> Deterministic Execution -> Repair/HITL`

核心原則：
- **Safety-first**：寧可拒答或澄清，不輸出不可信數字。
- **Determinism-first**：算術與規則檢核由 Python 契約執行。
- **Audit-first**：每次決策輸出診斷代碼、gate 行為與可追蹤證據。

## 4. 研究方法（Methodology）
### 4.1 M/N/F/E/I/C 錯誤分類學與多層防禦漏斗
| 漏斗層 | 代碼 | 定義 | 系統行動 |
|---|---|---|---|
| Intent | M | 意圖模糊，無法唯一映射公式族 | Refusal + 澄清問題 |
| Scope | N | 意圖明確，但超出當前 FIC 庫邊界 | Graceful Exit + 說明邊界 |
| Schema | F | 已選定公式，但必要欄位缺失/不可解析 | Slot-filling 補值 |
| Boundary | E | 欄位齊全但數值或尺度違反規則 | Deterministic Alert |
| Critic | I | 隱性語義歧義（匯率方向、期初/期末、時間基準） | Clarification Request |
| Logic | C | 進入執行後的程式/資料殘差問題 | Audit Log + 錯誤回報 |

### 4.2 FIC v2：三卡分工與合約欄位
VerifiQuant 以一題對應一組三卡（後續可擴展為去重併卡）：

1. **Retrieval Card**（語義映射）  
- 用途：RAG 選卡、邊界判定。  
- 重要欄位：`selection_hints`, `applicable_when`, `not_applicable_when`, `scope_boundaries`, `keywords`。

2. **Core Card**（計算與驗證）  
- 用途：定義輸入契約與 deterministic 執行。  
- 重要欄位：`inputs`, `output`, `execution.code`, `diagnostics.invariants`, `diagnostics.scale_checks`, `semantic_hints`。  
- `semantic_hints` 專供 I-gate Critic 使用，不進行直接算術。

3. **Repair Card**（修復交互）  
- 用途：將診斷結果映射為結構化提問與下一步行為。  
- 對應動作：例如 `request_missing_fields`, `declare_scope_boundary`, `present_clarification_options`。

### 4.3 Runtime Pipeline（線上推理）
對每筆 query，流程如下：
1. **Mapping（M/N）**：檢索候選卡並由 selector 決策 `select_card / abstain_m / abstain_n`。  
2. **Binding（F）**：從 question/context 抽取必要輸入，檢查 required slots。  
3. **Boundary（E）**：執行 deterministic 規則（尺度/不變量）。  
4. **Critic（I）**：用 `semantic_hints` 做歧義攔截，必要時回傳 `needs_clarification`。  
5. **Logic（C）**：通過前述 gate 後才執行 deterministic Python，輸出結果與 trace。  

### 4.4 方法對應的程式化架構（Code Architecture）
目前實作包含以下 pipeline：

1. **FIC 生成**：`dataset_case_to_fic`  
輸入 dataset 的 `question + context + python_solution (+ answer)`，輸出 `core/retrieval/repair`。

2. **卡片入庫/RAG**：`build_card_store`  
將三卡寫入 SQLAlchemy store，並提供檢索接口。

3. **Trap 擴展**：`expand_cases`  
由 clean case 擴展 `M/N/F/E/I` 測試題，形成壓力測試資料。

4. **診斷+執行主流程**：`run_error_classification_pipeline`  
同一支流程同時覆蓋：
- 錯誤攔截（M/N/F/E/I）  
- 正常執行與 deterministic trace（C/success）

5. **視覺化與指標**：`visualize_expand_eval`  
輸出 confusion、trap 命中率與整體對齊表現。

### 4.5 Oracle-in-the-Loop（O-ITL）自動修復實驗
資料集含 `question + context + code + result`。  
我們設計 Oracle Agent 僅可讀 `code + result`，不能直接替代主流程輸出答案，而是用於：
- 回答框架澄清問題  
- 重寫 `question/context` 後再送回系統  
- 最多迭代 3 輪

此設計讓修復過程可控、可重播、可比較。

### 4.6 兩個對照迭代流程（已拆分）
1. **Framework-guided self-improve**  
使用 VerifiQuant 診斷輸出（M/N/F/E/I）驅動重寫與再推理。  

2. **Pure CoT self-improve**  
不使用 VerifiQuant 診斷，僅依 CoT 自評不足與 Oracle 重寫循環。  

兩者都限制最多 3 輪，並以最終正確率與修復率比較。

## 5. 實驗設計（Experiments）
### 5.1 資料與測試集
- 原始資料：`question + context + code + answer (+ difficulty)`  
- 壓力測試：`Normal / M-trap / N-trap / F-trap / E-trap / I-trap`

### 5.2 Baselines
1. Vanilla CoT（單次）  
2. Pure CoT self-improve（多輪）  
3. VerifiQuant framework-guided（本文方法）

### 5.3 評估指標
- **Accuracy**：最終答案正確率  
- **Recovery Rate**：由錯轉對比例  
- **Interception Precision**：攔截是否精準  
- **Friction Index**：平均澄清/修復輪次  
- **Silent Wrong Rate**：未攔截但答案錯誤比例  
- **Reproducibility**：同輸入是否產生一致 deterministic 輸出

## 6. 研究貢獻（Contributions）
1. 提出金融高風險場景可用的 M/N/F/E/I/C 診斷分類學。  
2. 提出 FIC v2 三卡架構，將語義判定、計算、修復分離。  
3. 提供 O-ITL 自動化迭代評估，並與純 CoT self-improving 做可重現比較。  

## 7. 限制與後續工作（Limitations & Future Work）
- 當前仍以一題一卡為主，尚未全面啟用 AST-level 去重併卡。  
- O-ITL 仍依賴 Oracle 模型品質，需進一步控制其偏差。  
- 後續將導入資料切分與 cross-set 驗證，降低同源資料偏差風險。
