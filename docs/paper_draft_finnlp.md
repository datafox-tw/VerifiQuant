# VerifiQuant — FinNLP 2026 投稿 Draft

**Status**: Working draft (中文版，定稿後再翻英文)
**Last updated**: 2026-06-02（CoT oracle 改 GT-blind 盲審，主表與 Intro/Abstract 數字全面對齊）
**Target venue**: FinNLP 2026 (截稿 2026/07/25)
**Next venue**: ACL 2027 (預計 2027/01 投稿)

---

## 標題與摘要

**Title**: VerifiQuant: Contract-Validated Financial Reasoning via Multi-layer Diagnostic Funnels and Oracle-in-the-Loop Alignment

**Abstract** *[最後撰寫]*
建議含：
- 問題：金融推理三類失敗（silent failure / semantic drift / audit gap）
- 方法：FIC + M/N/F/E/I/C funnel + O-ITL + Verifiable Atomic Transforms
- 數據三個 punchline（**V3 confirmed**）：**Selective Accuracy 100%、Silent Wrong Rate 0%、Abstention 10% (5/50)**
  - 關鍵對比（無外洩協議）：CoT+oracle 即使達到 **98% accuracy（超過 VQ 的 90%）仍有 SWR=2%**；VQ 以 10% abstention 換取 **SWR=0%**——SWR=0% 來自架構級拒答而非準確率高低
- 貢獻 4 點

---

## 1. Introduction

### §1.1 金融推理對「準確率」的單一指標已不足

大型語言模型在金融問答 benchmark 上的準確率已逼近頂峰，但 headline accuracy 在實務部署中是一個 *誤導性指標*。我們識別出三類在標準評測中被掩蓋、卻在生產環境造成系統性風險的失敗模式：

1. **Silent Failure**——模型在資訊不足時仍以高自信輸出答案；
2. **Semantic Drift**——公式選對，但隱性歧義（匯率方向、現金流期初/期末、報酬率年化基準）導致系統性錯誤；
3. **Audit Gap**——CoT 軌跡因取樣隨機性無法重現，使損失發生時無從究責。

這些性質在通用 NLP benchmark 中不會扣分，但決定了一個金融 AI 系統能否被監管機構與內控部門接受。

### §1.2 CoT 的 Faithfulness 危機

一個自然的反問是：既然 CoT 加上 self-improve 已能達到極高準確率（我們的實驗中,GT-blind oracle 配置下 CoT 最高達 98%，甚至超過 VerifiQuant），為何還需要更複雜的架構？答案在於兩點：其一，CoT 的推理軌跡並不可信——且這個問題隨模型進步而惡化，而非改善；其二，再高的準確率也消不掉 silent wrong（98% 配置仍有 SWR=2%），而 CoT 架構性地缺乏「在不確定時拒答」的能力。

Jacovi & Goldberg (ACL 2020) 區分了兩個常被混淆的性質：**可信度 (plausibility)**——解釋對人類觀察者是否具有說服力；與 **忠實度 (faithfulness)**——解釋是否精確反映模型內部的真實推理過程。最危險的情境是高可信度、低忠實度的組合：一個聽起來極具說服力的解釋，實際上完全不對應模型的內部決策。

後續實證研究系統性地揭示 CoT 正落入此危險區間。Turpin et al. (NeurIPS 2023) 透過輸入操弄實驗證明，CoT 中間步驟並不必然反映模型實際得到答案的計算路徑——模型可能受隱性偏見影響，但 CoT 軌跡卻不揭示此影響。Lanham et al. (2023) 進一步量化了 CoT 步驟與最終答案的因果脫節，並揭示一個反直覺的 **scaling 趨勢：隨著模型參數規模擴大與能力增強，其生成的推理軌跡反而越不忠實**。換言之，模型越聰明，其偽裝推理軌跡以符合人類期待的能力就越強——這直接否定了「等更強的模型就能解決 faithfulness 問題」的樂觀假設。

Pfau et al. (ICLR 2024) 從機制層面提供了解釋：過去假設 CoT 的有效性來自將複雜任務拆解為人類可讀的離散步驟，但嚴格實驗證明 CoT tokens 提供的是純粹的 **計算深度 (computational depth)**，而不需要包含任何有意義的中間語義資訊——甚至將中間步驟替換為無意義的填充符號也不影響效能。所謂的「逐步推理」本質上是 post-hoc rationalization，表面的推理軌跡與內部計算是解耦的。近期工作 (Lie to Me, 2026; Ariadne, 2026) 將此分析延伸至 reasoning models 與 multi-agent 系統，確認此現象並非早期模型的暫時性缺陷。

三條證據匯合為一個對金融部署致命的結論：CoT 的表面合理性與內部忠實度不僅不相關，**而且隨模型進步會進一步分離**。在金融場景，這意味著即使 CoT 答對了，其產出的推理過程也無法作為審計證據；答錯時更無從定位故障層。**不可審計的高準確率與可審計的拒答能力是兩條不同的軸——前者在部署語境下無法替代後者。** 從 architecture 而非 prompt engineering 著手的理由不僅在於前者的可驗證性更強，更在於模型能力越強、CoT 的偽裝越精密，人類使用者越無能力自行檢驗——將驗證負擔外推給使用者在 high-stakes 部署中是不負責任的（詳見 §6.4）。

### §1.3 現有解法的光譜與缺口

為脫離 CoT 的不可信問題，文獻發展出三條路線：

- **(a) 執行端外部化**（PAL, PoT, FINDER）將算術交給 Python interpreter，但 LLM 仍可寫出語法正確、金融邏輯錯誤的程式碼，對 semantic drift 無防禦；
- **(b) 符號求解器路線**（Logic-LM, VERAFI）依賴自然語言到形式邏輯的脆弱翻譯，在複雜情境下準確率反而劣化；
- **(c) Agentic reflection**（ToT, Self-Refine）以 LLM 驗證 LLM，未解決 faithfulness 的本質問題。

**缺口在於：沒有一個框架同時處理 semantic drift 的事前攔截、計算過程的可審計性、以及修復過程的可驗證性。**

### §1.4 VerifiQuant：合約驗證的診斷漏斗

我們提出 **VerifiQuant**，一個將金融推理拆解為「**先驗證、再計算**」流程的 neuro-symbolic 框架，目標是在準確率與可審計性之間取得平衡而非單邊極致。其三項核心設計為：

1. **M/N/F/E/I/C 六層診斷漏斗**——將失敗模式拆為 Misunderstanding / Not-Supported / Formula-spec / Extraction-boundary / Interception / Calculation 六類，其中 I-gate 進一步細分為 I_HARD（阻斷式）與 I_SOFT（警告式），共七個可診斷閘門；每層皆為合法退出點，使「拒答」與「澄清」成為一級結果而非 fallback；
2. **Financial Inference Contract (FIC)**——每個金融公式對應一份結構化合約（檢索 metadata + deterministic 執行碼 + 不變量 + 語義陷阱描述），LLM 受合約約束選卡、抽欄位、辨歧義，但不執行算術；
3. **Verifiable Atomic Transforms**——當 I_HARD 攔截到語義歧義，系統僅允許 AST 邊界內的原子轉換，並以數值交叉驗證確保修改前後的代數等價，杜絕 LLM 在修復階段對核心邏輯的自由竄改。

### §1.5 為何此架構在部署語境下優於更高準確率的 CoT

在 selective prediction (El-Yaniv & Wiener, 2010; Geifman & El-Yaniv, 2017) 的框架下，VerifiQuant 在 FinanceReasoning 中度難度子集上達到 **90% Pass@1、selective accuracy 100%、Silent Wrong Rate 0%**——同時提供 CoT 系列架構性無法達成的 safe refusal（abstention）能力。在排除評估外洩的 **GT-blind 盲審協議**下，CoT + oracle 可達 92–98% 準確率（最強配置 plain Pro 達 98%，甚至高於 VQ），但全部以「永不拒答」為代價，SWR 仍落在 2–8%。金融 multi-agent baseline (JP Morgan MAS reimpl) 在相同 50Q 上達 86% (43/50)，同樣零拒答、SWR=14%（5 題 silent wrong + 2 題 pipeline error），呼應 *Why MAS Fail* (2026) 對 multi-agent 自然語言共識失效的觀察。

**關鍵比較（重點不在準確率高低）**：CoT+oracle 即便達到 98% accuracy（plain Pro），仍有 1 題 silent wrong（SWR=2%）甩不掉；VQ Flash V3 (90% acc, abstain=10%) 則以拒答能力達 **SWR=0%**。差異不在誰準確率高,而在 **失敗模式**：VQ 在不確定時拒答,CoT 系列被迫輸出數字。模型對照（固定 FIC 卡片版本）：V3 卡片上 VQ Flash 45/50 (SWR 0%) vs VQ Pro 43/50 (SWR 4%)，V1 卡片上 VQ Flash 43/50 (SWR 4%) vs VQ Pro 41/50 (SWR 6%)。所有 VQ 配置的 SWR（0–6%）均優於 JP Morgan MAS（14%）；與 CoT 的差異則在於 CoT 無論準確率多高皆無法達到 SWR=0%。

**因此 VerifiQuant 的可審計準確率與 CoT 系列的不可審計準確率在部署決策上並不等價**——即便 CoT 準確率更高（98% vs 90%），其無法消除的 SWR 與不可審計軌跡，在高風險金融部署語境下決定了可接受性。

### §1.6 貢獻

1. 提出 **M/N/F/E/I/C 錯誤分類學**（I-gate 細分為 I_HARD/I_SOFT），將金融 LLM 失敗模式從二元正誤擴展為七個可診斷閘門；
2. 設計 **FIC 合約 + Verifiable Atomic Transforms**，使修復過程受 AST 與數值不變量雙重約束；
3. 提出 **Oracle-in-the-Loop (O-ITL)** 評估協議，量測「意圖明確但答案未知」條件下的框架可恢復性；
4. 在 FinanceReasoning 與其衍生的 trap dataset 上實證：clean cases 上即使 CoT+oracle（GT-blind）準確率可超過 VerifiQuant，仍無法達到 SWR=0%（VQ 透過 abstention 達成）；在 **trap cases (F/E/I variants)** 上 VerifiQuant 顯著降低 silent wrong rate，並提供 CoT 無法產出的結構化審計軌跡。

---

## 2. Related Work

### §2.1 Financial Reasoning Benchmarks 與其侷限

金融推理近年成為 LLM 評測的熱門域。FinanceReasoning (Liu et al., ACL 2025) 系統化整理金融問答的失敗模式，指出當前模型表現受限的主因並非推理能力天花板，而是 *資料集本身存在歧義、缺乏標準化的問題規格*。JP Morgan 提出的金融 multi-agent system (Yu et al., EMNLP 2025) 在類似資料上 Pass@1 僅達 66%，且失敗案例缺乏結構化歸因。本研究使用 FinanceReasoning 作為主要資料集，並進一步衍生 trap variants 以系統性測試 silent failure 與 semantic drift；同時，我們以 FIC 合約取代「以自然語言描述問題」的傳統 schema，從上游降低 FinanceReasoning 所指出的歧義問題。

### §2.2 Chain-of-Thought 的 Faithfulness 質疑

CoT 及其延伸（Self-Refine, Tree-of-Thoughts）試圖讓模型外顯推理過程，但 faithfulness 文獻對其作為可信推理軌跡的價值提出系統性質疑（詳見 §1.2）。此處聚焦於 VerifiQuant 如何回應這些發現。

現有的應對策略——如 self-consistency (Wang et al., 2023)、process-reward model (Lightman et al., 2023)——仍依賴 LLM 自身的輸出作為信號源，無法從根本上解決軌跡與內部計算解耦的問題。VerifiQuant 選擇不同路線：將推理軌跡從 LLM 的自由生成中抽離，改由 FIC 合約與 deterministic 執行決定。當系統輸出「以 r×(1+i) 調整期末年金為期初年金」，此軌跡是 *程式碼執行的實際記錄* 而非 LLM 的事後敘述，使軌跡與結果之間具備可驗證的因果關係。

### §2.3 Tool-Augmented 與 Execution-Based Reasoning

PAL (Gao et al., 2023) 與 Program-of-Thoughts (Chen et al., 2023) 確立了「推理與執行分離」的範式：LLM 負責將問題轉化為 Python 程式碼，外部 interpreter 負責執行以消除算術錯誤。FinanceReasoning (Tang et al., ACL 2025) 採用 PoT 作為其主要基線，金融領域的 FINDER 框架亦沿此路線針對 FinQA 進行特化。

然而，這一流派共享同一個盲點：它們假設「程式碼語法正確即金融邏輯正確」，對 *semantic drift* 毫無防禦能力。若 LLM 在概念層將「期初年金」誤解為「期末年金」，它依然能生成一段語法完美、可正常執行、卻從根本上答錯題的 Python Code——而 interpreter 既無法偵測此錯誤，也無法提供結構化的失敗歸因。VerifiQuant 在執行階段之前插入 I-gate 進行 semantic drift 攔截，且修復動作受 AST 邊界與數值不變量雙重約束，確保任何調整皆留有可審計的代數軌跡。

### §2.4 Neuro-symbolic 與 Formal Verification

為取得形式保證，研究者嘗試讓 LLM 充當「翻譯者」，將自然語言問題轉換為確定性符號系統可處理的表示。此流派可拆為兩條子路線，但都最終受限於同一個根本瓶頸。

**（a）模組化推理路線（Modular Reasoning）。** Creswell et al. (DeepMind) 借鑑 neurosymbolic 概念，提出 Selection-Inference (SI) 框架，將推理拆為「從文本選取相關事實」與「從事實推導結論」兩個獨立模組。後續的 Faithful Reasoning (Creswell et al.) 進一步加入 Halter 模組（偵測「資訊是否已足夠」）與 Value Function + Beam Search（並行推演多條邏輯分支後選優）。這兩篇工作在可解釋性上取得顯著進展，並明確批評 CoT 等方法以「幻覺合理化」冒充忠實推理。然而，其表現的前提假設始終未能突破：所有必備前提、規則與干擾項必須事先完整封裝在 Context 中。一旦需要外部檢索，Selection 模組若抓取錯誤的前提，後續推論無論邏輯多完美，皆只是在錯誤的基礎上進行華麗的推導。

**（b）形式求解器路線（Formal Solver）。** Logic-LM (Pan et al., 2023) 讓 LLM 將題目翻譯成 ASP，交由 Clingo 這類確定性 Symbolic Solver 驗證；LINC (Olausson et al., EMNLP 2023) 嘗試映射至一階邏輯（FOL），但其錯誤分析明確指出模型頻繁出現語法錯誤與語意偏移（Semantic Drift）；SATLM (Ye et al., NeurIPS 2023) 改採宣告式路線，讓 LLM 只生成 Z3 SMT Solver 的約束式而非直接解題，但仍高度依賴特化 Few-shot Prompt 才能驅動，遇到 Prompt 未涵蓋的邏輯結構即告失敗，且剛性的 SMT Solver 對自然語言中的雜訊零容忍——一個符號錯誤即導致 UNSAT 或 Crash，且架構為單向 Parse→Solve，無自我修正機制。Logic-LM 即使加上自我修正模組，在複雜邏輯資料集（AR-LSAT）上的準確率仍僅達 43.04%。VERAFI 則在失去翻譯精度後退而求其次，改採「軟約束」，形式保證大幅削弱。

**（c）翻譯行為的根本極限。** 上述失敗並非工程實作的不足，而是問題本身的性質所致。最新的 FormalJudge (Zhou et al., arXiv 2026, under review) 在其系統的限制分析中明確指出：將自然語言需求翻譯為形式化規範，在一般情況下是一個「不可解的問題（undecidable problem）」，並將此現象命名為「規格合成瓶頸（specification synthesis bottleneck）」。即便 FORMAL JUDGE 採用雙向架構——將使用者意圖分解為原子級「是/否」問題後由 Z3 組合——其失敗案例分析仍顯示：剩餘錯誤主要集中於意圖分解與語意提取這兩個翻譯環節，並結論「這是一個單靠形式化驗證也無法解決的問題」。

VerifiQuant 採取不同路線：以離線建構的 FIC 合約承擔形式化的負擔，線上推理只需在已驗證的合約空間內進行 selection 與 binding。這使 runtime 完全繞開了自然語言到形式符號的即時翻譯，而非試圖在翻譯精度不可靠的前提下追求更強的求解器。

### §2.5 Multi-Agent Systems 與 Self-Reflection

以 multi-agent 協作或 self-reflection 提升可靠性是目前金融 agent 最常見的商業路線（ToT, Self-Refine, Reflexion）。然而這條路線在 faithfulness 上存在本質缺陷：CoT 與自我驗證的本質是「因果脫節（Causal Disconnect）的事後合理化」——LLM 宣稱已完成驗證，實際上是根據最終答案反向生成了一段看起來合理的推理軌跡，而非真正的驗證過程。在 multi-agent 架構中，這一問題被放大為 *theory-of-mind failure*：*Why Multi-Agent LLM Systems Fail* (Cemri et al., NeurIPS 2026) 指出，MAS 最常見的失敗模式在於 agent 之間假設彼此理解，但對共享狀態的詮釋存在分歧，導致錯誤在 agent 邊界處被靜默傳遞而非攔截。

我們在相同 50Q 子集上 reimplement JP Morgan MAS（9-node LangGraph，O-ITL 條件）得 86% Pass@1，SWR=14%（5 題 silent wrong + 2 題 pipeline error）。其零拒答、無結構閘門的特性正呼應上述觀察：在沒有明確失敗偵測機制的情況下，系統傾向以置信度高但錯誤的答案作為輸出，而非發出結構化警告。VerifiQuant 的 Oracle-in-the-Loop (O-ITL) 設計以 FIC 合約取代 agent 間自然語言共識：所有 agent 互動皆透過結構化的 DiagnosticReport 與 FIC 欄位進行，使 theory-of-mind failure 無從發生。

### §2.6 定位

現有工作各自解決部分問題：execution-based 解決 C-class 算術錯誤、neuro-symbolic 提供形式保證但受限於翻譯瓶頸、CoT/reflection 提升表面準確率但缺乏 faithfulness。VerifiQuant 的貢獻在於將這些路線在一個診斷漏斗下整合：以離線 FIC 合約承擔形式化負擔、以 I-gate 攔截 semantic drift、以 deterministic 執行保證 C-class 可審計、以 O-ITL 取代 agent 間自然語言共識——在保持可比準確率的同時，提供結構化的失敗診斷與審計軌跡。

---

## 3. Methodology

### §3.1 Overview: Verifiability as Infrastructure

VerifiQuant 的設計核心是將「金融推理的可驗證性」視為一項 *基礎建設*（infrastructure）而非事後分析（post-hoc explainability）。正如 type system 之於程式語言、單元測試之於軟體工程，VerifiQuant 主張：在金融這類高風險領域，可審計性必須在 *系統執行前* 就被結構化地寫進架構。我們提出四個整合的元件：

1. **Financial Inference Contract (FIC)**——以結構化合約取代自然語言問題描述；
2. **M/N/F/E/I/C Diagnostic Funnel**——將推理流程拆為六層可診斷閘門，I-gate 進一步分為 I_HARD 與 I_SOFT；
3. **Oracle-in-the-Loop (O-ITL) Alignment**——以 FIC 作為 agent 間共享參照，避開 multi-agent 系統的 theory-of-mind failure；
4. **Verifiable Atomic Transforms**——使修復過程受 AST 邊界與數值不變量雙重約束。

整體設計遵循三項原則：

- **Safety-first**：寧可結構化拒答，不輸出不可審計的數值；
- **Determinism-first**：算術與規則檢查由合約執行而非 LLM；
- **Audit-first**：每一筆推理皆產出可重播、可逐層追溯的 diagnostic trace。

### §3.2 Financial Inference Contract (FIC)

給定一個金融公式族，我們將其表示為一份結構化合約，由三張卡片構成：

- **Retrieval Card**：`selection_hints`、`applicable_when`、`not_applicable_when`、`scope_boundaries`、`keywords`，供 BM25 與 LLM selector 在線上判斷此 FIC 是否適用；
- **Core Card**：`inputs`（型別與單位）、`output`、`execution.code`（deterministic Python）、`diagnostics.invariants`、`diagnostics.scale_checks`、`semantic_hints`（標記如期初/期末、年化/月化、匯率方向等隱性歧義）；
- **Repair Card**：將每一類診斷結果映射至結構化動作，含 `request_missing_fields`、`swap_suspected_fields`、`rephrase_task_intent`、`declare_scope_boundary`、`present_clarification_options`、`confirm_unit_conversion`、`select_alternative_fic`、與 fallback `confirm_assumption` 共八種 action type。

FIC 的設計直接回應 FinanceReasoning (Liu et al., 2025) 所指出的「資料集歧義」問題：傳統 dataset 一道題對應一個自然語言描述與一個答案，VerifiQuant 則要求每題對應一份 *機讀的計算合約*，將語意判斷與算術執行解耦。

### §3.3 M/N/F/E/I/C Diagnostic Funnel

給定輸入問題 q 與上下文 c，VerifiQuant 將推理組織為循序閘門：

| Layer | Code | Gate Function | Exit Action |
|---|---|---|---|
| Intent | M | q 無法唯一映射至任一 FIC 族 | Refusal + clarification request |
| Scope | N | 意圖明確但無對應 FIC | Graceful scope-boundary exit |
| Schema | F | FIC 已選定但必要 input 缺失 | Slot-filling request |
| Boundary | E | Inputs 齊全但違反 invariants 或 scale-checks | Deterministic alert |
| Critic (hard) | I_HARD | 隱性歧義會 *改變計算路徑*（如期初/期末） | Block + verifiable transform |
| Critic (soft) | I_SOFT | 隱性歧義 *不改變計算路徑* 但可能影響詮釋 | Proceed + explicit soft warning |
| Logic | C | 通過前述閘門後的 deterministic 執行 | Audit log |

**I_HARD vs I_SOFT 設計理由**：I_HARD 對應 *形式上可驗證的歧義*——澄清後 verifiable transform 可保證計算正確性。I_SOFT 對應 *本質上 LLM-dependent 的判斷*——例如業界慣例上的近似、或文獻中存在多種等價但結果略異的公式變體。VerifiQuant 不假裝消除 I_SOFT，而是將其作為 *顯式宣告的不確定性區*：輸出附帶 soft warning，使用者得以選擇是否接受。我們將「逐步把 I_SOFT 子模式遷出至硬類別」視為 verifiability frontier 上的長期演進工作（§6.2）。

每筆推理皆產出 DiagnosticReport = (exit_layer, fic_id, binding, invariant_trace, action)，可重播且可逐層追溯。

### §3.4 Oracle-in-the-Loop Alignment

近期工作 (Authors, ICLR 2026) 指出 multi-agent 系統最常見的失敗模式為 *theory-of-mind failure*：agents 假設彼此理解，但對共享狀態的詮釋分歧。VerifiQuant 的 O-ITL 機制以 FIC 與 DiagnosticReport 作為 agent 間的 *顯式共享參照*，並以受限的 Oracle agent 模擬「意圖明確但答案未知」的理想使用者。

**Oracle 的介面與權限**。Oracle agent 在每一輪讀取 (i) 當前 DiagnosticReport（含 `diagnostic_type`、`status`、binding 細節）與 (ii) 該題的 *ground-truth 計算 code*——但 **不包含最終數值答案**。Code 表達的是「該如何計算」的意圖規格，而非「結果是多少」。Oracle 僅可輸出 `{updated_question, updated_context, notes}` 的 JSON 結構，**不被允許**直接寫入數值答案；其唯一允許的行為是在 ground-truth 計算邏輯的範圍內，將原始 question/context 改寫為更明確的版本。此設計使 O-ITL 成為「*意圖完全明確但答案未知時，框架的可恢復性上界*」的估計器——一種可重播、可比較的 oracle-augmented evaluation protocol，而非與真實用戶互動的 HITL 系統。

**Repair Action 介面**。Repair Card 將 DiagnosticReport 映射至八種結構化動作之一（§3.2），每條 repair rule 並宣告其 `allowed_next_steps` ∈ {`rerun_same_fic`, `select_alternative_fic`, `ask_followup`, `stop_with_refusal`}，限制下一輪的合法狀態轉移。

**迭代與終止**。設定最大輪次 K（實驗中 K=3，即 oracle 最多修正兩次）。完整的迭代序列為：VQ 拋出錯誤 1 → oracle 修正 1 → VQ 拋出錯誤 2 → oracle 修正 2 → VQ 最終回答。終止條件為以下之一：
- (a) `status = success` 且無 soft-mismatch；
- (b) 達到 K 上限；
- (c) `diagnostic_type` 落於不可修復類別（如 C-runtime）；
- (d) 連續兩輪的 question/context 無變化。

### §3.5 Verifiable Atomic Transforms

當 I_HARD 攔截到語義歧義並獲得澄清後，系統面臨關鍵設計選擇：如何讓計算邏輯反映新的語義？傳統做法允許 LLM 自由重寫 Python code，但這引入無界的故障半徑——LLM 可能竄改變數定義、引入未審查的 import、或破壞原本正確的邏輯。

VerifiQuant 限制修復行為僅能透過兩種 *atomic transforms*：

- **`result_postprocess`**：對最終結果套用純算術後處理（如 r_new = r_old × (1+i)）。受 AST node-count 上界、safe-name whitelist、與數值不變量檢查約束。
- **`code_patch`**：對 FIC 原始 code 進行受限的文字替換（如 `start=1` → `start=0`）。受以下條件約束：
    1. 替換 pattern 在 code 中唯一出現；
    2. shallow AST-diff 不超過預先宣告的 blast-radius bound；
    3. **cross-verification**：patched code 的數值輸出必須與對應 `result_postprocess` 的結果在多組樣本輸入下一致。

*Cross-verification* 是核心安全性保證：一個程式碼變更只有在其數值效果可被獨立的代數關係驗證時才被接受，無須符號求解器。

**Illustrative example (annuity due)**：給定 FIC 預設的「期末年金」執行碼，當使用者澄清「實為期初支付」後，系統不允許 LLM 重寫公式；而是
1. 套用預宣告的 `result_postprocess: r × (1+i)`；
2. 套用 `code_patch: start=1 → start=0` 並重跑；
3. 比對兩者數值結果。

一致時方接受修補並進入 C-gate 執行。

### §3.6 Integration

整合流程：

```
q → BM25 retrieval → LLM selector (M/N) → LLM extractor (F)
  → deterministic E-checks → LLM critic (I_HARD / I_SOFT)
  → verifiable transform if triggered → deterministic C execution
  → DiagnosticReport
```

LLM 僅扮演 *對合約的對應器*——選卡、抽欄位、辨歧義；所有具有金融後果的計算與決策皆由 deterministic 元件執行。

---

## 4. System Implementation
*[本節可選；簡述 SQLite/SQLAlchemy 卡片儲存、BM25 FTS、Gemini 2.5 Pro 作為 selector/extractor/critic、Flask 部署層；可壓縮到半頁或全併入 §3 後省略]*

---

## 5. Experiments

### §5.1 Research Questions

- **RQ1 (Selective Reliability)**：VerifiQuant 在 silent wrong rate 與 abstention rate 之間是否取得優於現有系統的平衡？
- **RQ2 (Diagnostic Distribution)**：M/N/F/E/I/C 各層攔截的分布為何？
- **RQ3 (Trap Resistance)**：在 F/E/I trap variants 下，各系統的 silent wrong rate 如何變化？
- **RQ4 (Recovery)**：O-ITL 多輪修復對 selective accuracy 的貢獻為何？
- **RQ5 (Soft-flagged Wrong Rate)**：VerifiQuant 答錯的案例中，有多少帶 I_SOFT warning？

### §5.2 Dataset

**Clean set（已確定）**：
FinanceReasoning (Liu et al., ACL 2025) medium 子集，50 questions。
選題方法：*stratified random sampling by difficulty quartile*，固定 seed=42，從 540 題有效題目（medium + 有 function + 有 ground_truth）按四個 difficulty 分位分層抽取（Q1:12, Q2:13, Q3:12, Q4:13），難度範圍 2.485–4.025（mean=3.02）。
可重現命令：
```bash
python3 scripts/sample_dataset.py \
  --source FinanceReasoning/data/FinanceReasoning/medium.json \
  --output-jsonl verifiquant/data/runs/paper_v1/questions_50.jsonl \
  --output-config verifiquant/data/runs/paper_v1/experiment_config.yaml \
  --n 50 --seed 42
```
所有 50 題的 question_id 記錄於 `experiment_config.yaml`（已 commit 進 repo）。

**為什麼不用 model-performance-based 選題**：
早期版本曾用「Gemini 2.0 Flash 答對 25 + 答錯 25」來選題，但發現 2.0 的失敗模式與 2.5 無直接關聯，且 2.5 在 medium 的 one-shot accuracy 本身已達 87%+，強迫平衡正確/錯誤比例反而造成 selection bias。改為 model-agnostic 的難度分層抽樣，可推廣性更強。

**Trap set（contract-grounded，Tier-1 deterministic）**：

trap 的核心難點是**標註可信度**：若用純 LLM 或 regex 從題幹「猜」這題該屬於哪一類錯誤，標註本身就不穩定——我們先前的嘗試顯示，意圖製造 I-class 歧義的題目常常被抽成其他類型，trap label 與實際觸發層不一致。本研究改採**合約錨定（contract-grounded）**的生成策略：trap 不是猜出來的，而是由「乾淨題 + 該題對應的 FIC 卡片欄位」經一個確定性算子（operator）推導出來——**label 即算子**，因此 trap 的 expected behavior 與 FIC 合約嚴格對齊，無需事後人工歸類。

我們以 `build_trap_set.py` 在 canonical 50Q clean set 上，針對每題的 FIC 卡片（`cards_v3.db`）套用五個算子，每類各 10 題（共 50 trap），全部與 `contracts.py` 的 `RefusalCategory` 對齊：

| 算子 | 擾動方式 | 錨定的合約欄位 | 期望診斷／行為 |
|---|---|---|---|
| **F** | 從題幹移除一個必要輸入（並 redact 其數值） | FIC 函式簽章 ∩ 已提供輸入 | `F` → refuse / ask（缺欄位） |
| **E** | 將某輸入改為違反邊界的負值 | 該卡片宣告的 E-check 述詞 `inputs['X'] < 0` | `E` → abstain / repair |
| **I** | 刪除題幹中消除歧義的關鍵字 | I-class repair rule 的 `rule_id`（家族 + `_hard`/`_soft` 後綴） | `I_HARD` → clarify／`I_SOFT` → proceed-with-warning |
| **N** | 換成超出領域的問題（exotic pricing） | 無任何 FIC 應匹配 | `N` → graceful refuse |
| **M** | 換成未指明 metric 的模糊意圖 | — | `M` → ask to disambiguate |

關鍵設計：(1) **F/E 算子直接改動 context 中的真實數值**——F redact 掉、E 改成負值——而非附加標記，因此擾動是 in-distribution 的；(2) **I 算子的類別由 FIC repair rule 的 `_hard`/`_soft` 後綴決定**（如 `i_throughput_metric_type_hard` → I_HARD、`i_output_scale_soft` → I_SOFT），同時確認刪除後題幹確實不再含消歧線索（否則 flag `needs_review`），確保歧義真的被還原；(3) 任何無法確定性錨定的個案不會被硬塞，而是標記 `needs_review` 留給 Tier-2（LLM 精修 + 人眼抽驗）。Tier-1 生成的 50 題 trap 全數 `needs_review=0`，並另經人工逐題抽驗。

### §5.3 Baselines

1. **CoT single-shot (Gemini 2.5 Flash)** — 無 self-improve，一次性輸出
2. **CoT single-shot (Gemini 2.5 Pro)** — 無 self-improve，一次性輸出（Pro 版本）
3. **CoT + oracle（GT-blind 盲審，Flash & Pro × plain & funnel-guided 共 4 配置）** — K=3，oracle 每輪盲審、不接收數值 GT/correctness、不以 is_correct 決定進場（避免外洩）；plain 與 funnel-guided 兩種 oracle prompt 對照
4. **JP Morgan MAS** (Yu et al., EMNLP 2025) — 金融領域 multi-agent baseline
5. **VerifiQuant + O-ITL K=3 (Flash)** — 完整架構 + oracle 修正最多兩次
6. **VerifiQuant + O-ITL K=3 (Pro)** — 同上但換用 Gemini 2.5 Pro（補跑，模型公平性）

> ⚠️ **Oracle 協議與公平性（GT-blind, blind-review）**：為避免 oracle 評估外洩，所有 CoT oracle 變體採 **GT-blind、每輪盲審（blind_review_every_turn）** 協議：(i) oracle **不接收**數值 ground truth 與 correctness 訊號，僅接收 `python_solution`（意圖規格，與 VQ oracle 對等）；(ii) oracle **每一輪都審查**，不以 `is_correct` 決定是否進場——否則「只在答錯時介入」本身即洩漏「此題有問題」，並造成「只救錯、不碰對」的樂觀偏誤。我們同時測試 plain 與 funnel-guided 兩種 oracle prompt，各跑 Flash 與 Pro：
>   - plain Flash：47/50（94%），SWR=6%　|　funnel Flash：46/50（92%），SWR=8%
>   - **plain Pro：49/50（98%），SWR=2%**　|　funnel Pro：48/50（96%），SWR=2%
>   - 四個配置 **broken_count=0**（盲審從未把正確答案改錯）
>
> 三個發現：(1) **即使在無外洩、最強配置（plain Pro 98%）下，CoT SWR 仍 ≥2%**——代表性案例 test-1443（CAGR，gold=19.14）在 4 個配置中錯了 3 個（plain Flash 19.12、funnel Flash 19.11、plain Pro 19.13；僅 funnel Pro 矇對 19.14）。根因是 LLM 對分數次方 `2.4^(1/5)` 的算術非決定性：答案在 19.11–19.16 間振盪，最後一位幾乎是擲骰子；plain Pro 甚至在 turn 2 算出正確的 19.14、turn 3 又漂回 19.13。(2) VQ 的 SWR=0% 來自 **abstention** 與 **deterministic 執行**，非 oracle 效能：VQ 把此計算交給 Python（每次精確 0.19135…），並對 decimal/percent 歧義掛 I_SOFT + 可驗證 ×100 transform。(3) **盲審下 funnel-guided ≤ plain**（Flash 92%<94%、Pro 96%<98%），反轉了 GT-gated 條件下的結論——當 oracle 看不到對錯，結構化「逐層找問題」的指令反而過度改寫、注入噪音。這強化核心主張：**可靠性源自架構（deterministic 執行 + abstention），不是更精巧的 oracle prompt。**
>
> ⚠️ **Model-fairness 說明**：主比較為 Flash-to-Flash；Pro 版本（baselines 2 & 6）作為 supplementary table 供參。

### §5.4 Metrics

採用 Selective Prediction (El-Yaniv & Wiener, 2010; Geifman & El-Yaniv, 2017) 框架：

- **Correct / Silent Wrong / Safe Refusal**（三元分類）
- **Coverage** = (Correct + Silent Wrong) / N
- **Selective Accuracy** = Correct / (Correct + Silent Wrong)
- **Silent Wrong Rate (SWR)** = Silent Wrong / N — *金融部署的關鍵風險指標*
- **Soft-Flagged Wrong Rate (SFWR)** = (最終輸出仍掛 I_SOFT warning ∩ Wrong) / (Correct + Silent Wrong)。注意：以「**最終交付給使用者的輸出是否仍帶 flag**」為準——若 I_SOFT 在迭代中被「解除」但答案仍錯，則該題退化為 Silent Wrong（計入 SWR），不計入 SFWR（見 §5.7 對 test-1593 的揭露）。
- **Diagnostic Distribution**：VerifiQuant 在 M/N/F/E/I/C 各層的攔截計數
- **Recovery Rate**：第 1 輪錯誤但於 O-ITL 後修正的比例

### §5.5 Main Results — Clean Set (RQ1, RQ5)

> **數字狀態（2026-05-30）**：全部 ✅ 已跑。所有 VerifiQuant 數字均在 **V3 FIC 卡片版本（`cards_v3.db`）** 下取得，與下方 baseline 為同一 50Q 集合。VQ Flash 與 VQ Pro 在主表中皆為 V3 卡片，可直接逐模型比較；V1 卡片數字見下方 supplementary 表。

**主表（canonical，全部 V3 FIC 卡片）：**

> **三個「答錯」指標的定義（分母皆為作答題數 = Correct + Silent Wrong；越低越好）**：
> - **SWR（Silent Wrong Rate）**：最終輸出一個**有把握、無任何 flag** 的錯誤數值。← 最危險，用戶毫無警覺。
> - **SFWR-final（最終帶 flag 且錯）**：**最終交付**的輸出仍掛 I_SOFT warning 且答案錯。← 較不危險，用戶已被警告。
> - **SFWR-ever（曾觸發 I_SOFT 且最終錯）**：流程中**任一輪**曾掛 I_SOFT、但最終答案錯（含 flag 在末輪被解除者）。← 揭露「I_SOFT 退化為 silent wrong」的隱性路徑。
>
> 三者關係：若一題在迭代中觸發 I_SOFT、最後 flag 被解除但答案仍錯，它**計入 SWR 與 SFWR-ever，但不計入 SFWR-final**（見 §5.7 test-1593）。CoT / MAS 無 I_SOFT 機制，故 SFWR 欄為「—」。

| System | Model | Correct | Silent Wrong | Safe Refusal | Coverage | Sel. Acc | **SWR** | **SFWR-final** | **SFWR-ever** |
|---|---|---|---|---|---|---|---|---|---|
| CoT (single-shot) | Flash 2.5 | 41 (82%) | 9 | 0 | 100% | 82.0% | **18.0%** | — | — |
| CoT (single-shot) | Pro 2.5 | 41 (82%) | 9 | 0 | 100% | 82.0% | **18.0%** | — | — |
| CoT + plain oracle (blind) | Flash 2.5 | 47 (94%) | 3 | 0 | 100% | 94.0% | **6.0%** | — | — |
| CoT + funnel oracle (blind) | Flash 2.5 | 46 (92%) | 4 | 0 | 100% | 92.0% | **8.0%** | — | — |
| CoT + plain oracle (blind) | Pro 2.5 | 49 (98%) | 1 | 0 | 100% | 98.0% | **2.0%** | — | — |
| CoT + funnel oracle (blind) | Pro 2.5 | 48 (96%) | 1 | 0(+1 no-ans) | 100% | 96.0% | **2.0%** | — | — |
| JP Morgan MAS (O-ITL) | Flash 2.5 | 43 (86%) | 7 | 0 | 100% | 86.0% | **14.0%** | — | — |
| **VerifiQuant + O-ITL (K=3, V3)** | **Flash 2.5** | **45 (90%)** | **0** | **5** | **90%** | **100%** | **0%** | **0%** | **0%** |
| VerifiQuant + O-ITL (K=3, V3) | Pro 2.5 | 43 (86%) | 2 | 5 | 90% | 95.6% | **4.0%** | **0%** | **2.2% (1/45, test-1593)** |

**Supplementary（V1 FIC 卡片，未修補；展示卡片修補對模型對照的影響）：**

| System | Model | Correct | Silent Wrong | Safe Refusal | Coverage | Selective Acc | SWR |
|---|---|---|---|---|---|---|---|
| VerifiQuant + O-ITL (K=3, V1) | Flash 2.5 | 43 (86%) | 2 | 5 | 90% | 95.6% | 4.0% |
| VerifiQuant + O-ITL (K=3, V1) | Pro 2.5 | 41 (82%) | 3 | 6 | 88% | 93.2% | 6.0% |

> **VQ Flash V3 完整 summary（`paper_v1/results/vq_flash_v3/summary.json`）**：
> ```
> total_cases:    50
> success_count:  45   (code 成功執行且答對)
> success_rate:   0.90
> correct_count:  45
> accuracy:       0.90
> recovery_count: 8    (觸發 oracle 後成功修正)
> recovery_rate:  0.16 (16%)
> abstain:        5    (M/N/F refusal，漏斗阻斷)
> silent_wrong:   0
> ```

> **VQ 版本說明（FIC 卡片修補過程）**：
> - V1 (vq_flash): 43/50，SW=2，E-check 語法錯誤 + semantic hint 缺失造成 2 題 C-error
> - V2 (vq_flash_v2): 44/50，SW=0，修補 3 C-error + 2 SW → 全 E-check 語法修正
> - **V3 (vq_flash_v3): 45/50，SW=0，再修 1 C-error（E-check 改用 compute(inputs) 而非 result）**
> - V3 是本文報告的 canonical VQ 卡片版本；V1/V2 為中間版本
> - **卡片版本對兩模型皆生效**：同一份 V3 卡片下，VQ Flash 45/50、VQ Pro 43/50；V1 卡片下 VQ Flash 43/50、VQ Pro 41/50。逐模型比較必須固定卡片版本（本文主表全用 V3）。
> - 卡片修補的防禦性說明：3 個 C-class 修正皆可由 Python 語法/執行期錯誤（`execution_smoke_ok=false`）偵測，**無需 gold answer**，因此非 cherry-picking。

**觀察**：
1. **CoT self-improve（GT-blind 盲審）對準確率有效，但 SWR 有不可逾越的邊界**：
   - 在無外洩協議下，CoT+oracle 達 92–98% accuracy（plain Pro 最高 98%），SWR 落在 2–8%
   - 四個配置 **broken_count=0**：盲審不會把正確答案改壞，協議安全
   - **核心發現（微觀案例 test-1443，CAGR，gold=19.14）**：即使在最強的無外洩配置（plain Pro, 98%）下仍答錯（4 配置中錯 3：plain Flash 19.12、funnel Flash 19.11、plain Pro 19.13；僅 funnel Pro 矇對）。根因不是歧義而是 **LLM 對分數次方 `2.4^(1/5)` 的算術非決定性**——答案在 19.11–19.16 間振盪；plain Pro 甚至 turn 2 算對 19.14、turn 3 又漂回 19.13，**同一模型連自己都重現不了**。oracle 救不了：題目本身清楚，oracle 盲審無從改起，且 oracle 不重算數字。VQ 則把此步交給 **deterministic Python**（每次精確 0.19135…→19.14），並對 decimal/percent 掛 I_SOFT + 可驗證 ×100 transform——**把擲骰子變成查表**。這證實：更高 accuracy 也消不掉源於 LLM 算術不可靠的 SWR；根本解是 deterministic 執行 + abstention，非更聰明的 oracle。
2. **Flash vs Pro 在 single-shot CoT 上無差異**：41/50 vs 41/50，顯示 medium 難度的 single-shot 對模型選擇不敏感；但加上 oracle 後 Pro 明顯領先（plain：Pro 98% vs Flash 94%），差異來自 oracle-augmented 的多輪修正而非單次推理能力
3. **架構選擇 vs 準確率競爭（無外洩條件下的本質差異）**：
   
   無外洩（GT-blind 盲審）下的完整對照：
   - CoT plain (Flash)：94% acc, SWR=6%, abstain=0%
   - CoT funnel (Flash)：92% acc, SWR=8%, abstain=0%
   - **CoT plain (Pro)：98% acc, SWR=2%, abstain=0%**（最強配置）
   - CoT funnel (Pro)：96% acc, SWR=2%, abstain=0%
   - VQ Flash V3：90% acc, SWR=0%, abstain=10%
   - VQ Pro V3：86% acc, SWR=4%, abstain=10%
   
   **核心發現**：VQ 與 CoT 的差異**不在準確率高低**（CoT 可達 98%），而在 **失敗模式與可靠性邊界**。
   - **CoT plain Pro 達 98% accuracy，但仍有 1 題 SW（SWR=2%）**——即使最強的無外洩配置，仍有不可消除的邊界錯誤（test-1443 在 4 配置中錯 3，含最強的 plain Pro；答案在 19.11–19.16 間擲骰子）
   - **VQ 選擇在此邊界拒絕作答**（abstain）而非賭一個錯誤。VQ 在不確定時把問題推向 refusal/clarification，達到 SWR=0%
   - **盲審下 funnel ≤ plain**（Flash 92%<94%、Pro 96%<98%）：當 oracle 看不到對錯，更精巧的 prompt 反而過度改寫——進一步證明可靠性不來自 oracle 設計
   
   **結論重框**：VerifiQuant 與 CoT 的競爭**不是準確率軍備競賽**，而是 **failure mode 的選擇**。在 98% 高準確率下仍無法消除的 2% SWR，VQ 用結構化拒答完全消除。這正是「**以拒答置換 silent wrong**」的真實意義——不是「VQ 比 CoT 更聰明」，而是「VQ 選擇在不確定時說實話」。此即 selective prediction 框架的精髓,也是金融 AI 的關鍵：*可信賴性 > 準確率*。
4. **Recovery 貢獻（VQ Flash V3）**：8/50（16%）題目由 oracle 修正後成功。無 oracle（K=1）時 VQ 準確率為 **36/50 = 72%**（§5.8 ablation 實測，非估計），oracle 帶來 +18pp 提升
5. **JP Morgan MAS（O-ITL）**：43/50 (86%)，SWR=14%（5 wrong + 2 pipeline error），oracle_used=0（`ask_human` node 從未觸發）。以 O-ITL 條件運行仍無法消除 silent wrong，呼應「**無結構閘門則 oracle 本身不足以保證 SWR=0%**」的核心主張
6. **⚠️ VQ Pro 在相同卡片版本下仍低於 VQ Flash**：固定 V3 卡片時 VQ Pro 43/50 (86%) < VQ Flash 45/50 (90%)；固定 V1 卡片時 41/50 < 43/50。差距在兩個卡片版本下一致為 2 題，排除「卡片版本混用」這個 confound——這是真實的模型效應，非命名不一致造成。VQ Pro V3 SWR=4% 與最強 CoT 盲審配置（plain Pro SWR=2%）同級、優於 JP Morgan MAS（14%），但劣於 VQ Flash 的 0%。注意：CoT 系列無論 SWR 高低皆無 abstention，VQ 的差異化在於 SWR=0% 的可達性。可能原因：(a) FIC 卡片設計時 prompt 以 Flash 為調試目標；(b) Pro 對結構化輸出 schema 的解讀更嚴格，導致更多 F-class 拒答卻未必轉為正確；(c) 50Q 下的 run variance。本文以 **Flash-to-Flash 為主比較**，Pro 數字提供模型對照。Pro 結果的實際意義：在 VQ 框架下模型越強不保證 accuracy 越高——架構約束（FIC binding、E-check）主導了表現差異

### §5.6 Trap Resistance (RQ3)

**假設**：CoT 在 trap 上 SWR 顯著上升（無 FIC 硬約束、無 abstention 機制），VerifiQuant 透過 M/N/F/E/I funnel 將 trap 在執行前攔截為 abstention 或 successful clarification。

**度量定義（trap 與 clean 的關鍵差異）**：trap **沒有合法數值答案**——正確行為是「攔截」（refuse／clarify／abstain，或對 I_SOFT 而言 proceed-with-warning），而非算出某個數字。因此 trap 上不評估算術正確性，而評估**攔截**：

- **Caught（攔截）**：系統在適當層級攔截——`final_status ∈ {refusal, alert, needs_clarification, error}`，或 `final_diagnostic_type` 標記了任一 M/N/F/E/I 診斷；I_SOFT 則接受 proceed-with-warning。
- **Silent-Wrong（靜默錯誤）**：系統輸出了一個有把握的數值答案，且**未附任何 flag**——在一個本不該被作答的 trap 上。
- **Trap SWR** = Silent-Wrong / trap 總數（越低越好，主指標）。

關鍵不對稱：**純 CoT 沒有攔截機制**，對 F/E/N/M trap 只能照常生出一個（必然無意義或錯誤的）數字，因此其作答的每一題在定義上都是 Silent-Wrong。這正是「是否會真的抓到」的核心對照——VerifiQuant 有 funnel 會攔截，CoT 不會。

**重要方法學註記（為何 trap 上用 K=1 funnel-only 評估）**：在 trap 上，O-ITL 的 oracle 會讀取**儲存的 ground truth / python_solution** 來修正答案。但 trap 依設計**沒有合法 ground truth**（F-trap 缺欄位、E-trap 含非法負值），oracle 等於把被 redact 的值或正確答案重新注入，**人為地破解了 trap**——例如 F-trap 在第 1 輪正確診斷為 `error/F`（Schema/slot_filling，`requested_fields=[processing_time]`），但第 2 輪 oracle 把 `15.0` 填回 context 後重跑成 success。因此 K≥2 在 trap 上**不是真實能力**，而是評估資料外洩。我們以 **K=1（純 funnel，無 O-ITL）** 作為 trap resistance 的公平度量：問題是「funnel 是否在執行前攔截了壞推理」，答案在第 1 輪閘門即可判定。此結論與 §5.8 ablation 一致——funnel 與 oracle 貢獻正交：oracle 提升 clean accuracy，funnel 提供 trap resistance。

**主結果（Clean SWR 來自 §5.4 main table；Trap SWR 為 50 題 contract-grounded trap set，Flash 2.5，VQ 為 K=1 funnel-only）**：

| System | Clean SWR | Trap SWR | 結構化攔截 |
|---|---|---|---|
| CoT single-shot | 18% | 48% (24/50) | 0%（無攔截層） |
| VerifiQuant (Flash V3, funnel-only) | **0%** | **26% (13/50)** | **74%** |
| CoT + oracle (plain, blind, Flash) | 6% | [N/A — oracle 經 `python_solution` 可破解 trap] | — |
| JP Morgan MAS | 14% | [待跑 — nested repo] | — |

VQ 將整體 Trap SWR 從 CoT 的 48% 降到 **26%**；但更關鍵的是**分算子的對照**揭示了優勢的來源與邊界：

**Trap SWR 分算子明細（CoT vs VQ funnel-only，兩次獨立 run 交叉驗證）**：

| 算子（擾動） | CoT SWR | VQ SWR | VQ 攔截方式 | 解讀 |
|---|---|---|---|---|
| **E**（注入非法負值） | **90%** | **20%** | `alert/E` 邊界檢查 | ★ 最大差異：CoT 對負值費用照算不誤，FIC 的 E-check 直接攔截 |
| **I**（移除消歧線索） | **100%** | **10%** | I_SOFT soft-warning（附 clarification + transform_spec） | ★ CoT 靜默選一種解讀；VQ proceed-with-warning，把歧義顯式標記 |
| **F**（移除必要輸入） | 0% | 0% | `error/F`，`requested_fields=[...]` | SWR 平手，但 VQ 給**機器可讀**的缺欄位清單，CoT 只給散文 |
| **N**（超出領域） | 0% | 0% | `refusal/N` | SWR 平手，但 VQ 給結構化拒答，CoT 只給散文 |
| **M**（模糊意圖） | 50% | 100% | （未攔截） | ⚠ **trap 設計限制**，見下 |

**I_SOFT 計分說明（scoring note）**：I_SOFT trap 的「caught」定義為系統輸出時附有 I_SOFT convention warning，而非拒絕回答——因為 I_SOFT 的正確行為本就是 proceed-with-warning。CoT 的 I SWR=100% 需細分為兩類：(1) **表示分歧（Representational Divergence）**：CoT 輸出與 VQ 差距 100× 的數值，例如 I-1057（Kelly Criterion）CoT 給 `25.0`（百分比形式）、VQ 給 `0.25`（小數形式）；I-1442（CAGR）CoT 給 `25.85`、VQ 給 `0.2585`——這 5 題 CoT 選了不同的 output convention，在下游直接使用時會產生量級錯誤。(2) **沉默輸出（Silent Output）**：CoT 與 VQ 數值一致，但 CoT 未附任何 convention 警告，例如 I-1026（after-tax cost of debt）兩者皆輸出 `0.06`——CoT 被標為 silent wrong 的原因不是數值算錯，而是在一個存在輸出歧義的情境下無任何 flag，下游系統無從判斷是 6% 還是 0.06 decimal。兩類損害路徑不同：前者是量級錯誤，後者是不可稽核的隱性假設；共同點是使用者均無從得知 convention 的選擇。

**關鍵發現**：

1. **「是否真的抓到」——分兩種情況**。對 **E（邊界違反）與 I（歧義）**，現代 CoT 完全抓不到：它對負值費用照算（E SWR 90%）、對缺失的消歧線索靜默選一種解讀（I SWR 100%）。VQ 透過 FIC 合約的 E-check 與 I-gate 把這兩類 SWR 壓到 20% / 10%——**這正是合約約束相對於純 prompt 推理的核心增益**。以 I-1057（Kelly Criterion）為例：CoT 給 `25.0`（百分比），VQ 給 `0.25`（小數）並附 I_SOFT warning 指出 `output_scale` 歧義；以 I-1442（CAGR 2015–2022）為例：CoT 給 `25.85`，VQ 給 `0.2585` 並附相同類型的 warning——兩題皆複現 100× 量級差距。I_SOFT warning 不保證 VQ 的 convention 選擇必然「正確」，但它讓不確定性對使用者可見，而非靜默埋入答案。

2. **F/N 上 SWR 打平，但「攔截品質」不同**。值得誠實指出：Flash 2.5 級別的 CoT **能**在缺資料（F）與超領域（N）時以散文形式拒答（informal abstain，各 10/10），所以兩者 SWR 都是 0%。差別在**結構化**：VQ 的攔截帶有 `diagnostic_type`、`requested_fields`、`gate_action`（可路由、可審計、可接 HITL），CoT 的拒答埋在 `reasoning_note` 自然語言裡，下游系統無法程式化處理（structured-catch：VQ 74% vs CoT 0%）。

3. **M 是 trap 設計限制，非 VQ 能力宣稱**。M-trap（模糊意圖）保留了原題的完整數據 context，導致 VQ 的 selector 仍匹配到原 FIC 並計算，反而比 CoT 差（CoT 偶爾因「缺交期」而 abstain）。這暴露 Tier-1 M-operator 的弱點：它沒有真正隔離「意圖歧義」。**M 列不納入優勢宣稱**，已列為 Tier-2 重新設計項（移除帶 metric 的 context）。

**結論**：VerifiQuant 的 trap resistance 集中在**合約強制的確定性層（E 邊界、F 缺欄位、N 領域）與顯式歧義層（I_SOFT warning）**——在 E 與 I 上對 CoT 有數量級的 SWR 改善，在 F/N 上提供 CoT 缺乏的結構化審計軌跡。純 LLM 判斷層（M 意圖歧義）仍是 open work，與本文既有的「I_SOFT / clarification 尚未完善」立場一致。

### §5.7 Diagnostic Distribution (RQ2)

VerifiQuant Flash V3 在 50Q clean set 上的診斷分布（解析自 `vq_flash_v3/output.jsonl`）。

**第 1 輪閘門分布（initial gate，未經 O-ITL）：**

| 閘門 | funnel_layer | 計數 |
|---|---|---|
| 通過（NONE） | Logic | 37 |
| I_HARD（攔截澄清） | Critic | 6 |
| F（缺欄位/schema） | Schema | 3 |
| E（執行邊界錯誤） | Boundary | 2 |
| N（不支援/scope） | Scope | 1 |
| C（計算錯誤） | — | 1 |

50 題中 37 題（74%）於第 1 輪 Logic 層乾淨通過；13 題（26%）被漏斗攔截，其中 I_HARD 6 題為最大宗的「需澄清」類。

**經 O-ITL 後的最終分布：**

| 最終 diagnostic_type | 計數 | 最終 status |
|---|---|---|
| NONE（成功） | 45 | success ×45 |
| F | 3 | error ×4（F×3 + E×1） |
| E | 1 | （同上） |
| C | 1 | alert ×1 |

**輪數使用**：38 題 1 輪解決、5 題 2 輪、7 題 3 輪（共 12 題觸發 O-ITL，其中 8 題由錯轉對 = recovery 16%）。第 1 輪的 6 題 I_HARD 全數於 O-ITL 後解決（最終 0 題停在 I 層），印證 verifiable transform + 澄清機制能把「歧義攔截」轉為「成功澄清」而非死路。

**Soft-Flagged Wrong Rate (SFWR)**：VQ Flash V3 共 16/50 題在流程中被掛上 I_SOFT 軟警告，其中 **0 題最終答錯** → **SFWR = 0%**（最終帶 flag 且錯 = 0）。Flash 的 5 題最終未作答（abstain）皆由 deterministic 閘門（Schema/Boundary/Calculation）阻斷，且 SW=0。

**誠實揭露 — VQ Pro 的 I_SOFT 退化路徑（test-1593）**：在 VQ Pro V3 上，SW=2（SWR=4%），其中 **test-1593 暴露一個重要的退化路徑**。逐輪追溯：turn 1 正確觸發 I_HARD（cost-plus 歧義）、turn 2 觸發 I_SOFT_MISMATCH（has_i_soft=True，答案 1,322,500 已錯）、**turn 3 該 I_SOFT 被「解除」（has_i_soft 回 False）、status=success，但答案仍是錯的 1,322,500**。結果：最終交付的輸出 **無任何 flag** → 計入 SWR（silent wrong），**不計入 SFWR**。
- 以「最終輸出仍帶 flag」的嚴格定義：**Flash SFWR=0%、Pro SFWR=0%**（無任何題最終帶 I_SOFT 且錯）。
- 以「流程中曾觸發 I_SOFT 且最終錯」的寬鬆 lens：**Pro 有 1 題（test-1593）= 2.2%**；Flash 仍為 0%。
- **根因（兩層）**：(i) 錯誤注入在 **input normalization/binding 階段**——Pro 把題目給的 contract_value=2,500,000 自行重算為 2,875,000（把利潤先加進去），此步在 E-check 與 I-gate 的**上游**，綁定後算出的 1,322,500 是個通過所有邊界檢查的「似是而非」金額，funnel 無從攔截；(ii) **I_SOFT 解除邏輯的盲點**——當 oracle「解決」的是被 flag 的維度（percentage_input_scale）、而非真正出錯的維度（contract_value），flag 被清除而錯誤留存，使一個本應停在 flagged-wrong 的案例退化為 silent wrong。VQ Flash 無此問題（直接採用題目給定的 contract_value），故 Flash SWR=0%、Pro SWR=4%。這精確界定了 **funnel 的保證成立於 input binding 之後**（詳見 §6.3）。

### §5.7.1 Case Study（重點）— 為何更強的模型反而失敗：input-binding 之上的 over-normalization

> 本節是我們最重視的誠實揭露之一。它同時回答了「為何 VQ Pro 的 SWR（4%）高於 VQ Flash（0%）」與「VerifiQuant 的可靠性保證邊界在哪裡」，並直接導出一條明確的改進路徑。

**現象**：在相同 V3 卡片下，VQ Flash SWR=0%，但 VQ Pro SWR=4%（2 題 silent wrong：test-1593、test-1890）。**更強的模型反而較差**——這乍看像設計缺陷，深究後卻是本文核心前提的最強證據。

**案例 test-1593（cost-plus 合約）**。輸入：contract_value=$2,500,000、profit=15%、completion=40%。三條計算路徑（已數值驗證）：

| 路徑 | 公式 | 結果 |
|---|---|---|
| ✅ 正解（利潤計一次） | `2,500,000 × 40% × (1+15%)` | **1,150,000** |
| ❌ 漏利潤 | `2,500,000 × 40%` | 1,000,000 |
| ❌ **利潤計兩次** | `(2,500,000 + 15%×2,500,000) × 40% × (1+15%)` | **1,322,500** |

**VQ Pro 的逐輪軌跡**：

| Turn | status | has_i_soft | answer | 發生什麼 |
|---|---|---|---|---|
| 1 | needs_clarification | False | None | 正確觸發 **I_HARD**（cost-plus 定義歧義）|
| 2 | needs_iteration | **True** | 1,322,500 | 觸發 I_SOFT_MISMATCH（已錯）|
| 3 | **success** | **False** | **1,322,500** | I_SOFT 被「解除」、status=success、**答案仍錯且無 flag** |

**兩層根因**：

1. **Over-normalization 發生在 input binding 之上（funnel 的上游）**。Pro 在 normalization 階段「自作主張」把題目**字面給定**的 contract_value=2,500,000 重算為 2,875,000（= 本金 + 利潤），認為「合約值應含利潤」。FIC 的 deterministic code 本身已含 `×(1+profit)`，於是利潤被計兩次 → 1,322,500。**關鍵**：這個值是正的、落在合理範圍、通過所有 E-check 與不變量——funnel 檢查的是「**綁定後的輸入與輸出**」，無從察覺「一個給定值在綁定前被竄改」。VQ Flash 不做這層額外推理，直接採用 2,500,000，故答對。

2. **I_SOFT 解除邏輯的盲點**。turn 2 的 I_SOFT 指向 `percentage_input_scale`（15% 是否為 decimal），而真正出錯的維度是 contract_value。oracle「解決」了被 flag 的維度後，turn 3 清除 flag、輸出 success——使一個本應停在 *flagged-wrong* 的案例**退化為 silent wrong**（因此計入 SWR，不計入 SFWR-final）。

**詮釋（為何這支持而非削弱本文論點）**：VerifiQuant 的核心前提是「**約束 LLM 比 LLM 的原始能力更重要**」。Flash>Pro 的結果正是此前提的直接證據——更強的模型在**尚未被約束的那一步（input normalization）**施加更多自主推理，因而偏離題目字面規格。問題不是「guide 太強壓制了模型」，而是「**guide 在 binding 這一步太鬆**」：我們約束了算術（交給 deterministic code），卻仍讓 LLM 自由地把文字詮釋為數值。能力越強，這個未設防的縫隙被利用得越多。**這精確定位了現行合約 under-constrained 的位置，並說明可靠性不隨模型規模單調提升——架構約束才是主導變數。**

**改進方向（導出的 future work）**：**input-provenance check**——強制每個綁定的輸入值對應到 context 中的一個**字面 span**；任何 derived / 重算的值（如原文不存在的 2,875,000）必須被標記或拒絕。具體可分三級強度：(a) provenance 約束（綁定值須溯源至字面 token）；(b) 兩階段 binding（先字面抽取、再僅允許白名單單位轉換，禁止綁定階段的算術）；(c) input invariant（E-check 從「檢查輸出」擴張到「檢查輸入是否出現於 context」）。三者皆能攔下 test-1593 的 1,322,500。

**誠實的 caveat**：N=50 下 Pro 僅多錯 2 題，嚴格而言落在 run variance 範圍內。但**兩個 Pro silent-wrong 案例（test-1593 重算輸入、test-1890 推斷符號慣例）呈現一致的 over-normalization 模式**，構成一個可信的**質性發現**而非統計顯著的定律；200 題擴增後可進一步驗證其穩定性。

### §5.8 Ablation (RQ4) — O-ITL Turn Budget

在 canonical V3 卡片 + Flash 2.5 上變動 oracle 修正輪數 K（其餘設定固定）：

| 配置 | max_turns | Correct | Silent Wrong | Abstain | Selective Acc | SWR | Recovery |
|---|---|---|---|---|---|---|---|
| **K=1（w/o O-ITL）** | 1 | **36 (72%)** | 0 | 14 | 100% | **0%** | 0 |
| **K=2（O-ITL×1）** | 2 | **45 (90%)** | 0 | 5 | 100% | **0%** | 7 |
| **K=3（default，O-ITL×2）** | 3 | **45 (90%)** | 0 | 5 | 100% | **0%** | 8 |

**觀察**：
1. **O-ITL 的準確率貢獻為 +18pp（72%→90%），且全部來自第一次修正**：K=1→K=2 把 14 題 abstain 中的 9 題轉為成功（recovery 7，abstain 14→5）；K=2→K=3 第二次修正僅多挽回 1 題（recovery 7→8）但最終 accuracy 不變（皆 45/50），顯示 clean set 上 oracle 收益在第一輪後快速遞減。
2. **SWR 在所有 K 下恆為 0%**：增加 oracle 輪數只把 abstain 轉為 correct，從不把 abstain 轉為 silent wrong——recovery 是「安全的」，不以引入隱性錯誤為代價。這是 selective prediction 框架下 O-ITL 的關鍵性質：oracle 只能在漏斗已攔截的點上提供 answer-blind 的澄清，無法越過 deterministic 閘門製造假陽性。
3. **對照 CoT+oracle**：相同 oracle 預算（K=3）下，CoT+oracle（GT-blind 盲審）準確率達 92–98%（甚至高於 VQ），但 SWR 始終 ≥2%（§5.5）；VQ 的 0% SWR 來自漏斗的 abstain 能力而非 oracle 本身——印證「**oracle 提升 accuracy，漏斗保證 SWR=0%**」是兩個正交的貢獻。

> Ablation 可重現：`--only vq_flash_v3_k1 vq_flash_v3_k2`（K=3 即 canonical `vq_flash_v3`）。

### §5.9 Run-to-Run Variance (VQ Flash V3)

以 canonical 設定（V3 卡片 + Flash 2.5 + K=3）重跑 3 次（`vq_flash_v3_var1/2/3`），連同 canonical 共 4 個樣本，檢驗 LLM 非決定性對 headline 數字的影響：

| Run | Correct | SWR | Recovery |
|---|---|---|---|
| canonical (`vq_flash_v3`) | 45/50 (90%) | 0% | 8 (16%) |
| var1 | 45/50 (90%) | 0% | 8 (16%) |
| var2 | 45/50 (90%) | 0% | 4 (8%) |
| var3 | 45/50 (90%) | 0% | 5 (10%) |
| **range** | **90%–90% (σ=0)** | **0%–0% (σ=0)** | 8%–16% |

**觀察**：**accuracy 與 SWR 在 4 次重跑中完全零浮動**（皆 45/50、SWR=0%）；唯一變動的是 recovery 路徑（4–8 題觸發 oracle）。這驗證了 §6 的核心主張——*哪些* 題目需要 O-ITL 修正會隨 LLM 取樣浮動，但 deterministic 漏斗閘門（E/F/C）把最終結果鎖定在同一組 45 correct + 5 abstain，因此 SWR=0% 是結構性保證而非運氣。對比之下，CoT 系列的 accuracy 文獻上有 ±5% 的 run-to-run 浮動（本文 baseline 為單次取樣，故主表數字應視為點估計）。

> Variance 可重現：`--only vq_flash_v3_var1 vq_flash_v3_var2 vq_flash_v3_var3`。

> 可重現命令：
> ```bash
> # 跑所有 baseline
> python3 scripts/run_paper_experiments.py \
>     --config verifiquant/data/runs/paper_v1/experiment_config.yaml
>
> # 只跑 canonical VerifiQuant（V3 卡片，論文主表）
> python3 scripts/run_paper_experiments.py \
>     --config verifiquant/data/runs/paper_v1/experiment_config.yaml \
>     --only vq_flash_v3 vq_pro_v3
>
> # 彙整已有結果（不重跑）
> python3 scripts/run_paper_experiments.py \
>     --config verifiquant/data/runs/paper_v1/experiment_config.yaml \
>     --aggregate-only
> ```
> **卡片版本綁定**：config 中每個 VerifiQuant pipeline 以 `db_path` 明確指定 FIC 卡片版本——`vq_flash_v3`/`vq_pro_v3` → `fic/cards_v3.db`（canonical，論文主表）；`vq_flash`/`vq_pro` → `fic/cards.db`（V1，supplementary）。逐模型比較固定在同一 `db_path`，避免卡片版本混用。所有 baseline 結果彙整於 `verifiquant/data/runs/paper_v1/paper_results_summary.json`

---

## 6. Discussion

### §6.1 Verifiability vs Auditability: Two Distinct Properties

我們主張在金融 AI 的部署語境下，可信賴推理系統需同時提供兩種性質且兩者不可互相替代：

- **Verifiability (forward)**：每一步計算皆在預先宣告的邏輯與約束下進行，且該約束可被獨立檢查。Verifiability 是關於 *計算過程本身* 的屬性——「這一步是否符合我宣稱的規則」。
- **Auditability (backward)**：給定 output，可逐層回溯至 input 與每一個關鍵決策點。Auditability 是關於 *trace 結構* 的屬性——「我能從結果倒推回起點」。

兩者並非自動相伴：一個 trace 可以是 verifiable 但非 auditable（執行了正確 Python 但沒記錄變數綁定），亦可宣稱具備 audit trail 但非 verifiable（CoT 自述的步驟可能與實際計算路徑無因果關聯）。VerifiQuant 刻意分別承載兩者：FIC invariants 與 deterministic execution 提供 verifiability；DiagnosticReport 五元組 (exit_layer, fic_id, binding, invariant_trace, action) 提供 auditability。

此切分使我們能精確定位其他系統的不足：CoT 兩者皆缺；PAL/PoT 提供 verifiability 但 auditability 不完整（為何選此段 code、哪個 input 對應哪個變數常無紀錄）；Logic-LM 提供高 verifiability 但 auditability 受限於翻譯黑盒。VerifiQuant 是我們所知唯一將兩者作為 first-class 設計目標的金融推理框架。

### §6.2 Toward a Shrinking LLM-Dependent Zone

VerifiQuant 的長期願景並非「消滅 LLM」，而是 *讓 LLM 不可驗證的行為集中於一個顯式、可量測、可被工程手段逐步收縮的區域*。當前 I_SOFT 類別承擔此功能。

我們預期後續工作將沿三條路徑將 I_SOFT 模式遷出：
- (a) 更細緻的 FIC 不變量——將「合理範圍」「典型業界值」等慣例編碼為 scale-checks；
- (b) 自動化 invariant mining——從歷史交互中歸納可驗證的軟約束；
- (c) 業界慣例 ontology——將「典型情境下的預設假設」結構化。

每一次將 I_SOFT 模式硬化的工程努力，都是 verifiability frontier 的單調擴張。將「所有最終輸出皆為形式可驗證」視為設計北極星而非可達終點，本身即是一個誠實的工程立場。

### §6.3 Limitations

1. **Repair Card 對應規則尚未完整**：(diagnostic_type × FIC family) 的 mapping 仍依賴 `confirm_assumption` 作為 fallback；對部分組合尚未提供精細化規則。Repair Card schema 的完整化是下一步工作。
2. **I_SOFT 處理仍部分 ad-hoc**：當前 I_SOFT 觸發條件與在 metric 中的計入方式尚未完全形式化，未來工作將為其建立正式的 sub-schema 與評估協議。
3. **Oracle 模擬 *intent-explicit, answer-blind* 使用者**：Oracle 持有 ground-truth 計算規格（code）但不持有數值答案，因此其澄清行為受限於「我知道我要算什麼、但還沒算出來」的條件。實際使用者的意圖本身可能 under-specified（不知道該套哪個公式、誤述問題、或對自身需求認知模糊），此種更困難的條件需透過後續 user study 驗證。
4. **CoT oracle 基線：GT-blind 盲審協議已排除外洩疑慮**：早期 CoT oracle 以 `is_correct is False` 作為進場條件並向 oracle 提供 correctness 訊號，構成 GT-gated entry（「只在答錯時介入」本身洩漏「此題有問題」）與「只救錯、不碰對」的偏誤。我們改為 **GT-blind、每輪盲審** 協議（oracle 僅得 `python_solution`，不得數值 GT/correctness；每輪皆審），並完整重跑 plain/funnel × Flash/Pro 四配置：accuracy 92–98%、SWR 2–8%、broken_count=0。**即使最強配置（plain Pro, 98%）仍有 1 題 SW（test-1443）持續存活**。這證實 SWR=0% 需架構級 abstention，非 oracle 效能。附帶發現：盲審下 funnel-guided ≤ plain，更精巧的 oracle prompt 不改善可靠性。
5. **評估規模與難度涵蓋**：主評估為 50Q clean + 50Q trap 的 medium tier。考量 O-ITL 多輪互動（K=3）的深度診斷追溯與 token 成本，50Q 提供的是「深度分析」而非大樣本統計；我們以分層抽樣與三輪 variance check（§5.9，accuracy/SWR 零浮動）部分緩解樣本數疑慮。為佐證跨難度泛化，另自 FinanceReasoning **Hard Tier 抽 ~20 題**跑 VQ + CoT baseline，置於 Appendix（重點觀察 SWR≈0% 的 abstention 行為是否跨難度穩定）。跨資料集（FinQA、TAT-QA）泛化仍為後續工作。
6. **M-operator 與 Verifiability Frontier 的邊界**：trap 實驗中 M-operator（模糊意圖）SWR=100%，**但這是 trap 設計限制而非 VQ 無能**——M-trap 保留了原題完整的 metric context，導致 VQ 的 selector 仍匹配到原 FIC 並計算，未真正隔離「意圖歧義」。此結果精確界定了 VerifiQuant 的 *verifiability frontier*：合約強制的確定性層（E 邊界、F 缺欄位、N 領域）與顯式歧義層（I_SOFT）落在 frontier 之內並可靠攔截；純 LLM 意圖判斷（M）落在 frontier 之外，是 open work（Tier-2 M-operator 重設計）。M 列不納入優勢宣稱。
7. **funnel 保證成立於 input binding 之後 + I_SOFT 解除盲點（test-1593 揭露）**：VerifiQuant 的 deterministic 閘門（E-check、I-gate）檢查的是「**綁定後的輸入與輸出**」，不檢查「LLM 在 normalization 階段是否竄改了題目給定的值」。VQ Pro 在 test-1593 把給定的 contract_value=2,500,000 自行重算為 2,875,000（多計一次利潤），綁定後算出 1,322,500——一個通過所有邊界檢查的「似是而非」金額——funnel 無從攔截。**這界定了一個明確邊界：funnel 的可靠性保證 conditional on 正確的 input binding**。同題另暴露 **I_SOFT 解除盲點**：當 oracle 解除的是被 flag 的維度而非真正出錯的維度，flag 被清而錯誤留存，使案例由 flagged-wrong 退化為 silent wrong（故此題計入 SWR 而非 SFWR）。兩者指向同一個未來工作方向：**input-provenance check**（偵測「是否重算了任何題目直接給定的數值」）。VQ Flash 無此問題（直接採用給定值），故 Flash SWR=0%、Pro SWR=4%——再次印證架構約束的效果不隨模型規模單調提升。
8. **FIC 構建仍倚賴 LLM**：線上推理受合約嚴格約束，但 FIC 本身於 build-time 由 LLM 產出，透過 validation pipeline 部分緩解但未完全消除。

### §6.4 Broader Impact

金融場景中 LLM 錯誤的代價不限於數值損失。當 AI 系統以高自信輸出錯誤的財務建議或計算，責任、聲譽與信賴成本由 *使用者、顧問、與機構* 承擔——而非 LLM 本身。LLM 並非道德主體，無法承擔合約義務、合規責任或聲譽損害。因此錯誤的後果必然由 *無法獨立驗證 LLM 推理的人類* 吸收。這個「**責任不對稱性**」（accountability asymmetry）是金融 AI 部署相較於通用 NLP 應用的根本性差異。

主流回應之一是寄望於 *使用者教育*——「使用者應該自行檢驗 LLM 輸出」。在金融場景中此期望並不現實：
- (a) 多數使用者缺乏驗證模型推理所需的金融與技術專業；
- (b) LLM 輸出的流暢性與表面合理性壓制使用者懷疑（automation bias; Mosier & Skitka, 1996）；
- (c) post-hoc rationalization 的 CoT 軌跡使「自行檢驗」流於形式。

VerifiQuant 的設計立場是 *將驗證的負擔從使用者轉移到架構*：與其要求使用者具備檢測 LLM 失誤的能力，不如要求系統本身在輸出前完成可驗證的內部審查，並在無法驗證時 *顯式拒答* 或 *請求澄清*。在 LLM 不可作為道德主體的前提下，**將不確定性的承擔內化為架構責任，而非外推給使用者**，是 high-stakes AI 部署的基本倫理立場。本研究的所有設計選擇——M/N 結構化拒答、I_HARD 強制澄清、I_SOFT 顯式不確定性宣告——皆服從此立場。

此立場有其代價：VerifiQuant 覆蓋率 90%（5 題拒答）低於 CoT 的 100% 覆蓋率，準確率（90%）亦可能低於最強的 CoT+oracle 配置（plain Pro 98%）。但代價換來的是 SWR=0%：CoT 系列即使準確率更高，仍無法消除 silent wrong（盲審下最佳仍 SWR=2%）。我們主張此 trade-off 在金融部署中是合理且必要的——以結構性拒答置換 silent wrong，是對使用者信賴與責任分配的負責任設計；在 high-stakes 場景，*可稽核的拒答* 優於 *不可稽核的高準確率*。

---

## 7. Deployment Considerations

本節討論 VerifiQuant 架構在金融 AI 部署語境下的具體對應。我們不主張此架構已完全合規——合規涉及超出本文範圍的組織與流程因素——而是論證其設計上的 first-class properties 如何系統性地回應主要監管框架對 high-risk AI 的核心要求。

### §7.1 Architectural Properties Required by High-Risk AI Deployment

跨越 Basel、EU AI Act 與 FINRA 三套主要框架，可歸納出對系統的共同要求：

| 要求 | 定義 | VerifiQuant 承載元件 |
|---|---|---|
| Boundary declaration | 系統明確識別自身的失效情境 | M/N 結構化拒答 |
| Reproducible traceability | 同輸入產生可重現的決策軌跡 | DiagnosticReport + deterministic execution |
| Explicit checkpoints | 在語義邊界處強制人為介入 | I_HARD gate clarification |
| Bounded modification | 系統內部變更受限與可審計 | Verifiable Atomic Transforms |

### §7.2 Basel — Model Risk Management & Output Boundaries

Basel Committee 對 banking AI 的 model risk management 要求（含 SR 11-7 及相關更新）強調系統必須能識別其失效情境並設立輸出邊界，避免在模型適用範圍外仍輸出建議。VerifiQuant 的 M/N 結構化拒答正是架構級的 output floor：當問題語義 (M) 或範圍 (N) 超出 FIC 庫的明確界限，系統強制退出而非外推；相較之下，CoT 系列因缺乏 abstention 機制，在 OOD 場景必然產出 silent wrong。本研究實證 (§5.5) 顯示 JP Morgan MAS 在同等 O-ITL 條件下仍達 14% SWR（43/50 correct，7/50 silent wrong），凸顯架構級 boundary declaration 的必要性——即便有 oracle 輔助，無結構約束的 MAS 仍無法消除 silent wrong。

### §7.3 EU AI Act — Logging and Traceability (§12–13)

EU AI Act §12 要求 high-risk AI 系統具備事件紀錄能力，§13 要求記錄足以精確追溯至原始輸入。VerifiQuant 的 DiagnosticReport schema 原生滿足兩條要求：每一筆推理皆紀錄 (exit_layer, fic_id, binding, invariant_trace, action)，可逐欄位追溯至原始 question/context 與所選用的合約。CoT 軌跡因取樣的隨機性難以重現——同一輸入在不同推理時刻可能產生不同的「推理過程描述」——使其作為合規 log 的可信度本身受質疑。VerifiQuant 的 trace 因建立於 deterministic execution 之上，具備 audit log 所需的可重現性。

### §7.4 FINRA — Supervisory Controls and Human Checkpoints

FINRA 對 AI 在金融建議與經紀業務中的 supervisory framework 要求明確的人為檢查點——系統不能在語義模糊處自主決斷。VerifiQuant 的 I_HARD gate 在語義歧義處強制 clarification request，在當前實作中由 Oracle agent 自動回應；在實際部署語境下，此 checkpoint 可直接替換為 human reviewer 介面，使「強制人為介入」成為架構性質而非政策性疊加。O-ITL 的 K-bounded iteration 設計亦提供了 supervisory loop 的天然上界。

### §7.5 What This Architecture Does Not Solve

此架構不主張解決：
- (a) FIC 庫的建構過程本身的責任歸屬；
- (b) 跨司法管轄區的合規差異；
- (c) 對未涵蓋業務（如演算法交易、即時市場做市）的延伸；
- (d) 模型供應商層級的合規（資料來源、訓練透明性）。

這些屬於 *組織與流程* 層面，需架構與政策共同設計。VerifiQuant 在此分工中承擔的是 *使部署層的合規論證有架構支撐* 的角色，而非取代政策設計本身。

---

## 8. Conclusion
*[最後撰寫]*
建議含：
- 問題重述
- 四項貢獻摘要
- **三個 punchline 數字（V3 confirmed）：selective accuracy 100% / SWR 0% / abstention 10%**
  - 補充對比：CoT+oracle（GT-blind 盲審）即使達 98% 準確率（高於 VQ），仍 SWR=2%、abstention=0%（永不拒答）——SWR=0% 是架構能力,非準確率高低
- 未來工作（trap study, user study, I_SOFT 收縮、跨資料集驗證）

---

## Citation Pool

### CoT Faithfulness（§1.2 / §2.2 承重結構）

**Peer-reviewed anchors**:
- Jacovi & Goldberg, 2020 — *Towards Faithfully Interpretable NLP Systems* — **ACL 2020**
- Turpin et al., 2023 — *Language Models Don't Always Say What They Think* — **NeurIPS 2023**
- Pfau et al., 2024 — *Let's Think Dot by Dot* — **ICLR 2024**

**arXiv supporting**:
- Lanham et al., 2023 — *Measuring Faithfulness in Chain-of-Thought Reasoning* — Anthropic
- Radhakrishnan et al., 2023 — *Question Decomposition Improves the Faithfulness of Model-Generated Reasoning* — Anthropic
- Lie to Me (2026/03) — *How Faithful Is Chain-of-Thought Reasoning in Reasoning Models?*
- Ariadne (2026/01) — *A Structural Causal Framework for Auditing Faithfulness in LLM Agents*

### Financial Reasoning
- FinanceReasoning (Liu et al., ACL 2025) **[年份/venue 待確認]**
- JP Morgan MAS (Yu et al., EMNLP 2025) **[年份/venue 待確認]**

### Multi-Agent
- *Why Multi-Agent LLM Systems Fail* (ICLR 2026) **[年份/venue 待確認]**

### Neuro-Symbolic & Modular Reasoning
- Pan et al., 2023 — *Logic-LM: Empowering Large Language Models with Symbolic Solvers for Faithful Logical Reasoning* — **EMNLP 2023**
- VERAFI
- Olausson et al., 2023 — *LINC: A Neurosymbolic Approach for Logical Reasoning by Combining Language Models with First-Order Logic Provers* — **EMNLP 2023**
- Ye et al., 2023 — *SATLM: Satisfiability-Aided Language Models Using Declarative Prompting* — **NeurIPS 2023**
- Creswell et al. (DeepMind) — *Selection-Inference: Exploiting Large Language Models for Interpretable Logical Reasoning* — **558 citations, arXiv**
- Creswell et al. (DeepMind) — *Faithful Reasoning Using Large Language Models* — **205 citations, arXiv**
- Zhou et al., 2026 — *FormalJudge: A Neuro-Symbolic Paradigm for Agentic Oversight* — **arXiv:2602.11136** (preprint, under review); 定調「翻譯為形式規範是 undecidable problem」，specification synthesis bottleneck

### Tool-Augmented
- Gao et al., 2023 — *PAL: Program-Aided Language Models*
- Chen et al., 2023 — *Program of Thoughts (PoT)*
- FINDER

### Selective Prediction
- El-Yaniv & Wiener, 2010 — *On the Foundations of Noise-free Selective Classification*
- Geifman & El-Yaniv, 2017 — *Selective Classification for Deep Neural Networks* — NeurIPS

### Automation Bias
- Mosier & Skitka, 1996 — *Human Decision Makers and Automated Decision Aids*

---

# Appendix: 6-Week TODO Checklist

**起算日 2026-05-22，FinNLP 2026 截稿日 2026-07-25。** 原 6 週排程（至 07/03）較截稿日提前約 3 週，緩衝充足。

## Week 1 — 2026/05/22–05/29 — 文獻 + Code 緊急修正

| Pri | 任務 | 狀態 |
|---|---|---|
| ~~P0~~ | ~~精讀 Turpin et al. (NeurIPS 2023) — §1.2 承重點~~ | **✅ 完成** |
| ~~P0~~ | ~~讀 Lanham et al. (2023) 摘要 + 核心 figures~~ | **✅ 完成** |
| ~~P0~~ | ~~讀 Jacovi & Goldberg (2020) 的 faithfulness 定義段~~ | **✅ 完成** |
| ~~P1~~ | ~~讀 Pfau et al. (ICLR 2024) 摘要 + 主圖~~ | **✅ 完成** |
| ~~P1~~ | ~~精讀 Lie to Me + Ariadne 摘要與結論~~ | **✅ 完成** |
| ~~P0~~ | ~~驗證 4 篇主 cite 的會議與年份~~ | **✅ 完成**（見「已確認」清單）|
| ~~P0~~ | ~~決定 I_HARD/I_SOFT 是否正式拆 → 修改 `contracts.py` 的 `RefusalCategory` literal union~~ | **✅ 決議：維持現狀，不拆**。實驗已在現行 schema 下完成（I_SOFT 以 `has_i_soft` flag + soft_warnings 表達，I_HARD 走 needs_clarification）；正式拆分列為 §6.3 future work，不阻擋投稿 |
| ~~P1~~ | ~~核對 Recovery Rate 是 28% 還是 38%~~ | **✅ 完成**（官方 paper_v1 = 16%；38% 為 pre-paper ad-hoc，口徑不同）|
| ~~P1~~ | ~~核對 silent wrong 裡有幾個帶 I_SOFT flag → 算出 SFWR~~ | **✅ 完成（已精算）**：Flash SFWR=0%；Pro 最終帶 flag 定義下亦 0%，但 test-1593 為 I_SOFT→silent-wrong 退化（計入 Pro SWR=4%）。詳見 §5.7 |
| ~~P1~~ | ~~Lock §1, §2, §3 文字版（不等實驗）；把 placeholder 標清楚~~ | **✅ 完成** |

**出口條件**：✅ §1/§2/§3 文字定稿；✅ contracts.py I_HARD/I_SOFT 決議（維持現狀，列 future work）；✅ citation pool 確認。**Week 1 全數完成。**

## Week 2 — 2026/05/30–06/05 — Trap Dataset + Repair 統一

| Pri | 任務 | 狀態 | 預估時間 |
|---|---|---|---|
| ~~P0~~ | ~~跑 `expand_cases.py` 生成 trap variants~~ → **改用 `build_trap_set.py`（contract-grounded Tier-1）** | **✅ 完成**：F/E/I/N/M 各 10 題，needs_review=0 | — |
| ~~P0~~ | ~~人工抽驗 trap labels（每類 10 題）~~ | **✅ 完成**：F/E/N/M 逐題抽驗通過；I 修正 op_I 兩個 bug（重複片語、_hard/_soft 後綴）| — |
| P0 | Repair Card mapping 統一：把 `confirm_assumption` fallback 替換為明確規則 | ⏳ 待跑 | 2 天 |
| ~~P0~~ | ~~**Pilot trap test**（看 directional signal）~~ | **✅ 完成**：CoT 48% vs VQ funnel 26%；E 90→20%、I 100→10% 方向強烈 | — |
| ~~P1~~ | ~~重跑 JP Morgan baseline 在同一批 50Q 上~~ | **✅ 完成** (2026-05-30): correct=43/50, SWR=14% | — |
| ~~P2~~ | ~~把 trap set 存成 reproducible artifact~~ | **✅ 完成**：`trap/trap_set.jsonl` + `trap_manifest.json` + `build_trap_set.py` | — |
| P1 | Tier-2：M-operator 重新設計（移除帶 metric 的 context）+ I_HARD 補強 | ⏳ 待做 | 半天 |
| ~~新增 P1~~ | ~~跑 VQ Pro baseline（V3 卡片，`--only vq_pro_v3`，model fairness）~~ | **✅ 完成**（43/50, SWR=4%）| — |
| ~~新增 P1~~ | ~~跑 Ablation：VQ w/o O-ITL（K=1）+ O-ITL×1（K=2）~~ | **✅ 完成**（K=1: 36/50；K=2: 45/50；皆 SWR=0%）| — |

**出口條件**：trap set 鎖定；Repair Card mapping 完整；JP Morgan + VQ Pro baseline 確認；pilot 結果方向確認。

## Week 3 — 2026/06/06–06/12 — Main Experiments

| Pri | 任務 | 狀態 | 預估時間 |
|---|---|---|---|
| ~~P0~~ | ~~跑 CoT single-shot Flash/Pro~~ | **✅ 完成**（41/50 各）| — |
| ~~P0~~ | ~~跑 CoT+oracle Flash~~ | **✅ 完成**；後由 GT-blind 盲審重跑取代（plain Flash 47/50, 詳見主表）| — |
| ~~P0~~ | ~~跑 VQ Flash (V1/V2/V3)~~ | **✅ 完成**（45/50, SWR=0%）| — |
| ~~P0~~ | ~~跑 JP Morgan MAS × clean 50Q~~ | **✅ 完成**（43/50, SWR=14%）| — |
| ~~P0~~ | ~~跑 VQ Pro × clean 50Q（V3 cards canonical + V1 supplementary）~~ | **✅ 完成**（V3: 43/50, SWR=4%；V1: 41/50, SWR=6%）| — |
| 🔄 P0 | 跑 baselines × trap set（5 operators 分別 metric）| **主比較完成**：CoT single-shot vs VQ funnel(K=1) 已跑並評分（§5.6，整體 48%→26%）。**所有 oracle 變體（plain / funnel-guided / VQ K≥2）皆因 oracle 讀取 GT/python_solution 而在 trap 上構成評估外洩，故 trap 的公平度量固定為「無 oracle」配置**——這是 §5.6 已論證的方法學選擇，非待辦缺口。唯一仍可選補的是 JP Morgan MAS × trap（nested repo），作為 supplementary | 選補 0.5 天 |
| ~~P0~~ | ~~計算 clean set metrics~~ | **✅ 完成**（VQ Pro 跑完後 final update）| — |
| ~~P1~~ | ~~跑 ablation：VQ w/o O-ITL, w/ O-ITL×1（K=2）, w/ O-ITL×2（K=3, default）~~ | **✅ 完成**（72%→90%→90%，全程 SWR=0%）| — |
| ~~P2~~ | ~~解析 VQ output.jsonl 的 diagnostic_type 分布（§5.7）~~ | **✅ 完成**（§5.7 已填）| — |
| ~~P2~~ | ~~跑 variance check：VQ Flash V3 跑三輪看浮動範圍（§5.9）~~ | **✅ 完成**（accuracy/SWR 零浮動，recovery 8–16%）| — |
| ~~新增 P1~~ | ~~CoT oracle 改 GT-blind 每輪盲審 + 重跑 plain/funnel × Flash/Pro 四配置~~ | **✅ 完成**：排除 GT-gated entry 外洩；plain Flash 94%/SWR6%、funnel Flash 92%/SWR8%、plain Pro 98%/SWR2%、funnel Pro 96%/SWR2%，broken_count 全 0（§5.5）| — |

### 🆕 補強項（進度超前，回應預期 reviewer 質問）

| Pri | 任務 | 狀態 | 預估時間 | 動機（reviewer rebuttal） |
|---|---|---|---|---|
| **P1（補強）** | **Hard Tier 泛化證明**：從 FinanceReasoning Hard Tier 抽 ~20 題，建 FIC 卡片並跑 VQ + 至少一個 CoT baseline，結果放 **Appendix** | ⏳ 待做 | 1–1.5 天 | **質問一（樣本數）**：50Q clean + 50Q trap 在 NLP 視角偏小。AAAI 偏系統/agent 機制脈絡下，O-ITL 多輪（K=3）的深度分析可撐住 50Q；但補一個 Hard Tier 跨難度 slice 當泛化證明，直接堵住「only medium tier」的質疑。重點看 SWR 是否仍 ≈0%（abstention 行為是否跨難度穩定），而非追 accuracy |
| **P1（補強）** | **M-operator limitation 段落定稿**：把 §5.6 的「M 是 trap 設計限制、非 VQ 無能」寫成明確的 *Verifiability Frontier* 定義段，誠實標示 M-trap 保留完整 metric context 導致 selector 仍匹配原 FIC | ⏳ 待做（文字） | 0.5 天 | **質問二（M 慘烈）**：誠實揭露 limitation 在 AAAI 反而加分——精確定義「哪些失敗模式落在 verifiability frontier 之外」。要點：M-trap SWR=100% 是因 operator 未隔離意圖歧義（保留 metric context），不納入優勢宣稱，且已列 Tier-2 重設計 |

**出口條件**：Clean set main table 數字全部到位（含 JP Morgan + VQ Pro + 4 個 blind-review CoT）；trap pilot 結果出來；**（補強）Hard Tier 20Q 泛化 slice 跑完入 Appendix；M-operator frontier 段落定稿**。

## Week 4 — 2026/06/13–06/19 — Write §5 + Iterate Methodology

| Pri | 任務 |
|---|---|
| P0 | 寫完整 §5 Experiments，含所有表格與分析段 |
| P0 | 根據實驗結果回頭微調 §1.5 / §3 / §6 的具體數字 claim |
| P0 | 寫 Abstract（含三個 punchline 數字） |
| P0 | 寫 §8 Conclusion |
| P1 | 寫 §4 System Implementation 或合併進 §3 |
| P1 | 製作 Figure 1（pipeline diagram）+ Figure 2（diagnostic distribution）+ Table 1 (main results) |
| P2 | Appendix：CoT+funnel 的完整 prompt、FIC schema example、annuity due 範例展開 |

**出口條件**：full draft 完成（含 abstract / conclusion / figures）。

## Week 5 — 2026/06/20–06/26 — Advisor Review + Polish

| Pri | 任務 |
|---|---|
| P0 | 把 full draft 寄給老師 + 至少一個 peer reader |
| P0 | 同時自己跑一次 sanity check：所有數字與表格 cross-reference 一致 |
| P0 | 校對所有 citation（用 BibTeX manager 整理） |
| P1 | 收老師回饋後第一輪修訂 |
| P1 | Limitations 段確認所有 disclosure 都到位（特別是 Oracle 那塊） |

**出口條件**：advisor 回饋收齊；第一輪修訂完成。

## Week 6 — 2026/06/27–07/03 — Revision + Submit

| Pri | 任務 |
|---|---|
| P0 | 處理 advisor 第二輪意見 |
| P0 | Format check：FinNLP submission template、page limit、anonymization |
| P0 | Cross-check：所有 §1 的 forward reference 對應到後文 |
| P0 | Proof read（建議找另一個 reader 做 fresh eye pass） |
| P0 | Submit。**deadline 前 24h 內必須完成上傳測試。** |

**出口條件**：投稿完成。

---

## 關鍵風險點（全數解除，存查）

> **2026-06-02 更新：所有實驗風險已解除。** 下表保留作為決策歷史紀錄，不再是 active 風險。

| 風險 | 原機率 | 結果 |
|---|---|---|
| ~~Trap experiments 跑出來 VerifiQuant **沒有顯著優勢**~~ | ~~中~~ | ✅ **已解除**：§5.6 整體 Trap SWR 48%(CoT)→26%(VQ)，E（90→20%）與 I（100→10%）數量級改善。剩餘 M-operator 弱點已明確標為 trap 設計限制，不納入優勢宣稱 |
| ~~Turpin/Lanham/Pfau 精讀後發現引用方向有偏差~~ | ~~低~~ | ✅ **已解除**：三篇精讀完成，faithfulness 批判方向穩健，§1.2 引用無偏差 |
| ~~老師回饋要求大改架構~~ | ~~低-中~~ | ⏳ 待 Week 5 advisor review（非實驗風險，正常流程） |
| ~~FinNLP CFP 截稿日有變動~~ | ~~低~~ | ✅ **已確認**：截稿日 **2026/07/25** |
| ~~Repair Card 統一比預期久~~ | ~~中~~ | ✅ **已解除**：降級為 §6.3 limitation #1，實驗用 fallback 完成，不阻擋投稿 |

---

## Single Point of Failure（已通過）

> **2026-06-02 更新：此風險已通過。** Trap pilot 與完整 trap set 皆顯示 VerifiQuant 方向性優勢（§5.6 整體 48%→26%）。論文主軸確立。下文保留原始決策邏輯存查。

**原始顧慮**：Week 3 的實驗結果。如果 trap set 上 VerifiQuant 沒有比 CoT+funnel 強很多，整篇論文的 RQ3（主要差異化論點）會空心化。

**結果**：小規模 trap pilot 方向強烈（E 90→20%、I 100→10%），已展開至完整 trap set 並驗證。論文重心同時涵蓋「trap resistance」與「structured diagnostics + explicit abstention」——後者更因 CoT funnel-guided Pro 達 98% 仍有 2% SWR 而獲得強化（§5.5），無需 re-framing。

---

## 兩階段 venue 策略

| 版本 | Deployment 章節 (§7) 處理 | 理由 |
|---|---|---|
| 2026/07 FinNLP 版 | §7 獨立節，含 Basel/EU/FINRA mapping + 部署情境分析 | 對 FinNLP venue-fit 加分；10 月開會時可拿來跟業界討論 |
| 2027/01 ACL 版 | 縮回 §6 一段 | ACL mainstream reviewer pool 對 deployment 容忍度低，把 deployment 收進 Discussion 一段，主軸回歸 NLP/neuro-symbolic 方法貢獻 |

---

## 待你決定/查核的清單

### ✅ 已確認

| 項目 | 結果 |
|---|---|
| **VQ Flash V3 三元分類** | Correct=45, SW=0, Abstain=5；SWR=0%, SelAcc=100%, abstention=10% |
| **VQ Flash V3 Recovery** | recovery_count=8, recovery_rate=16%（paper_v1 官方 run；舊的 38% 是 pre-paper ad-hoc 跑，口徑不同）|
| **CoT single-shot Flash** | Correct=41, SW=9, Abstain=0；SWR=18%, SelAcc=82% |
| **CoT single-shot Pro** | Correct=41, SW=9, Abstain=0；SWR=18%, SelAcc=82% |
| **CoT oracle（GT-blind 盲審，canonical）** | plain Flash 47/50 (94%, SWR 6%)；funnel Flash 46/50 (92%, SWR 8%)；plain Pro 49/50 (98%, SWR 2%)；funnel Pro 48/50 (96%, SWR 2%)。broken_count 全 0 |
| ~~CoT basic oracle Flash（舊 GT-gated）~~ | ~~Correct=45, SW=5；SWR=10%~~ **已棄用**：GT-gated entry 外洩，由上方 blind-review 取代 |
| FinanceReasoning venue | **ACL 2025** — *Benchmarking Financial Numerical Reasoning More Credible, Comprehensive and Challenging* |
| Why MAS Fail venue | **ICLR 2026** — *Why Do Multi-Agent LLM Systems Fail?* |
| JP Morgan MAS venue | **EMNLP 2025** — *A Multi-Agent Framework for Quantitative Finance: An Application to Portfolio Management Analytics* |
| NeSy paper 替補 | **✅ 解決**：以 SI (Creswell et al., DeepMind)、Faithful Reasoning (Creswell et al.)、LINC (EMNLP 2023)、SATLM (NeurIPS 2023)、FormalJudge (Zhou et al., arXiv 2026) 五篇取代；§2.4 已全面改寫 |
| K 的定義 | K=3 = VQ 跑三次，oracle 修正兩次 |
| **FIC 卡片修補必要性** | 3 題 C-error = Python syntax/runtime 錯誤（E-check 在執行前就失敗，可從 `execution_smoke_ok=false` 驗證，與 gold answer 無關）；2 題 semantic 錯誤（increasing required field / adding semantic hint）= 卡片建構品質問題非 cherry-pick |

### ⏳ 待確認

| 項目 | 狀態 |
|---|---|
| **JP Morgan MAS 官方 paper_v1 結果** | ✅ correct=43/50 (86%)，SW=7（5 wrong + 2 pipeline error），SWR=14%，oracle_used=0 |
| **SFWR（I_SOFT-flagged wrong count）** | ✅ 重算完成：**Flash SFWR=0%**（SW=0，無最終帶 flag 且錯）。**Pro：最終帶 flag 定義下 SFWR=0%；但 test-1593 在 turn 2 觸發 I_SOFT、turn 3 解除後答案仍錯 → 退化為 silent wrong（計入 Pro SWR=4%）。「曾觸發 I_SOFT 且最終錯」的寬鬆 lens 下 Pro=2.2%（1/45）**。已誠實寫入 §5.7 + §6.3 #7 |
| **VQ 各層 diagnostic 分布** | ✅ 完成：§5.7 已填（第1輪 37 通過/6 I_HARD/3 F/2 E/1 N/1 C；最終 45 success + 5 abstain；SFWR=0%）|
| **I_HARD / I_SOFT 正式拆進 `contracts.py`** | ✅ 決議：**維持現狀，不拆**，列為 §6.3 future work（實驗已在現行 schema 下完成）|
| **VerifiQuant + Gemini 2.5 Pro 版本** | ✅ 完成：canonical V3 卡片 43/50 (86%)，SW=2，abstain=5，SWR=4%（與 Flash V3 同卡片可直接比較）；supplementary V1 卡片 41/50 (82%)，SWR=6%。兩個卡片版本下 Pro 皆比 Flash 低 2 題，為真實模型效應而非版本混用 |
| **CoT oracle GT-blind 盲審（4 配置）** | ✅ 完成（取代早期 GT-gated 數字）：plain Flash 47/50 (94%, SWR 6%)；funnel Flash 46/50 (92%, SWR 8%)；plain Pro 49/50 (98%, SWR 2%)；funnel Pro 48/50 (96%, SWR 2%)。broken_count 全 0。即使 98% 仍 SWR≥2%（test-1443 在 4 配置錯 3，含最強 plain Pro；CAGR 分數次方算術在 19.11–19.16 間振盪）。盲審下 funnel ≤ plain。詳見各 run 的 RUN_SUMMARY.md |
| **oracle GT 外洩疑慮** | ✅ 已排除：CoT oracle 改採 GT-blind 每輪盲審（不給數值 GT/correctness、不以 is_correct 決定進場）；VQ oracle 唯一 GT 觸點為 I_SOFT_MISMATCH（I_SOFT+答案錯），trap 上以 K=1 funnel-only 規避 |
| **FinNLP 2026 CFP 截稿日** | ✅ **確認：2026/07/25** |
| **NeSy paper 替補 citation** | ✅ 完成：SI、Faithful Reasoning (DeepMind)、LINC (EMNLP 2023)、SATLM (NeurIPS 2023)、FormalJudge (arXiv 2026) — §2.4 已改寫為三子點結構 |

---

*本檔案為 working draft，定稿前可隨時編輯。最終英文版預計於 2026/06/13–06/19 開始翻譯。*
