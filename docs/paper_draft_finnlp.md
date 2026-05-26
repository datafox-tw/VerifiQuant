# VerifiQuant — FinNLP 2026 投稿 Draft

**Status**: Working draft (中文版，定稿後再翻英文)
**Last updated**: 2026-05-26
**Target venue**: FinNLP 2026 (預計 2026/07 投稿)
**Next venue**: ACL 2027 (預計 2027/01 投稿)

---

## 標題與摘要

**Title**: VerifiQuant: Contract-Validated Financial Reasoning via Multi-layer Diagnostic Funnels and Oracle-in-the-Loop Alignment

**Abstract** *[最後撰寫]*
建議含：
- 問題：金融推理三類失敗（silent failure / semantic drift / audit gap）
- 方法：FIC + M/N/F/E/I/C funnel + O-ITL + Verifiable Atomic Transforms
- 數據三個 punchline：Selective Accuracy 95.7%、Silent Wrong Rate 4%、Abstention 8%
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

一個自然的反問是：既然 CoT 加上 self-improve 已能達到 90% 以上準確率，為何還需要更複雜的架構？關於 LLM 推理 faithfulness 的研究 (Jacovi & Goldberg, 2020) 已建立了「解釋應如實反映實際推理過程」的形式定義。Turpin et al. (NeurIPS 2023) 透過輸入操弄實驗證明，CoT 中間步驟並不必然反映模型實際得到答案的計算路徑——模型可能受隱性偏見影響但 CoT 軌跡卻不揭示此影響。Lanham et al. (2023) 進一步量化了 CoT 步驟與最終答案的因果脫節。Pfau et al. (ICLR 2024) 顯示 CoT tokens 可被無意義填充符號替代而效能不降，暗示 CoT 的表面合理性可能與內部計算無關。近期工作 (Lie to Me, 2026; Ariadne, 2026) 延伸此分析至 reasoning models 與 multi-agent 系統。

在金融場景，這意味著即使 CoT 答對了，其產出的推理過程也無法作為審計證據；答錯時更無從定位故障層。**90% 的 CoT 準確率與 88% 的可審計準確率在部署語境下並不可比。** 從 architecture 而非 prompt engineering 著手的理由不僅在於前者的可驗證性更強，更在於 high-stakes 部署中不能將驗證負擔外推給使用者（詳見 §6.4）。

### §1.3 現有解法的光譜與缺口

為脫離 CoT 的不可信問題，文獻發展出三條路線：

- **(a) 執行端外部化**（PAL, PoT, FINDER）將算術交給 Python interpreter，但 LLM 仍可寫出語法正確、金融邏輯錯誤的程式碼，對 semantic drift 無防禦；
- **(b) 符號求解器路線**（Logic-LM, VERAFI）依賴自然語言到形式邏輯的脆弱翻譯，在複雜情境下準確率反而劣化；
- **(c) Agentic reflection**（ToT, Self-Refine）以 LLM 驗證 LLM，未解決 faithfulness 的本質問題。

**缺口在於：沒有一個框架同時處理 semantic drift 的事前攔截、計算過程的可審計性、以及修復過程的可驗證性。**

### §1.4 VerifiQuant：合約驗證的診斷漏斗

我們提出 **VerifiQuant**，一個將金融推理拆解為「**先驗證、再計算**」流程的 neuro-symbolic 框架，目標是在準確率與可審計性之間取得平衡而非單邊極致。其三項核心設計為：

1. **M/N/F/E/I/C 六層診斷漏斗**——將失敗模式拆為 Misunderstanding / Not-Supported / Formula-spec / Extraction-boundary / Interception / Calculation 六類，I-gate 進一步分為 I_HARD 與 I_SOFT；每層皆為合法退出點，使「拒答」與「澄清」成為一級結果而非 fallback；
2. **Financial Inference Contract (FIC)**——每個金融公式對應一份結構化合約（檢索 metadata + deterministic 執行碼 + 不變量 + 語義陷阱描述），LLM 受合約約束選卡、抽欄位、辨歧義，但不執行算術；
3. **Verifiable Atomic Transforms**——當 I_HARD 攔截到語義歧義，系統僅允許 AST 邊界內的原子轉換，並以數值交叉驗證確保修改前後的代數等價，杜絕 LLM 在修復階段對核心邏輯的自由竄改。

### §1.5 為何此架構在部署語境下優於更高準確率的 CoT

在 selective prediction (El-Yaniv & Wiener, 2010; Geifman & El-Yaniv, 2017) 的框架下，VerifiQuant 在 FinanceReasoning 中度難度子集上達到 88% Pass@1、4% silent wrong、8% safe refusal——亦即 selective accuracy 95.7%，僅略低於 CoT+funnel prompt 的 96%，但提供 CoT 系列架構性無法達成的 abstention 能力。同時，金融 multi-agent baseline (JP Morgan) 在無結構約束下達 34% silent wrong，呼應 *Why MAS Fail* (2026) 對 multi-agent 自然語言共識失效的觀察。

**因此 VerifiQuant 88% 的可審計準確率與 CoT 92% 的不可審計準確率在部署決策上並不等價。**

### §1.6 貢獻

1. 提出 **M/N/F/E/I/C 錯誤分類學**（含 I_HARD/I_SOFT 二分），將金融 LLM 失敗模式從二元正誤擴展為七層可診斷狀態；
2. 設計 **FIC 合約 + Verifiable Atomic Transforms**，使修復過程受 AST 與數值不變量雙重約束；
3. 提出 **Oracle-in-the-Loop (O-ITL)** 評估協議，量測「意圖明確但答案未知」條件下的框架可恢復性；
4. 在 FinanceReasoning 與其衍生的 trap dataset 上實證：clean cases 上 VerifiQuant 與 CoT 達到可比 selective accuracy，但在 **trap cases (F/E/I variants)** 上 VerifiQuant 顯著降低 silent wrong rate，並提供 CoT 無法產出的結構化審計軌跡。

---

## 2. Related Work

### §2.1 Financial Reasoning Benchmarks 與其侷限

金融推理近年成為 LLM 評測的熱門域。FinanceReasoning (Liu et al., ACL 2025) 系統化整理金融問答的失敗模式，指出當前模型表現受限的主因並非推理能力天花板，而是 *資料集本身存在歧義、缺乏標準化的問題規格*。JP Morgan 提出的金融 multi-agent system (Yu et al., EMNLP 2025) 在類似資料上 Pass@1 僅達 66%，且失敗案例缺乏結構化歸因。本研究使用 FinanceReasoning 作為主要資料集，並進一步衍生 trap variants 以系統性測試 silent failure 與 semantic drift；同時，我們以 FIC 合約取代「以自然語言描述問題」的傳統 schema，從上游降低 FinanceReasoning 所指出的歧義問題。

### §2.2 Chain-of-Thought 的 Faithfulness 質疑

一條主流路線是透過 CoT 與其延伸（Self-Refine, Tree-of-Thoughts）讓模型外顯推理過程。然而 NLP faithfulness 文獻對此持系統性質疑：Jacovi & Goldberg (ACL 2020) 確立 faithfulness 的形式定義，Turpin et al. (NeurIPS 2023) 與 Lanham et al. (2023) 透過實驗證據展示 CoT 軌跡可與實際推理脫節，Pfau et al. (ICLR 2024) 進一步顯示 CoT 的表面結構可能與內部計算無關。Lie to Me (2026) 與 Ariadne (2026) 將此分析延伸至 reasoning models 與 multi-agent 系統。在金融部署語境下，這意味著 CoT 軌跡無法作為審計證據。本研究將推理軌跡從 LLM 的自由生成抽離，改由 FIC 合約與 deterministic 執行決定，使軌跡與結果之間具備可驗證的因果關係。

### §2.3 Tool-Augmented 與 Execution-Based Reasoning

PAL (Gao et al., 2023)、Program-of-Thoughts (Chen et al., 2023) 以及金融領域的 FINDER 將算術交給外部 Python interpreter 執行，解決了 LLM 在算術上的不可靠性。然而這類方法假設「程式碼正確即邏輯正確」，無法防禦 LLM 雖然語法正確但金融語義錯誤的情境——例如將期初年金誤寫為期末年金。VerifiQuant 與其差別在於：執行階段之前先插入 I-gate 進行 semantic drift 攔截，且僅允許 AST 邊界內的 verifiable atomic transforms 進行修復。

### §2.4 Neuro-symbolic 與 Formal Verification

為了取得形式保證，Logic-LM (Pan et al., 2023)、VERAFI 等工作嘗試將自然語言題目翻譯為符號邏輯或 SMT 規則交給 solver 處理。NeSy 2025 的近期分析指出此類方法的核心障礙：使用者意圖以非正式的自然語言表達，難以無損地映射為形式規則，使翻譯瓶頸成為系統可靠性的上限。VerifiQuant 採取折衷路線——以 FIC 作為 *中間層合約*，將形式化的負擔集中於離線階段的卡片建構，線上推理只需在已驗證的合約空間內進行 selection 與 binding，避開了 runtime 的形式化翻譯瓶頸。

### §2.5 Multi-Agent Systems 與 Self-Reflection

另一條路線以 multi-agent 協作或 self-reflection 提升可靠性 (ToT, Self-Refine, Reflexion)。然而 *Why Multi-Agent LLM Systems Fail* (Authors, ICLR 2026) 指出，MAS 最常見的失敗模式為 *theory-of-mind failure*——agent 之間假設彼此理解，但對共享狀態的詮釋分歧；JP Morgan MAS 的 66% Pass@1 與 34% silent wrong rate 呼應此觀察。VerifiQuant 的 Oracle-in-the-Loop (O-ITL) 設計避開了 agent 間自然語言共識的脆弱性：所有 agent 互動皆透過結構化的 DiagnosticReport 與 FIC 欄位進行。

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

**Repair Action 表面**。Repair Card 將 DiagnosticReport 映射至八種結構化動作之一（§3.2），每條 repair rule 並宣告其 `allowed_next_steps` ∈ {`rerun_same_fic`, `select_alternative_fic`, `ask_followup`, `stop_with_refusal`}，限制下一輪的合法狀態轉移。

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

**Trap set（待跑）**：
由 `expand_cases.py` 從上述 50Q 衍生的 F-trap、E-trap、I-trap variants，每類 N 題 **[待填]**；trap 標註將經人工抽驗（每類至少 10 題 sanity check）。

### §5.3 Baselines

1. **CoT single-shot (Gemini 2.5 Flash)** — 無 self-improve，一次性輸出
2. **CoT + basic oracle (Flash)** — 給予兩次自我修改機會（prompt 層級，無 funnel 概念）
3. **CoT + VQ funnel oracle (Flash)** — 在 oracle prompt 中注入六層漏斗概念，但無 FIC 結構約束
4. **CoT + VQ funnel oracle (Pro)** — 同上但換用 Gemini 2.5 Pro
5. **JP Morgan MAS** (Yu et al., EMNLP 2025) — 金融領域 multi-agent baseline
6. **VerifiQuant** (Ours, Gemini 2.5 Flash) — 完整架構 + 最多 K=3（oracle 修正兩次）

> ⚠️ **Model-fairness 問題待決**：VerifiQuant 目前使用 Flash，最強的 CoT baseline 使用 Pro。公平比較應同時跑 VerifiQuant + Pro 版本。建議在 Week 3 補跑。

### §5.4 Metrics

採用 Selective Prediction (El-Yaniv & Wiener, 2010; Geifman & El-Yaniv, 2017) 框架：

- **Correct / Silent Wrong / Safe Refusal**（三元分類）
- **Coverage** = (Correct + Silent Wrong) / N
- **Selective Accuracy** = Correct / (Correct + Silent Wrong)
- **Silent Wrong Rate (SWR)** = Silent Wrong / N — *金融部署的關鍵風險指標*
- **Soft-Flagged Wrong Rate (SFWR)** = (I_SOFT-flagged ∩ Wrong) / (Correct + Silent Wrong)
- **Diagnostic Distribution**：VerifiQuant 在 M/N/F/E/I/C 各層的攔截計數
- **Recovery Rate**：第 1 輪錯誤但於 O-ITL 後修正的比例

### §5.5 Main Results — Clean Set (RQ1, RQ5)

> ⚠️ **數字狀態說明**：以下表格中，CoT 各行為目前已有數據；VQ 的 Silent Wrong / Safe Refusal 精確分布待重跑確認；JP Morgan 的 silent wrong 分布待確認。

| System | Model | Correct | Silent Wrong | Safe Refusal | Coverage | Selective Acc | SWR |
|---|---|---|---|---|---|---|---|
| CoT (single-shot) | Flash 2.5 | 30 (60%) | 20 (40%) | 0 | 100% | 60.0% | 40% |
| CoT + basic oracle | Flash 2.5 | 44 (88%) | 6 (12%) | 0 | 100% | 88.0% | 12% |
| CoT + VQ funnel oracle | Flash 2.5 | 46 (92%) | 4 (8%) | 0 | 100% | 92.0% | 8% |
| CoT + VQ funnel oracle | Pro 2.5 | 48 (96%) | 2 (4%) | 0 | 100% | 96.0% | 4% |
| **VerifiQuant + O-ITL (K=3)** | **Flash 2.5** | **44 (88%)** | **[待確認]** | **[待確認]** | **92%** | **95.7%** | **[待確認]** |
| JP Morgan MAS | - | 33 (66%) | [待確認] | [待確認] | 100%? | 66.0% | [待確認] |

**現有 VQ 精確數據（來自 summary JSON）**：
```
total_cases:        50
success_count:      47   (code 成功執行)
success_rate:       0.94
correct_count:      44   (執行且答對)
accuracy:           0.88
recovery_count:     19   (觸發 oracle 後成功修正)
correct_in_one_shot: 28  (直接通過，無 funnel 攔截)
recovery_rate:      0.38
```

**6 個失敗案例分析**（共 6 = 50 - 44 correct）：
- 1 題：golden code 本身錯誤（資料集問題，非系統失敗）
- 2 題：需要期初/期末 atomic transform，功能尚未完成（預計 Week 2 解決）
- 1 題 F-class：oracle 修正後仍失敗
- 1 題 E-class：oracle 修正後仍失敗
- 1 題 I-class：oracle 修正後仍失敗

> **K 的定義說明**（需在 §3.4 統一）：K=3 = VQ 跑三次，oracle 修正兩次（oracle 修正次數 = K-1）。本文所有「K=2」改為「K=3」，or 改用「oracle 修正最多兩次」的說法。

**觀察**：
1. CoT single-shot 60% → basic oracle 88% → VQ funnel oracle 92%：self-improve 與 funnel knowledge 均對準確率有獨立貢獻
2. 同 Flash model 下，VQ (88%) 與 CoT+funnel oracle (92%) 相差 4%，但 VQ 具備 abstention；CoT 永遠不拒答
3. JP Morgan MAS 66% 呼應 *Why MAS Fail* 的 theory-of-mind failure 觀察

### §5.6 Trap Resistance (RQ3) — 待跑

**假設**：CoT+funnel 在 trap 上 SWR 顯著上升（無 FIC 硬約束），VerifiQuant 透過 I-gate 與 verifiable transforms 將 trap 轉為 abstention 或 successful clarification。

**[實驗待補]**

### §5.7 Diagnostic Distribution (RQ2)

VerifiQuant 在 50Q clean set 上的分布：success / I_HARD / I_SOFT / E / F / C / M / N **[數據待補]**

### §5.8 Ablation (RQ4)

- VerifiQuant w/o O-ITL（max_turns=1）
- VerifiQuant w/ O-ITL×1（max_turns=2，oracle 修正一次）
- VerifiQuant w/ O-ITL×2，即 K=3（default，oracle 修正最多兩次）

**[實驗待跑]**

> 可重現命令：
> ```bash
> # 跑所有 baseline
> python3 scripts/run_paper_experiments.py \
>     --config verifiquant/data/runs/paper_v1/experiment_config.yaml
>
> # 只跑特定 baseline
> python3 scripts/run_paper_experiments.py \
>     --config verifiquant/data/runs/paper_v1/experiment_config.yaml \
>     --only vq_flash vq_pro
>
> # 彙整已有結果（不重跑）
> python3 scripts/run_paper_experiments.py \
>     --config verifiquant/data/runs/paper_v1/experiment_config.yaml \
>     --aggregate-only
> ```
> 所有 baseline 結果彙整於 `verifiquant/data/runs/paper_v1/paper_results_summary.json`

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
4. **評估規模有限**：當前評估集中於 FinanceReasoning medium 子集；對 hard tier 與跨資料集（FinQA、TAT-QA）的泛化能力有待後續驗證。
5. **FIC 構建仍倚賴 LLM**：線上推理受合約嚴格約束，但 FIC 本身於 build-time 由 LLM 產出，透過 validation pipeline 部分緩解但未完全消除。

### §6.4 Broader Impact

金融場景中 LLM 錯誤的代價不限於數值損失。當 AI 系統以高自信輸出錯誤的財務建議或計算，責任、聲譽與信賴成本由 *使用者、顧問、與機構* 承擔——而非 LLM 本身。LLM 並非道德主體，無法承擔合約義務、合規責任或聲譽損害。因此錯誤的後果必然由 *無法獨立驗證 LLM 推理的人類* 吸收。這個「**責任不對稱性**」（accountability asymmetry）是金融 AI 部署相較於通用 NLP 應用的根本性差異。

主流回應之一是寄望於 *使用者教育*——「使用者應該自行檢驗 LLM 輸出」。在金融場景中此期望並不現實：
- (a) 多數使用者缺乏驗證模型推理所需的金融與技術專業；
- (b) LLM 輸出的流暢性與表面合理性壓制使用者懷疑（automation bias; Mosier & Skitka, 1996）；
- (c) post-hoc rationalization 的 CoT 軌跡使「自行檢驗」流於形式。

VerifiQuant 的設計立場是 *將驗證的負擔從使用者轉移到架構*：與其要求使用者具備檢測 LLM 失誤的能力，不如要求系統本身在輸出前完成可驗證的內部審查，並在無法驗證時 *顯式拒答* 或 *請求澄清*。在 LLM 不可作為道德主體的前提下，**將不確定性的承擔內化為架構責任，而非外推給使用者**，是 high-stakes AI 部署的基本倫理立場。本研究的所有設計選擇——M/N 結構化拒答、I_HARD 強制澄清、I_SOFT 顯式不確定性宣告——皆服從此立場。

此立場有其代價：VerifiQuant 88% Pass@1 低於不具備拒答機制的 CoT。我們主張此 trade-off 在金融部署中是合理且必要的——少量覆蓋率降低換取結構性的 silent-wrong-rate 控制，是對使用者信賴與責任分配的負責任設計。

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

Basel Committee 對 banking AI 的 model risk management 要求（含 SR 11-7 及相關更新）強調系統必須能識別其失效情境並設立輸出邊界，避免在模型適用範圍外仍輸出建議。VerifiQuant 的 M/N 結構化拒答正是架構級的 output floor：當問題語義 (M) 或範圍 (N) 超出 FIC 庫的明確界限，系統強制退出而非外推；相較之下，CoT 系列因缺乏 abstention 機制，在 OOD 場景必然產出 silent wrong。本研究實證 (§5.5) 顯示 JP Morgan MAS 在無此邊界機制下達 34% silent wrong rate，凸顯架構級 boundary declaration 的必要性。

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
- selective accuracy 95.7% / SWR 4% / abstention 8% 三個 punchline 數字
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

### Neuro-Symbolic
- Pan et al., 2023 — *Logic-LM*
- VERAFI
- ~~Formal Verification for LLM Reasoning (NeSy 2025)~~ **[已放棄：CoRR 2025，非頂會；改去找更強的 faithfulness 或 neuro-symbolic 頂會論文替補]**

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

**起算日 2026-05-22，目標 2026-07-XX 投稿。**

## Week 1 — 2026/05/22–05/29 — 文獻 + Code 緊急修正

| Pri | 任務 | 預估時間 |
|---|---|---|
| P0 | 精讀 Turpin et al. (NeurIPS 2023) — §1.2 承重點 | 3h |
| P0 | 讀 Lanham et al. (2023) 摘要 + 核心 figures | 1.5h |
| P0 | 讀 Jacovi & Goldberg (2020) 的 faithfulness 定義段 | 1h |
| P1 | 讀 Pfau et al. (ICLR 2024) 摘要 + 主圖 | 1h |
| P1 | 精讀 Lie to Me + Ariadne 摘要與結論 | 2h |
| P0 | 驗證 4 篇主 cite 的會議與年份 | 2h |
| P0 | 決定 I_HARD/I_SOFT 是否正式拆 → 修改 `contracts.py` 的 `RefusalCategory` literal union | 3h |
| P1 | 核對 Recovery Rate 是 28% 還是 38% | 1h |
| P1 | 核對 88/4/8 中 4 個 silent wrong 裡有幾個帶 I_SOFT flag → 算出 SFWR | 2h |
| P1 | Lock §1, §2, §3 文字版（不等實驗）；把 placeholder 標清楚 | 半天 |

**出口條件**：§1/§2/§3 文字定稿；contracts.py I_HARD/I_SOFT 決議；citation pool 確認。

## Week 2 — 2026/05/30–06/05 — Trap Dataset + Repair 統一

| Pri | 任務 | 預估時間 |
|---|---|---|
| P0 | 跑 `expand_cases.py` 在 full 50Q clean set 上生成 F-trap, E-trap, I-trap variants | 1 天 |
| P0 | 人工抽驗 trap labels（每類 10 題，確認 trap 真的是 trap） | 半天 |
| P0 | Repair Card mapping 統一：把 `confirm_assumption` fallback 替換為明確規則 | 2 天 |
| P0 | **Pilot trap test**（5-10 題 I-trap 跑一輪，看 directional signal） | 半天 |
| P1 | 重跑 JP Morgan baseline 在同一批 50Q 上，確認 66/34/0 分布 | 半天 |
| P2 | 把 expand_cases.py 生成的 trap set 存成 reproducible artifact | 2h |

**出口條件**：trap set 鎖定；Repair Card mapping 完整；JP Morgan baseline 確認；pilot 結果方向確認。

## Week 3 — 2026/06/06–06/12 — Main Experiments

| Pri | 任務 | 預估時間 |
|---|---|---|
| P0 | 跑 4 baselines × clean 50Q：CoT / CoT+funnel / VQ / JP Morgan，輸出三元分類 | 2 天 |
| P0 | 跑 4 baselines × trap set（F/E/I 三類分別 metric） | 2 天 |
| P0 | 計算所有 metrics：Correct/SW/SR, Coverage, Selective Acc, SWR, SFWR | 1 天 |
| P1 | 跑 ablation：VQ w/o O-ITL, w/ K=1, w/ K=2 | 1 天 |
| P2 | 跑 variance check：CoT 跑三輪看浮動範圍 | 1 天 |

**出口條件**：所有 main table 數字到位；trap resistance 結果出來。

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

## 關鍵風險點

| 風險 | 機率 | 緩解 |
|---|---|---|
| Trap experiments 跑出來 VerifiQuant **沒有顯著優勢** | 中 | Week 3 結果出來時若 trap SWR 沒差很多，立刻 framing 退守到 "explicit abstention" 與 "structured diagnostics" 兩個論點，trap section 改成 qualitative case study |
| Turpin/Lanham/Pfau 精讀後發現引用方向有偏差 | 低 | Week 1 結束就會知道；這三篇對 faithfulness 的批判都很穩健，風險低 |
| 老師回饋要求大改架構 | 低-中 | Week 4 結束時就先 informal 跟老師同步主軸 |
| FinNLP CFP 截稿日有變動 | 低 | Week 1 確認官方時間軸 |
| Repair Card 統一比預期久 | 中 | Week 2 若 slip，把「完整 mapping」降級為 §6.3 limitation #1，先用 fallback 跑實驗 |

---

## Single Point of Failure

**Week 3 的實驗結果**。如果 trap set 上 VerifiQuant 沒有比 CoT+funnel 強很多，整篇論文的 RQ3（主要差異化論點）會空心化。

**保險措施**：Week 2 結束時先跑 *小規模 trap pilot*（5-10 題 I-trap），看 directional signal。如果方向對，再展開到完整 trap set；如果完全沒差，**Week 3 就要重新 framing**——把論文重心從「trap resistance」改為「structured diagnostics + explicit abstention」，避免 Week 5 才發現問題。

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
| Recovery Rate | **38%** (recovery_count=19, 以最新 JSON 為準) |
| FinanceReasoning venue | **ACL 2025** — *Benchmarking Financial Numerical Reasoning More Credible, Comprehensive and Challenging* |
| Why MAS Fail venue | **ICLR 2026** — *Why Do Multi-Agent LLM Systems Fail?* |
| JP Morgan MAS venue | **EMNLP 2025** — *A Multi-Agent Framework for Quantitative Finance: An Application to Portfolio Management Analytics* |
| NeSy paper | **放棄**，CoRR 2025 非頂會，待找替補 |
| K 的定義 | K=3 = VQ 跑三次，oracle 修正兩次 |

### ⏳ 待確認

| 項目 | 狀態 |
|---|---|
| **SFWR（I_SOFT-flagged wrong count）** | 待重跑 pipeline 後確認 |
| **VQ 三元分類精確數字**（Silent Wrong / Safe Refusal 各幾題） | 待重跑確認 |
| **JP Morgan silent wrong 分布** | 你的 reimplementation 只有 correct=33，其餘 17 題分布待確認 |
| **I_HARD / I_SOFT 正式拆進 `contracts.py`** | 計畫拆，尚未動作 |
| **VerifiQuant + Gemini 2.5 Pro 版本** | ⚠️ 建議補跑，解決 model fairness 問題 |
| **FinNLP 2026 CFP 截稿日** | 尚未確認 |
| **NeSy paper 替補 citation** | 待找（方向：neuro-symbolic 頂會，或 formal verification + NLP 頂會） |
| **recovery_count=19 vs 留言說 14 的矛盾** | JSON 說 19（38%），留言說「14 題有 F/E/I 類錯誤的經過 oracle 處理後變得精準」。兩者定義是否不同？請確認 `recovery_count` 的統計口徑 |

---

*本檔案為 working draft，定稿前可隨時編輯。最終英文版預計於 2026/06/13–06/19 開始翻譯。*
