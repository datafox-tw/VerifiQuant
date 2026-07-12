# ICAIF 2026 — VerifiQuant 8 頁 Outline（Governance-by-Execution 版）

**Status**: Planning doc（定稿前隨時編輯）
**Created**: 2026-07-12
**Deadlines**: **ICAIF 2026 = 08/02**（主目標，剩 21 天）· FinNLP 2026 = 08/11（次目標，剩 30 天）
**Format**: ACM `sigconf`（⚠️ 依 CFP 確認：頁數是否含 references、是否 double-blind、是否允許 appendix）
**母文件**: `docs/paper_draft_finnlp.md`（方法與 50Q 數字的 single source of truth）

---

## 0. 現況分析（2026-07-12，逐項驗證過）

### 已完成（paper_v1，50Q）— 論文級完整度

| 資產 | 狀態 | 位置 |
|---|---|---|
| Clean set 主表（VQ/CoT×6/JPM 共 10+ 配置） | ✅ 全跑完 | `verifiquant/data/runs/paper_v1/results/`（20 個 run 目錄） |
| VQ Flash V3 canonical | ✅ 45/50、SWR=0%、abstain=5、recovery 16% | `results/vq_flash_v3/` |
| CoT 盲審 grid（plain/funnel × Flash/Pro） | ✅ 92–98%、SWR 2–8%、broken=0 | `results/cot_*_blindreview_*` |
| JPM MAS reimpl | ✅ 86%、SWR=14%、oracle_used=0 | `results/jpmorgan/` |
| Trap set 50 題（contract-grounded, 5 算子×10） | ✅ E: 90→20%、I: 100→10% | `paper_v1/trap/` |
| K-ablation（K=1/2/3 截斷自 canonical） | ✅ 74→84→90%，全程 SWR=0% | 同上 |
| Variance（4 runs） | ✅ accuracy/SWR 零浮動 | `vq_flash_v3_var1/2/3` |
| Refusal prompt ablation | ✅ 2026-06-09 完成 | `docs/results/2026-06-09_refusal_prompt_ablation.md` |
| FinNLP 中文 draft | ✅ ~90%（§1–§8 + citation pool + TODO） | `docs/paper_draft_finnlp.md` |

### 已就緒但未跑（paper_v2_250）

| 資產 | 狀態 |
|---|---|
| `questions_250.jsonl`（180 medium + 70 hard，canonical 50 為子集，seed 42） | ✅ |
| FIC 卡片 core/retrieval/repair = 250/250/1960（1522 diagnostic rules） | ✅ |
| 沙盒驗證：`execution_smoke failed=0`、零重複、kernel dedup 零碰撞 | ✅ `fic/validation_report.json` |
| **250Q 任何跑分** | ❌ `results/` 為空 |
| Card store（cards.db）建置 | ❌ |
| 250Q trap set / VQ Pro 250 / IMR instrumentation / JPM 250 | ❌ |

**難度缺口提醒**：hard tier difficulty 4.16–7.19，50Q 上限才 4.03——hard 是全新區間，
VQ 行為（abstention 率、卡片良率、SWR）都是未知數。§7.2 的 claim fallback 規則因此存在。

### 關鍵路徑

```
A0 card store + lint + relations（本地，半天）
 → A1 cot_single_shot_flash 250（便宜，驗 hard-tier parsing）
 → A2 vq_flash K=3 + cot_basic_oracle 250（最重，估 50Q 的 5×）
 → A3 修卡 loop（只修無-gold-可偵測項）→ 重跑
 → A4 主表 + medium/hard 分層分析          ← 必須在 ~07/26 前完成才趕得上 8/2
 → A5 擇優：VQ Pro 250 > IMR > trap 250 > JPM 250
```

---

## 1. Venue 策略與時程

### 1.1 兩投風險（08/02 前必查）

ICAIF（8/2 投）與 FinNLP（8/11 投）審稿期重疊。若兩篇「substantially similar」，
多數 venue 的 concurrent-submission 條款會踩線。**兩條路擇一**：

- **(a) 實質差異化（建議）**：ICAIF = governance 敘事 + 250Q 主結果 + 難度分層分析；
  FinNLP = NLP/faithfulness 敘事 + 50Q 深度分析 + trap 生成方法學。兩篇主表、RQ、
  貢獻宣稱皆不同。仍需逐字檢查兩邊 CFP 的 dual-submission 條款。
- **(b) 序列投**：8/2 只投 ICAIF；FinNLP 待 ICAIF 結果（若 FinNLP 有下一輪）或改投 ACL 2027。

### 1.2 倒數 21 天（07/12 → 08/02）

| 日期 | Track A（實驗） | Track B（寫作） |
|---|---|---|
| 07/12–07/14 | A0 card store + lint + relations；A1 cot single-shot 250 | 本 outline 定稿；LaTeX sigconf skeleton；§2 governance 脊椎表英文化 |
| 07/15–07/19 | A2 vq_flash 250（K=3）+ cot_basic_oracle 250（機器勿睡眠） | §1/§3 英文初稿（方法不動，直接從 FinNLP draft 翻+縮） |
| 07/20–07/22 | A3 修卡 loop → `cards_v2_v2.db` → 重跑受影響題 | §4/§5 用 50Q 數字佔位寫死結構；§6 case studies |
| 07/23–07/26 | A4 250Q 主表 + 分層分析；（若順）A5-① VQ Pro 250 | §5 換裝 250Q 數字；Fig 2 難度分層圖 |
| 07/27–07/30 | 冷凍實驗；只補圖表 | 全文 polish、§7 limitations、abstract 定稿、advisor pass |
| 07/31–08/01 | — | format check、anonymization、references、上傳測試 |
| **08/02** | — | **Submit ICAIF** |
| 08/03–08/11 | — | FinNLP 版差異化改寫（母稿已 90%，重點是與 ICAIF 版拉開） |

**斷點規則**：07/26 若 250Q 全量未完成 → 主表用「50Q full + 250Q partial（已完成子集）」
或退回 50Q preliminary + 註明 250Q in progress。結構上 §5 的表格欄位設計已預留兩種情況（§7.2）。

---

## 2. 核心敘事轉向

### 2.1 一句話主張

> From "financial reasoning accuracy" to **"governance-by-execution for financial LLM workflows."**

金融 LLM workflow 的部署風險不是答錯，而是**無法治理**：沒有明確邊界、沒有可重播 trace、
沒有結構化拒答、沒有可驗證修復。VerifiQuant 用 FIC + diagnostic funnel + deterministic
execution 把 governance controls **編譯進推理流程本身**——governance 是被執行的，不是被描述的。

**Strongest claim（全文收束句）**：
> Accuracy alone is not a sufficient governance metric for financial LLM workflows. A deployable
> system must expose *when* it does not know, *why* it refuses, *which* computation contract it
> used, and *whether* a repair changed the underlying formula. VerifiQuant turns these requirements
> into executable controls — and we show that injecting the same diagnostic knowledge into prompts
> does not reproduce them.

### 2.2 Governance 脊椎（貫穿 §2→§3→§5/§6 的三欄結構，即 Table 1）

| Governance control | 承載機制（§3） | 實驗證據（§5/§6） | 對應監管要求 |
|---|---|---|---|
| Boundary declaration | M/N 結構化拒答 | trap N/M；JPM MAS 對照（無閘門→SWR 14%） | Basel SR 11-7 output boundaries |
| Input & boundary validation | E-check（deterministic invariants/scale-checks） | trap E：CoT 90% → VQ 20% | model risk management |
| Ambiguity disclosure | I_HARD 阻斷 / I_SOFT 顯式警告 | trap I：CoT 100% → VQ 10%；SFWR=0% | FINRA supervisory checkpoints |
| Verified repair | Verifiable Atomic Transforms（AST bound + 數值 cross-verify） | annuity 579.98→577.10 worked example | bounded modification |
| Replayable trace | DiagnosticReport 五元組 + deterministic 執行 | variance study：accuracy/SWR 零浮動 | EU AI Act §12–13 logging |
| Abstention as a control | selective prediction 框架化拒答 | CoT 98% 仍 SWR≥2% vs VQ SWR=0%（abstain 10%） | accountability asymmetry |

> 寫法：§2 定義左邊兩欄（控制項 + 監管出處，一張表收掉 Basel/EU/FINRA，不寫成法規論文）；
> §3 逐機制展開；§5/§6 每個實驗開頭一句話點名它驗證哪一列。Reviewer 可以沿這張表垂直讀完全文。

---

## 3. Title Options

1. **VerifiQuant: Governance-by-Execution for Financial LLM Workflows** ← 首選，最貼 ICAIF
2. Executable Governance Cards for Reliable Financial LLM Reasoning
3. From Prompted Reasoning to Governed Execution in Financial AI

---

## 4. Abstract Draft（英文，數字帶 placeholder）

> Large language models are increasingly embedded in financial workflows, yet they are governed —
> if at all — by prompts and post-hoc review. We argue the central deployment risk is not accuracy
> but *governability*: a production financial AI system must declare its scope boundaries, refuse
> structurally when it cannot verify its own reasoning, expose replayable decision traces, and
> mathematically verify any repair it applies. We present **VerifiQuant**, a framework that compiles
> these governance controls into the reasoning pipeline itself. Each financial formula family is
> represented as a **Financial Inference Contract (FIC)** — retrieval metadata, deterministic
> executable code, invariants, and declared semantic ambiguities — and every query passes through a
> six-layer diagnostic funnel (M/N/F/E/I/C) in which each layer is a legitimate, auditable exit.
> Semantic repairs are restricted to **verifiable atomic transforms** whose numerical effect is
> cross-checked against a declared algebraic identity before acceptance. On a [50-question →
> **250-question (180 medium + 70 hard)**] FinanceReasoning subset, VerifiQuant achieves [90%]
> accuracy with **zero silent-wrong answers** (selective accuracy [100%], abstention [10%]), while
> the strongest CoT+oracle baseline reaches [98%] accuracy yet cannot eliminate silent wrongs
> (SWR ≥ [2%]) and never abstains. On contract-grounded adversarial traps, VerifiQuant reduces
> silent-wrong rates from [90%→20%] (boundary violations) and [100%→10%] (semantic ambiguity).
> Critically, injecting the same diagnostic taxonomy into oracle prompts fails to reproduce these
> guarantees — under blind review it *degrades* accuracy — indicating that governance must be
> executed by architecture, not described in prompts.

（若 250Q hard-tier 跑出 SWR>0，"zero silent-wrong" 改為 §7.2 的 fallback 措辭。）

---

## 5. Section-by-Section Outline（8 頁預算 + 素材來源）

| § | 頁數 | 內容 | 素材來源與動作 |
|---|---:|---|---|
| Abstract | 0.15 | 上方 draft | 新寫 |
| **1. Introduction** | 0.8 | 金融 LLM 的三類部署失敗（silent wrong / audit gap / boundary failure）→ governance gap → VQ 三設計 → 貢獻 4 點 | FinNLP §1.1/§1.4/§1.6 **搬+縮**；§1.2 faithfulness **砍到 1 段**（只留 Turpin+Lanham 一句、「越強越不忠實」一句）；§1.5 併入 §6 |
| **2. The Governance Problem** | 0.8 | 定義 6 個 governance controls（Table 1 脊椎表）+ 監管出處一欄收掉 Basel/EU/FINRA；related work 壓成半頁（execution-based 無 semantic 防禦 / neuro-symbolic 翻譯瓶頸 / MAS theory-of-mind failure，各 2–3 句） | **新寫**（骨架來自 FinNLP §7.1 表 + §2 壓縮）；FinNLP §2.4 三子點結構砍成一段 |
| **3. Method: VerifiQuant** | 1.7 | FIC 三卡結構；M/N/F/E/I/C funnel（I_HARD/I_SOFT）；O-ITL 協議（壓縮）；Verifiable Atomic Transforms + annuity worked example（唯一展開的例子）；integration 流程圖 | FinNLP §3.2–§3.6 **直接搬**，O-ITL（§3.4）壓到 0.3 頁；每小節開頭加一句「本機制承載 Table 1 第 X 列」 |
| **4. Experimental Protocol** | 0.9 | FinanceReasoning 250Q（180M+70H，分層抽樣 seed 42，canonical 50 子集）；trap set 生成（contract-grounded，算子表**保留**、細節砍）；baselines（single-shot / blind oracle / VQ；50Q 上另有完整 oracle grid + JPM，引用之）；metrics（三元分類 + SWR/SFWR + coverage/selective accuracy） | FinNLP §5.1–§5.4 **搬+縮**；§5.2 trap 生成細節砍 60%（算子表留，N 算子誠實標註留一句）；oracle 盲審協議細節壓成一個 protocol box |
| **5. Results** | 1.2 | (i) 主表（Table 2，250Q 或 50Q prelim）；(ii) **medium/hard 難度分層**（Fig 2：難度上移時 abstention↑、SWR 平坦——ICAIF 版獨有貢獻）；(iii) trap resistance（Table 3，分算子）；(iv) K-ablation 一小段（oracle 提升 accuracy、funnel 保證 SWR，正交） | FinNLP §5.5/§5.7–§5.9 **搬+大縮**；診斷分布收進主表附註；variance 一句話帶過 |
| **6. Governance Analysis** | 1.0 | (i) **Prompt ≠ governance**：盲審下 funnel-guided ≤ plain（Flash 92<94、Pro 96<98）——把 funnel 寫進 prompt 反而注入噪音；(ii) 三個 case studies：E（負費用 88.46%）、N（Heston→答成 21.5）、I_HARD（HERO verified transform）；(iii) 兩個微觀案例：test-1443（CAGR 算術非決定性——prompt 再好也救不了）、test-1593（input binding 邊界 + I_SOFT 解除盲點——誠實界定保證範圍） | `prompt_comparison_all_systems.md` §四/§五 **重寫成正文**；`2026-06-15_demo_script.md` 軸 3/4/5 **轉寫**；FinNLP §5.5 觀察 1 + §5.6 **壓縮搬入**（表格砍、留敘事） |
| **7. Limitations** | 0.5 | input binding 上游無防禦（→ input-provenance check）；FIC build-time 依賴 LLM；M-operator 為 trap 設計限制；coverage/abstention 代價；50Q→250Q 樣本規模說明 | FinNLP §6.3 **挑 5 條搬**，每條 2 句 |
| **8. Conclusion** | 0.25 | safe failure > headline accuracy；governance-by-execution 一句收束 | FinNLP §8 **縮** |
| References | 0.8–1.0 | 壓核心 ~25 篇：faithfulness 3、fin-bench 2、MAS 2、NeSy 4、tool-aug 3、selective prediction 2、監管 3–4 | citation pool 篩選；⚠️ **FinanceReasoning 作者不一致待修**（draft §2.1 寫 Liu et al.、§2.3 寫 Tang et al.——查證後統一） |

### 明確砍掉（FinNLP 有、ICAIF 沒有）

- §1.2/§2.2 CoT faithfulness 文獻戰線（Jacovi/Pfau/Lie to Me/Ariadne 全砍，只留 1 段 2 cites）
- §2.4 神經符號三子點長篇（SI/Faithful Reasoning/LINC/SATLM/FormalJudge → 一段）
- §5.2 trap 標註可信度論證長篇（留「contract-grounded、label 即算子」兩句）
- §5.6 over-normalization 完整逐輪表（留 test-1593 敘事 3–4 句 + 指向 limitation）
- §5.10 variance 表（留一句「4 runs accuracy/SWR 零浮動」）
- §6.1 verifiability vs auditability 概念節（融進 §2 的兩句話）
- §6.4 broader impact / §7 監管全節（→ Table 1 的一欄 + §2 各一句）
- 兩階段 venue 策略、TODO checklist（planning 內容，不進論文）

---

## 6. Figures & Tables 清單

| # | 內容 | 來源/狀態 |
|---|---|---|
| **Fig 1** | Pipeline 圖：q → retrieval → M/N → F → E → I_HARD/I_SOFT → transform → C，每個閘門標註對應 governance control（Table 1 列號） | 需新製（可從 demo UI 流程改繪） |
| **Table 1** | Governance 脊椎表（§2.2，control × mechanism × evidence × regulation） | 本文件已有，直接排版 |
| **Table 2** | 主結果表（見 §7.1 設計） | 50Q 數字現成；250Q 待 A4 |
| **Fig 2** | 難度分層行為圖：x=難度分位（medium Q1–Q4 + hard），y=stacked correct/abstain/silent-wrong 佔比 | **250Q 獨有**，待 A4；若 250Q 未及，退為 medium-only 就沒有此圖 → 全力保 A2 |
| **Table 3** | Trap resistance 分算子（CoT vs VQ funnel-only，E/I 為 SWR 差異化、F/N 為 auditability、M 標 trap-limitation） | 50Q 現成；若 A5-③ 完成換 250 版 |
| **Table 4** | Prompt vs architecture ablation（plain/funnel × Flash/Pro 盲審 + VQ，欄：Acc/SWR/Abstain） | 現成（`prompt_comparison_all_systems.md` §一） |
| **Box 1**（可選） | I_HARD verified transform worked example（annuity 579.98→577.10，invariant + code_patch cross-verify） | demo script 軸 5 轉寫 |

---

## 7. 主表設計與 50Q→250Q 替換計畫

### 7.1 Table 2 設計（欄位對兩種情況通用）

| System | N | Correct | Silent Wrong | Safe Refusal | Coverage | Sel. Acc | SWR |
|---|---:|---:|---:|---:|---:|---:|---:|
| CoT single-shot (Flash) | 50→250 | 41 → [___] | 9 → [___] | 0 → [___] | 100% → [___] | 82% → [___] | 18% → [___] |
| CoT + blind oracle (Flash, K=3) | 50→250 | 47 → [___] | 3 → [___] | 0 → [___] | 100% → [___] | 94% → [___] | 6% → [___] |
| CoT + blind oracle (Pro)† | 50 | 49 | 1 | 0 | 100% | 98% | 2% |
| JP Morgan MAS† | 50 | 43 | 7 | 0 | 100% | 86% | 14% |
| **VerifiQuant (Flash, K=3)** | 50→250 | **45 → [___]** | **0 → [___]** | **5 → [___]** | **90% → [___]** | **100% → [___]** | **0% → [___]** |
| VerifiQuant (Pro, K=3)‡ | 50→(250) | 43 → [___] | 2 → [___] | 5 → [___] | 90% → [___] | 95.6% → [___] | 4% → [___] |

† 50Q-only 配置（完整 oracle grid 與 JPM 成本高，保留在 50Q 作 protocol validation，表註說明）。
‡ VQ Pro 250 = A5-①，若完成則 over-normalization 從質性發現升級為統計驗證；未完成則保留 50Q 列。

**加一張分層子表（250Q 版才有）**：同欄位 × {medium 180 / hard 70}，
punchline = *abstention 隨難度上升、SWR 維持平坦* → governance 行為在 distribution shift 下穩定。

### 7.2 替換來源與 claim fallback 規則

- 數字來源：`verifiquant/data/runs/paper_v2_250/results/<baseline>/summary.json`（A4 產出彙整）。
- 替換順序：Abstract → §5 主表 → §5 分層 → §6 引用數字 → Conclusion。全文用 `[V2:xxx]`
  標記待換數字，submit 前 grep `[V2:` 確認歸零。
- **Fallback（若 250Q VQ 出現 silent wrong）**：
  - "zero silent-wrong" → "reduces SWR to [x]% (vs [y]% for the strongest baseline), with every
    residual failure attributed to a named funnel layer"。
  - SWR>0 的每一題必須做逐題歸因（哪一層漏的、是否 input-binding 上游問題），寫進 §7——
    governance 敘事下「可歸因的失敗」仍是賣點，不可歸因才是敗筆。
  - 若 hard-tier 卡片良率明顯低（修卡 >20 張）：§4 誠實報告良率 + 修卡紀律（僅修
    execution-smoke 可偵測項，無 gold 介入），沿用 paper_v1 的防禦寫法。

### 7.3 Track A gate 摘要（詳見 runbook）

| Gate | 條件 | 未過的處置 |
|---|---|---|
| A0 | `lint_echecks` + `validate_relations` 零錯誤 | 修卡後重驗，卡版本號 +1 |
| A1 | cot single-shot 250 跑通，hard-tier answer-matching 正常 | 修 scoring，不動卡片 |
| A2→A3 | VQ 250 完跑；C-error 卡片數 ≤ ~25（10%） | 超過 → 檢討 hard-tier 卡片生成 prompt，報告良率 |
| A4 | 07/26 前主表 + 分層完成 | 未達 → 主表退 50Q prelim + 250Q partial，Fig 2 砍 |

---

## 8. 與 FinNLP 版的差異化（兩投防線）

| 維度 | ICAIF 版（8/2） | FinNLP 版（8/11） |
|---|---|---|
| 敘事 | governance-by-execution | contract-validated reasoning + CoT faithfulness 批判 |
| 主數據 | 250Q（180M+70H）+ 難度分層 | 50Q 深度分析（逐輪軌跡、O-ITL 協議、trap 方法學） |
| 獨有內容 | Table 1 governance 脊椎、Fig 2 難度分層、監管對應 | §1.2/§2.2 faithfulness 文獻、§5.6 over-normalization 完整案例、trap 生成可信度論證 |
| 貢獻宣稱 | executable governance controls | 錯誤分類學 + O-ITL 評估協議 |

⚠️ 投稿前逐字核對兩邊 CFP 的 concurrent/dual-submission 條款；若任一邊禁止，改序列投。

---

## 9. 投稿前檢查清單

- [ ] ICAIF CFP：頁數（含/不含 ref）、double-blind、appendix 政策、dual-submission 條款
- [ ] FinanceReasoning 引用作者統一（Liu vs Tang，查 ACL Anthology）
- [ ] 全文 `[V2:` placeholder 歸零
- [ ] 主表所有數字與 `summary.json` cross-check（寫個 assert script 更穩）
- [ ] anonymization：repo 連結、demo URL、姓名（若 double-blind）
- [ ] Figure 字級 ≥ 8pt、色盲友善
- [ ] deadline 前 24h 完成上傳測試
