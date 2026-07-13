# ICAIF Reviewer 八大痛點 × 我們的回應(Points 1–8)

模擬嚴格 reviewer 的第一輪質疑,逐點判定 + 回應方式 + 落點。
**Points 1–2 以新實驗回應;Points 3–8 以論文修改回應(⚠️ AI 代擬、作者尚未逐句審閱,
逐處清單見文末附錄)。** `paper/main.tex` 已套用全部修改,tectonic 編譯通過(10 頁)。

## 總覽表

| # | 痛點 | 判定 | 回應 | 落點 |
|---|------|------|------|------|
| 1 | 把 benchmark 解題空間編譯進系統? | 部分成立 | OOD family 實驗 + coverage 揭露 + claim 收窄 | 新 §oodfam、§protocol |
| 2 | Oracle 看 GT computation code = 洩漏? | 成立 | 250Q K-ablation(K=1 vs K=3 分離)+ oracle 強度揭露 | 新 §kablation、Limitations |
| 3 | 「mathematically verify」講過頭 | 成立 | 降為 numerically cross-verify + *Scope of the guarantee* 段 | abstract、§transforms 等 6 處 |
| 4 | Risk–coverage 只有單一 operating point | 部分成立 | K=1/2/3 三點 + perfect-rejector 上界圖 + matched coverage | 新 §kablation 圖表、4 處限定 |
| 5 | 「Replayable」與 LLM 隨機性矛盾 | 成立(且論文自打) | 三層 replay 定義,消掉 bit-identical 矛盾句 | §method、§fiftyq、§deployment 等 5 處 |
| 6 | attribution 只是事後分類 | 成立 | runtime vs post-hoc 二分 + 架構先定 taxonomy 辯護 | §attribution 等 8 處 |
| 7 | Contract 成本被低估 | 成立 | *Contract engineering cost, quantified* 段 | §protocol |
| 8 | 250 題 ≠ workflow governance | 成立 | scope 收斂到 formula-grounded numerical core | abstract、intro *Scope of claims*、Limitations |

**貫穿全部八點的統一 reframe:不賣 accuracy 霸權,賣 silent-wrong rate(6–8 倍低)+
校準的 abstention。** 兩大攻擊(1、2)只有在宣稱 accuracy 優勢時才咬得住;
換回 thesis(predictable, verifiable failure behavior)後同時失去牙齒。

---

## Point 1 — 你是不是把 benchmark 的解題空間編譯進系統了?

**判定:部分成立。** 250Q 實驗用的是 250 卡庫(每 family 一張、覆蓋率 ~100%)——這本身
就是 reviewer 說的 task-aware library construction 的證據。AST dedup 只證明 instance 層
獨立,不證明 family-selection 層獨立。

**回應(實驗):OOD formula-family eval** — reviewer 三分法的 setting (c)。
以 53 卡庫對 250Q 自然切分:199 個 family 不在庫(OOD)、51 個在庫(ID),同題同餵
CoT baseline(明確允許拒答)。K=1 單發。
- **VQ abstention selectivity:OOD 拒 88.9% vs ID 只拒 19.6% → 69.3 點分離**
  (拒答追蹤的是庫覆蓋,不是題目難度)
- CoT 允許拒答:60.3% vs 49.0% → 只有 11.3 點分離(無差別避險;ID 準確率崩到 45.1%)
- VQ 真 silent-wrong(單位校正後):OOD 2.5% / ID 2.0%
- 三分法對應:(a) 已知 family 未見 instance = 主結果;(b) 未見表述 = trap set;
  (c) 未見 family = 本實驗

**回應(論文):**
- §protocol 警鈴句 `auto-generated per question family` → `per formula family present in
  the reference corpus`,並新增 *Coverage disclosure*:主結果測的是 **in-library**
  routing/binding/gating,非 open-world reasoning
- Abstract 加 selectivity 句;新 §oodfam(`tab:oodfam` + 三個誠實 caveats:
  out-of-library ≠ out-of-knowledge、demo 庫、pristine 重跑計畫)

**Artifacts:** `scripts/run_ood_family_eval.py`(config 全記錄,K 變化一鍵重跑)、
`docs/2026-06-15_ood_family_eval.md`、`verifiquant/data/runs/ood_eval/`。

---

## Point 2 — Oracle 看到 ground-truth computation code,真的沒洩漏嗎?

**判定:成立。** computation code = 公式選擇 + 變數角色 + 語義解釋 = 答案最難的那一半。
K=3 的 93.6% 是「系統 + 持有 intent spec 的澄清代理」的 recoverability,不是
autonomous accuracy。

**回應(實驗):250Q K-ablation** — 從既有 run 的 `framework_guided.history` 精算重建,
零重跑:

| Budget | Correct | SW | Abstain | Coverage | Sel. Acc |
|---|---|---|---|---|---|
| K=1(autonomous) | 203 (81.2%) | 7 (2.8%) | 40 | 84.0% | 96.7% |
| K=2 | 231 (92.4%) | 9 (3.6%) | 10 | 96.0% | 96.2% |
| K=3(canonical) | 234 (93.6%) | 8 (3.2%) | 8 | 96.8% | 96.7% |

Recovery:31/250(12.4%),獲救案例平均 2.1 輪;全體平均 1.20 輪。
關鍵解讀:**K=1 autonomous(81.2%)≈ CoT 帶 oracle(82.0%)——accuracy 打平,
但 SWR 是它的 1/6**。Recoverability 是獨立、有標價的能力,不是 autonomous 數字的隱藏成分。

**回應(論文):**
- 新 §kablation(`tab:kablation250`);abstract 揭露 K=1 autonomous 數字;
  headline 定位改為「93.6% = clarification-assisted recoverability」
- Limitations 明寫:oracle 讀 computation code、**比任何真實使用者都強**
  (真實使用者通常不知道自己需要哪個計算)——這正是分開報 K=1 的原因;
  realistic-user oracle(只看題面)列為 future work

**Artifacts:** `docs/2026-06-15_250Q_k_ablation.md`。

---

## Point 3 — 「mathematically verify repair」講得比實際機制強

**判定:成立,必須改。**

**機制事實(程式碼):**
- `verify_transform` = AST node-count 上限 + safe-name whitelist + 數值決定性檢查 +
  **在 sampled inputs 上檢查數值不變量**(stage_transform.py:417)。
- Runtime(demo endpoint, app.py)只用 **1 個樣本點 = 使用者當下的 binding**。
- **不變量本身是 build-time 由 Gemini 生成**(normalization filter 是 hard gate)。
  Reviewer 的第二刀「identity 錯了,兩個一致的錯誤實作會被接受」**屬實**。

**但有一個比 reviewer 預期更強的誠實辯護(已寫進論文):**
repair 只在**單一 binding** 上被套用。在該 binding 上檢查不變量,對「這一次套用」是
**充分的**——不需要 all-input equivalence;需要 all-input 等價的只有 code_patch 進 library
的 acceptance(那裡用 multiple samples)。精確 claim = **point-of-use validation,
conditional on the declared invariant**:信任邊界從「runtime LLM 自由改碼」移到
「build-time 可審查、版本化的靜態 artifact」。

**已套用 patches:**
- Abstract:`mathematically verify any repair` → `numerically cross-verify … against a
  contract-declared algebraic relation at the exact binding being repaired`
- §transforms 新增 *Scope of the guarantee* 段(bounded validation, not formal
  equivalence;雙側同錯會通過、單側錯會被抓;連回 §rationale 的「為何放棄 CAS proof」)
- worked example `provable`→`cross-checked`;case study `proof`→`invariant check`;
  C4 表格列 → `cross-verified repair (AST bounds + point-of-use numeric check)`;conclusion 同步

---

## Point 4 — Risk–coverage tradeoff 只展示一個 operating point

**判定:部分成立;答案已建好。** "dominates on both axes" 在被比較的兩點上字面成立
(點對點 Pareto),但 reviewer 要曲線脈絡。

**回應(實驗 + 圖):**
- **多 operating points**:VQ K=1/2/3 三個真實點(`fig:riskcov`,
  `paper/figures/risk_coverage_250q.png`)
- **Matched coverage 且比 threshold sweep 更強**:CoT 沒有 continuous confidence 可掃,
  改畫它的 **perfect-abstention 上界**(神級 rejector)。VQ **真實** K=1 點
  (96.7% @ 84%)高於 CoT single-shot 的**理論上界**(92.4% @ 84%)——沒有任何實際
  threshold 能贏過 perfect rejector。
- **誠實 caveat(已寫)**:CoT+oracle 上界(97.6% @ 84%)高 VQ-K=1 約 1 pt,但那是
  「無 oracle 的 VQ」對「有 oracle 的 CoT 理論最佳」;在 VQ 自己的 K=3 設定下,
  同 coverage 再贏 ~12 pts。
- **「讓 baseline 也拒答」**:已有的 prompted-refusal ablation + 新 OOD 實驗的
  refusal-permitted CoT(非選擇性、ID 崩盤)= 實證關門。

**已套用 patches:** 新 §kablation(表 + 圖 + matched-coverage 論證);
abstract/contributions/§results/conclusion 四處 `dominates…simultaneously` →
`at their respective operating points`(+ matched-coverage 指標)。

---

## Point 5 — 「Replayable」與 LLM gates 隨機性之間有矛盾

**判定:成立,且論文自打。** §attribution 自己寫了 resampling 下 class-D 翻正、
「unflagged coin flip」——同稿一邊 claim replayable 一邊記錄 run-to-run 變異。
C5 的「4 reruns variance = 0」是 aggregate 穩定,不是 trace 相同。

**程式碼支持的真實保證(reviewer 三分法直接採用):**
1. Deterministic compute + invariant evaluation:給定 stored binding **位元級重播**
   (DiagnosticReport 存 code + inputs + raw_output)
2. LLM 決策(selector/extractor/critic):**logged**,歷史決策可照記錄重執行
3. 從原始問題 fresh rerun:**不保證**同路徑

**已套用 patches:**
- §method(DiagnosticReport)寫入三分法;統一改用 **audit-replayable**
  (abstract、C5 表格列、conclusion)
- §fiftyq:`bit-identical…only the recovery path varies` 矛盾句 → 「outcome 相同、
  路徑變異,正是 C5 保證的形狀:funnel 而非 path 擁有安全性」(把 coin-flip 觀察
  反轉成支持證據)
- §deployment:`four bit-identical reruns` → outcome-identical + 「EU AI Act 的
  traceability 要求重建 how an output was produced,不是 sampling determinism——
  後者任何含 LLM 的系統都不提供」

---

## Point 6 — 「Every residual failure is attributed」只是事後分類

**判定:成立。** B/C/D 的歸類確實是看過 gold 之後做的 forensic analysis,
不能包裝成 deployment-time attribution。

**辯護核心(已寫進論文):taxonomy 是架構先定的,不是看到錯才發明的。**
funnel 裡 LLM 只剩三個動作——selection / binding / hint-recall——先於任何實驗固定;
分派由 stored trace 機械讀出(錯 `fic_id`→selection、錯 binding 值→binding、
宣告了沒觸發→recall)。

**已套用 patches:**
- §attribution 開頭新增二分:*runtime diagnostic attribution*(gate 出口即產出、
  無需 gold)vs *post-hoc research attribution*(用 gold + trace 歸類);
  誠實註明單一標註者、但大多由 trace 強制決定
- `zero unattributed` → `none falling outside these classes`
- 收尾句(採用作者建議):**~300 題內每個 observed silent wrong 都落在三類內;
  不宣稱開放世界窮盡,只說 300 題沒出現第四類**——且三類一一對應 LLM 仍被委任的位置
- abstract/contributions/limitations/conclusion 的 `attributed` 統一改
  **`post-hoc localized`**;Limitations 明寫「attribution 用了 gold,部署時沒有」

---

## Point 7 — Contract 建構與維護成本被低估了

**判定:成立。** 論文把 offline formalization 寫成優勢,卻沒量化 offline 負擔;
而自己披露的 5/50 手修、type defect、98/250 hint-trigger 正說明 contract quality
是 first-order 瓶頸。

**回應(把「250Q 沒逐張檢查」變成量化賣點):** §protocol 新增
*Contract engineering cost, quantified* 段——
- 250 卡**全自動生成**(每 family 3 個 staged LLM call + deterministic normalization),
  此規模下**零逐卡人工審查**
- 品質控制 = gate battery,實測記錄:smoke 0/250、E-check lint 修 4 expr/2 卡、
  type audit 依 pre-declared policy 自動修 4 卡(2 flagged、3 false positives cleared)
- 手修率隨 gate 遞減:50Q 時代(battery 前)5/50,250Q 由自動 audit 接手
- 殘存 = **semantic spec error = B/C/D frontier**;hint-trigger pre-pass 是其第一個機械化削減
- 直面交換:「runtime hallucination 換 build-time spec error——後者 static、
  per-family localized、部分可無 gold 稽核;上述記錄量化『部分』」
- 誠實列未量測:inter-author consistency(單一 LLM author)、跨域遷移成本

---

## Point 8 — FinanceReasoning 250 題離「financial workflow governance」還很遠

**判定:成立。** 實驗是 benchmark numerical QA,deployment 語言遠大於實驗場景。

**已套用 patches(scope 收斂到 formula-grounded):**
- Abstract:`financial AI system` → `financial reasoning system`;VerifiQuant 定界為
  **formula-grounded numerical core**(答案 = 對可抽取輸入的確定性計算)
- Introduction 結尾新增 *Scope of claims* 段:列出真實 workflow 還缺什麼
  (多文件、表格/PDF、conflicting sources、data freshness、jurisdiction、非數值判斷),
  **註明 deployed demo(verifiquant.com)已有 PDF/表格擷取,但本文 claims 不依賴它**
- Limitations 加 scope 條目:「evaluation 是單題 benchmark QA;workflow surface 是
  demo 部分實作的部署脈絡,本文不評測」
- Conclusion 對齊
- **標題未動**(作者決定):若要更窄,備選
  *Governance-by-Execution for Formula-Grounded Financial Reasoning*

---
---

# 附錄:已套用 patch 逐處清單(審閱用 checklist)

> **狀態:AI 代擬並套用,作者尚未逐句審閱。** tectonic 編譯通過,10 頁
> (ICAIF 上限 8 頁全含,策略 = 先有再刪,[ICAIF-CUT] 標記可用)。

## 第一批(Points 3–5 + 1–2 的論文落點,2026-07-13)
- **Point 3(6 處)**:abstract 降語氣;§transforms 新增 *Scope of the guarantee* 段;
  worked example `provable`→`cross-checked`;case study `proof`→`invariant check`;
  C4 表格列;conclusion。
- **Point 4(7 處)**:新 §sec:kablation(`tab:kablation250` K=1/2/3 + `fig:riskcov`)
  + matched-coverage 論證;新 §sec:oodfam(`tab:oodfam` + 3 caveats);abstract 加 K=1
  與 selectivity 兩句;4 處 `dominates` 限定;§protocol 警鈴句改寫 + *Coverage
  disclosure*;Limitations 補 oracle 強度揭露。
- **Point 5(5 處)**:DiagnosticReport 三分法;§fiftyq 消 `bit-identical` 矛盾;
  §deployment 補 traceability ≠ sampling determinism;C5 列與 abstract/conclusion
  統一 `audit-replayable`。
- **審閱重點**:新數字直接寫死正文(81.2/2.8/84.0、96.7/92.4/97.6、
  88.9/19.6/60.3/49.0、31 recovered/2.1 rounds),未做進 numbers.tex macro;
  數據來源見 `2026-06-15_250Q_k_ablation.md` 與 `2026-06-15_ood_family_eval.md`。

## 第二批(Points 6–8,同日套用)
- **Point 6(8 處措辭 + 2 段新內容)**:§attribution 開頭 runtime vs post-hoc 二分;
  taxonomy = 架構先定的三個 LLM 步驟;分派由 trace 機械判定、單一標註者;
  `zero unattributed` → `none falling outside these classes`;300 題收尾句;
  abstract/contributions/limitations/conclusion 的 `attributed`→`post-hoc localized`。
- **Point 7**:§protocol 新增 *Contract engineering cost, quantified* 段。
- **Point 8(5 處)**:abstract 2 處定界;intro *Scope of claims* 段;Limitations
  scope 條目;conclusion 對齊;標題留作者決定。

## 第三批(作者初審回饋後的精修,2026-07-13)
- **Perfect-rejector 論證正面化(§kablation)**:先承認真實 proxy 存在(answer entropy、
  self-consistency vote margins、verifier/external-calibrator scores),再指出任何 proxy
  誘導的 rejector 都被 perfect rejector 支配 → 「贏過上界 = 贏過該 coverage 下所有可實現的
  confidence heuristic」;註明具體 proxy benchmark 為 future work,並引 §promptsnotcontrols
  與 §oodfam 的實測 refusal frontier(遠低於上界)作實證關門。
- **Contract cost 段語義修正**:`no per-card human review` 與 `2 flagged for review` 矛盾
  → 改為「manual inspection was not performed exhaustively; automated audits surface a
  small candidate set for human review」+ 明寫 5 個候選交人工裁定(2 張 deliberately-array
  確認為預期、3 個 false positives 清除);`manual-repair rate has fallen`(pre-battery vs
  post-battery 不等價比較)→ 改為「division of labor shifted rather than the cost
  vanishing」。
- **兩個過強句軟化**:abstract 尾句 `Governance must be executed by architecture, not
  described in context` → `In our setting, describing governance in prompts did not
  reproduce the selectivity or attribution of executed controls`(+ 前句加 in our
  experiments);contributions `derive six executable controls from regulatory requirements`
  → `operationalize recurring regulatory governance concerns into six executable controls`。
- **保留未動**:abstract 長度(作者判斷先留);§intro `Our position is…`(明示為立場、
  非結論,可守)。
