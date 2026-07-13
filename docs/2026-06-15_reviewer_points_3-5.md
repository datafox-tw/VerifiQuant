# Reviewer points 3–5 — verified against code & paper, with exact patches

Checked against `paper/main.tex` and the actual mechanisms in
`verifiquant/preprocessing/stage_transform.py` / `app.py`. Verdict per point, then
line-referenced text patches.

---

## Point 3 — "mathematically verify repair" 講得比機制強 → **成立,必須改**

**機制事實(程式碼):**
- `verify_transform` = AST node-count 上限 + safe-name whitelist + 數值決定性檢查 +
  **在 sampled inputs 上檢查數值不變量**(stage_transform.py:417)。
- Runtime(demo endpoint, app.py:803)只用 **1 個樣本點 = 使用者當下的 binding**:
  `verify_transform(spec, core, [inputs])`。
- **不變量本身是 build-time 由 Gemini 生成**(normalization filter 是 hard gate,見
  CLAUDE.md)。Reviewer 的第二刀「identity 錯了,兩個一致的錯誤實作會被接受」**屬實**。

**但有一個比 reviewer 預期更強的誠實辯護(建議寫進論文):**
repair 只在**單一 binding** 上被套用。在該 binding 上檢查不變量,對「這一次套用」是
**充分的**——不需要 all-input equivalence;需要 all-input 等價的只有 code_patch 進 library
的 acceptance(那裡用 multiple samples)。所以精確的 claim 是
**point-of-use validation, conditional on the declared invariant**:
信任邊界從「runtime LLM 自由改碼」移到「build-time 可審查、版本化的靜態 artifact」。
這是真實且可守的貢獻,不是 formal proof。

**Patches(main.tex):**
- **L29(abstract)** `mathematically verify any repair it applies`
  → `numerically cross-verify every repair it applies against a contract-declared algebraic
  relation at the exact binding being repaired`
- **L163(worked example)** `its numerical effect is provable without trusting the LLM`
  → `its numerical effect is cross-checked against the declared relation without trusting the
  LLM that proposed it`
- **§transforms 加一句(限定範圍,先發制人):**
  `This is bounded repair validation, not formal equivalence: the check is numerical, at the
  point of use, and conditional on the contract-declared invariant (a static, reviewable,
  versioned artifact authored at build time). Two coincident errors—wrong invariant and a
  patch wrong in exactly the compensating way—would pass; single-sided errors are caught.`
- **C4 表格列(L107)** `verified repair` → `cross-verified repair (bounded, point-of-use)`
- 全文避免 `proof / provable / mathematically`,改 `numerically cross-verified /
  invariant-checked / bounded validation`。(§171 的 symbolic-proof 段是講被放棄的設計,
  可保留,反而支持誠實敘事。)

---

## Point 4 — risk–coverage 只有單一 operating point → **部分成立;我們已把答案做出來了**

**事實:** L33/84/210/471 的 "dominating on both axes simultaneously" 在**被比較的兩個
operating points 上是字面成立的**(93.6/3.2 vs 82.0/14.8,同時較高 acc 且較低 SWR =
點對點 Pareto dominance)。Reviewer 要的是曲線脈絡,而這已在
`docs/figures/risk_coverage_250q.png` + `docs/2026-06-15_250Q_k_ablation.md` 建好:
- **多 operating points**:VQ K=1/2/3 三個真實點(cov 84.0→96.8%,sel-acc ≈96.7%)。
- **Matched coverage 且比任何 threshold sweep 更強**:CoT 沒有 continuous confidence 可掃,
  我們畫它的 **perfect-abstention 上界**(假設有神級 rejector)。VQ 的**真實** K=1 點
  (96.7% @ 84%)仍高於 CoT single-shot 的**理論上界**(92.4% @ 84%)——沒有任何實際
  threshold 能贏過 perfect rejector,所以這比 matched-coverage sweep 更難反駁。
- **誠實 caveat(要寫)**:同 coverage 下 CoT+oracle 的理論上界(97.6%)高於
  VQ-autonomous(96.7%)約 1pt——但那是「無 oracle 的 VQ」對「有 oracle 的 CoT 理論最佳」。
- 論文已有的 prompted-refusal ablation(§promptsnotcontrols)= 真實給 baseline 拒答旋鈕的
  結果(non-selective),正好補「讓 baseline 也調閾值」那問。

**Patches:**
- L33/84/210 在 `dominating ... simultaneously` 後加限定:
  `at their respective operating points; under matched coverage, VerifiQuant's deployed
  abstention policy exceeds even a perfect-rejector upper bound for the single-shot baseline
  (\S risk–coverage)`
- 收一張 risk–coverage 圖(figure 已產)+ K=1/K=3 兩列進主表(K-ablation 文件裡有現成表)。
- 若要更多 VQ points:`m_min_top_score`(retrieval 信心閾值)是現成**連續**旋鈕,
  加上 I_HARD/I_SOFT 政策開關可再生 3–4 個 operating points——每點需重跑(~250 LLM 呼叫),
  overnight 等級,非必要但可加。

---

## Point 5 — "replayable" 與 LLM 隨機性矛盾 → **成立,而且論文自己有內部矛盾**

**事實:**
- C5(L108)`4 reruns: accuracy/SWR variance = 0` 是 **aggregate 穩定性**,不是 trace 相同。
- **論文自打**:§attribution(L284)明寫 resampling 下 class-D 兩例翻正、
  「behaves as an unflagged coin flip」——同一份稿一邊說 replayable、一邊記錄 run-to-run
  binding 變異。Reviewer 一定圈這裡。
- 程式碼支持的真實保證:DiagnosticReport 存 `fic_id + binding(inputs)+ code +
  raw_output + invariant trace` → **給定 stored binding,deterministic compute 可
  bit-level 重播**(pure Python)。LLM 層(selector/extractor/critic)是被 **logged**,
  不是 deterministic。

**Patches(reviewer 給的三分法直接採用):**
- **C5 表格列(L108)** `replayable trace ... 4 reruns: accuracy/SWR variance = 0`
  → `audit-replayable trace & deterministic layers bit-replayable from stored bindings;
  outcome-stable across 4 reruns (accuracy/SWR variance = 0)`
- **L140** `replayable, layer-attributable, machine-readable`
  → `layer-attributable, machine-readable, and audit-replayable: deterministic execution and
  invariant evaluation replay exactly from the stored binding; LLM-mediated selection and
  binding decisions are logged so any historical decision can be re-executed as recorded;
  fresh reruns from raw text are not guaranteed to take identical paths`
- 在 §governance 或 C5 附近補一句對齊法規:
  `EU AI Act traceability requires reconstructing how an output was produced, not sampling
  determinism; our guarantee is exactly the former.`
- 把 §attribution 的 coin-flip 觀察**引回來當支持**:aggregate 不變 + path 變異正是
  「funnel 擁有安全性、LLM 只擁有路徑」的證據(L295 已有此語,連起來即可)。

---

## 優先序
1. **Point 3 文字修改** — 純 wording,10 分鐘,不修一定被圈。
2. **Point 5 三分法** — 純 wording + 消內部矛盾,15 分鐘。
3. **Point 4** — 圖與表已做好,收進論文 + 加限定句;可選加 m_min_top_score sweep(需重跑)。

---

## ✅ 已套用至 paper/main.tex(2026-07-13)— ⚠️ 尚未經作者逐句審閱

> **狀態:AI 代擬並套用,作者尚未完整看過。審閱時逐處檢查下列 15 處。**
> tectonic 編譯通過,目前 10 頁(ICAIF 上限 8 頁全含,策略=先有再刪)。

- **Point 3(6 處)**:abstract 降語氣;§transforms 新增 *Scope of the guarantee* 段
  (bounded validation / point-of-use / conditional on build-time invariant / 雙側同錯會通過);
  worked example `provable`→`cross-checked`;case study `proof`→`invariant check`;
  C4 表格列;conclusion。
- **Point 4(7 處)**:新 §sec:kablation(`tab:kablation250` K=1/2/3 + `fig:riskcov`
  risk–coverage 圖,圖檔已入 `paper/figures/risk_coverage_250q.png`)+ matched-coverage 論證;
  新 §sec:oodfam(`tab:oodfam` OOD selectivity + 3 caveats);abstract 加 K=1 與 selectivity
  兩句;4 處 `dominates…simultaneously`→`at their respective operating points`;
  §protocol 警鈴句改寫 + *Coverage disclosure*;Limitations 補 oracle 強度揭露。
- **Point 5(5 處)**:DiagnosticReport 三分法(exact replay / logged decisions / fresh
  rerun 不保證);§fiftyq 消掉 `bit-identical` 自相矛盾句;§deployment 補「traceability ≠
  sampling determinism」;C5 表格列與 abstract/conclusion 統一 `audit-replayable`。
- **審閱重點**:新數字直接寫死正文(81.2/2.8/84.0、96.7/92.4/97.6、88.9/19.6/60.3/49.0、
  31 recovered/2.1 rounds),未做進 numbers.tex macro;數據來源見
  `2026-06-15_250Q_k_ablation.md` 與 `2026-06-15_ood_family_eval.md`。
