# 補實驗 Implementation Plan(回應外部審查,2026-07-13)

外部審查結論:不是實驗太少,是**證據配置失衡**——C4 只有一個 worked example、
library 幾乎全覆蓋 benchmark、主表只有 Flash。本計畫把審查的優先序轉成可執行的
批次(wave),原則:**先做零 API 成本的、再做隔夜跑的、最後做要改架構的**;
每項標明可復用資產與「做不到時的降級方案」。

**已完成(本 session,直接抵免)**:53 卡 OOD probe(→被 Wave 1 的 sweep 升級取代)、
250Q K-ablation、risk–coverage 圖、refusal-permitted CoT baseline。

**關鍵盤點(已確認)**:250 卡庫含 **191 個 transform_spec / 159 個 family**
(RepairBench 原料充足);hint-trigger pre-pass 在
`run_error_classification_pipeline.py:947 _lexical_hint_triggers`(holdout 可行);
VQ/CoT 兩邊 250Q per-case correctness 都在(統計檢定零成本)。

---

## Wave 0 — 零 API、離線可完成(~2–3 天,順手全做)

### 0-A. RepairBench:C4 系統性驗證 【審查 P0,最高優先】
- **目標**:回答「cross-verification 到底抓得到多少錯誤 patch」——valid-accept、
  invalid-reject、**false-accept** 三率。
- **設計**(全部 deterministic Python,不需 LLM):
  - 原料:250 卡庫 191 個 transform_spec(取 30–60 個、跨 ≥10 family;
    另含 paper_v1 的 37 個 transform_map)。
  - 每個合法 transform 配 5 類**程式化變異**的錯誤 patch:
    錯 operator(×↔÷、+↔−)、錯係數(r/12→r/6、1+r→1−r)、錯變數、
    identity/no-op、白名單外名稱注入(`__import__` 等)+ 超大 AST。
  - code_patch 側:multi-occurrence、錯 site、import 注入、超 blast-radius。
  - **兩個加測**:(i) point-of-use(1 樣本)vs multi-sample(5 隨機輸入)的
    false-accept 差;(ii) 「雙側同錯」盲點量化——expr 與 invariant 一致地錯,
    預期通過,誠實報告此率(對應論文 *Scope of the guarantee* 段)。
- **復用**:`stage_transform.verify_transform / verify_code_patch`、`test_transform_poc.py`。
- **交付**:一張表(per-mutation-class rejection rate)+ 1–2 句進 §transforms。
- **降級方案**:若不做,C4 從 headline contribution 降為 design prototype/case study
  (審查原話)。但成本極低,應做。

### 0-B. 統計檢定 + 標註一致性(審查 P2,但零成本 → 提前)
- Paired bootstrap CI(accuracy、SWR)+ **McNemar test**(VQ vs CoT per-case,
  資料已在 `results/*/output.jsonl`)。「statistically indistinguishable」句要有檢定撐。
- 第二標註者盲標 8 個 silent-wrong 的 B/C/D 歸類(給 trace 不給我們的標籤),
  報 agreement——成本 = 找一個人 30 分鐘。
- **交付**:主表加 CI;§attribution 加 annotation protocol 一句。

### 0-C. 成本/延遲盤點(審查 P3,可從 log 重建大半)
- 從既有 run 的 `elapsed_seconds` + history 輪數重建:每題 LLM calls、K=1/K=3
  平均延遲、對比 CoT;contract build = 3 calls/family 一次性成本。
- **交付**:Deployment 節一小段或附表。

### 0-D. 子集卡庫產生器(Wave 1 的前置,純 SQL/Python)
- 從 250 卡庫按 family 分層抽樣建 10%/25%/50%/75%/100% 五個 cards.db
  (走 `card_store.ingest_artifacts` 保 FTS 同步;seed 固定)。
- 同時解決 OOD probe 的「demo-mutated 庫」caveat——sweep 全用 pristine 子集。

---

## Wave 1 — 隔夜 API 跑(~4–6 晚,harness 都是現成或小改)

### 1-A. Library-size sweep 【審查 P0「跨資料驗證」的最小可行版】
- 5 個子集庫 × 250Q,`run_ood_family_eval.py` 直接吃 `--lib-db`(已支援)。
- 畫審查建議的核心圖:ID acc / ID abstain / OOD abstain / OOD SW /
  coverage-separation gap vs library size——把「abstention tracks library
  coverage」從單點升級成曲線,**取代並升級現有 §oodfam 單點結果**。
- 成本:每點 ≈ 1 小時(不帶 baseline)/ 5 晚內完成;斷點續跑靠既有 incremental flush。

### 1-B. 強模型 CoT 250Q(審查 P0 baseline 補強)
- Gemini 2.5 Pro CoT single-shot(+若預算許可 oracle K=3)跑滿 250Q。
- 現成 runner 換 model 參數即可。順帶回應「VQ Pro 反而差」的敏感性疑問
  (50Q 已有,250Q 補 CoT 側)。

### 1-C. PAL/PoT 執行式 baseline 250Q(審查 P0)
- 新 runner(~半天):LLM 生成 Python → 沙箱執行 → 數值答案;同 ternary 計分。
- 這是「不讓 LLM 算數學」的直接對照,把 deterministic-execution 的貢獻與
  contracts/gates 分開的第一刀。

### 1-D. Self-consistency@k baseline(第三批精修時許諾的 future work,便宜)
- 每題 k=5 採樣、多數決 + 分歧率當 confidence → 真實 rejector 的
  risk–coverage 實測點,補在 perfect-rejector 上界圖上。

---

## Wave 2 — 需改 pipeline 的實驗(各 1–3 天工 + 1–2 晚跑)

### 2-A. Component ablation ladder 【審查 P0 的核心科學問題】
- 在 pipeline 加 gate 開關(config flags):
  `FIC+compute only → +F/E → +I → +M/N → full → +hint-trigger`。
- 每階 250Q 一跑(6 晚)。交付:一張「哪個元件貢獻哪類 SWR 下降」的階梯表——
  審查說這張表「非常有殺傷力」。
- 工程註記:monolith 裡各 gate 的 early-exit 點清楚(M/N :1318-1400、F :1490、
  E :1660、I :1737),加 flag 跳過即可,但要小心 skip 後的 fallthrough 語義。

### 2-B. Hint-trigger held-out 驗證(審查 P1)
- protocol:medium 抽 60 題當 dev 設計/凍結 trigger 詞表 → 在其餘 medium+hard
  + paraphrase 上測;報 **trigger precision/recall**(不只 critic miss rate)
  + false-positive 專項(字面命中但非真 ambiguity)+ 4 個轉 abstention 案例覆盤。
- 復用:`_lexical_hint_triggers` 已是獨立函式,詞表抽出即可凍結。

### 2-C. Natural paraphrase holdout(審查 P0-B 的另一半)
- 對已覆蓋 family 用 LLM 改寫 60–100 題(換變數名、敘述順序、加 distractor、
  單位轉換),人工抽查 20%;跑 full pipeline 對照原題。
- 直指 selection+binding 的真實難度,比 deterministic deletion traps 更接近實戰。

---

## Wave 3 — 方法補強(下一個 method contribution 等級)

### 3-A. Input provenance / span-grounded binding(審查 P1,架構閉環)
- Binding schema 加 `value / source_span / source_offsets / normalization_operation`;
  規則:literal 必須 exact/normalized match 原文 span;derived 必須標記;
  contract 未允許 derivation 不得改寫;多候選 span → abstain/clarify。
- Binding trap set(每類 10–20):quarterly vs annual 同現、90 天 tenor vs 360
  annualization、$2.5M vs 含 15% profit、25% vs 0.25、per-share vs total、
  正負號/WC direction、distractor 公司與前年值。
- 交付:span-grounding 前後 SWR 對照;這是論文自己指名的「最明確架構破口」的補完。

---

## Backlog(明確延後)

### B-1. M-gate 重做 【user 決定延後】
- 困難:真正「無唯一 metric intent」的題目只能自寫或 LLM 生成後人工審。
- **設計筆記(留檔)**:模板 =「投入 X、每年回收 Y、n 年、折現/通膨 r,
  我會不會賺?」→ 正確行為 = 列出 NPV/IRR/payback 候選並反問;
  變體 = context 同含 ROI/ROE/ROA 所需欄位 +「calculate the return」。
  Demo showcase 已有 1 題驗證可觸發 abstain_m(本 session 實測),
  生成配方 = 取共享輸入簽名的已覆蓋 families、剝掉 metric 名;每類目標 30–50 題。
- 在此之前:論文維持現狀(M 排除於 claims、Limitations 已列 trap-M 缺陷)。

### B-2. Realistic clarification / human study(審查 P2)
- 最小版:oracle 只能看原始題目 + DiagnosticReport(不看 computation code),
  250Q 重跑一晚 → 量化 privileged-info 貢獻;真人研究(2–3 位財金背景 × 20–30 題)
  再往後。

---

## 建議執行順序與資源估算

| 週次 | 內容 | API 成本 |
|---|---|---|
| 第 1 週 | Wave 0 全部(RepairBench、統計、成本盤點、子集庫) | 0 |
| 第 1–2 週夜間 | Wave 1(sweep 5 晚 + Pro CoT 1 晚 + PAL 1 晚 + SC@k 1 晚) | ~8 晚 |
| 第 2–3 週 | Wave 2(ablation flags + 6 晚跑;trigger holdout;paraphrase) | ~8 晚 |
| 之後 | Wave 3 provenance;Backlog 視餘裕 | — |

**若只能做三件事(審查的「最小可投實驗包」對應)**:
1. Wave 2-A ablation ladder(科學核心)
2. Wave 1-A library sweep + 2-C paraphrase(externality)
3. Wave 0-A RepairBench(C4;做不到就降級 C4 文字)

Wave 0 與 Wave 1 互不阻塞,可立即並行啟動。
