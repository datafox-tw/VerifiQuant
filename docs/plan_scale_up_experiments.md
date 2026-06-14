# 草稿：擴大實驗規模 + Pro<Flash 驗證計畫

狀態：DRAFT（2026-06-10）。回應兩個 open item：(1) 為什麼 VQ Pro < VQ Flash；(2) 50Q→~250Q 擴增，medium/hard 是否都納入。

資料池（filter = has_function AND has_ground_truth 後的「可用」題數）：
- medium：540 可用（raw medium.json 1000）
- hard：130 可用（raw hard.json 238）

---

## Part 1 — 為什麼 Pro < Flash：把質性發現變成可量化指標

你的診斷（over-normalization 發生在 input-binding，funnel 上游；funnel 只檢查「綁定後的輸入與輸出」，察覺不到「給定值在綁定前被竄改」）已經很完整。缺的是**證據強度**：目前只有 2 個案例（test-1593、test-1890），N=50 下嚴格說落在 run variance 內。

### 關鍵主張：可以不需要 gold answer 就量化 over-normalization
做一個 **input-provenance 量測**（純診斷、不評分）：
- 在 binding 階段 log 出每個被綁定的數值，與 context 的字面 token 比對。
- 定義 **Input-Mutation Rate (IMR)** = 綁定值無法溯源到 context 任一字面 span 的題數 / 總題數。
- 跨 Flash vs Pro 在同一份 250Q 上比較 IMR。

預期：Pro 的 IMR 顯著 > Flash。這把「Pro 想太多」從 2 個軼事變成一條**模型級的量化趨勢**，且**完全不需 gold**（所以便宜、可在全 250 上跑、不受 Pro 推理成本以外的限制）。這也直接驗證了你提的 future work（provenance check 三級強度 a/b/c）能攔下哪些題。

### 建議的 Pro<Flash 驗證實驗（不必全量 Pro 跑分）
1. **IMR 量測**：Flash vs Pro，binding-only instrumentation，全 250Q。便宜、決定性。
2. **配對錯誤分析**：找出 Pro 錯但 Flash 對的題（paired），人工確認是否皆為 over-normalization 模式。
3. （可選）實作 provenance check level (a)，重跑 Pro，看 IMR→0 後 SWR 是否回到 Flash 水準。這是「架構修正有效」的正面證據。

---

## Part 2 — 擴大到 ~250Q：設計與分階段

### 2.1 抽樣設計
- **建議規模**：250Q。**superset 原則** —— 新題集包含現有 canonical 50Q，使舊結果成為子分析、可前後對照。
- **medium/hard 都納入，但分開報告**（難度是 SWR 故事的變因；hard 預期放大 CoT 的算術 silent-wrong，強化 VQ 相對優勢）。
- **建議切分**：**180 medium + 70 hard**（medium 從 540 抽、hard 從 130 抽）。
  - 保留分層：沿用 `stratified_random_by_difficulty_quartile`，在每個 tier 內分層。
  - hard 取 70 而非全 130：留 60 題 hard 作為未來 OOD/holdout，且 70 已足夠成一個可報告的 cell。
  - 若想要 hard 統計力更強：150 medium + 100 hard（hard 池吃緊、holdout 只剩 30）。

### 2.2 成本結構（這是擴增的真正瓶頸）
每題成本依系統差異極大：

| 系統 | 每題 LLM 呼叫 | 250Q 量級 | 備註 |
|---|---|---|---|
| FIC card build（一次性） | ~3（core/retrieval/repair） | ~750 | **可重用**，所有 VQ run 共享；用 Flash 生卡即可 |
| CoT single-shot | 1 | 250/cell | 便宜 |
| CoT + oracle K=3 | ~5 | 1250/cell | 中等 |
| VQ K=3 | ~每輪 selector+extractor+judge(+oracle) ≈ 4×3 | ~3000/run | 最貴，尤其 Pro |
| JPMorgan MAS | multi-agent，數十 | 高 | 視需要 |

**Pro 成本策略**（你最擔心的）：
- 卡片用 **Flash 生**（卡是 model-agnostic 產物，不需要 Pro）。
- **VQ Flash 跑全 250**。
- **VQ Pro 不必全 250 跑分**：Pro 的角色是驗證 over-normalization，而 IMR 量測（Part 1）已能在全量上便宜完成。VQ Pro 完整跑分只需一個**分層子集（~100）**即可報「Pro 在更大樣本上仍 ≤ Flash」。省下約 60% Pro 成本。

### 2.3 卡片品質在規模下的風險（必須先解決）
V1→V3 是**人工修補** 3 個 C-error 才達到 SWR=0。**250 題不可能手動修卡**。擴增前必須：
- 完全倚賴 `execution_smoke_ok` / `validate_relations.py` 的**自動閘門**過濾壞卡。
- 量化並報告 **card-quality**：250 張卡有幾張 smoke-test 失敗、幾張 relation 不一致。把「卡片良率」當成一個透明指標，而不是偷偷修到好。
- 這同時是防 cherry-picking 的論述（卡的修正皆可由語法/執行期錯誤偵測，無需 gold）。

### 2.4 分階段執行（每階段可獨立失敗、可中止）
- **Phase 0**：定稿 250Q 抽樣（superset of 50）→ 新 `experiment_config.yaml`。
- **Phase 1**：用 Flash 生 250 張 FIC 卡 → build_card_store → 跑 validate_relations + smoke test，輸出卡品質報告。
- **Phase 2（便宜先行）**：CoT 全變體（single-shot / oracle）on Flash，先得主準確率/SWR 表 + 難度拆分。
- **Phase 3**：VQ Flash 全 250（K=3）。
- **Phase 4（貴、可子集）**：VQ Pro on 分層 100 子集 + IMR 全量量測。
- **Phase 5**：trap set 在新卡上重建（build_trap_set）→ score_trap_set，這是最大差異點主戰場。

### 2.5 統計
- N=250 下，Pro vs Flash 的差距可做檢定：配對用 **McNemar**，比率報 **bootstrap 95% CI**。
- 難度分層各自報 SWR / selective-acc，看 VQ 優勢是否隨難度擴大。

---

## 已拍板（2026-06-10）
1. **規模**：**180 medium + 70 hard**，沿用 `scripts/sample_dataset.py` 的同一抽樣法（stratified_random_by_difficulty_quartile, seed 42），且為現有 50Q 的 superset。
2. **Pro**：**這輪不跑 Pro 跑分**。只做 VQ Flash 全 250 + **全量 IMR（Flash vs Pro，僅 binding 階段）**。Pro 的 over-normalization 用 IMR 驗證，不需 Pro 完整跑分。
3. **MAS**：**不重跑**。沿用舊 50Q 的 JPMorgan 數字；本輪主戰場是 CoT vs VQ。

目標定位：**AAAI tier-1.5 等級**；FinNLP（7 月底）時程吃緊，故分階段、先求把故事跑出來。

### 因此本輪的最小可發表路徑（phase 重排）
- **Phase 0**：抽樣 180m+70h（superset of 50）→ 新 config。【純抽樣、無 API】
- **Phase 1**：Flash 生 250 卡 + 自動閘門 + 卡品質報告。
- **Phase 2**：CoT 全變體（Flash）→ 主表 + 難度拆分。
- **Phase 3**：VQ Flash 全 250（K=3）。
- **Phase 4**：**全量 IMR（Flash vs Pro，binding-only）** —— 取代 Pro 跑分，便宜且決定性。
- **Phase 5**：trap set 在新卡上重建 + score。
- （延後）VQ Pro 完整跑分、MAS 重跑、provenance check 實作。
