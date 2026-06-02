# Advisor Meeting Brief — 數據重跑後的進度報告

> 一句話開場：「我把整個 pipeline 重跑成乾淨可重現的版本。重點不是某個數字變了,而是**重跑過程逼我修掉一個方法論漏洞,結果核心主張反而更強、更乾淨了**。」

---

## 報告脈絡（建議講述順序，約 15–20 分鐘）

### Part 1 — 我重跑了什麼、為什麼（2 分鐘）
- 全部實驗收斂到單一 canonical pipeline（`paper_v1`），所有 run 有可重現 artifact + RUN_SUMMARY。
- 固定 FIC 卡片版本（V3 canonical），讓 Flash/Pro 可逐模型公平比較。
- **動機**：之前數字是多次 ad-hoc 跑累積的，口徑不一致；重跑是為了讓主表經得起 reviewer 追問。

### Part 2 — 過程中發現並修掉的方法論漏洞（4 分鐘，這段是重點，展現嚴謹度）
- **發現**：原本 CoT baseline 的 oracle 是「答錯才進場」（用 `is_correct` 當開關），而且 prompt 裡塞了 ground truth code。
- **問題**：(1) oracle 只在答錯時出現 → 進場本身就洩漏「這題是錯的」；(2) 只救錯、從不碰對 → accuracy 被高估。這是**對 baseline 不公平、且評估外洩**。
- **修法**：改成 **GT-blind 每輪盲審**——oracle 每輪都審、不知對錯、不給數值 GT（只給 `python_solution` 意圖規格，與 VQ oracle 對等）。
- **講法**：「我發現這個設定其實對我的 baseline 不公平、而且有外洩疑慮,所以我主動把它修嚴格。」← 這句讓老師知道你會自己抓自己的洞。

### Part 3 — 公平重跑的結果，以及為什麼這對我們是好消息（5 分鐘，核心）
- 四配置（plain/funnel × Flash/Pro）blind-review 結果：
  - plain Flash 94% / SWR 6%
  - funnel Flash 92% / SWR 8%
  - **plain Pro 98% / SWR 2%**
  - funnel Pro 96% / SWR 2%
- **關鍵轉折**：CoT+oracle 現在甚至能達到 98%，**超過 VQ 的 90%**。
- **為什麼這是好消息**：它逼我把主張從「同準確率、VQ 的 SWR 更低」升級成更強的版本——
  > **「準確率不是軸。即使 CoT 準確率更高（98%），它仍消不掉 silent wrong（SWR=2%）；只有 VQ 靠 abstention 達到 SWR=0%。可靠性是架構能力，不是準確率競賽。」**

### Part 4 — 三個讓主張站穩的硬證據（3 分鐘）
1. **broken_count = 0**（四配置全 0）：盲審不會把對的改錯 → 協議是安全的，不是靠亂改衝分。
2. **test-1443（CAGR，gold=19.14）在 4 配置錯 3**（含最強 plain Pro）：根因是 LLM 對分數次方 `2.4^(1/5)` 的算術非決定性——答案在 19.11–19.16 間擲骰子；plain Pro 甚至 turn 2 算對 19.14、turn 3 又漂回 19.13，連自己都重現不了。VQ 交給 deterministic Python（永遠 0.19135…→19.14）→ 把擲骰子變查表。這是最有力的微觀案例。
3. **盲審下 funnel-guided ≤ plain**（Flash 92<94、Pro 96<98）：當 oracle 看不到對錯，更精巧的結構化 prompt 反而過度改寫 → 證明可靠性不來自 oracle prompt 設計，而來自架構。

### Part 5 — Trap 實驗（差異化的主戰場）（3 分鐘）
- 整體 Trap SWR：CoT 48% → VQ 26%。
- **E（邊界違反）90%→20%、I（歧義）100%→10%**：數量級改善，這是 FIC 合約 + funnel 的核心增益。
- F/N：SWR 打平（CoT 也能散文式拒答），但 VQ 給結構化、機器可讀的攔截。
- **M（模糊意圖）誠實揭露**：VQ 在 M-trap 反而差（SWR 100%）→ 但這是 trap 設計限制（M-operator 保留了完整 metric context，沒真正隔離意圖歧義），**不納入優勢宣稱**。這精確定義了我們的 *verifiability frontier*。

### Part 6 — 需要老師拍板的決定（3 分鐘，把球交給老師）
1. **Venue：AAAI main vs FinNLP workshop**
   - 內容成熟度像 main conference 規格；老師覺得 AAAI 有機會。
   - 但 FinNLP 是主場（領域對口）、是 workshop（份量輕）。
   - **要先確認**：(a) AAAI 只能投 AAAI-27（約 2026/08 截稿，比 FinNLP 07/25 晚）；(b) FinNLP 是否 non-archival → 若是，可先 FinNLP 再 AAAI 兩段走。
2. **跨資料集泛化要做到什麼程度**：已排 Hard Tier 20Q（Appendix）。要不要再加 FinQA/TAT-QA？這是唯一的硬骨頭，工作量大，需老師定範圍。

---

## 一頁速查（meeting 桌上版）

| 系統 | Acc | SWR | Abstain |
|---|---|---|---|
| CoT single-shot (Flash) | 82% | 18% | 0% |
| CoT plain oracle blind (Flash) | 94% | 6% | 0% |
| CoT plain oracle blind (Pro) | **98%** | 2% | 0% |
| JP Morgan MAS | 86% | 14% | 0% |
| **VQ Flash V3** | 90% | **0%** | 10% |
| VQ Pro V3 | 86% | 4% | 10% |

**一句話結論**：CoT 準確率可達 98%（贏 VQ），但 SWR 永遠 >0%；VQ 用 abstention 達 SWR=0%。**在金融部署，可稽核的拒答 > 不可稽核的高準確率。**

---

## 如果老師質疑，預備好的三個答案
- **「50 題太少」** → O-ITL 多輪（K=3）深度分析 + 分層抽樣 + 三輪零浮動 variance + Hard Tier 20Q 泛化 slice。
- **「VQ accuracy 還輸」** → 那正是論點：準確率不是軸，SWR=0% 的 abstention 能力才是；CoT 98% 仍 SWR 2%。
- **「卡片是為這些題量身打造（cherry-pick）」** → C-class 修補皆由 `execution_smoke_ok` 偵測、無需 gold answer；Hard Tier 20Q 現建卡片可進一步證明卡片建構不依賴看過題目。
