# VerifiQuant 口頭報告大綱（依 `paper/main_v2.tex` 8 頁版逐節展開）

**單一來源**：`paper/main_v2.tex`（2026-08-06 版）。所有數字都取自該檔內嵌的 result macro，未引入任何論文外的數據。
**時間設計**：主軸 17 張投影片，約 18 分鐘；後面 8 張 backup 只在 Q&A 取用。
**壓縮版**：要壓到 12 分鐘，刪 S5、S8，S13 只講成本一句，S14 只留右半的 OOD 表。
**擴充版**：要拉到 28 分鐘（口試），把 backup B1/B2/B3/B6 併入主軸，並在 S9 後加一張實際 trap 範例走查。

**全場唯一一句話（每張投影片都要能連回這句）**：
> 乾淨題目上，governance 不會讓你更準；它的價值出現在「這題根本不該給出數字」的時候。

---

## Slide 1 — 標題

**標題**：VerifiQuant: Governance-by-Execution for Financial LLM Workflows

**版面**：題目 + 姓名/指導教授/系所 + 一行 subtitle。

**Subtitle（放在題目下方）**：
> When should a financial LLM workflow be allowed to output a number?

**表格**：無。

**口說稿（約 30 秒）**：
> 各位老師好，我今天報告的題目是 VerifiQuant，一個把「治理」直接編譯成可執行程式碼的金融 LLM 框架。這份研究要回答的問題只有一句：一個金融 LLM 工作流，什麼時候才有資格輸出一個數字？我會先用一個真實的失敗案例說明為什麼這是部署上的核心問題，接著介紹方法，然後用四個研究問題把證據攤開，包含對我們自己不利的那一部分。

---

## Slide 2 — 一個具體的失敗（動機）

**標題**：A Program That Runs, Parses, and Is Completely Wrong

**版面**：左邊題目文字（Heston option pricing），右邊 baseline 輸出一個乾淨的數字，中間一個紅色標記「no signal」。

**內容要點**：
- 題目要求用 Heston 隨機波動度模型定價選擇權。
- 一個強的 program-writing baseline 改去計算上下文裡一個毫不相關的製造業比率。
- 程式**通過語法檢查、成功執行、回傳一個乾淨的數字**。
- 輸出裡沒有任何東西標記它是錯的。

**表格**：無（用一張「question → generated program → clean number」的三格圖）。

**口說稿（約 60 秒）**：
> 先看一個我們 trap study 裡真實發生的案例。題目要求用 Heston 隨機波動度模型為一個選擇權定價。一個很強的、會自己寫程式的 baseline，最後去計算了上下文裡一個完全不相關的製造業比率。重點是：那支程式語法正確、成功執行、回傳一個乾淨漂亮的數字。從輸出上，你完全看不出它是錯的。
> 這不是一個算術錯誤，所以「模型再準一點」救不了它。這也是我整份研究的起點：金融工作流在生產環境掛掉，通常不是因為它算錯，而是因為**沒有人能治理它究竟該不該輸出數字**。

---

## Slide 3 — 重新定義問題：可治理性

**標題**：The Problem Is Governability, Not Accuracy

**版面**：上半三分支圖，下半四個 metric 名詞卡。

**內容要點**：
- 每一次回應只會落在三種結果之一：
  1. **Correct**：正確的數值答案
  2. **Silent wrong**：有信心、沒標記、錯的數字 ← 部署風險的來源
  3. **Structured abstention / clarification**：說明「為什麼沒有給數字」
- 四個量測軸：Coverage（回答比例）、Selective accuracy（回答中的正確率）、**SWR**（silent-wrong rate，全體分母）、**FNAR**（false numeric acceptance rate，只在 trap 上算）。
- 所有系統的 informal prose refusal 一律算 abstention，不算 silent wrong（對 baseline 有利的算法）。

**表格**：無（三分支圖 + metric 卡片即可）。

**口說稿（約 75 秒）**：
> 所以我把問題重新定義。給定一個金融問題和它的 context，什麼時候輸出一個數字是**正當的**？
> 我把每一次回應歸到三類。第一類是正確答案。第二類我稱為 silent wrong，一個有信心、格式完美、沒有任何標記的錯誤數字，這就是部署風險真正的來源。第三類是有結構的拒答或澄清，它會說明為什麼沒有給數字。
> 對應的量測有四個。Coverage 是回答比例，selective accuracy 是有回答的題目裡的正確率，SWR 是 silent wrong 佔全體的比例，這是我全篇最關鍵的指標。最後 FNAR，false numeric acceptance rate，只用在那些「本來就不該有數字」的輸入上。
> 這裡先說明一個對 baseline 有利的計分慣例：任何系統只要用自然語言說它做不到，我都算它 abstention，不算 silent wrong。

---

## Slide 4 — 四個研究問題（全場路線圖）

**標題**：Four Questions, One of Which We Lose

**版面**：四列，每列一個 RQ + 一句話答案（答案先用灰色，講到時再變色）。**這張投影片在 S10、S11、S14、S15 各回來一次**。

**內容要點（表格化呈現）**：

| RQ | 問題 | 一句話答案 |
|---|---|---|
| RQ1 | 乾淨題目上，governance 有沒有讓準確率變高？ | **沒有。** 這是預期中的結果，它精確界定了我們主張的邊界 |
| RQ2 | 在不該產生數字的輸入上，**執行**治理有沒有用？ | **有，而且這是本文的主結果**（FNAR 27.5% → 11.3%） |
| RQ3 | 這份安全性來自哪裡，代價是什麼？ | 來自 gate，代價是約 15 個百分點的 clean coverage |
| RQ4 | 獨立的人類專家標註能不能佐證這個邊界？ | 能，方向一致（Human-FNAR 最低 8.7%），但也暴露 retrieval 的代價 |

**口說稿（約 45 秒）**：
> 我用四個研究問題組織全部證據，而且我要先講清楚，第一個問題我們是「輸」的。
> RQ1 問：在乾淨、良好定義的題目上，加治理會不會更準？答案是不會。RQ2 問：在不該產生數字的輸入上，把治理**執行**出來有沒有用？答案是有，而且這是本文的主結果。RQ3 問這份安全來自哪裡、代價是什麼。RQ4 用一組完全獨立的人類專家標註，檢查這個邊界是不是我們自己造出來的假象。
> 這張圖我等一下會回來四次。

---

## Slide 5 — 相關工作定位（可壓縮）

**標題**：Why Existing Approaches Do Not Close This Gap

**版面**：四列比較表，最右欄是「缺什麼」。

**表格內容**：

| 路線 | 代表工作 | 解決了什麼 | 沒解決什麼 |
|---|---|---|---|
| Program-aided reasoning | PAL, PoT | 把算術交給直譯器，消除計算誤差 | 不會**檢查題目本身**。程式完美執行，公式、綁定或慣例仍可能是錯的，而且沒有訊號 |
| Neuro-symbolic | Logic-LM, LINC, SAT-LM | 用 solver 換取保證 | 翻譯步驟本身是主要誤差來源，規格合成一般而言不可判定 |
| Selective prediction | E-Y & Wiener, Geifman & El-Yaniv | 答或不答的框架 | 依賴一個 confidence 門檻，拒答不會說明**是哪個檢查失敗** |
| Multi-agent finance | Kundu et al. | 覆蓋更廣的任務 | 自然語言共識，錯誤在 agent 之間靜默傳播 |

**口說稿（約 60 秒）**：
> 快速定位一下。第一條路是 PAL 和 Program-of-Thoughts，把算術丟給直譯器執行，這解決了計算誤差，但它不會去質疑題目本身。一支完美執行的程式，公式、變數綁定、或是計息慣例仍然可以是錯的，而且沒有任何訊號。
> 第二條是 neuro-symbolic，用 solver 換取保證，但翻譯那一步本身就是最大的誤差來源。VerifiQuant 的做法是把形式化整個搬到離線：contract 在 build time 就建好並驗證過，runtime 只在一個已驗證的空間裡做選擇和綁定。
> 第三條是 selective prediction，它通常靠一個信心分數的門檻。我們不靠機率，我們靠 contract 宣告的檢查來拒答，而且拒答會指名是哪一個檢查失敗。

---

## Slide 6 — 方法一：Financial Inference Contract

**標題**：Financial Inference Contract (FIC): Formalization Moved Offline

**版面**：一張卡片視覺，四個欄位堆疊，右側一條「build-time governance」的直線。

**內容要點**：
- 一個 FIC 用 `fic_id` 封裝一個公式家族的四個要素：
  1. **適用/不適用條件** + retrieval metadata（決定這張 contract 適不適用）
  2. **typed inputs**（型別化輸入）
  3. **deterministic compute**（純 Python，決定性計算）
  4. **numerical invariants + 宣告過的語意歧義**（期別時點、百分比 vs 小數、單位基準），每個歧義帶解法選項與 hard/soft 等級
- **LLM 的角色只有「映射進 contract」**：selecting、binding、flagging。它**永遠不做金融計算**。
- Build-time 治理：compute code 過 smoke test，invariant 運算式對 runtime 命名空間做 lint gate，輸出型別政策由 audit 強制。
- 本文使用的 250 張卡片庫**全自動生成**（每個家族三次分階段 LLM 呼叫 + 決定性正規化），**從未讀過任何一題的解題過程或標準答案**。

**表格**：無（用卡片結構圖）。

**口說稿（約 75 秒）**：
> 方法的第一塊是 Financial Inference Contract，簡稱 FIC。一張 contract 封裝一個公式家族的四件事：第一，適用與不適用條件，加上檢索用的 metadata，這決定這張卡到底適不適用於眼前這題。第二，型別化的輸入。第三，決定性的計算程式碼，就是純 Python。第四，數值不變式，以及**事先宣告好的語意歧義**，例如期初期末、百分比還是小數、單位基準，每個歧義都帶著解法選項和 hard 或 soft 的等級。
> 這裡最重要的設計是：LLM 的角色只有「映射進 contract」，也就是選擇、綁定、標記。它從頭到尾不做金融計算。
> 另外要強調兩件事。第一，contract 是在 build time 被治理的，程式碼要過執行 smoke test，invariant 要對 runtime 的求值命名空間做 lint。第二，這 250 張卡片是全自動生成的，而且生成過程從來沒有讀過任何一題的解題過程或標準答案。

---

## Slide 7 — 方法二：可執行的診斷漏斗

**標題**：The Executable Diagnostic Funnel (M/N/F/E/I/C)

**版面**：直接用 `paper/figures/fig1_pipeline_loop_v3.pdf`（論文 Figure 1）。逐 gate 動畫進場。

**內容要點**：
- Gate 依序觸發，每一關產生一份 **DiagnosticReport**。
- **M / N**：是否恰好存在一張適用的 contract。出口是結構化的 disambiguation 請求，或 scope 邊界拒答。
- **F / E**：輸入是否完整、是否滿足 contract 的決定性檢查。出口是 slot-filling 請求，或指名被違反規則的 boundary alert。
- **I**：未解決的宣告歧義會不會**改變計算結果**（blocking I_HARD）還是只改變詮釋（warning I_SOFT）。
- **C**：帶稽核日誌的決定性執行。
- 檢索是純詞彙式：SQLite FTS5 BM25，**沒有任何一處被訓練或針對評測集調參**，且 VerifiQuant、R-PAL、R-PAL⁺ 用**同一個 retriever、同一個 top-4 深度**。
- Abstention 不是 fallback：五種漏斗條件可以終止於結構化的非答案出口。
- 稽核性的精確範圍：決定性執行與 invariant 求值可從儲存的 binding **完全重播**；LLM 的選擇與綁定決策有 log，可依記錄重跑；但**從原始文字重新跑一次不保證走同一條路徑**。

**表格**：無（用論文 Figure 1）。

**口說稿（約 90 秒）**：
> 第二塊是可執行的診斷漏斗。一個問題進來，gate 依序觸發，每一關都產生一份 DiagnosticReport。
> M 和 N 問的是：是不是恰好存在一張適用的 contract。出口是結構化的釐清請求，或者範圍外拒答。F 和 E 問的是：輸入完整嗎，滿不滿足 contract 自己宣告的決定性檢查。出口是要求補值，或者指名被違反的那條規則。I 問的是：還沒解決的歧義，會改變計算結果，還是只改變詮釋。前者擋下來，後者帶警告放行。最後 C 才做決定性執行，並留下稽核日誌。
> 這裡有兩個我要主動說明的設計選擇。第一，檢索是純詞彙式的 BM25，沒有任何一處針對評測集調過參數，而且 VerifiQuant 和後面兩個 baseline 共用同一個 retriever、同一個 top-4 深度，所以檢索品質是被固定住的變數，不是我這邊可以優化的東西。
> 第二，關於稽核性我的說法有嚴格界線。決定性執行和不變式求值，可以從儲存的 binding 完全重播。LLM 的選擇和綁定有留下 log，可以依記錄重跑。但是從原始文字重新跑一次，我**不保證**它會走同一條路徑，這一點我在論文裡有明確寫出來。

---

## Slide 8 — 方法三：有界修復與 RepairBench（可移到 backup）

**標題**：Bounded Repair: A Patch Must Match a Declared Identity

**版面**：左邊流程（hard ambiguity → AST-bounded transform → numerical cross-verify → accept/reject），右邊表格。

**內容要點**：
- 當 hard ambiguity 改變了該做的計算，自由重新生成等於把剛關上的門再打開。
- 因此修復採用 **AST-bounded transform**：接受前，補丁的數值效果必須在多組取樣輸入上，符合 contract 宣告的代數關係。
- **誠實界線**：這是 bounded numerical validation，**不是形式等價**。它抓得到單邊錯誤，但一個錯誤的 invariant 配上一個互相抵消的補丁會通過。
- RepairBench：179 個可合成的 transform spec，6 種補丁錯誤類別。

**表格內容（論文 Table 1 原樣）**：

| Mutation class | Gen. | Eq. | Rej./non-eq. | FA |
|---|---|---|---|---|
| function-call injection | 179 | 0 | 179/179 (100%) | 0 |
| oversized AST | 179 | 0 | 179/179 (100%) | 0 |
| operator swap | 171 | 8 | 159/163 (97.5%) | 4 |
| coefficient perturbation | 164 | 9 | 152/155 (98.1%) | 3 |
| variable swap | 34 | 2 | 29/32 (90.6%) | 3 |
| identity (no-op) | 179 | 8 | 161/171 (94.2%) | 10 |
| **Total** | **906** | **27** | **859/879 (97.7%)** | **20** |

**口說稿（約 60 秒）**：
> 第三塊是有界修復。當一個 hard ambiguity 確定會改變計算方式時，如果讓模型自由重新生成，等於把漏斗剛關上的門又打開。所以修復只能用 AST 上有界的 transform，而且在接受之前，這個補丁的數值效果必須在多組取樣輸入上，符合 contract 自己宣告的代數關係。
> 我用 906 個變異體去攻擊這個驗證器，其中 27 個在算術上等價所以排除，剩下 879 個裡它擋掉 97.7%，兩種結構性攻擊是 100% 擋下。剩下 20 個誤放，集中在條件式 transform，因為五組取樣的 binding 沒有走到那個分支。
> 我要主動說清楚它的界線：這是有界的數值驗證，不是形式等價證明。一個錯誤的不變式，配上一個剛好互相抵消的補丁，是會通過的。

---

## Slide 9 — 實驗設計：把變因鎖死

**標題**：Holding Everything Fixed Except the Thing We Claim

**版面**：上半 baseline 階梯（每一階只多一個東西），下半 trap 運算子 + K 設定。

**內容要點（表格化呈現）**：

| System | Retrieval | Deterministic execution | Governance taxonomy | Executed gates |
|---|---|---|---|---|
| CoT (single-shot) | ✗ | ✗（LLM 自己算） | ✗ | ✗ |
| PAL | ✗ | ✓ | ✗ | ✗ |
| **R-PAL** | ✓ 同一組 top-4 | ✓ | ✗ | ✗ |
| **R-PAL⁺** | ✓ 同一組 top-4 | ✓ | ✓ **完整 M/N/F/E/I 寫進 prompt** | ✗ |
| **VerifiQuant** | ✓ 同一組 top-4 | ✓ | ✓ | ✓ |

- 共用：backbone（Gemini 2.5 Flash）、題目與 context、top-4 檢索、執行環境、計分器（1% output-scale-normalized 容差）。**唯一的差別是最後一欄。**
- 資料：FinanceReasoning 250 題（180 medium / 70 hard），function-linked 子集，每題附公式層級的參考實作。Contract library 由那些參考函式建成，所以 clean set 是 **in-library by construction**，量的是 routing、binding、gating，不是未見公式的推導能力。
- Trap 由決定性運算子從乾淨題目導出：**F** 移除必要輸入、**E** 注入違反宣告檢查的值、**I** 刪掉宣告過的釐清資訊、**N** 換成範圍外的請求。標籤**就是**運算子。平衡 80 題（各 20）。
- **M 不做量化宣稱**，因為 under-specified 的題目仍保留帶指標的 context，無法產生可靠的 trap。
- **K=1 是唯一的可部署設定，也是所有 trap 評測使用的設定。**
- **K=3 是理想化的可回復性分析**：受限 oracle 可以看到題目的 intent specification（ground-truth 計算程式碼），但**永遠看不到數值答案**，且只能改寫題目與 context。

**口說稿（約 90 秒）**：
> 實驗設計的核心是把變因鎖死。一個有 contract 的系統，同時也帶著檢索和決定性執行，所以我如果只贏過一個裸的生成式 baseline，我沒辦法說是哪一個元件賺到的。
> 所以我建了一個階梯。CoT 什麼都沒有。PAL 有決定性執行但沒有檢索。R-PAL 拿到跟 VerifiQuant **完全相同**的 top-4 contract，自己寫一支程式決定性地執行，但沒有 gate。最關鍵的是 R-PAL⁺，它把完整的 M/N/F/E/I 治理分類**整套寫進 prompt**，並要求它在生成程式前明確決定要執行、拒絕、還是澄清。它有分類，但沒有可執行的 gate。
> 所以 backbone、題目、檢索、執行環境、計分器全部一樣，最後只剩下一個差別：**這些檢查到底是被執行，還是只被描述**。
> Trap 這邊，我用決定性運算子從乾淨題目導出：F 拿掉必要輸入，E 注入違反宣告檢查的值，I 刪掉釐清資訊，N 換成範圍外的請求。標籤就是運算子本身。M 我不做任何量化宣稱，因為 under-specified 的題目仍然保留帶指標的 context，做不出可靠的 trap，這一點我在論文裡明說。
> 最後，K=1 是唯一可部署的設定，也是**所有** trap 評測使用的設定。K=3 是一個理想化的上界分析，它的 oracle 可以看到題目的意圖規格，但永遠看不到數值答案。

---

## Slide 10 — RQ1：乾淨題目上，我們沒有比較準

**標題**：RQ1 — No, Governance Does Not Buy Clean Accuracy

**版面**：Panel A 表格，R-PAL 那列用綠色標記（它贏），VerifiQuant K=1 那列用中性色。**K=3 那列要有一個顯眼的角標。**

**表格內容（論文 Table 2 Panel A）**：

| System | Acc | SWR | Sel. Acc |
|---|---|---|---|
| CoT (single-shot) | 77.6% | 18.0% | 81.2% |
| PAL (no retrieval) | 91.2% | 7.2% | 92.7% |
| R-PAL (top-4 retrieval) | **96.8%** | 2.4% | 97.6% |
| R-PAL⁺ (taxonomy prompt) | 92.0% | 0.9% | 99.0% |
| **VerifiQuant (K=1, autonomous)** | 81.2% | 2.8% | 96.7% |
| VerifiQuant (K=3) ⚠️ | 93.6% | 3.2% | 96.7% |

> ⚠️ **投影片上必須加的註記（也同步加進論文 Table 1 caption）**：
> *K=3 denotes an intent-oracle upper bound; K=1 is the fully autonomous deployable setting.*

**內容要點**：
- R-PAL 幾乎每題都答，準確率 96.8%。VerifiQuant 自主模式是 81.2%，coverage 84.0%。
- R-PAL 與 K=3 上界在統計上**無法區分**：accuracy Δ=+3.2 點，95% CI [−0.4, 6.8]，exact McNemar p=.134；SWR Δ=−0.8 點，CI [−3.6, 2.0]，p=.791。
- 解讀不是「我們差」，而是這條軸**已經飽和**：公式一旦被正確檢索並決定性執行，in-library 的良好題目已經沒有留給治理層的空間。
- 換更大的模型也不解決 silent wrong：Flash 換成 Gemini 2.5 Pro，單次 CoT accuracy 從 77.6% 升到 82.4%，但 SWR 只從 18.0% 降到 15.2%。
- R-PAL⁺ 在乾淨集上表現同樣好（92.0% / 0.9% SWR），**但它在 trap 上接受的錯誤數字是 VerifiQuant 的兩倍以上**。站在乾淨集上，你無法預測一個系統的治理行為。

**口說稿（約 75 秒）**：
> RQ1，乾淨題目上治理有沒有讓我們更準？沒有，而這個預期中的結果，正好精確界定了我們主張的邊界。
> 表上 R-PAL 幾乎每題都答，準確率 96.8%。VerifiQuant 的自主模式是 81.2%，coverage 84%。我這裡要特別提醒一件事：K=3 這一列是 intent-oracle 的上界分析，不是自動化結果，K=1 才是可部署的設定。而且即使是那個上界，它跟 R-PAL 在統計上也**無法區分**，McNemar 檢定 p 值 0.134。
> 我的解讀是：這條軸已經飽和了。公式一旦被正確檢索出來並且決定性地執行，in-library 的良好題目已經沒有空間留給任何治理層。
> 而且換更大的模型也救不了 silent wrong。把 Flash 換成 Pro，CoT 的準確率從 77.6 升到 82.4，但它的 silent-wrong rate 只從 18 降到 15.2。
> 最後看 R-PAL⁺ 這一列，它在乾淨集上表現一樣好，但等一下你會看到，它在 trap 上接受的錯誤數字是我們的兩倍以上。**站在乾淨集上，你完全無法預測一個系統的治理行為。**

---

## Slide 11 — RQ2：主結果

**標題**：RQ2 — Describing Governance Is Not Executing It

**版面**：Panel B 大字呈現三個 FNAR（48.8% → 27.5% → 11.3%），下方接 per-operator 表。

**表格內容 1（論文 Table 2 Panel B，80 題平衡 trap 集）**：

| System | FNAR | Diag | Action |
|---|---|---|---|
| R-PAL (plain) | 48.8% | — | — |
| R-PAL⁺ (taxonomy prompt) | 27.5% | 65.4% | 55.8% |
| **VerifiQuant (K=1)** | **11.3%** | **75.0%** | **73.8%** |

> 16.3 個百分點的差距，base-question-clustered 95% CI [5.8, 26.4]。

**表格內容 2（論文 Table 3，每個運算子 n=20）**：

| Operator | FNAR: R-PAL | FNAR: R-PAL⁺ | FNAR: VQ | VQ Diag | VQ Act | R-PAL⁺ Diag |
|---|---|---|---|---|---|---|
| E (illegal boundary) | 90% | 30% | **10%** | 75% | 75% | 63% |
| I (deleted disambiguator) | 95% | 80% | **20%** | 60% | 60% | 10% |
| F (removed input) | 10% | 0% | 10% | 90% | 85% | 100% |
| N (out-of-scope ask) | 0% | 0% | 5% | 75% | 75% | 88% |
| **Overall** | **48.8%** | **27.5%** | **11.3%** | **75.0%** | **73.8%** | **65.4%** |

**口說稿（約 105 秒）**：
> 這是本文的主結果。在這 80 題本來就不該有數字的輸入上，裸的 R-PAL 有 48.8% 的時候會硬湊出一支程式，回傳一個有信心的數字。
> 接著關鍵的一步：我把**完整的**治理分類寫進 R-PAL 的 prompt 裡，它只降到 27.5%。而把同樣這些檢查**執行**出來，VerifiQuant 是 11.3%，差距 16.3 個百分點，以 base question 做 cluster 的 95% 信賴區間是 5.8 到 26.4。而且它給出正確結構化出口的比例也高得多。
> 這一格同時回答了三件事：**執行不等於治理，拿得到檢索不等於治理，而把治理寫在 prompt 裡，也不等於執行治理。**
> 下面那張表說明差距集中在哪裡。在 E 和 I，也就是正確與否取決於一個邊界或一個慣例的時候，裸的 R-PAL 幾乎每次都硬答，連加了分類的版本在 I 上都還有 80%，而我們的執行檢查守得住。
> 我也要講對我們不利的兩列。在 F 和 N 上，我們其實**略遜於** prompt 版本，F 是 10% 對 0%，N 是 5% 對 0%。原因不是 gate 沒跑，而是 binding 那一關：模型把缺掉的欄位用一個看起來合理的鄰近值填了進去，完整性檢查看到的就是一個滿足的 schema。這正是我等一下會定位的三個 runtime 前沿之一。
> 另外一個誠實的補充：這裡的 trap 是 contract 導出的，所以這是一個**針對機制的壓力測試**，不是自然部署發生率的估計。這也是為什麼我還需要 RQ4。

---

## Slide 12 — RQ3-a：安全與效用的取捨平面

**標題**：RQ3 — The Safety Is Not Free

**版面**：直接用 `paper/figures/safety_utility_250q.pdf`（論文 Figure 2）。三個點標註清楚。

**內容要點**：
- 橫軸：250 題乾淨題目的 coverage。縱軸：80 題 trap 的 FNAR。**兩軸刻意分開，不假設任何 clean-to-trap 混合比例。**
- 三個自主系統落在同一條單調前沿上：
  - 裸 R-PAL：coverage 99.2%，FNAR 48.8%
  - R-PAL⁺：coverage 92.9%，FNAR 27.5%
  - VerifiQuant：coverage 84.0%，FNAR 11.3%
- 一句話結論：VerifiQuant 相對於純執行放棄了約 15 個百分點的 clean coverage，換到的是純執行與 prompt 治理都達不到的錯誤數字接受率。

**表格**：無（用論文 Figure 2）。

**口說稿（約 60 秒）**：
> RQ3 問這份安全來自哪裡、代價是什麼。這張圖把三個自主系統放在同一個平面上，橫軸是乾淨題目的 coverage，縱軸是 trap 上的錯誤數字接受率。兩軸我刻意分開畫，因為我不想假設任何一個 clean 對 trap 的混合比例，那會是我自己編出來的部署情境。
> 三個點落在一條單調的前沿上。裸 R-PAL 幾乎全答，但接受了將近一半的 trap。加了 prompt 分類往左下移動一些。把 gate 執行出來，走到 11.3% 的 FNAR，代價是 84% 的 coverage。
> 這份安全不是免費的。相對於純執行，我放棄了大約 15 個百分點的乾淨題 coverage。

---

## Slide 13 — RQ3-b：Gate 階梯與成本

**標題**：Gates Buy Safety With Coverage, Not With Accuracy

**版面**：左表（gate ladder），右邊四個成本數字。

**表格內容（論文 Table 4，同一個 backbone、同一組 250 題累加）**：

| Configuration | Acc | SWR | Abst | Latency |
|---|---|---|---|---|
| CoT single-shot (LLM 自己算) | 77.6% | 18.0% | 11 | — |
| FIC select + bind + execute (no gates) | 87.6% | 6.4% | 15 | 6.6 s |
| ＋F/E gates | 86.8% | 4.4% | 22 | +0.2 s |
| ＋I gate | 79.6% | 4.4% | 40 | 12.7 s |
| ＋M/N gates (full funnel) | 79.2% | **4.0%** | 42 | 12.2 s |
| ＋hint triggers（僅為說明，非 held-out） | 80.0% | **3.6%** | 41 | — |

**對照成本**：R-PAL 10.5 s／R-PAL⁺ 8.2 s／self-consistency K=5 38.1 s。

**必須講的但書**：這是一次獨立的 gate-disable run，它的全漏斗那一階 SWR 是 4.0%（10/250），與 canonical K=1 的 2.8%（7/250）不同。**主張的是同一次 run 內的趨勢，不是絕對端點相等。**

**口說稿（約 75 秒）**：
> 接著是機制上的兩個視角。第一個是 gate 的累加消融，同一個 backbone、同一組 250 題，一層一層把控制加上去。
> 你會看到 silent-wrong rate 每一階都在降，從 CoT 的 18% 一路降到全漏斗的 4%，而準確率在 I 和 M/N 這兩階往下掉。這正是設計上想要的形狀：**gate 是用 coverage 去買安全，不是用 accuracy 去買。** 掉下去的那些題目變成了 abstention，不是變成錯答。
> 這裡要說明一個技術細節：這是一次獨立的 gate-disable run，它全漏斗那一階是 4.0%，跟 canonical run 的 2.8% 不同，所以我主張的是同一次 run 內的趨勢，不是絕對數值相等。
> 同一個階梯也幫我把成本標出來。選擇、綁定加執行是每題 6.6 秒。加上 F 和 E 只多 0.2 秒，因為那些是執行的 Python，不是模型呼叫。成本集中在 I，因為它多一次模型呼叫。全漏斗 12.2 秒，對照 R-PAL 的 10.5 秒，大約 1.2 倍，而且遠低於 self-consistency K=5 的 38 秒。

---

## Slide 14 — RQ3-c：庫覆蓋掃描與 OOD 的誠實代價

**標題**：Selective Abstention, and What It Costs Out of Library

**版面**：左邊 `paper/figures/libsweep_250q.png`（論文 Figure 3），右邊固定 OOD split 的對照表。

**內容要點（左）**：
- 在 10/25/50/75/100% 五個庫大小上重建 contract library（25/62/125/188/250 個公式家族，固定 seed，巢狀包含），每次都重跑全部 250 題。
- **每個庫大小下 abstention 都是選擇性的**：OOD abstention 91.6% → 80.6%，ID abstention 8.0% → 14.4%，全程維持 67 到 84 個百分點的選擇性差距。
- ID accuracy 在庫成長十倍的過程中是平的（81.6–84.0%）。
- 唯一可見的代價是誠實的：OOD 的 silent wrong 從 1.3% 微升到 4.8%，因為語意鄰近的 contract 讓「近鄰硬湊」更有吸引力。

**表格內容（右，固定 OOD split：50 家族庫、199 題家族缺席）**：

| | R-PAL | VerifiQuant |
|---|---|---|
| Abstain | 0.5% | 88.9% |
| Correct | 95.0% | 7.0% |
| Silent wrong | 4.5% | 4.0% |
| **Selective accuracy** | **95.5%** | **63.6%** |

**這張表要主動講出來的一句話**：VerifiQuant 在庫外**並沒有比較準**，這是最限制我們主張的一個結果。

**口說稿（約 75 秒）**：
> 第二個機制視角，是把「範圍訊號」變成一個可量測的性質。我在五個不同大小的 contract library 上重建卡片庫，每次都重跑全部 250 題。左圖的重點是：**在每一個庫大小下，abstention 都是選擇性的**。庫外題目的拒答率維持在八到九成，庫內題目維持在一成上下，全程有 67 到 84 個百分點的差距。而且庫成長十倍的過程中，庫內準確率是平的。
> 唯一可見的代價很誠實：庫外的 silent wrong 從 1.3% 微升到 4.8%，因為庫越密，語意鄰近的 contract 就越容易被硬湊上去。
> 右邊這張表是我最想主動講的。在固定的 199 題庫外題目上，matched 的執行 baseline 幾乎全答，而且靠參數知識答對了 95%。VerifiQuant 拒答了 88.9%，只答對 7%。所以請注意，**VerifiQuant 在庫外並沒有比較準**，它自己在庫外的 selective accuracy 是 63.6%，比 baseline 的 95.5% 低，而且它的殘留 silent-wrong 也沒有比較好。這是最限制我們主張的一個結果。
> 我們買到的不是庫外的正確率，而是一條**已驗證與未驗證的分界線**。那個 baseline 在這裡很準，靠的正是參數知識，而參數知識錯的時候，完全不帶訊號。

---

## Slide 15 — RQ4：獨立人類標註的外部效度

**標題**：RQ4 — An Independent Human Panel, and the Bill It Presents

**版面**：左邊標註流程（三個 blind 標記），右邊結果表。

**內容要點（流程）**：
- 一個獨立的 GPT-5.2 作者把 FinanceReasoning 主題改寫成 82 題自然語氣的查詢，**沒有**接觸我們的 contract、M/N/F/E/I 分類、標準答案或任何系統輸出。作者模型刻意不同於所有系統的 solver backbone（Gemini 2.5 Flash），使「凌亂程度」與「哪些題對 solver 容易」不相關。
- 五位受過金融訓練的標註者，對來源題目、生成條件、所有系統輸出全部 blind，各自獨立指派四種行動之一並附一句理由。
- **多數標籤在任何系統被評分之前就凍結**。82 題全部第一輪就達成多數，不需要仲裁。
- 信度：**Fleiss' κ = 0.84**（almost perfect）。多數標籤分布：36 answer、20 request-info、10 clarify、16 out-of-scope，也就是 **82 題裡有 46 題不該只給一個數字**。

**表格內容（論文 Table 5）**：

| System | Human-FNAR | Match | Sel. Acc |
|---|---|---|---|
| CoT (single-shot) | 21.7% | 87.8% | 15.2% |
| R-PAL (top-4 retrieval) | 56.5% | 68.3% | 11.3% |
| R-PAL⁺ (taxonomy prompt) | 13.0% | 65.9% | 17.2% |
| **VerifiQuant (K=1)** | **8.7%** | 31.7% | 9.1% |

**必須主動說的三件事**：
1. 這 46 題上實際的錯誤接受數是 4、6、10、26，所以**只有跟裸 R-PAL 的差距個別顯著**（Fisher exact p < 10⁻⁴），VerifiQuant 對 R-PAL⁺ 是 p = .74。RQ4 建立的是**方向的外部效度**，不是第二次有檢定力的比較。
2. VerifiQuant 的 Match 只有 31.7%，答題數 11/82，在人類判定可答的 36 題裡只答了 7 題。
3. 29 次短少裡有 **26 次是 top-4 檢索根本沒撈到適用的 contract**（例如 LIFO 成本題只撈到 FIFO 的卡），所以它是正確地拒答而不是硬湊。這是可歸因的 retrieval 覆蓋成本，不是隨機分布的謹慎。

**口說稿（約 90 秒）**：
> RQ4 處理一個很直接的質疑：那 80 題 trap 是我自己的 contract 導出的，標籤會不會只是把我自己的檢查重新編碼一次？
> 所以我做了一組外部效度檢驗。一個獨立的 GPT-5.2 作者把題目改寫成 82 題自然語氣的查詢，它看不到我們的 contract、分類法、標準答案或任何系統輸出，而且它刻意跟所有系統的 solver 用不同的模型。然後五位受過金融訓練的標註者，對來源、生成條件、系統輸出全部 blind，各自獨立標行動。多數標籤在任何系統被評分之前就凍結。Fleiss kappa 是 0.84，屬於 almost perfect，82 題全部第一輪就達成多數。結果是 82 題裡有 46 題不該只給一個數字。
> 結果表上，VerifiQuant 的人類定義 FNAR 是 8.7%，最低，順序跟 Panel B 完全一致。
> 但我要主動講三件事。第一，這 46 題上實際的數字是 4、6、10、26 次，所以只有跟裸 R-PAL 的差距個別顯著，我們對 prompt 版本的 p 值是 0.74。RQ4 建立的是**方向的外部效度**，不是第二次有檢定力的比較。
> 第二，我們的 Match 只有 31.7%，人類認為可答的 36 題我們只答了 7 題。第三，也是最重要的：29 次短少裡有 26 次，是 top-4 檢索根本沒有撈到適用的 contract，例如一題 LIFO 成本問題只撈到 FIFO 的卡。所以那是正確地拒答，不是隨機的謹慎，但在這組資料上，同一個機制也把大部分效用一起拿走了。

---

## Slide 16 — 邊界與失敗分析

**標題**：Where the Claim Holds, and Where It Breaks

**版面**：左邊三個成立條件，右邊三個 runtime 前沿（圖示對應到 funnel 的位置）。

**內容要點**：
- **主張成立的三個條件**：目標公式家族存在於庫中；contract 的規格與不變式足夠正確；問題屬於「以公式為基礎的決定性數值推理」。三者不成立時，VerifiQuant 至多提供一個範圍訊號，不提供正確答案。
- **殘留失敗只落在三個預先宣告的、由 LLM 中介的 runtime 前沿**：
  1. **Selection**：選錯或硬湊到語意近鄰的 contract
  2. **Binding**：抽取或縮放一個值時出錯
  3. **Declared-hint recall**：沒有觸發 contract 已經宣告過的歧義
- 實證上，250 題乾淨題與 80 題 trap 的**每一個**殘留都落在這三類。K=1 是 7 個 silent wrong；K=3 是 8 個，拆解為 3 selection、2 binding、3 hint-recall。
- 明確的界線：**我們不驗證輸入的 provenance**，只確認一個值滿足 schema 與邊界，不確認它取自正確的來源欄位。這正是 binding 前沿存在的原因。
- 與此不同的是 **build-time defect**（例如漏宣告一個歧義、寫錯一個不變式，本庫中有三張指標卡回傳 series 而非純量）。這些由 contract audit 與型別政策檢查在 runtime 之前擋下，不是由漏斗擋下。
- Hint-recall 這一前沿可以部分機械化：決定性的 trigger pre-pass 在慣例詞彙逐字出現時直接觸發。**這是說明性的，不是驗證過的結果**，held-out 驗證是未來工作。

**表格**：無（用兩欄圖示）。

**口說稿（約 60 秒）**：
> 最後是邊界與失敗分析。我的主張要成立需要三個條件：目標公式家族在庫裡；contract 的規格和不變式夠正確；問題是以公式為基礎的決定性數值推理。這三個條件不成立的時候，VerifiQuant 最多給你一個範圍訊號，不會給你正確答案。
> 而剩下來的 silent wrong，全部落在三個我事先宣告的 runtime 前沿：contract 的**選擇**、輸入的**綁定**，以及**宣告歧義的召回**。250 題乾淨題加 80 題 trap，每一個殘留都落在這三類裡。這件事的價值在於，它告訴一個部署團隊下一步該修哪裡，而不是丟給他們一個不透明的總分。
> 這裡有一個我要明講的界線：**我們不驗證輸入的 provenance**。我們的檢查只確認一個值滿足 schema 和邊界，不確認它是從正確的來源欄位取來的。這正是 binding 前沿存在的原因。

---

## Slide 17 — 結論

**標題**：Governance Is Worth Buying When a Wrong Number Costs More Than a Declined One

**版面**：三行大字，最後一行是條件句（刻意不做無條件宣稱）。

**內容要點**：
- 金融 LLM 的部署瓶頸，比較少來自模型的錯誤，比較多來自系統**治理**那個錯誤的能力。
- VerifiQuant 把 scope、schema、boundary、ambiguity 四類檢查編譯成一個可執行的漏斗，讓拒答、澄清、修復成為**可歸因**的結果。
- 它的價值不在良好定義的 in-library 題目上（那裡 matched 的執行已經夠了），而在兩個地方：不該產生數字的輸入，以及庫的邊界。
- 條件式結論：**當一個沒有依據的數字比一次拒答更昂貴時，它值得；當 coverage 才是主要限制時，它不值得。**

**口說稿（約 45 秒）**：
> 總結。金融 LLM 的部署瓶頸，比較少來自模型會犯錯，比較多來自系統有沒有能力治理那個錯誤。VerifiQuant 把範圍、結構、邊界、歧義四類檢查編譯成一個可執行的漏斗，讓拒答、澄清和修復變成可以歸因的結果。
> 它的價值不出現在良好定義的 in-library 題目上，因為在那裡 matched 的執行 baseline 已經夠了。它出現在兩個地方：不該產生數字的輸入，以及知識庫的邊界。
> 所以我的結論刻意是條件式的：**當一個沒有依據的數字，比一次拒答更昂貴的時候，這套治理值得買；當 coverage 才是主要限制的時候，它不值得。** 這是部署情境的性質，不是方法的性質。謝謝。

---

# Backup 投影片（只在 Q&A 取用）

| # | 標題 | 內容 | 對應的預期提問 |
|---|---|---|---|
| B1 | K ablation | K=1: 81.2/2.8/coverage 84.0/SelAcc 96.7 ・ K=2: 92.4/3.6/96.0/96.2 ・ K=3: 93.6/3.2/96.8/96.7。**oracle 看得到 intent spec，看不到數值答案** | 「K=3 是不是作弊？」 |
| B2 | 檢索深度 | 250 題乾淨集上，gold contract 中位數排名第 1，top-4 命中 94.8%，放寬到 8 只多 2.4 點、到 16 多 4.0 點。**故固定 k=4 而不調參**。dense/hybrid retriever 未測 | 「是不是 k 調小才贏？」 |
| B3 | 統計檢定 | R-PAL vs VQ K=3：acc Δ=+3.2, CI[−0.4, 6.8], McNemar p=.134；SWR Δ=−0.8, CI[−3.6, 2.0], p=.791。Panel B gap 16.3 點, base-clustered CI [5.8, 26.4]。RQ4 Fisher: vs 裸 R-PAL p<10⁻⁴, vs R-PAL⁺ p=.74 | 「有沒有做統計檢定？」 |
| B4 | 計分慣例 | 1% output-scale-normalized 容差，一個計分器套用到所有系統。R-PAL 若程式無法 parse、拋錯、逾時、或回「cannot solve」，一律算 abstention 不算 silent wrong。**FNAR 完全不依賴容差**，因為它只看有沒有吐數字 | 「容差是不是對你有利？」 |
| B5 | 人類標註規則 | 四標籤 top-down 決策：不是「算一個數字」→ out-of-scope；缺必要數值→ request-info；值都在但詮釋選擇會改變數字→ clarify；否則 answer。兩組對比明確訓練過。M-refusal 一律算 out-of-scope（保守讀法；改讀成 clarify 會把 Match 從 31.7% 拉到 34.1%） | 「標註標準是什麼？會不會偏袒你？」 |
| B6 | 為什麼 M 不做量化 | under-specified 的題目仍保留帶指標的 context，決定性運算子做不出可靠的 trap，因此對 M 不做任何數值宣稱 | 「六層漏斗為什麼只測四層？」 |
| B7 | 釋出的成果 | 250 張 contract library 與建庫的分階段 prompt、82 題人類評測集與 provenance、五位標註者各自的標籤檔、凍結的多數 ground truth、一致性報告、計分與彙總腳本與稽核軌跡（含標籤凍結時點） | 「可重現性？」 |
| B8 | 倫理 | 題目為合成、不含個資，只蒐集對題目的標籤與理由；自願參與，標註者匿名為 Annotator 1–5；經與機構諮詢認定不需 ethics-committee 審查 | 「有沒有 IRB？」 |

---

# 預期提問與回答骨架

**Q1「你在乾淨集上輸給 baseline，這篇的貢獻是什麼？」**
> 貢獻正是那個「輸」被界定得很乾淨。我的 load-bearing baseline 拿到跟我完全一樣的 backbone、檢索、執行環境和計分器，還額外把整套治理分類寫進 prompt。唯一剩下的差別是檢查有沒有被執行。乾淨集上它贏，trap 上它接受的錯誤數字是我的兩倍以上。這組對照把 gain 定位在「執行」，而不是定位在分類法本身，也不是定位在準確率。

**Q2「trap 是你自己的 contract 導出的，這不是循環論證嗎？」**
> 這正是 RQ4 存在的理由。我另外做了一組 82 題，由一個獨立的模型改寫、由五位對所有系統 blind 的金融背景標註者標記、多數標籤在評分前凍結、Fleiss kappa 0.84。順序完全複製了 Panel B。我也誠實說明它的檢定力：只有跟裸 R-PAL 的差距個別顯著，所以我主張的是方向的外部效度，不是第二次有檢定力的比較。

**Q3「拒答率這麼高，實務上誰要用？」**
> 我不會主張所有情境都該用。我的結論是條件式的：當一個沒有依據的數字比一次拒答更昂貴時它值得，例如估值、法遵報告、面向客戶的試算。當 coverage 才是主要限制時，把分類寫進 prompt 更便宜，而且我把那條路能走多遠也量出來了（92.9% coverage、27.5% FNAR）。

**Q4「這只是換一個更保守的 threshold 而已吧？」**
> 不是。threshold 式的 selective prediction 拒答時不會告訴你是哪個檢查失敗。我的拒答會指名是哪一條宣告的檢查、哪一個缺失欄位、哪一個未解決的慣例，而且可以從儲存的 binding 重播。另外，庫覆蓋掃描顯示這個拒答在每個庫大小下都是**選擇性**的，庫外和庫內之間維持 67 到 84 個百分點的差距，這不是一個一律變保守的門檻做得到的。

**Q5「換成更強的模型是不是就解決了？」**
> 我測過。Flash 換成 Gemini 2.5 Pro，單次 CoT 準確率從 77.6 升到 82.4，但 silent-wrong rate 只從 18.0 降到 15.2。準確率那條軸有進步，但危險的失敗模式幾乎沒動。

**Q6「你們的『可稽核』到什麼程度？」**
> 界線我寫得很嚴格。決定性執行和不變式求值，可以從儲存的 binding 完全重播。LLM 的選擇與綁定有 log，可以依記錄重跑。但從原始文字重新跑一次，我不保證走同一條路徑。另外我不驗證輸入的 provenance，只驗證 schema 和邊界，這也是 binding 前沿仍然存在的原因。

---

# 給簡報製作的執行備註

1. **K=3 標籤要一路貫穿**：Slide 10 與 backup B1 只要出現 K=3，就必須帶上「intent-oracle upper bound；K=1 才是全自動」的角標。論文端已於 2026-08-09 補進 `paper/main_v2.tex` 主結果表的 caption。
2. **可直接沿用的圖檔**：`paper/figures/fig1_pipeline_loop_v3.pdf`（S7）、`paper/figures/safety_utility_250q.png`（S12）、`paper/figures/libsweep_250q.png`（S14）。
3. **配色一致性**：整場固定 R-PAL、R-PAL⁺、VerifiQuant 三個顏色，從 S9 的階梯表開始就用同一組，讓 S12 的三個點不需要再解釋一次。
4. **顏色語意**：SWR 與 FNAR 是「越低越好」，建議所有這兩欄用同一種「風險紅」深淺，避免觀眾在 S10 誤讀成 VerifiQuant 全面落後。
5. **口說稿字數**：以每分鐘約 230 字估算，全場主軸約 4,100 字，對應 18 分鐘。壓縮到 12 分鐘時刪 S5、S8，並把 S13 的成本段縮成一句。
