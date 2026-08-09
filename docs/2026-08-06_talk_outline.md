# VerifiQuant 口頭報告大綱（依 `paper/main_v2.tex` 8 頁版逐節展開）

**修訂紀錄**
- `2026-08-06` 初版：17 張主軸 + 8 張 backup。
- `2026-08-09` v2 擴充：新增 S9（三個系統 = 三組儀器）、Part II（V1 → V2 敘事轉變）、
  Part III（審稿視角的優缺點綜合比較與缺口分級），backup 從 8 張擴到 14 張，
  新增 Appendix A–D（provenance、被砍內容庫存、統計細節、可離線補算的實驗）。
  主軸投影片重新編號（舊 S9–S17 → 新 S10–S18）。

**單一來源**：`paper/main_v2.tex`（2026-08-09 版）。所有進投影片的數字都取自該檔內嵌的 result macro。
Part II 與 Appendix B 會引用 **V1 才有、V2 因頁數被砍掉** 的數字，這些一律標明來源，
且**不得混進主軸投影片**，只能在 Q&A 取用。

**時間設計**：主軸 18 張投影片，約 19 分鐘；Part II／Part III 共 5 張，只在口試與內部報告進主軸；
14 張 backup 只在 Q&A 取用。

| 版本 | 長度 | 內容 |
|---|---|---|
| **會議版** | 18 分 | S1–S18，刪 S5、S8（S8 移 backup），S14 成本只講一句 |
| **標準版** | 19 分 | S1–S18 全上 |
| **口試版** | 30 分 | S1–S18 ＋ Part II（S19–S20）＋ Part III（S21–S23），S10 後加一張 trap 實例走查 |
| **內部／組會版** | 40 分 | 口試版 ＋ backup B1／B2／B3／B9／B10 併入主軸 |

---

## 全場唯一一句話（每張投影片都要能連回這句）

> 乾淨題目上，governance 不會讓你更準；它的價值出現在「這題根本不該給出數字」的時候。

## 全場心智模型（S9 會正式展開，但講者要從 S1 就守住）

三個系統不是三個競爭者，是**一組儀器**。每一個負責隔離出一個變因。

| 角色 | 系統 | 它單獨證明的一件事 |
|---|---|---|
| **Execution control** | R-PAL | 光是「決定性執行 + 拿到正確的公式」就已經買到非常多（clean 96.8%） |
| **Prompted-governance control** | R-PAL⁺ | **描述治理不等於執行治理**（clean 一樣好，trap 上錯誤接受率是兩倍以上） |
| **Executed-governance system** | VerifiQuant | 把前兩件事組成一個可部署的架構，並且付出可量化的代價 |

這個框架決定了三件事，講者要能隨時回到它：
1. **R-PAL 不是要被打敗的對手**，它是用來把 gain 從「檢索」和「執行」身上扣掉的減項。
2. **R-PAL⁺ 才是載重 baseline**。它拿到完整的 M/N/F/E/I 分類法當 prompt，所以它一贏，
   我們的貢獻就不是分類法本身；它一輸，剩下的唯一解釋就是「執行」。
3. **VerifiQuant 不主張最準**。它主張的是一個可歸因、可重播、可定價的邊界。

---

---

# Part I — 主軸投影片（S1–S18）

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

**版面**：四列，每列一個 RQ + 一句話答案（答案先用灰色，講到時再變色）。**這張投影片在 S11、S12、S15、S16 各回來一次**。

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

> **講者注意**：第三列是審稿人最容易追打的一列（見 Part III 的 G1）。
> 如果被問「你有沒有實測 selective prediction」，答案是**有，就在這一段**：
> 論文已納入 SC@5 的門檻掃描。細節與界線見 backup **B9**。

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

> **講者注意**：審稿人明確指出「缺少具體 FIC 範例」（Part III 的 G3）。
> 這張投影片右側**強烈建議直接放一張真實的卡**（backup **B12** 已備妥版面），
> 口試場合直接放進主軸，不要只放結構圖。

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
  **blocking 等級是 contract 在 build time 宣告的**，runtime 只決定「這題留下哪些未解決的 hint」，
  不決定嚴重度。這一點被審稿人點名要求操作化，判準與邊界案例見 backup **B13**。
- **C**：帶稽核日誌的決定性執行。
- 檢索是純詞彙式：SQLite FTS5 BM25，**沒有任何一處被訓練或針對評測集調參**，且 VerifiQuant、R-PAL、R-PAL⁺ 用**同一個 retriever、同一個 top-4 深度**。
- Abstention 不是 fallback：五種漏斗條件可以終止於結構化的非答案出口。
- 稽核性的精確範圍：決定性執行與 invariant 求值可從儲存的 binding **完全重播**；LLM 的選擇與綁定決策有 log，可依記錄重跑；但**從原始文字重新跑一次不保證走同一條路徑**。

**表格**：無（用論文 Figure 1）。

**口說稿（約 90 秒）**：
> 第二塊是可執行的診斷漏斗。一個問題進來，gate 依序觸發，每一關都產生一份 DiagnosticReport。
> M 和 N 問的是：是不是恰好存在一張適用的 contract。出口是結構化的釐清請求，或者範圍外拒答。F 和 E 問的是：輸入完整嗎，滿不滿足 contract 自己宣告的決定性檢查。出口是要求補值，或者指名被違反的那條規則。I 問的是：還沒解決的歧義，會改變計算結果，還是只改變詮釋。前者擋下來，後者帶警告放行，而且**嚴重度是卡片在建庫時就宣告好的，不是 runtime 現場判的**。最後 C 才做決定性執行，並留下稽核日誌。
> 這裡有兩個我要主動說明的設計選擇。第一，檢索是純詞彙式的 BM25，沒有任何一處針對評測集調過參數，而且 VerifiQuant 和後面兩個 baseline 共用同一個 retriever、同一個 top-4 深度，所以檢索品質是被固定住的變數，不是我這邊可以優化的東西。
> 第二，關於稽核性我的說法有嚴格界線。決定性執行和不變式求值，可以從儲存的 binding 完全重播。LLM 的選擇和綁定有留下 log，可以依記錄重跑。但是從原始文字重新跑一次，我**不保證**它會走同一條路徑，這一點我在論文裡有明確寫出來。

---

## Slide 8 — 方法三：有界修復與 RepairBench（會議版移 backup）

**標題**：Bounded Repair: A Patch Must Match a Declared Identity

**版面**：左邊流程（hard ambiguity → AST-bounded transform → numerical cross-verify → accept/reject），右邊表格。

**內容要點**：
- 當 hard ambiguity 改變了該做的計算，自由重新生成等於把剛關上的門再打開。
- 因此修復採用 **AST-bounded transform**：接受前，補丁的數值效果必須在多組取樣輸入上，符合 contract 宣告的代數關係。
- **誠實界線**：這是 bounded numerical validation，**不是形式等價**。它抓得到單邊錯誤，但一個錯誤的 invariant 配上一個互相抵消的補丁會通過。
- **第二層誠實界線（審稿人點出、論文目前沒寫）**：這 906 個變異體是**非適應性**的，
  它們不知道驗證器長什麼樣。一個知道 AST 節點上界與 invariant 形式的適應性對手沒有被測過。
  見 Part III 的 G4。
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
> 我要主動說清楚它的界線：這是有界的數值驗證，不是形式等價證明。一個錯誤的不變式，配上一個剛好互相抵消的補丁，是會通過的。而且這批變異體是非適應性的，一個知道驗證器內部參數的對手，我沒有測過。

---

## Slide 9 — 三個系統 = 三組儀器（NEW）

**標題**：Three Systems, Three Instruments

**版面**：三欄，每欄一個系統。上方是角色名稱（大字），中間是「它拿到什麼」，
下方是「它單獨證明什麼」。三欄用整場固定的三個顏色。

**內容要點**：

| | **R-PAL** | **R-PAL⁺** | **VerifiQuant** |
|---|---|---|---|
| 角色 | Execution control | Prompted-governance control | Executed-governance system |
| 拿到什麼 | 同一組 top-4 contract、決定性執行 | 上述 ＋ **完整 M/N/F/E/I 分類法寫進 prompt** | 上述 ＋ **gate 真的被執行** |
| 它單獨證明什麼 | 「執行 + 正確檢索」本身就買到很多 | **描述治理 ≠ 執行治理** | 把前兩者組成可部署架構，並定價 |
| 對本文的功能 | 把 gain 從檢索與執行身上**扣掉** | 把 gain 從**分類法本身**身上扣掉 | 剩下的殘差就是「執行」的貢獻 |

**這張投影片要建立的三個約束**：
1. **R-PAL 不是對手，是減項。** 它贏 clean set 是設計的一部分，不是我們的失敗。
2. **R-PAL⁺ 是載重 baseline。** 它拿到跟我們一模一樣的分類法。它一贏，我們的貢獻就不是分類法；
   它一輸，剩下的唯一解釋就是執行。
3. **VerifiQuant 不主張最準。** 它主張一個可歸因、可重播、可定價的邊界。

**口說稿（約 75 秒）**：
> 在進實驗之前，我要先把三個系統的角色講清楚，因為接下來每一張結果投影片都建立在這個框架上。
> 這三個系統不是三個競爭者，它們是一組儀器，每一個負責隔離出一個變因。
> R-PAL 是 execution control。它拿到跟我完全相同的 top-4 contract，自己寫程式決定性執行，但沒有任何 gate。它告訴我們的是：光是「執行」加上「拿到正確的公式」，就已經買到非常多。所以 R-PAL 不是我要打敗的對手，它是我用來把 gain 從檢索和執行身上扣掉的減項。
> R-PAL⁺ 是 prompted-governance control。我把**完整的** M/N/F/E/I 治理分類法整套寫進它的 prompt，要求它在生成程式前明確決定要執行、拒絕，還是澄清。它有分類法，但沒有可執行的 gate。它存在的目的只有一個：如果它贏，我的貢獻就不是這套分類法本身。
> VerifiQuant 是 executed-governance system。它是把前面兩件事組成一個可部署架構的主體。
> 所以整場的邏輯很簡單：R-PAL 扣掉檢索與執行的功勞，R-PAL⁺ 扣掉分類法的功勞，剩下的殘差就是「執行」這件事的貢獻。

---

## Slide 10 — 實驗設計：把變因鎖死

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
- Trap 由決定性運算子從乾淨題目導出：**F** 移除必要輸入、**E** 注入違反宣告檢查的值、**I** 刪掉宣告過的釐清資訊、**N** 換成範圍外的請求。標籤**就是**運算子。平衡 80 題（各 20），來自 52 個 base question，每個 base 最多 2 個運算子。
- **M 不做量化宣稱**，因為 under-specified 的題目仍保留帶指標的 context，無法產生可靠的 trap。
- **K=1 是唯一的可部署設定，也是所有 trap 評測使用的設定。**
- **K=3 是理想化的可回復性分析**：受限 oracle 可以看到題目的 intent specification（ground-truth 計算程式碼），但**永遠看不到數值答案**，且只能改寫題目與 context。

**口說稿（約 90 秒）**：
> 承接上一張的儀器框架，我把它做成一個階梯。CoT 什麼都沒有。PAL 有決定性執行但沒有檢索。R-PAL 拿到跟 VerifiQuant **完全相同**的 top-4 contract。R-PAL⁺ 再加上完整的治理分類法當 prompt。VerifiQuant 再加上真的被執行的 gate。
> 所以 backbone、題目、檢索、執行環境、計分器全部一樣，最後只剩下一個差別：**這些檢查到底是被執行，還是只被描述**。
> 資料這邊要說清楚：contract library 是由這個 benchmark 的參考函式建成的，所以乾淨集在建構上就是 in-library。我量的是 routing、binding 和 gating，不是未見公式的推導能力，這一點我不會含糊過去。
> Trap 這邊，我用決定性運算子從乾淨題目導出：F 拿掉必要輸入，E 注入違反宣告檢查的值，I 刪掉釐清資訊，N 換成範圍外的請求。標籤就是運算子本身，80 題來自 52 個不同的 base question，避免同一題被重複計分。M 我不做任何量化宣稱，因為 under-specified 的題目仍然保留帶指標的 context，做不出可靠的 trap，這一點我在論文裡明說。
> 最後，K=1 是唯一可部署的設定，也是**所有** trap 評測使用的設定。K=3 是一個理想化的上界分析，它的 oracle 可以看到題目的意圖規格，但永遠看不到數值答案。

---

## Slide 11 — RQ1：乾淨題目上，我們沒有比較準

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

## Slide 12 — RQ2：主結果

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

**I 的殘差要主動拆開講**：20 個 I trap 裡，**9 個被降級成 soft warning、4 個直接通過**。
所以我們在 I 上的殘差是**嚴重度誤判**，不是「沒發現有歧義」。R-PAL⁺ 的側寫剛好相反
（F 100%、N 88%，但 I 崩到 10%），它是**根本沒看見**那個歧義。

**口說稿（約 105 秒）**：
> 這是本文的主結果。在這 80 題本來就不該有數字的輸入上，裸的 R-PAL 有 48.8% 的時候會硬湊出一支程式，回傳一個有信心的數字。
> 接著關鍵的一步：我把**完整的**治理分類寫進 R-PAL 的 prompt 裡，它只降到 27.5%。而把同樣這些檢查**執行**出來，VerifiQuant 是 11.3%，差距 16.3 個百分點，以 base question 做 cluster 的 95% 信賴區間是 5.8 到 26.4。而且它給出正確結構化出口的比例也高得多。
> 這一格同時回答了三件事：**執行不等於治理，拿得到檢索不等於治理，而把治理寫在 prompt 裡，也不等於執行治理。**
> 下面那張表說明差距集中在哪裡。在 E 和 I，也就是正確與否取決於一個邊界或一個慣例的時候，裸的 R-PAL 幾乎每次都硬答，連加了分類的版本在 I 上都還有 80%，而我們的執行檢查守得住。
> 而且這兩個系統在 I 上的失敗**性質不同**。我們的 20 個 I trap 裡，有 9 個被降級成軟警告、4 個直接通過，所以我們的殘差是嚴重度誤判，是漏斗內部的分級問題。R-PAL⁺ 的 I 診斷率只有 10%，它是根本沒看見那個歧義存在。
> 我也要講對我們不利的兩列。在 F 和 N 上，我們其實**略遜於** prompt 版本，F 是 10% 對 0%，N 是 5% 對 0%。原因不是 gate 沒跑，而是 binding 那一關：模型把缺掉的欄位用一個看起來合理的鄰近值填了進去，完整性檢查看到的就是一個滿足的 schema。這正是我等一下會定位的三個 runtime 前沿之一。
> 另外一個誠實的補充：這裡的 trap 是 contract 導出的，所以這是一個**針對機制的壓力測試**，不是自然部署發生率的估計。這也是為什麼我還需要 RQ4。

---

## Slide 13 — RQ3-a：安全與效用的取捨平面

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

> **口試版加分做法**：這張平面圖是放 backup **B9**（信心門檻式 selective baseline）的最佳位置。
> 如果被問到 selective prediction，直接在同一張圖上多畫一條曲線。

**口說稿（約 60 秒）**：
> RQ3 問這份安全來自哪裡、代價是什麼。這張圖把三個自主系統放在同一個平面上，橫軸是乾淨題目的 coverage，縱軸是 trap 上的錯誤數字接受率。兩軸我刻意分開畫，因為我不想假設任何一個 clean 對 trap 的混合比例，那會是我自己編出來的部署情境。
> 三個點落在一條單調的前沿上。裸 R-PAL 幾乎全答，但接受了將近一半的 trap。加了 prompt 分類往左下移動一些。把 gate 執行出來，走到 11.3% 的 FNAR，代價是 84% 的 coverage。
> 這份安全不是免費的。相對於純執行，我放棄了大約 15 個百分點的乾淨題 coverage。

---

## Slide 14 — RQ3-b：Gate 階梯與成本

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

**成本結構的重點**：F/E 只花 +0.2 秒，因為它們是執行的 Python，不是模型呼叫。
成本幾乎全部集中在 **I gate 的 +5.9 秒**，因為它引入第三次模型呼叫。
這正是審稿人說「I gate 貢獻大部分延遲且非決定性」的地方（Part III 的 G5），
而 hint trigger 那一階就是把這一步部分機械化的方向。

**口說稿（約 75 秒）**：
> 接著是機制上的兩個視角。第一個是 gate 的累加消融，同一個 backbone、同一組 250 題，一層一層把控制加上去。
> 你會看到 silent-wrong rate 每一階都在降，從 CoT 的 18% 一路降到全漏斗的 4%，而準確率在 I 和 M/N 這兩階往下掉。這正是設計上想要的形狀：**gate 是用 coverage 去買安全，不是用 accuracy 去買。** 掉下去的那些題目變成了 abstention，不是變成錯答。
> 這裡要說明一個技術細節：這是一次獨立的 gate-disable run，它全漏斗那一階是 4.0%，跟 canonical run 的 2.8% 不同，所以我主張的是同一次 run 內的趨勢，不是絕對數值相等。
> 同一個階梯也幫我把成本標出來。選擇、綁定加執行是每題 6.6 秒。加上 F 和 E 只多 0.2 秒，因為那些是執行的 Python，不是模型呼叫。成本幾乎全部集中在 I，因為它多一次模型呼叫，一次就是 5.9 秒。全漏斗 12.2 秒，對照 R-PAL 的 10.5 秒，大約 1.2 倍，而且遠低於 self-consistency K=5 的 38 秒。
> 這一格也指出我們的下一步在哪裡：I gate 同時是最貴的和唯一非決定性的一關，最後那一階的 hint trigger 就是把它部分機械化的方向。

---

## Slide 15 — RQ3-c：庫覆蓋掃描與 OOD 的誠實代價

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

> **講者注意**：這是審稿人最重的一項扣分（Part III 的 G2）。
> 最強的補充證據在 backup **B10**（near-OOD 24 probes），它是 V1 有、V2 因頁數被砍的內容。
> 那組數字顯示，當目標公式家族被整個移出庫時，R-PAL 的 force-fit 率是 **100%**，
> VerifiQuant 是 **0%**。它證明「範圍訊號」是真的，不是保守度的副作用。

**口說稿（約 75 秒）**：
> 第二個機制視角，是把「範圍訊號」變成一個可量測的性質。我在五個不同大小的 contract library 上重建卡片庫，每次都重跑全部 250 題。左圖的重點是：**在每一個庫大小下，abstention 都是選擇性的**。庫外題目的拒答率維持在八到九成，庫內題目維持在一成上下，全程有 67 到 84 個百分點的差距。而且庫成長十倍的過程中，庫內準確率是平的。
> 唯一可見的代價很誠實：庫外的 silent wrong 從 1.3% 微升到 4.8%，因為庫越密，語意鄰近的 contract 就越容易被硬湊上去。
> 右邊這張表是我最想主動講的。在固定的 199 題庫外題目上，matched 的執行 baseline 幾乎全答，而且靠參數知識答對了 95%。VerifiQuant 拒答了 88.9%，只答對 7%。所以請注意，**VerifiQuant 在庫外並沒有比較準**，它自己在庫外的 selective accuracy 是 63.6%，比 baseline 的 95.5% 低，而且它的殘留 silent-wrong 也沒有比較好。這是最限制我們主張的一個結果。
> 我們買到的不是庫外的正確率，而是一條**已驗證與未驗證的分界線**。那個 baseline 在這裡很準，靠的正是參數知識，而參數知識錯的時候，完全不帶訊號。

---

## Slide 16 — RQ4：獨立人類標註的外部效度

**標題**：RQ4 — An Independent Human Panel, and the Bill It Presents

**版面**：左邊標註流程（三個 blind 標記），右邊結果表。

**內容要點（流程）**：
- 一個獨立的 GPT-5.2 作者把 FinanceReasoning 主題改寫成 80 題自然語氣的查詢，**沒有**接觸我們的 contract、M/N/F/E/I 分類、標準答案或任何系統輸出。作者模型刻意不同於所有系統的 solver backbone（Gemini 2.5 Flash），使「凌亂程度」與「哪些題對 solver 容易」不相關。
- 五位受過金融訓練的標註者，對來源題目、生成條件、所有系統輸出全部 blind，各自獨立指派四種行動之一並附一句理由。
- **多數標籤在任何系統被評分之前就凍結**。80 題全部第一輪就達成多數，不需要仲裁。
- 信度：**Fleiss' κ = 0.84**（almost perfect）。多數標籤分布：34 answer、20 request-info、10 clarify、16 out-of-scope，也就是 **80 題裡有 46 題不該只給一個數字**。

**表格內容（論文 Table 5）**：

| System | Human-FNAR | Match | Sel. Acc |
|---|---|---|---|
| CoT (single-shot) | 21.7% | 87.5% | 11.4% |
| R-PAL (top-4 retrieval) | 56.5% | 67.5% | 8.3% |
| R-PAL⁺ (taxonomy prompt) | 13.0% | 66.3% | 14.3% |
| **VerifiQuant (K=1)** | **8.7%** | 32.5% | 9.1% |

**必須主動說的三件事**：
1. 這 46 題上實際的錯誤接受數是 4、6、10、26，所以**只有跟裸 R-PAL 的差距個別顯著**（Fisher exact p < 10⁻⁴），VerifiQuant 對 R-PAL⁺ 是 p = .74。RQ4 建立的是**方向的外部效度**，不是第二次有檢定力的比較。
2. VerifiQuant 的 Match 只有 32.5%，答題數 11/80，在人類判定可答的 34 題裡只答了 7 題。
3. 27 次短少裡有 **24 次是 top-4 檢索根本沒撈到適用的 contract**（例如 LIFO 成本題只撈到 FIFO 的卡），所以它是正確地拒答而不是硬湊。這是可歸因的 retrieval 覆蓋成本，不是隨機分布的謹慎。

> **這是全篇最重要的「下一步」訊號**：24/27 的短少是**檢索**，不是**治理**。
> 換一個更好的 retriever，是唯一一個能同時提升 coverage 而不動 FNAR 的方向。
> 這正是審稿人 G2 的核心訴求，見 Part III 的行動表 A5。

**口說稿（約 90 秒）**：
> RQ4 處理一個很直接的質疑：那 80 題 trap 是我自己的 contract 導出的，標籤會不會只是把我自己的檢查重新編碼一次？
> 所以我做了一組外部效度檢驗。一個獨立的 GPT-5.2 作者把題目改寫成 80 題自然語氣的查詢，它看不到我們的 contract、分類法、標準答案或任何系統輸出，而且它刻意跟所有系統的 solver 用不同的模型。然後五位受過金融訓練的標註者，對來源、生成條件、系統輸出全部 blind，各自獨立標行動。多數標籤在任何系統被評分之前就凍結。Fleiss kappa 是 0.84，屬於 almost perfect，80 題全部第一輪就達成多數。結果是 80 題裡有 46 題不該只給一個數字。
> 結果表上，VerifiQuant 的人類定義 FNAR 是 8.7%，最低，順序跟 Panel B 完全一致。
> 但我要主動講三件事。第一，這 46 題上實際的數字是 4、6、10、26 次，所以只有跟裸 R-PAL 的差距個別顯著，我們對 prompt 版本的 p 值是 0.74。RQ4 建立的是**方向的外部效度**，不是第二次有檢定力的比較。
> 第二，我們的 Match 只有 32.5%，人類認為可答的 34 題我們只答了 7 題。第三，也是最重要的：27 次短少裡有 24 次，是 top-4 檢索根本沒有撈到適用的 contract，例如一題 LIFO 成本問題只撈到 FIFO 的卡。所以那是正確地拒答，不是隨機的謹慎，但在這組資料上，同一個機制也把大部分效用一起拿走了。
> 而這件事對我來說是好消息，因為它指出瓶頸在**檢索**，不在**治理**。這是我下一步唯一想做的事情。

---

## Slide 17 — 邊界與失敗分析

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

## Slide 18 — 結論

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

---

# Part II — V1 → V2 的敘事轉變（口試／組會版進主軸，會議版放 backup）

> **為什麼要講這一段**：口試委員與組會聽眾看過 V1。如果不主動說明轉變，
> 他們會以為主結果變弱了（93.6% 掉到 81.2%）。實際發生的事情相反：
> **主張變窄，但證據變硬**。這一段的功能是把「數字變小」翻譯成「宣稱變可辯護」。

## Slide 19 — 一年內我們把主張收窄了兩次

**標題**：What We Stopped Claiming, and Why

**版面**：兩欄對照表，中間一條垂直的「收窄」箭頭。

**表格內容（核心對照）**：

| 面向 | **V1**（約 11 頁，至 2026-07-30） | **V2**（8 頁，2026-08-09） |
|---|---|---|
| 敘事裝置 | **4-lens balance**：只有 VQ 四個透鏡都守得住 | **execution vs description**：唯一變因是檢查有沒有被執行 |
| 主結果 | 分散在四個 lens，沒有單一格 | 單一格：**FNAR 27.5% → 11.3%** |
| Results 開場 | clean lens（先講我們跟 R-PAL 打平） | **RQ1「我們輸」**（先講對自己不利的） |
| 載重 baseline | CoT ×6 grid ＋ CoT+oracle ＋ JPM MAS，R-PAL 後期才加 | **R-PAL⁺ 是主角控制組**，CoT／PAL 降為階梯的低階 |
| 頭條數字 | VQ 93.6%、SWR 3.2%，對 CoT 5.6× | K=1 的 81.2%／2.8%，頭條讓給 trap 的 FNAR |
| K=3 | 主表數字 | **intent-oracle 上界**，K=1 才是 canonical |
| OOD | 一度當作 advantage（selective abstention） | **明確降級為 disclosed trade-off**（selective acc 63.6% vs 95.5%） |
| 外部效度 | 無，trap 全部由自己的 contract 導出 | **RQ4 80 題人類標註**，5 位標註者，κ = 0.84，標籤評分前凍結 |
| 結論句 | 無條件（governance 值得） | **條件句**（一個沒有依據的數字比拒答貴時才值得） |

**口說稿（約 75 秒）**：
> 這一段是給看過前一版的人的。表面上我們的頭條數字從 93.6% 變成 81.2%，看起來像退步。實際發生的是相反的事情：主張變窄了，但證據變硬了。
> V1 的敘事裝置是「四個透鏡的平衡」，主張只有 VerifiQuant 四個透鏡都守得住。這個框架的問題是它沒有單一的分離點，而且四個透鏡各自用不同的對照系統，讀者很難檢查。
> V2 換成一個單一的隔離設計：唯一的變因是檢查有沒有被執行。頭條也跟著換掉，從乾淨集的準確率換成 trap 上的錯誤數字接受率。
> 三個最重要的收窄是：K=3 從主表數字降為 oracle 上界；OOD 從 advantage 降為明確揭露的 trade-off；結論句從無條件改成條件句。
> 同時我們也加了一件 V1 沒有的東西，就是完全獨立的人類標註外部效度集。

---

## Slide 20 — 轉變的三個觸發點，與它的代價

**標題**：Three Runs That Forced the Reframe

**版面**：上半三個觸發事件（時間軸），下半「被砍掉的內容」清單。

**三個觸發事件**：

| 日期 | 事件 | 對敘事的衝擊 |
|---|---|---|
| 2026-07-29 | **R-PAL matched baseline 跑出來**（`docs/2026-07-29_retrieval_pal_matched_baseline.md`） | R-PAL 在 clean set 上追平甚至超過 VQ。「beat CoT」的敘事當場失效 |
| 2026-08-04 | **R-PAL⁺ 跑完 clean 250**（`docs/2026-08-04_rpal_plus_clean250.md`） | prompted governance 在 clean 上一樣好（92.0／0.9），但 trap 上是 27.5%。**真正的分離點被找到** |
| 2026-08-03 | **RQ4 人類標註完成**（`docs/2026-08-03_rq4_messy_external_validity.md`） | 循環論證的質疑第一次有了外部答案，同時也暴露 retrieval 是瓶頸 |

**V2 為了 8 頁付出的代價（全部進了 Appendix B 的庫存清單）**：

| 被砍的內容 | V1 的數字 | 為什麼可惜 |
|---|---|---|
| **near-OOD 24 probes 專節** | R-PAL force-fit **100%**，VQ **0%**（bare）／**8.3%**（near-miss） | 這是 scope signal 最強的單一證據，比 trap-N 有力得多 |
| CoT OOD selectivity 對照 | VQ 選擇性 gap **69.3** vs CoT **11.3** | 直接反駁「只是換一個更保守的門檻」 |
| risk–coverage 與 perfect-rejector 分析 | VQ 實際操作點超過單次 baseline 的完美拒答上界 | 對 selective-prediction 審稿人最有說服力 |
| MAS 深入分析 | 剩 related work 一句話 | 失去一個金融領域讀者熟悉的對照 |
| Deployment considerations 專節 | 整節刪除 | ICAIF 讀者最在意的落地段落沒了 |
| difficulty tier 分層 | medium SWR 2.8% vs hard 4.3%，比值 5.0× / 6.7× | Figure 2 讓位給 safety–utility 平面 |
| hint-trigger canonical run | SWR **1.2%**、SelAcc **98.7%** | 只剩 gate ladder 最後一階（3.6%），最好的數字沒露出 |

**口說稿（約 75 秒）**：
> 這個轉變不是寫作偏好，是被三次實驗逼出來的。
> 第一次是七月底 R-PAL 的 matched baseline 跑出來，它在乾淨集上追平甚至超過我們。那一刻「打敗 CoT」這條敘事就失效了，因為我沒辦法說 gain 是治理賺到的。
> 第二次是八月初 R-PAL⁺ 跑完乾淨集，它在乾淨集上一樣好，但 trap 上是 27.5%。這一次才真正找到分離點，而且分離點的位置跟我原本以為的不一樣。
> 第三次是人類標註完成，它同時做了兩件事：回答了循環論證的質疑，也告訴我瓶頸在檢索。
> 但這個轉變有代價。為了壓到八頁，我砍掉了幾塊我其實很喜歡的內容。最可惜的是 near-OOD 那一節，那組數字顯示當目標公式家族整個被移出庫時，R-PAL 的 force-fit 率是 100%，我們是 0%。它是範圍訊號最強的單一證據，現在只能放在 backup 裡。所以如果等一下有人問到 OOD 或問到保守度，我會直接把那張叫出來。

---

---

# Part III — 審稿視角：這篇現在的強項、缺口與回應（口試／組會版進主軸）

> **來源**：一份完整的 AI reviewer 審稿意見（Summary／Strengths／Weaknesses／Detailed Comments）。
> 本段把它的每一條對照到論文現況，分成三級：**已在論文中**、**便宜可補**、**真未來工作**。
> 這一段同時是投稿前的 checklist 與口試的防守稿。

## Slide 21 — 審稿人認可的四件事（以及哪張投影片在承載）

**標題**：What the Review Already Credits

**版面**：四列，每列左邊是強項，右邊是「承載它的投影片」。

| 審稿人認可的強項 | 承載它的投影片 | 講者注意 |
|---|---|---|
| **關注點分離**：LLM 只做選擇與綁定，所有金融算術與治理檢查在模型外決定性執行 | S6、S7 | 已經講夠，不要再加時間 |
| **受控的載重 baseline**：R-PAL 與 R-PAL⁺ 共用 retriever、top-k、backbone、執行環境、計分器，唯一變因是 gate 有沒有執行 | **S9、S10** | 這是審稿人給分最重的一項。**S9 是新加的，不要跳過** |
| **四層證據**：clean in-library、contract-derived traps、獨立人類標註集、庫覆蓋掃描 | S11、S12、S15、S16 | 「四層」這個說法可以直接借用，比「四個 RQ」更容易被記住 |
| **明確劃清主張界線**：明說不宣稱 clean-set 準確率優勢，明確列出三個 runtime 前沿 | S11、S17 | 誠實度本身被算成強項，這是保留 RQ1「我們輸」開場的理由 |

**一句話**：審稿人給分最高的地方，不是任何一個數字，而是**實驗設計的隔離度**與**主張的自我設限**。
所以 S9 與 S11 是全場不能砍的兩張。

---

## Slide 22 — 我們自己知道的缺口（分三級）

**標題**：Known Gaps, Triaged

**版面**：一張三色分級表。綠 = 已在論文中、黃 = 便宜可補、紅 = 真未來工作。

| # | 審稿意見 | 論文現況 | 分級 | 我的回應 |
|---|---|---|---|---|
| **G1** | 缺 selective prediction／conformal abstention 的**實測**對照 | **已補**（2026-08-10）：§2 selective prediction 段落納入 SC@5 門檻掃描，Limitations 同步改寫 | 🟢 已回答 | 零 API 呼叫，數字見 backup **B9**。仍缺 retrieval-matched 版與真正的 conformal 程序 |
| **G2** | Retrieval 只有 BM25，無 dense／hybrid，且 RQ4 顯示它是主要瓶頸 | 有深度掃描（top-4 94.8%、+8 → +2.4、+16 → +4.0），無 retriever 家族對照 | 🔴 真未來工作 | **最高 ROI 的下一步**。RQ4 的 24/27 短少就是檢索沒撈到 |
| **G3** | 缺具體 FIC／invariant 範例，難以判斷規格完整度與建庫可行性 | 論文只有結構描述，沒有完整範例卡 | 🟡 便宜可補 | 最便宜的一項。放一張真實的卡進 appendix，見 backup **B12** |
| **G4** | Bounded repair 的驗證是取樣的，對手可能設計繞過的補丁 | §3.3 已明說不是形式等價 | 🟢 已揭露，但可再誠實一級 | 現有 906 個變異體是**非適應性**的。適應性對手未測，應寫進 Limitations |
| **G5** | I gate 靠 LLM critic，非決定性且貢獻大部分延遲 | Gate ladder 已量化（+5.9 s），hint trigger 是部分機械化但非 held-out | 🟢 已揭露 + 有方向 | S14 已經主動講。held-out 驗證留給下一篇 |
| **G6** | I gate 的 hard／soft 判準需要更明確的操作化 | §3.2 只說「是否會改變計算」 | 🟡 便宜可補 | 補判準 + 兩個邊界案例。見 backup **B13** |
| **G7** | 1% 容差統一套用，某些金融情境需要更嚴或慣例感知的容差 | 已說明一個 scorer 套所有系統，且 FNAR 完全不依賴容差 | 🟡 便宜可補 | 已存 run 有 `numeric_output` 與 `gold`，**離線重算**即可給敏感度。見 backup **B11** |
| **G8** | OOD 效用損失嚴重，庫外 baseline 常能答對 | §5.3 主動揭露並列為最限制主張的結果 | 🟢 已揭露 | 補充證據在 backup **B10**（near-OOD，V1 被砍） |
| **G9** | 缺 neuro-symbolic 的小規模 head-to-head | 只有論述定位 | 🔴 真未來工作 | 可辯護：我們的形式化在離線，head-to-head 會測到另一條 pipeline 的翻譯品質 |
| **G10** | 單一 backbone 家族 | Limitations 已寫，且有 Flash→Pro 的 CoT 對照 | 🟢 已揭露 | 跨 backbone 是下一篇 |

**口說稿（約 90 秒）**：
> 這一張是我自己的缺口清單，我把它分成三級。
> 綠色是論文已經誠實揭露的，包含 OOD 的效用損失、修復驗證不是形式等價、I gate 的非決定性、以及單一 backbone。這些我不打算辯護，它們就是這篇的邊界。
> 黃色是便宜可補的，四項。第一，selective prediction 的實測對照，這是審稿人最在意的一項，而我發現我們現有的 self-consistency run 已經存了五個樣本和一致度，可以**不打任何一次 API** 就掃出一條門檻曲線，等一下我會給數字。第二，一張完整的 FIC 範例卡，這是最便宜的一項。第三，hard 跟 soft 的判準操作化。第四，容差的敏感度分析，因為我們已存的輸出檔裡有數值和標準答案，離線重算就好。
> 紅色是兩項真的未來工作。第一是換 retriever，這是最高 ROI 的一項，因為 RQ4 告訴我瓶頸在那裡。第二是 neuro-symbolic 的正面對照。
> 我想強調的是，這張表上沒有任何一項會推翻主結果。主結果是一個受控比較，它不依賴檢索品質、不依賴容差、也不依賴修復模組。

---

## Slide 23 — 行動清單（依 ROI 排序）

**標題**：What We Do Next, in Order

**版面**：三個時程欄（投稿前／camera-ready／下一篇），每格 1–2 個動作。

### 🟢 投稿前可做：零外部 API 呼叫，純離線重算

| # | 動作 | 回答哪一項 | 成本 | 產出 |
|---|---|---|---|---|
| ~~**A1**~~ ✅ | 從 `sc_baseline/sc5_gemini_25_flash.jsonl` 的 `share` 欄掃門檻 | **G1** | 已完成 2026-08-10 | 已寫進 §2，assert guard 已加。Figure 2 疊線仍可選做 |
| **A2** | 一張完整的 FIC 範例（inputs／code／invariants／declared hints）進 appendix | **G3** | < 1 小時 | 半欄的圖 |
| **A3** | 用已存的 `numeric_output`／`gold` 重跑 0.1% 與 benchmark 原生慣例的容差 | **G7** | < 1 小時 | 一個註腳 |
| **A4** | hard／soft 判準 + 兩個邊界案例寫進 §3.2 或 appendix | **G6** | < 1 小時 | 三句話 |
| **A4b** | Limitations 補一句「RepairBench 的變異體是非適應性的」 | **G4** | 10 分鐘 | 一句話 |

> **A1 的關鍵**：run 檔存了每題 5 個樣本 (`answers`) 與勝出者一致度 (`share`)，
> 所以掃門檻**完全不需要重新呼叫模型**。這是目前投報率最高的一項。

### 🟡 Camera-ready 或 rebuttal：需要 API 呼叫

| # | 動作 | 回答哪一項 | 估計呼叫數 |
|---|---|---|---|
| **A5** | 一個 dense／hybrid retriever，**只跑 RQ4 的 80 題**，看 24/27 的檢索短少能收回多少 | **G2**（最高 ROI） | 80 題 × 1 系統 |
| **A6** | SC@5 跑在 **R-PAL** 之上（而非無檢索 CoT），做嚴格 matched 的 selective baseline | **G1** 的嚴格版 | (250 + 80) × 5 ≈ 1,650 |
| **A7** | near-OOD 24 probes 復活成 appendix 一段 | **G8** | 0（V1 已跑完） |

### 🔴 下一篇（ACL）

| # | 動作 | 回答哪一項 |
|---|---|---|
| **A8** | 適應性對手攻擊 RepairBench 驗證器 | **G4** |
| **A9** | held-out trigger-phrase 集驗證 hint trigger | **G5** |
| **A10** | neuro-symbolic 共享子集 head-to-head | **G9** |
| **A11** | 跨 backbone／跨 vendor 複製 | **G10** |
| **A12** | 手工建構的 I_HARD 集（補足 6 → 10+），見 backup **B14** | I_HARD 樣本過小 |

**口說稿（約 60 秒）**：
> 最後是行動清單，依投報率排序。
> 投稿前這一欄我特別想強調，因為它全部不需要打任何一次 API。最重要的是 A1：我們的 self-consistency run 已經把每題五個樣本和一致度存下來了，所以我可以直接掃門檻，畫出一條信心門檻式的 risk–coverage 曲線，疊在原本的 safety–utility 平面上。這一項一小時內做得完，而它回答的是審稿人最在意的缺口。
> 需要 API 的那一欄，最高投報率的是 A5：換一個 dense 或 hybrid retriever，而且只跑 RQ4 的 80 題就好。因為 RQ4 已經精確告訴我，27 次短少裡有 24 次是檢索沒撈到，所以這 80 題是投報率最集中的地方，82 次呼叫就能知道我的 coverage 天花板在哪裡。
> 剩下的四項留給下一篇。

---

---

# Backup 投影片（只在 Q&A 取用）

| # | 標題 | 內容 | 對應的預期提問 |
|---|---|---|---|
| B1 | K ablation | K=1: 81.2/2.8/coverage 84.0/SelAcc 96.7 ・ K=2: 92.4/3.6/96.0/96.2 ・ K=3: 93.6/3.2/96.8/96.7。**oracle 看得到 intent spec，看不到數值答案** | 「K=3 是不是作弊？」 |
| B2 | 檢索深度 | 250 題乾淨集上，gold contract 中位數排名第 1，top-4 命中 94.8%，放寬到 8 只多 2.4 點、到 16 多 4.0 點。**故固定 k=4 而不調參**。dense/hybrid retriever 未測 | 「是不是 k 調小才贏？」 |
| B3 | 統計檢定 | R-PAL vs VQ K=3：acc Δ=+3.2, CI[−0.4, 6.8], McNemar p=.134；SWR Δ=−0.8, CI[−3.6, 2.0], p=.791。Panel B gap 16.3 點, base-clustered CI [5.8, 26.4]。RQ4 Fisher: vs 裸 R-PAL p<10⁻⁴, vs R-PAL⁺ p=.74 | 「有沒有做統計檢定？」 |
| B4 | 計分慣例 | 1% output-scale-normalized 容差，一個計分器套用到所有系統。R-PAL 若程式無法 parse、拋錯、逾時、或回「cannot solve」，一律算 abstention 不算 silent wrong。**FNAR 完全不依賴容差**，因為它只看有沒有吐數字 | 「容差是不是對你有利？」 |
| B5 | 人類標註規則 | 四標籤 top-down 決策：不是「算一個數字」→ out-of-scope；缺必要數值→ request-info；值都在但詮釋選擇會改變數字→ clarify；否則 answer。兩組對比明確訓練過。M-refusal 一律算 out-of-scope（保守讀法；改讀成 clarify 會把 Match 從 32.5% 拉到 35.0%） | 「標註標準是什麼？會不會偏袒你？」 |
| B6 | 為什麼 M 不做量化 | under-specified 的題目仍保留帶指標的 context，決定性運算子做不出可靠的 trap，因此對 M 不做任何數值宣稱 | 「六層漏斗為什麼只測四層？」 |
| B7 | 釋出的成果 | 250 張 contract library 與建庫的分階段 prompt、80 題人類評測集與 provenance、五位標註者各自的標籤檔、凍結的多數 ground truth、一致性報告、計分與彙總腳本與稽核軌跡（含標籤凍結時點） | 「可重現性？」 |
| B8 | 倫理 | 題目為合成、不含個資，只蒐集對題目的標籤與理由；自願參與，標註者匿名為 Annotator 1–5；經與機構諮詢認定不需 ethics-committee 審查 | 「有沒有 IRB？」 |
| **B9** | **信心門檻式 selective baseline** | 見下方專節。**已寫進論文 §2**（2026-08-10） | 「你有沒有跟 selective prediction／conformal abstention 比？」 |
| **B10** | **near-OOD force-fit（NEW，V1 被砍）** | 24 個 probe，目標公式家族從庫中移除。force-fit = 沒有適用公式時仍輸出有信心的數字。R-PAL：bare 100%（24/24）、near-miss 100%；VQ K=1：bare **0%**（全部走 N gate）、near-miss 8.3%（2/24，selection 前沿滑動）。來源 `verifiquant/data/runs/nearood/` | 「OOD 你們效用那麼差，範圍訊號是真的嗎？」 |
| **B11** | **容差敏感度（NEW，待跑）** | 已存 run 檔含 `numeric_output` 與 `gold`，可離線重跑 0.1%／原生慣例容差。**FNAR 本來就與容差無關**，所以敏感度只會動 Panel A，不動主結果 | 「1% 對金融來說太鬆了」 |
| **B12** | **一張完整的 FIC 範例（NEW，待做）** | 一張真卡：`fic_id`、適用/不適用條件、typed inputs、deterministic Python、numerical invariants、declared hints（含 hard/soft 與 transform_spec） | 「contract 到底長什麼樣？建庫可行嗎？」 |
| **B13** | **I gate hard／soft 判準（NEW，待寫）** | 判準：未解決的宣告歧義**會改變數值輸出** → I_HARD（阻擋，發非數值澄清）；只改變**詮釋或呈現** → I_SOFT（放行 + 警告）。**等級由卡片在 build time 宣告**，runtime 只判斷「哪些宣告的 hint 未被解決」。邊界案例應各給一個：期初/期末（改變數字 → hard）、名目/實質標示（多半只改詮釋 → soft） | 「hard 跟 soft 怎麼分？誰決定？」 |
| **B14** | **I_HARD 為什麼只有 6 題（NEW）** | 243 個 selected case 中 40 個 FIC 宣告 hard ambiguity，但只有 **6** 可用 phrase-deletion ground：17 個「宣告了但題目沒有可刪的片語」（慣例藏在 context），19 個「無對應片語」（method／basis choice，如 ROA 定義、cash-flow method）。所以 hard、會改變計算的歧義在可刪片語的 trap 裡天生稀少。補到 10+ 需要手工建構 | 「I_HARD 只有 6 題，結論可信嗎？」 |

---

## B9 專節 — 信心門檻式 selective baseline（**已於 2026-08-10 寫進論文 §2 selective prediction 段落**）

**這是什麼**：審稿人指出論文缺少 selective prediction／conformal abstention 的實測對照。
我們現有的 self-consistency run 已經把**每題 5 個樣本**與**勝出者一致度**存進
`verifiquant/data/runs/sc_baseline/sc5_gemini_25_flash.jsonl`
（欄位 `answers`（長度 5）、`share`、`winner`、`correct`、`gold`）。
一致度就是離散輸出上最標準的 non-conformity score，所以掃門檻 τ 完全不需要重新呼叫模型。

**掃出來的曲線（250 題乾淨集，與 `sc5_gemini_25_flash_summary.json` 的
`risk_coverage_points` 一致）**：

| 門檻 τ（要求的一致度） | Coverage | Sel. Acc | SWR |
|---|---|---|---|
| 0.0（無門檻） | 88.8%（222/250） | 91.0% | 8.0%（20/250） |
| ≥ 0.6 | 86.8% | 92.2% | 6.8% |
| ≥ 0.8 | 84.4% | 92.4% | 6.4% |
| **= 1.0（5/5 全數一致）** | **84.0%**（210/250） | **92.4%** | **6.4%**（16/250） |
| **VerifiQuant K=1（對照）** | **84.0%** | **96.7%** | **2.8%**（7/250） |

> **注意 τ=0 那一列不是 100% coverage**：250 題裡有 **28 題五次取樣都產不出可解析的數字**，
> 所以這個 baseline 的 coverage 上限本來就是 88.8%。計分沿用 run 自己的慣例
> （`winner is None` 不算作答）。另外查證過：加上論文計分器的 ×100 output-scale 等價後，
> 這四列數字**完全不變**，所以計分慣例的差異在這個 run 上不構成疑慮。

**兩個可以講的結論，但強度不同**：

1. **強、且與檢索無關的結論（建議主打）**：這個門檻**幾乎沒有可用的行程**。
   把 τ 從 0 拉到 1（最緊），coverage 只從 88.8% 移到 84.0%，SWR 只從 8.0% 降到 6.4%，
   整條門檻總共買到 **1.6 個百分點**。原因是 250 題裡有 **210 題是 5/5 全數一致**，
   分數本身已經沒有解析度可花。信心門檻買不到更多安全，因為模型在答錯的題目上一樣很一致。
   這正是「拒答應該由宣告的檢查驅動，而不是由一個分數驅動」的機制性證據。
2. **弱、且被混淆的結論（不要主打）**：在**完全相同的 84.0% coverage** 上，
   VerifiQuant 的 selective accuracy 是 96.7% 對 92.4%，SWR 是 2.8% 對 6.4%。
   **但這不是 matched 比較**，因為這個 SC baseline **沒有檢索**（它是 CoT 家族的 self-consistency）。
   R-PAL 單獨就有 96.8% 準確率，所以這個差距不能歸給治理。
   嚴格的 matched 版本是「SC@5 跑在 R-PAL 之上」，需要約 1,650 次呼叫（行動清單 A6）。
   論文裡這一段已經用「reference point rather than a matched comparison」明確標示這條界線。

**重現指令**（純離線；`winner is not None` 這個過濾是必要的）：

```bash
python3 -c "
import json
rows=[json.loads(l) for l in open('verifiquant/data/runs/sc_baseline/sc5_gemini_25_flash.jsonl')]
for tau in (0.0,0.6,0.8,1.0):
    a=[r for r in rows if r['winner'] is not None and r['share']>=tau]
    c=sum(1 for r in a if r['correct'])
    print(f'tau={tau}: cov={len(a)/len(rows):.3f} selacc={c/len(a):.3f} SWR={(len(a)-c)/len(rows):.3f}')
"
```

論文端的數字由 `scripts/assert_paper_numbers.py` 的
「Selective-prediction reference point」區塊守住，改動 run 或巨集會直接讓 guard 失敗。

**如果被當場問到，一句話版本**：
> 我有做，在 related work 那一段。我把 self-consistency 的一致度當分數掃門檻，
> 全部離線、不需要額外呼叫模型。結果是這個門檻幾乎沒有行程可走：250 題裡有 210 題模型五次完全一致，
> 所以從最鬆到最緊，coverage 只動 4.8 個百分點，silent-wrong 只從 8.0% 降到 6.4%。
> 在跟我們完全相同的 84% coverage 上，它的 selective accuracy 是 92.4%，我們是 96.7%。
> 但我會誠實說，那個 baseline 沒有檢索，所以它是一個參考點，不是 matched 比較。
> 真正的重點是機制：門檻式拒答在錯的題目上一樣有信心，而且它拒答時不會告訴你是哪一個檢查失敗。

---

# 預期提問與回答骨架

**Q1「你在乾淨集上輸給 baseline，這篇的貢獻是什麼？」**
> 貢獻正是那個「輸」被界定得很乾淨。我的 load-bearing baseline 拿到跟我完全一樣的 backbone、檢索、執行環境和計分器，還額外把整套治理分類寫進 prompt。唯一剩下的差別是檢查有沒有被執行。乾淨集上它贏，trap 上它接受的錯誤數字是我的兩倍以上。這組對照把 gain 定位在「執行」，而不是定位在分類法本身，也不是定位在準確率。

**Q2「trap 是你自己的 contract 導出的，這不是循環論證嗎？」**
> 這正是 RQ4 存在的理由。我另外做了一組 80 題，由一個獨立的模型改寫、由五位對所有系統 blind 的金融背景標註者標記、多數標籤在評分前凍結、Fleiss kappa 0.84。順序完全複製了 Panel B。我也誠實說明它的檢定力：只有跟裸 R-PAL 的差距個別顯著，所以我主張的是方向的外部效度，不是第二次有檢定力的比較。

**Q3「拒答率這麼高，實務上誰要用？」**
> 我不會主張所有情境都該用。我的結論是條件式的：當一個沒有依據的數字比一次拒答更昂貴時它值得，例如估值、法遵報告、面向客戶的試算。當 coverage 才是主要限制時，把分類寫進 prompt 更便宜，而且我把那條路能走多遠也量出來了（92.9% coverage、27.5% FNAR）。

**Q4「這只是換一個更保守的 threshold 而已吧？」**
> 不是，而且我有兩組證據。第一，庫覆蓋掃描顯示這個拒答在每個庫大小下都是**選擇性**的，庫外和庫內之間維持 67 到 84 個百分點的差距，這不是一個一律變保守的門檻做得到的。第二，我離線掃過一條真正的信心門檻曲線：250 題裡有 210 題模型五次完全一致，所以門檻很快就用完解析度，從最鬆到最緊 coverage 只動 4.8 個百分點，silent-wrong 只從 8.0% 降到 6.4%。另外，threshold 式的拒答不會告訴你是哪個檢查失敗，我的會指名是哪一條宣告的檢查、哪一個缺失欄位、哪一個未解決的慣例，而且可以從儲存的 binding 重播。

**Q5「換成更強的模型是不是就解決了？」**
> 我測過。Flash 換成 Gemini 2.5 Pro，單次 CoT 準確率從 77.6 升到 82.4，但 silent-wrong rate 只從 18.0 降到 15.2。準確率那條軸有進步，但危險的失敗模式幾乎沒動。

**Q6「你們的『可稽核』到什麼程度？」**
> 界線我寫得很嚴格。決定性執行和不變式求值，可以從儲存的 binding 完全重播。LLM 的選擇與綁定有 log，可以依記錄重跑。但從原始文字重新跑一次，我不保證走同一條路徑。另外我不驗證輸入的 provenance，只驗證 schema 和邊界，這也是 binding 前沿仍然存在的原因。

**Q7「OOD 你們效用損失這麼大，這套還有價值嗎？」（新增）**
> 我先承認這是最限制我主張的結果，庫外我們的 selective accuracy 是 63.6%，比 baseline 的 95.5% 低。但我要補一個 backup 的數字：當目標公式家族被整個移出庫的時候，那個 baseline 的 force-fit 率是 100%，我們是 0%。所以我們買到的不是庫外的正確率，是一條已驗證與未驗證的分界線。baseline 在庫外很準，靠的是參數知識，而參數知識錯的時候完全不帶訊號。要不要買這條分界線，是部署情境的性質。

**Q8「Retrieval 只用 BM25，換更好的 retriever 你的結論會不會垮？」（新增）**
> 不會垮，而且會變好。第一，主結果是受控比較，三個系統共用同一個 retriever，換掉它是三個系統一起換，分離點不受影響。第二，RQ4 明確告訴我 27 次短少裡有 24 次是檢索沒撈到，所以更好的 retriever 是直接把 coverage 加回來，而不會動到 FNAR。這是我目前投報率最高的下一步。

**Q9「你的 repair 驗證只是取樣，對手可以繞過吧？」（新增）**
> 可以，我不主張形式等價。而且我要說得比論文更清楚一點：那 906 個變異體是非適應性的，它們不知道驗證器的 AST 節點上界和 invariant 形式長什麼樣。一個知道這些參數的適應性對手，我沒有測過。目前擋得住的是無意的錯誤和結構性攻擊，不是一個有針對性的攻擊者。

---

---

# Appendix A — 主軸數字的 provenance（不進投影片，備查）

| 數字 | 來源 |
|---|---|
| Clean 250Q（VQ canonical） | `verifiquant/data/runs/paper_v2_250/results/vq_flash/` |
| R-PAL clean | `verifiquant/data/runs/retrieval_pal_baseline/rpal_top4_gemini_25_flash*.jsonl` |
| R-PAL⁺ clean | `verifiquant/data/runs/governance_eval/prompted_clean250_run1_summary.json` |
| 80-trap confirmatory | `verifiquant/data/runs/governance_confirmatory/confirmatory_summary.json` |
| Gate ladder + latency | `verifiquant/data/runs/gate_ladder/r{1..5}_*_summary.json` |
| Self-consistency K=5 | `verifiquant/data/runs/sc_baseline/sc5_gemini_25_flash*` |
| near-OOD（V1，已砍） | `verifiquant/data/runs/nearood/nearood_{vq,rpal}_run1_summary.json` |
| RQ4 messy set | `verifiquant/data/runs/messy_query_pilot/` |
| Trap 生成與 manifest | `verifiquant/pipeline/build_trap_set_v2.py`、`governance_confirmatory/trap_manifest.json` |

相關記錄文件：`docs/2026-07-12_experiment_record_master.md`（主帳）、
`docs/2026-07-29_phase_split_icaif_acl.md`（P0/P1/P2 與 ICAIF/ACL 邊界）、
`docs/2026-08-04_review_response_map.md`（上一輪審稿對應表）、
`docs/2026-08-03_rq4_messy_external_validity.md`（RQ4 方法與檢定力）。

---

# Appendix B — V1 被砍內容的庫存（隨時可召回）

| 內容 | 數字 | 檔案 |
|---|---|---|
| near-OOD 24 probes | R-PAL 100%/100% force-fit；VQ 0%/8.3% | `runs/nearood/` |
| CoT OOD selectivity | CoT abst OOD 60.3 / ID 49.0（gap 11.3）；VQ 88.9 / 19.6（gap 69.3） | `main_v2.tex` 巨集（body 未用） |
| hint-trigger canonical | acc 94.0%、SWR 1.2%、SelAcc 98.7%、abstain 12 | `main_v2.tex` 巨集（僅 TODO 註解） |
| difficulty tier | medium SWR 2.8% / hard 4.3%；vs CoT 5.0× / 6.7× | `main_v2.tex` 巨集 |
| MAS（JPMorgan 重現） | 50Q：86% acc、SWR 14%；原論文 0.46 Pass@1 | `runs/paper_v1/results/jpmorgan/` |
| risk–coverage / perfect rejector | VQ 實際操作點超過單次 baseline 的完美拒答上界 | `scripts/plot_risk_coverage.py` |
| Deployment considerations | 整節（V1 §7） | `paper/main_v1/main_body.tex` |

**使用規則**：這些一律只在 Q&A 出現，且開口必須先說「這一段因為頁數限制沒有進論文」。
不要在主軸投影片混用 V1 與 V2 的數字。

---

# Appendix C — 統計檢定細節（B3 的完整版）

| 比較 | 統計量 | 值 |
|---|---|---|
| R-PAL vs VQ (K=3)，clean accuracy | Δ / 95% CI / exact McNemar | +3.2 點 / [−0.4, 6.8] / p = .134 |
| R-PAL vs VQ (K=3)，clean SWR | Δ / 95% CI / p | −0.8 點 / [−3.6, 2.0] / p = .791 |
| R-PAL⁺ vs VQ，trap FNAR | Δ / base-clustered 95% CI | 16.3 點 / [5.8, 26.4] |
| VQ vs 裸 R-PAL，Human-FNAR（46 題） | Fisher exact | 4 vs 26，p < 10⁻⁴ |
| VQ vs R-PAL⁺，Human-FNAR（46 題） | Fisher exact | 4 vs 6，p = .74 |
| I_HARD / I_SOFT FNAR | VQ vs R-PAL⁺ | 16.7% / 21.4% vs 50.0% / 92.9%（hard n = 6） |

**必須一起講的但書**：RQ4 的檢定力只夠支撐對裸 R-PAL 的比較。
I_HARD 只有 6 題，所以 hard/soft 一律分開報，不對小 stratum 做泛化。

---

# Appendix D — 待補實驗的成本估計（行動清單的依據）

| 動作 | API 呼叫 | 預估時間 | 阻塞條件 |
|---|---|---|---|
| A1 SC 門檻曲線 | **0** | < 1 h | 無 |
| A2 FIC 範例卡 | **0** | < 1 h | 無 |
| A3 容差敏感度 | **0** | < 1 h | 無 |
| A4 hard/soft 判準 | **0** | < 1 h | 無 |
| A5 dense/hybrid retriever（僅 80 題） | ~82 | 半天（含建 index） | 需選定 embedding 模型 |
| A6 SC@5 on R-PAL（matched） | ~1,650 | 約 3–4 h | 需 egress 批准 |
| A7 near-OOD 復活 | **0** | < 1 h | 只是把 V1 的段落搬回來 |

---

# 給簡報製作的執行備註

1. **K=3 標籤要一路貫穿**：Slide 11 與 backup B1 只要出現 K=3，就必須帶上「intent-oracle upper bound；K=1 才是全自動」的角標。論文端已於 2026-08-09 補進 `paper/main_v2.tex` 主結果表的 caption。
2. **可直接沿用的圖檔**：`paper/figures/fig1_pipeline_loop_v3.pdf`（S7）、`paper/figures/safety_utility_250q.png`（S13）、`paper/figures/libsweep_250q.png`（S15）。
3. **配色一致性**：整場固定 R-PAL、R-PAL⁺、VerifiQuant 三個顏色，**從 S9 的儀器表開始**就用同一組，讓 S13 的三個點不需要再解釋一次。
4. **顏色語意**：SWR 與 FNAR 是「越低越好」，建議所有這兩欄用同一種「風險紅」深淺，避免觀眾在 S11 誤讀成 VerifiQuant 全面落後。
5. **口說稿字數**：以每分鐘約 230 字估算，主軸 S1–S18 約 4,400 字，對應 19 分鐘。壓縮到 12 分鐘時刪 S5、S8，並把 S14 的成本段縮成一句。Part II 約 900 字，Part III 約 1,000 字，各約 4 分鐘。
6. **S9 不能砍**：它是整場的邏輯地基，也是審稿人給分最重的一項（實驗隔離度）。所有壓縮版本都保留它。
7. **Part III 的雙重用途**：它是口試的防守稿，也是投稿前的 checklist。行動清單 A1–A4b 全部零 API 呼叫，建議在定稿前完成。
