# VerifiQuant Demo Script (MECE 版) — 2026-06-15

對應 `http://127.0.0.1:6222/demo` 的 **Conversation** 分頁。每個展品只示範一種「計算之外」的失敗
機制,彼此互斥、合起來窮盡賣點。對照組 = 直接 prompting / GPT 接計算機。

> **核心定位(開場一句)**：GPT 加計算機只保證**算術**正確,但不能保證**用對公式、看懂模糊、
> 擋住壞資料、拒絕越界**。VerifiQuant 治理的是計算外面那層「推論合約」,而且連修正都經數學驗證。

---

## 六個 MECE 展品 → 網站對應案例

下拉選單分兩群:**Curated (one per class)** 與 **Trap set**。逐字對照:

| # | 能力軸(互斥) | 下拉群組 → 選項(逐字) | 對照組(raw AI / +計算機)會怎樣 | VQ 會怎樣 |
|---|---|---|---|---|
| 1 | 乾淨題持平 | Curated → **C · Clean compute (verified)** | 也答 2.0 | `success` 2.0 ✓(不退化) |
| 2 | 選錯公式(意圖) | Curated → **M · Which method? (project profitability)** | 直接挑 NPV 算 -2103.68 | `refusal / M`,反問 NPV/payback/ROI |
| 3 | 越界/捏造(範圍) | Curated → **N · Won't invent a formula (refuse)** | 發明一個 60/40 公式並計算 | `refusal / N` 誠實拒絕 |
| 4 | 無效輸入值(邊界) | Curated → **E · Trap: negative expense (alert)** | 丟負號算 **88.46%** | `alert / E` 攔下 |
| 5 | 解讀模糊 + 可驗證修復 | Curated → **I · Timing ambiguity → atomic transform (HERO)** | 默默假設期末給數字 | `I_HARD` 攔截 → 驗證式轉換 577.10 + 稽核報告 |
| 6 | 系統性穩健(非精挑) | Trap set → 逐題(見下) | 五類各跌 | 五類各接(可重現) |

**第 6 軸的 Trap set 選項(逐字,已一類一題)**：
- `trap · M · Overall, is this worth doing? Please evaluate.`
- `trap · N · Please price this scenario using a Heston stoch…`
- `trap · F · What is the total throughput time required to p…`
- `trap · E · What is the working ratio for ABC Manufacturing…`
- `trap · I_HARD · What is the total throughput time required to p…`

> 旗艦備案(同屬軸 5,主流程**不要**和 HERO 連放):
> Curated → **I · Metric ambiguity: time vs rate (trap)**(throughput,time vs rate,1/result 轉換)。

---

## 運鏡台詞(每軸只演一次)

**開場(0:40)** 勾選頂端 **Compare with raw AI**;Control model 選 GPT(沒設 `OPENAI_API_KEY`
就用 Gemini)。「右邊是沒有護欄的現成 AI,左邊是 VerifiQuant。」

**軸 1 · C 持平(30s)** 選 C·Clean → Send。「簡單題我們不退化,baseline 也對——我們不是來
扯算術後腿的。」

**軸 4 · E 無效輸入(60s,最乾淨的一槍)** 選 E·Trap negative expense → Send。
「總費用 -3,000,000 是異常值。GPT、甚至接計算機,都默默丟掉負號算出 **88.46%**——自信但錯。
VQ 的 E 閘門擋下。計算機只保證括號裡算對,不會質疑輸入根本不該長這樣。」

**軸 2 · M 選錯公式(60s)** 選 M·Which method → Send。「『值不值得投資』能套進 NPV、折現回收期、
ROI。對照組替你選了 NPV;VQ 把選擇權還給你。」(可現場打 `use NPV` 接 I_HARD,展示多輪記憶。)

**軸 3 · N 越界(60s)** 選 N·Won't invent a formula → Send。「沒有對應合約,對照組會**發明**一條
公式並算得煞有介事——接計算機只會讓假公式更精確。VQ 誠實說做不到。」

**軸 5 · I_HARD 旗艦(150s,給最多時間)** 選 I·Timing (HERO) → Send →【攔截,不給數字】→
點 **Beginning of period(verified transform 標籤)**。「補上假設後,做的是**數學驗證過的原子轉換**
579.98 → 577.10,附不變量證明。」展開 **Diagnostic Report**:採用合約、人類介入時間戳、公式+Python。
「Tool 的輸出你得相信;我們的修正可證明。這是 FINRA 人為檢查點,天生符合 EU AI Act §13。」

**軸 6 · Trap 連環(60–90s)** 切到 Trap set 群組。「以上不是挑題——這是 50 題對抗式資料集,
每類 10 題。我注入五類陷阱,看現成 AI 連環跌、我們連環接。」最戲劇化先跑 **trap · N · Heston**:
「它根本沒做 Heston,回頭把製造情境算成 **21.5**——答了一個完全不同的問題還很自信。」
再補 **trap · E**(88.46)收束:「可重現,不是運氣。」

**收尾(45s)** 「我們不是更會算的 ChatGPT,也不是 LLM+計算機。我們治理計算外的推論合約:
篩選→估算→求解一體成型,連修正都經驗證,還能從答案反推回用了哪張合約、哪些人類決策。」

---

## 時間預算
- **~7 分鐘版**：開場 + 軸 4(E) + 軸 5(I_HARD,完整)+ 軸 6(trap N+E)。其餘口頭帶過。
- **~11 分鐘版**：六軸全演(上面台詞)。
- 真的很趕：只留 **軸 5(I_HARD)** + **軸 6(trap N)**,這兩個涵蓋「攔截+可驗證修復」與「系統性」。

---

## 為什麼「GPT + 計算機」每一軸都一樣會跌
| 軸 | 現成 AI / 加計算機 | VQ |
|---|---|---|
| 2 沒指定方法 | 直接挑 NPV 硬算 | 交還選擇權 |
| 3 越界/捏造 | 答成另一題 / 發明公式 | 誠實拒絕 |
| 4 越界值 | 丟負號算 88.46 | E 閘門 alert |
| 5 解讀模糊 | 默默賭一種解讀 | 攔截 + 驗證式轉換 |
> 工具呼叫只治理**算術(軸 1)**;軸 2–5 全發生在計算之外,計算機只會把錯前提算得更漂亮。

---

## 重要：F 類不要當打臉案例
研究顯示 **真‧缺值的 F 類**,Raw Gemini 會**自己停下不答** → 雙方都拒絕,**沒有對比張力**。
只有 **E 類(有給值但 size 怪)** raw AI 會照算 → 那才是「AI 自信跌倒」的乾淨對照。
- 主流程用 **E** 當輸入驗證的代表;F 若要提,一句話帶過(「缺值我們會精準點名欄位並可續算」)。
- Trap set 裡的 `trap · F`(throughput 刪一個輸入)因為**還留著其他數字**,raw AI 會幻覺補值算出
  7.5——那是「半缺值」,可秀但別當主軸。

## 避免重複(維持 MECE)
- E:curated-E 與 trap-E 同軸 → 主流程演 curated-E 一次,trap-E 只在軸 6 聚合時帶過。
- I:HERO(付款期初末)與 throughput(time vs rate)同軸 → 主流程只放 HERO,throughput 當備案。
- M / N:curated 與 trap 同軸 → 各演一次即可。

---

## 錄影前 30 秒檢查
1. 伺服器在跑:`http://127.0.0.1:6222/demo`(啟動:`GEMINI_API_KEY=… PORT=6222 python3 app.py`)。
2. **Cmd+Shift+R 硬重新整理**(載入最新 JS/CSS:gold badge、多題分組、trap 一類一題)。
3. 設 `OPENAI_API_KEY` 才有 GPT 對照組;沒設就 Control model 切 Gemini。
4. 每次從下拉選題 = 自動開新對話;答完一題按 **New conversation** 再選下一題。
5. M / N / I_HARD 觸發是 LLM 決定的 → 錄影前**各跑一次**確認有觸發(沒中就重送)。
   實測(Gemini 2.5 Flash):M→abstain、N→refuse、E→alert、I_HARD→attempt 攔截+577.10、
   trap-N→raw 答 21.5 / VQ refuse,皆正常。
