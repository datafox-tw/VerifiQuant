# VerifiQuant Demo 完整提詞稿(約 11 分鐘) — 2026-06-15

邊錄邊念。每幕含:🖱️ 操作 / 🗣️ 逐字旁白 / ✅ 預期畫面。
網址 `http://127.0.0.1:6222/demo` → Conversation 分頁。錄影前 Cmd+Shift+R 硬重新整理。

---

## 00:00–00:45 ｜ 開場與定位
🖱️ 操作：停在 Conversation 分頁。勾選頂端 **Compare with raw AI**;Control model 選 **GPT**
(若沒設 OpenAI key 就選 Gemini)。
🗣️ 旁白：
> 「大家好。今天我要展示的不是一個更會算數學的 ChatGPT,也不是『AI 接一台計算機』。
> 金融場景真正出事的地方,從來不在算術,而在算術**之外**——有沒有用對公式、看不看得懂模糊、
> 擋不擋得住壞資料、敢不敢承認自己不會。接下來我會把同樣的問題,同時丟給沒有護欄的現成 AI
> (右邊)和我們的 VerifiQuant(左邊),你會看到差別。」
✅ 預期畫面:聊天區空白,下方有 Showcase / Trap 下拉、Compare 勾選框與模型選單。

---

## 00:45–01:30 ｜ 軸 1：乾淨題持平(我們不退化)
🖱️ 操作：Showcase 下拉 → **C · Clean compute (verified)** → 按 **Send**。
🗣️ 旁白：
> 「先講清楚:我們不是來扯算術後腿的。這是一題資訊完整、毫無歧義的固定資產周轉率。
> 兩邊都算出 2.0——簡單題我們和現成 AI 一樣準。重點是,我們的 2.0 旁邊有一個綠色
> 『✓ verified』:這個數字是用一張可追溯的合約、跑真實 Python 算出來、再驗證過的。」
✅ 預期畫面:VQ 泡泡 `success`,`fixed_asset_turnover_ratio = 2`,綠色 ✓;右側 raw AI 也給 2.0。
題目上方出現綠色 gold badge「✦ 此題有正確答案:2.00」。

---

## 01:30–02:45 ｜ 軸 4：無效輸入(AI 自信地跌倒)
🖱️ 操作：先按 **New conversation** → Showcase → **E · Trap: negative expense (alert)** → **Send**。
🗣️ 旁白：
> 「現在動手腳。這題的總費用是 **負三百萬**——一個現實中不該出現的值。看右邊:現成 AI
> 默默把負號丟掉,算出一個漂亮的 88.46%。注意,就算你幫它接上計算機,結果一樣——
> 因為計算機只會忠實地把錯誤的輸入算對。
> 再看左邊:VerifiQuant 沒有給數字,而是觸發了 E 類邊界閘門,告訴你『費用不該是負的』。
> 這就是差別:工具保證括號裡算對,我們質疑括號該不該長這樣。」
✅ 預期畫面:右側 raw AI = 88.46% 配紅色 ✗;VQ 泡泡 `alert / E`,綠色「✓ correctly raised E-gate」。

---

## 02:45–03:50 ｜ 軸 2：選錯公式(把選擇權還給人)
🖱️ 操作：**New conversation** → Showcase → **M · Which method? (project profitability)** → **Send**。
🗣️ 旁白：
> 「這題只說『我想看這專案值不值得投資』,沒有指定用哪個指標。右邊的 AI 自作主張選了 NPV,
> 直接算出一個負數。問題是——值不值得,可以看 NPV、看折現回收期、也可以看 ROI,
> 每個答案的意義完全不同。VerifiQuant 不替你賭:它在 M 閘門停下來,反問你『要用哪一個?』。
> 這在合規上,就是把決策權交還給人類。」
✅ 預期畫面:右側 raw AI 給 NPV 數字 + ✗;VQ 泡泡 `refusal / M`,反問列出 NPV / 折現回收期 / ROI。

---

## 03:50–04:55 ｜ 軸 3：範圍外(拒絕憑空捏造)
🖱️ 操作：**New conversation** → Showcase → **N · Won't invent a formula (refuse)** → **Send**。
🗣️ 旁白：
> 「這題要求一個我們資料庫裡根本不存在的『自訂財務健康分數』。最危險的失敗就在這:
> 現成 AI 會**發明**一條公式,然後一本正經地算給你——接上計算機只會讓這個捏造的公式
> 算得更精確、更可信。VerifiQuant 在 N 閘門誠實說:『我沒有對應的合約,我不會自己編。』
> 在金融場景,一句誠實的『我不會』,比一個自信的錯誤值錢太多。」
✅ 預期畫面:右側 raw AI 給一個捏造公式的數字 + ✗;VQ 泡泡 `refusal / N`。

---

## 04:55–07:40 ｜ 軸 5：解讀模糊 + 可驗證修復(皇冠,給最多時間)
🖱️ 操作:**New conversation** → Showcase → **I · Timing ambiguity → atomic transform (HERO)** → **Send**。
🗣️ 旁白(第一段):
> 「這是今天的重頭戲。一筆貸款月付金,題目沒講還款發生在『期初』還是『期末』。
> 右邊的 AI 默默假設期末,直接給數字。左邊的 VerifiQuant——」
✅ 預期畫面:VQ 泡泡 `needs_clarification / I_HARD`,**沒有數字**,跳出期初/期末選項。

🖱️ 操作:在 VQ 泡泡點 **Beginning of period**(帶 `verified transform` 標籤的按鈕)。
🗣️ 旁白(第二段):
> 「——它先攔截,完全不給數字,把這個會影響估值幾十萬的假設攤開來問你。我選『期初』。
> 接下來是關鍵:系統不是重算,而是做一個**數學驗證過的原子轉換**,把期末的 579.98
> 轉成期初的 577.10,並且附上不變量證明:result × (1 + r/12) 必須等於原值,驗證通過。」
✅ 預期畫面:新泡泡 `Diagnostic Report · verified transform`,`579.98 → 577.10`,綠色 ✓ verified。

🖱️ 操作:展開該泡泡的 **Diagnostic Report** 區塊(已預設展開)與「How this was derived & verified」。
🗣️ 旁白(第三段):
> 「再看這份稽核報告:採用的合約是哪一張、使用者在哪個時間戳補了什麼假設、用了什麼公式、
> 跑了哪段 Python、不變量是否成立——全部攤開。Tool-calling 的輸出你只能選擇相信;
> 我們的修正是**可以證明**的。這正是 FINRA 要求的人為檢查點,也天生符合 EU AI Act
> 第 13 條的可追溯性。稽核人員一眼就能回溯這個數字怎麼來的。」
✅ 預期畫面:報告列出 採用合約(FIC 2164 Loan)、人類介入時間戳、Transform 公式、不變量 verified、Python。

---

## 07:40–09:40 ｜ 軸 6：Trap 連環(系統性,不是精挑)
🖱️ 操作:**New conversation** → Showcase 下拉捲到 **Trap set** 群組 → 選 **trap · N · Please price this scenario using a Heston stoch…** → **Send**。
🗣️ 旁白(第一段):
> 「你可能會想,這些是不是挑過的題?不是。這是我們的對抗式 trap 資料集,50 題、五類各 10 題,
> 由真實題目系統性地注入陷阱。先看這題:我把一個製造業情境,換成要求用 Heston 隨機波動模型定價。
> 看右邊——現成 AI 根本沒做 Heston,它回頭把製造情境硬算成 21.5,**答了一個完全不同的問題還很自信**。
> VerifiQuant 直接在 N 閘門拒絕。」
✅ 預期畫面:右側 raw AI = 21.5(離題)+ ✗;VQ `refusal / N`,綠色 ✓。

🖱️ 操作:**New conversation** → Trap set → 選 **trap · E · What is the working ratio for ABC Manufacturing…** → **Send**。
🗣️ 旁白(第二段):
> 「再一題:同樣的工人比率,注入越界的負費用。現成 AI 又算出 88.46,VerifiQuant 又擋下。
> 重點是『又』——五類陷阱、同一套行為:現成 AI 五種跌法,我們五種接法。這不是運氣,
> 是**可重現、可預測的失敗行為**。這才是金融 AI 能上線的前提。」
✅ 預期畫面:右側 raw AI = 88.46 + ✗;VQ `alert / E`,✓。

---

## 09:40–10:40 ｜ 收束:可確定性 + 可稽核性
🖱️ 操作:回到旗艦那則(或任一 transform 泡泡),展開「How this was derived & verified」,
指向 M→N→F→E→I→C 的漏斗圖示。
🗣️ 旁白：
> 「最後把架構講清楚。每一個答案,都跑過這條六層漏斗:先篩選意圖與範圍(M、N)、
> 再驗證輸入(F、E)、處理語意歧義(I),最後才計算(C)。我們不是只在最後丟一個數字,
> 而是讓整條推論鏈——從問題到答案——都可追溯、可驗證。」
✅ 預期畫面:漏斗 stepper 正確亮起對應層、卡片選擇與 Python 執行可見。

---

## 10:40–11:00 ｜ 結語
🗣️ 旁白：
> 「總結一句:VerifiQuant 治理的是計算**之外**那層推論合約——篩選、估算、求解一體成型,
> 連修正都經過數學驗證,而且能從答案反推回它用了哪張合約、經過哪些人類決策。
> 可驗證、可稽核、可預測的失敗,這就是我們和『LLM 加計算機』真正的差別。謝謝。」

---

## 一頁速查(口袋小抄)
| 時間 | 操作(下拉選項) | 預期 VQ | 預期 raw AI |
|---|---|---|---|
| 00:45 | C · Clean compute | success 2.0 ✓ | 2.0 |
| 01:30 | E · Trap negative expense | alert / E ✓ | 88.46 ✗ |
| 02:45 | M · Which method? | refusal / M(問 NPV/payback/ROI) | 選 NPV ✗ |
| 03:50 | N · Won't invent a formula | refusal / N | 捏造公式 ✗ |
| 04:55 | I · Timing (HERO) → 點 Beginning | I_HARD → 577.10 verified + 報告 | 默認期末數字 ✗ |
| 07:40 | trap · N · Heston | refusal / N ✓ | 21.5(離題)✗ |
| 08:40 | trap · E · working ratio | alert / E ✓ | 88.46 ✗ |

## 錄影注意
- 每題開始前按 **New conversation**;從下拉選題會自動開新題。
- M / N / I_HARD 觸發由 LLM 決定,**錄影前各跑一次**確認;沒中就重送。
- 沒設 `OPENAI_API_KEY` → Control model 切 Gemini。
- 7 分鐘精簡版:只留開場 + 軸 4(E) + 軸 5(I_HARD 完整) + 軸 6(trap N+E)+ 結語。
