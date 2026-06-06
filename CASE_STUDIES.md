# VerifiQuant — 三個微觀案例（meeting / 論文 case study）

> 每個案例都標清楚：gold 是什麼、哪些配置對/錯、答對時是運氣還是能力、VQ 怎麼處理。

---

## 案例 A（★ 主打）— ARPU 負收入：E-class 邊界攔截 = 優雅 abstention

**來源**：trap set，E-operator（E-test-1035）。對應的乾淨題 test-1035 收入為正、gold = **+76.19**。

**陷阱**：E-operator 把 total revenue 改成 **−$1,200,000**（違反「收入不得為負」）。訂戶 15,000 → 16,500。

**這題有沒有 ground truth？**
- **沒有合法 GT**。負收入使這題在語義上無法回答；正確行為是「攔截 / 請求修正」，而非輸出數字。
- −76.19 不是答案，是 CoT 把乾淨題答案 76.19 機械翻符號的產物。

**各系統反應**：

> ⚠️ **數據來源說明**：本案例來自 **trap set**。trap 上只跑了兩個配置——**CoT single-shot Flash** 與 **VQ K=1 funnel-only**（所有 oracle 變體因在 trap 上會經 `python_solution` 洩漏 GT 而排除，Pro 亦未在 trap 上運行）。故下表僅有 Flash single-shot 與 VQ K=1 兩列；這是 §5.6 的公平度量設計，非數據缺漏。

| 系統 | 輸出 | 判定 |
|---|---|---|
| CoT single-shot Flash | **−76.19**（自信回報「每用戶平均收入 = 負 76 美元」）| ❌ silent wrong |
| VQ Flash (K=1 funnel) | **status=alert, diag=E**，觸發 `arpu_val_001`「Total Revenue Non-Negative Check」 | ✅ 攔截 |

（CoT 計算：`−1,200,000 / ((15,000+16,500)/2) = −1,200,000 / 15,750 = −76.19`）

**VQ 的攔截訊息（逐字）**：
> "Total revenue is negative. While technically possible in some accounting scenarios, this may indicate an input error. Is the negative 'Total Revenue' value correct, or would you like to provide a non-negative value?"

**重要釐清**：Pro **沒有**跑過 trap set（trap 只跑 CoT single-shot Flash + VQ K=1，因 oracle 在 trap 上洩漏 GT）。且即使 Pro 跑 single-shot 也會吐 −76.19——single-shot CoT 無論模型多強都無 abstention 機制。**報錯能力來自架構（E 層 boundary check），不是模型強度。**

**一句話**：收入欄位是負一百二十萬（疑似上游符號錯誤）。CoT 自信地算出「每用戶收入 −$76.19」交出去；VQ 的 E 層邊界檢查直接攔下並回問使用者要不要修正。**這就是優雅 abstention：系統知道自己不該算，把判斷權還給人。**

---

## 案例 B（C-class，精度/非決定性）— CAGR

**來源**：clean set，test-1443。**gold = 19.14**（題目要求以百分比表示）。
正確計算：`(1,200,000 / 500,000)^(1/5) − 1 = 0.1913578… → 19.14%`

**各配置結果（7 個 CoT 配置中錯 6）**：

| 系統 | 輸出 | 對? |
|---|---|---|
| CoT single-shot Flash | 19.13 | ❌ |
| CoT single-shot Pro | 19.13 | ❌ |
| CoT plain blind Flash | 19.12 | ❌ |
| CoT funnel blind Flash | 19.11 | ❌ |
| CoT plain blind Pro | 19.13（turn 2 曾算對 19.14、turn 3 又漂回）| ❌ |
| CoT funnel blind Pro | 19.14 | ✅（**矇對**）|
| VQ Flash V3 | 0.19135…（→19.14）| ✅（**deterministic，每次一樣**）|
| VQ Pro V3 | 0.19135… | ✅ |

**答對時是運氣還是能力？**
- CoT funnel-blind-Pro 的 19.14 是**運氣**：答案在 19.11–19.16 間振盪，最後一位是擲骰子；plain Pro 甚至 turn 2 算對、turn 3 又漂回 19.13，**連自己都重現不了**。
- VQ 的對是**能力**：把分數次方交給 deterministic Python，每次精確 0.19135…，零浮動。同時對 decimal/percent 掛 I_SOFT + 可驗證 ×100 transform。

**一句話**：CoT 在做 `2.4^(1/5)` 的 token 心算，答案擲骰子；VQ 把擲骰子變成查表。

---

## 案例 C（I_HARD，意圖歧義）— Cost-Plus 合約

**來源**：clean set，test-1593。**gold = 1,150,000**。
**輸入**：總約值 contract_value = $2,500,000、利潤 profit = 15%、完工 completion = 40%。
**題目的歧義**：「cost-plus 合約」（利潤按實際成本計）vs「利潤是總約值的 15%」（固定費用）——兩種解讀的計算路徑不同。

**三條計算路徑（數學已驗證）**：

| 路徑 | 公式 | 結果 |
|---|---|---|
| ✅ 正解（利潤計一次）| `2,500,000 × 40% × (1+15%)` | **1,150,000** |
| ❌ 漏利潤 | `2,500,000 × 40%` | 1,000,000 |
| ❌ 利潤計兩次 | `(2,500,000 + 15%×2,500,000) × 40% × (1+15%)` | 1,322,500 |

**各配置結果**：

| 系統 | 輸出 | 對? | 錯在哪 / 對在哪 |
|---|---|---|---|
| CoT single-shot Flash | 1,000,000 | ❌ | reasoning 寫「總約值已含 15% 利潤，所以發票=總約值×40%」→ **漏掉 ×1.15** |
| CoT single-shot Pro | 1,000,000 | ❌ | 同上，同樣的誤讀 |
| CoT plain/funnel blind (Flash & Pro) | 1,150,000 | ✅ | oracle turn 2 改寫成「base 2.5M + 利潤 375k = 2,875,000，再 ×40%」= 1,150,000（等價於正解的重排）|
| VQ Flash V3 | 1,150,000 | ✅ | turn 1 觸發 **I_HARD** 澄清 cost-plus 歧義 → 綁定 contract_value=2,500,000（用題目給的原值）→ FIC code 算對 |
| VQ Pro V3 | **1,322,500** | ❌ | **過度 normalize 輸入**：把 contract_value 重算成 2,875,000（=2.5M+利潤），FIC code 又自帶 ×(1+15%) → **利潤算兩次** |

**這題誠實揭露兩件事（兩者都值得寫進論文）**：

1. **I_HARD 攔截的價值**：VQ turn 1 就偵測到 cost-plus 的定義衝突並明確提問，而 CoT single-shot 直接選一種解讀（且選錯，漏了利潤）卻不告知它假設了什麼。VQ 的 I_HARD 提問（逐字）：
   > 「'cost-plus' 通常指利潤按實際成本計算，但'總約值 15%'暗示固定費用。完成部分的發票應該怎麼算？」

2. **VQ Pro 在這題反而錯（誠實揭露）**：錯因明確——Pro 在輸入 normalization 階段「自作聰明」把利潤加進 contract_value（2.5M→2.875M），而 FIC 的 deterministic code 本身已含 ×(1+profit)，導致利潤被計兩次。**VQ Flash 沒有這個問題**（它直接用題目給的 2,500,000）。這正是 §5.5「VQ Pro < VQ Flash」的根因個案：更強的模型在 normalization 階段過度推理反而出錯。寫進論文顯示我們不挑數據，反而強化「架構約束 > 模型強度」的論點。

**注意**：本題 oracle 變體（plain/funnel blind）能修對，是因為這是 **clean set**（有合法 GT，oracle 可用 `python_solution` 在不洩漏數值答案下澄清計算邏輯）。trap 上則不適用 oracle（見案例 A）。

---

## 三案例的論文定位

| 案例 | 對應 funnel 層 | 講的能力 | 強度 |
|---|---|---|---|
| **A — ARPU 負收入** | **E（Boundary）** | 攔截髒資料 + 優雅 abstention | ★★★ 主打 |
| C — Cost-plus | **I_HARD（Critic）** | 意圖歧義的顯式澄清 | ★★ 配套 |
| B — CAGR | C（Logic/精度）| deterministic 執行消除算術非決定性 | ★ 補充 |

**建議**：meeting 與論文 §5.6 以 **A（E-class）為主打**、**C（I_HARD）為配套**，B 作為「即使算術這種小事 LLM 也不可靠」的補充。A 的畫面感（負收入、優雅退回）最能一秒說服人。
