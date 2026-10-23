# 教授追問 A／B：分析結果與修改提案（2026-10-17）

所有數字都由程式重算：`scripts/analyze_trap_sensitivity.py`（輸出 `governance_confirmatory_1014/trap_sensitivity.json`），計分函式與 bootstrap（seed 20260729、10,000 次、依基礎題群集）和主表相同。論文本身**尚未修改**，下方提案等你決定。

---

## A. F trap 刪掉的欄位若在合約裡是 optional，這題還符合定義嗎？

### A-1 檢查結果：20 題 F trap 中有 3 題不符合「刪除 required input」的定義

| Trap | 刪掉的欄位 | 合約 | 通用函式預設 | 判斷 | 各系統表現 |
|---|---|---|---|---|---|
| F-test-1263 | `quantity`（100） | optional，說明寫「Defaults to 100 shares, corresponding to one option contract」 | `quantity: int = 100` | **預設值正當，trap 標籤有誤**：照預設作答本來就對 | VerifiQuant 三次都答出正確答案（被算成 FNAR）；R-PAL 也答對；R-PAL+、Staged、Staged+Spec 都沒答 |
| F-test-1559 | `quantity`（100） | 同上（同一張卡 fic_article_1395） | `quantity: int = 100` | 同上 | VerifiQuant 三次都答對；其他系統都沒答 |
| F-test-1271 | `number_of_units`（15000） | optional | `number_of_units: int = 0` | **標籤正確，合約規格有誤**：題目是「每單位固定權利金」，單位數必填，0 只是佔位值；合約應標為條件式必填 | VerifiQuant 三次都正確要求補欄位 |

### A-2 預設值是正當的，還是來源題目的殘留？

- 這三題的預設值都寫在**公開的通用函式簽名**裡，不是建庫時從來源題目帶進來的。
- 全部 250 張卡共有 38 個 optional 或有預設值的輸入：34 個的預設值來自函式簽名，4 個沒有寫預設值，**只出現在來源題目或解題程式裡的殘留預設值：0 個**。
- 但有 6 個 optional 輸入的函式預設值是佔位值（0、None、False）。其中 `refundable=False`、`annualize=False` 是合理的旗標；`number_of_units=0`（即 F-1271 那張卡）、`shares_bought_price=0`、`exchange_rate=None` 可能和 F-1271 一樣其實是條件式必填。80 題 trap 中只有 F-1271 碰到這類欄位。

結論：A 的兩種可能中，F-1263 和 F-1559 屬於「預設值正當，trap 建構與標籤要修正」；F-1271 屬於另一個問題，是合約把條件式必填的欄位標成 optional（建庫規格問題，影響範圍上限是上述 6 個欄位中的佔位型）。

### A-3 建構檢查與敏感度（教授建議的做法）

新增 trap 建構規則：**被刪除的欄位必須在合約中標為 required，且合約與函式都沒有可直接套用的預設值。** 用這條規則重算：

| 範圍 | VerifiQuant | R-PAL+ | 差距（95% CI） | Staged+Spec 減 VerifiQuant |
|---|---|---|---|---|
| 全部 80 題（目前論文） | 10.4% | 27.5% | 17.1 [6.8, 27.0] | −1.7 [−8.3, 4.8] |
| 拿掉 2 題預設值正當的（78 題） | 8.1% | 28.2% | 20.1 [11.0, 29.2] | +0.9 [−4.7, 6.8] |
| 依建構規則拿掉 3 題（77 題） | 8.2% | 28.6% | 20.3 [11.1, 29.5] | +0.9 [−4.8, 6.9] |

拿掉後 VerifiQuant 的 F 類 FNAR 變成 0%，主結論不變，差距反而變大；「Staged+Spec 與 VerifiQuant 打平」也不變。

**建議**：保留 80 題 benchmark 與主表數字（和投稿版、審稿回覆一致），正文 §5.2 改寫那兩個 F acceptance 的說明並給敏感度數字（提案 P7），附錄 D 放完整的建構檢查與三種範圍的表。

---

## B. 人工集到底驗證了哪一種能力？

### B-1 人工標籤 × 系統出口（80 題）

| 人工標籤 | 題數 | VerifiQuant | R-PAL+（單一 prompt） |
|---|---|---|---|
| ANSWER | 34 | 停在 N 24；答出數字 7（4 題附 I_SOFT 警告）；E 1、M 1、F 1 | 答出數字 22；READY 2；N 9；I_SOFT 1 |
| REQUEST-INFO | 20 | 停在 N 13；**F 5**；M 1；答出數字 1 | **F 16**；N 4 |
| CLARIFY | 10 | 停在 N 4；M 3；答出數字 2（附警告）；F 1；**I_HARD 0** | 答出數字 4；M 2；N 3；I_HARD 1 |
| OUT-OF-SCOPE | 16 | 停在 N 11；E 2；F 1；M 1；答出數字 1 | N 11；其他 5 |

### B-2 只看 VerifiQuant 有選到合約的題目（沒有停在 N）

| 人工標籤 | 題數 | VerifiQuant | R-PAL+ |
|---|---|---|---|
| REQUEST-INFO | 7 | F 5／7，答出數字 1，M 1 | F 7／7 |
| CLARIFY | 6 | I_HARD 0，M 3，附警告答 2，F 1 | I_HARD 0，M 2，答 3，N 1 |

**誠實的結論**：人工集上偏低的 human FNAR 主要來自「沒有選到合約就拒答」。在自然語言的題目上，VerifiQuant 辨識缺漏資訊的能力**沒有比單一 prompt 好**（F 5/7 對 7/7），歧義的硬性澄清兩邊都幾乎沒觸發。這和 §5.4 現有的那句「this set supports the scope signal, not the claim that those checks work on natural phrasing」一致，只是現在有數字。

**限制**：「VerifiQuant 選到合約」是代理指標，不等於「庫裡確實有適用合約」。要照教授說的「有／無適用合約」切開，需要逐題判定。目前判定過的只有 ANSWER 類停在 N 的 24 題（17 題庫裡沒有合約、7 題有但沒被檢索到）；REQUEST-INFO、CLARIFY、OUT-OF-SCOPE 停在 N 的 28 題還沒判定。**這一步需要你決定要不要做**，判定標準沿用 `scripts/attribute_messy_shortfalls.py`（合約宣告的用途涵蓋所問的量，且程式能用題目給的輸入算出）。

### B-3 scope signal：宣稱不等於正確

- 已知至少 7 個 scope signal 是錯的：ANSWER 類停在 N 的 24 題中，有 7 題其實庫裡有適用合約。
- REQUEST-INFO 和 CLARIFY 類停在 N 的 17 題，宣稱是否正確尚未判定。
- 所以「庫外會說明沒有合約而不猜測」要改成「找不到適用合約時會說明，而不猜測」，並在定義處寫明這是一個可能出錯的宣稱。

---

## 修改提案（改前 → 改後）

| 編號 | 位置 | 改前 | 改後 |
|---|---|---|---|
| P1 | 摘要 | Outside that library the system states that no contract applies rather than guessing. | When it finds no applicable contract, it says so instead of guessing. |
| P2 | §2 scope signal 定義 | We call this a *scope signal*. It is defined only for systems that carry a library, so not for CoT. | We call this claim a *scope signal*, which can be wrong when a contract exists but goes unretrieved (§5.4). It is defined only for systems that carry a library. |
| P3 | 貢獻段 | …it mainly offers a *scope signal*, a statement that no contract applies, made instead of a guess. | …it mainly offers a *scope signal*, a claim that no contract applies, made instead of a guess. |
| P4 | §2 | …so runtime only selects and binds within a pre-verified space. | …so runtime only selects and binds within a contract-backed, declared scope. |
| P5 | §5.3 | What it provides is a *contract-grounded verified/unverified boundary*: it emits a number only when a contract applies, whereas execution and prompting emit confident numbers they cannot vouch for. | What it provides is a *contract-backed boundary*: it emits a number only when it selects a contract and otherwise declines as outside its declared scope, whereas execution and prompting emit confident numbers that no contract backs. |
| P7 | §5.2 | VerifiQuant's two F acceptances are not silent wrongs. Both come from one contract that declares the share count optional with a default of 100, which is also the value the trap removed, so the system returns the correct number although the input no longer states it. | VerifiQuant's two F acceptances come from traps that break the F definition, since both remove a share count the contract marks optional with a default of 100. Without them its F rate is 0% and the gap widens to 20.1 points (95% CI [11.0, 29.2], App. D). |
| P8 | §6 | …apart from the declared-default case of §5.2. | …apart from the two F traps of §5.2. |
| P9 | §5.4 | Of the 20 items the panel labels REQUEST-INFO, only 5 exit at F. | （後面加）Even where it selects a contract, it asks for the missing input on 5 of 7 such items, against 7 of 7 for R-PAL+ (App. E). |

**附錄（arXiv 版，不佔 ACM 頁數）**
- 附錄 D：F trap 建構規則、3 題的明細（A-1 表）、三種範圍的敏感度表（A-3 表）、預設值來源的全庫統計（A-2）。
- 附錄 E：B-1、B-2 兩張交叉表，以及 scope signal 已知錯誤數（B-3）。

**頁數（實際編譯）**

| 組合 | ACM 頁數 |
|---|---|
| P1–P5、P7–P9 | 10 頁（多 5 行） |
| 加上 B5（§4.2 容忍度縮成一句，排行榜慣例那句移附錄 H） | 9 頁 |
| 加上 B2（§2 自洽性細節移附錄 H） | 9 頁 |
| 加上 B5 與 B2 | 9 頁，並多留約 4 行 |

---

## 需要你決定

1. 是否套用 P1–P5、P7–P9。
2. 為了留在 9 頁，要搭 B5、B2，還是兩個都要（兩個都要可以多留約 4 行給之後新增）。
3. 是否要我逐題判定剩下 28 題停在 N 的人工集題目「庫裡有沒有適用合約」，讓 B 可以照教授要的「有／無適用合約」完整切開。判定結果會列成表給你確認，再放進附錄 E。
