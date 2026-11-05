# PoC 結果：harness 在 50 題上（run `full_v4_r1`）

## G1 泛用性、G2 安全性：三個系統在同一批題目上

| 系統 | 正確輸出 | 輸出但錯（silent wrong） | 追問 | 拒答 | 錯誤 |
|---|---|---|---|---|---|
| harness（自動卡片，單次，不回答追問） | 34/50 (68%) | 1/50 (2%) | 3/50 (6%) | 12/50 (24%) | 0 |
| VQ（論文，公式卡，含模擬使用者多輪） | 48/50 (96%) | 1/50 (2%) | — | 1/50 (2%) | |
| CoT 單次（論文） | 37/50 (74%) | 9/50 (18%) | — | 4/50 (8%) | |

harness 有輸出時的正確率：34/35 (97%)。

| 難度 | 題數 | 正確 | silent wrong | 追問 | 拒答 |
|---|---|---|---|---|---|
| hard | 14 | 6 | 0 | 2 | 6 |
| medium | 36 | 28 | 1 | 1 | 6 |

### 漏斗：每題停在哪一步

| 停在 | 題數 |
|---|---|
| 走完全程（輸出） | 35 |
| 6 驗證（沒有程式通過） | 9 |
| 2 意圖規格（出處／卡片不適用） | 3 |
| 8 檢查員不一致 | 2 |
| 7 歧義（追問或無法區分） | 1 |

選卡正確率（選到出題用的那張卡）：45/50 (90%)；選了別張卡：5 題；選 none：0 題。

### silent wrong 逐題

| 題號 | 輸出 | 標準答案 | 選卡正確 | 卡片完整 | 卡片殺死率 | 人工檢查 |
|---|---|---|---|---|---|---|
| test-1490 | 0.036 | 3.35 | True | None | None | Average of 3.25/3.45/3.30/3.40% is 3.35%; plus 25 bp gives 3.60%. The gold solution adds 25/10000 = 0.0025 to 3.35 (a percent number), mixing units, and reports 3.35. Checked by hand 2026-10-31. |

其中 1 題經人工檢查判定為標準答案有誤（上表「人工檢查」欄），表中仍照原始評分計入 silent wrong。
## G4 追問校準

追問 3/50 (6%)。逐題（需人工判斷是否為真歧義）：

- test-1704：slots → fees, interest；選項中有正確答案：None
- test-2134：slots → cash_flows, periods；選項中有正確答案：None
- test-2097：slots → contributions, annual_limits；選項中有正確答案：None

欄位狀態統計：grounded 153、mismatch 4、missing 2

## G5 不變量的貢獻：跟「多數決」比

只看候選程式多數決（CodeT 式，不管不變量）：對 44、錯 3（有候選可投票的 47 題）。

- 多數決會錯、harness 沒有輸出錯答案：2 題 test-1593（harness correct）, test-2021（harness abstain）
- 多數決會對、harness 卻沒輸出（驗證的代價）：11 題 test-1560（停在 verify）, test-1422（停在 verify）, test-1828（停在 verify）, test-1448（停在 inspect）, test-1546（停在 inspect）, test-1032（停在 verify）, test-2218（停在 verify）, test-2009（停在 verify）, test-2095（停在 ambiguity）, test-2236（停在 verify）, test-2077（停在 verify）

