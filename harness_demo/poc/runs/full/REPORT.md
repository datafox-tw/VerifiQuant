# PoC 結果：harness 在 50 題上（run `full`）

## G1 泛用性、G2 安全性：三個系統在同一批題目上

| 系統 | 正確輸出 | 輸出但錯（silent wrong） | 追問 | 拒答 | 錯誤 |
|---|---|---|---|---|---|
| harness（自動卡片，單次，不回答追問） | 27/50 (54%) | 1/50 (2%) | 3/50 (6%) | 19/50 (38%) | 0 |
| VQ（論文，公式卡，含模擬使用者多輪） | 48/50 (96%) | 1/50 (2%) | — | 1/50 (2%) | |
| CoT 單次（論文） | 37/50 (74%) | 9/50 (18%) | — | 4/50 (8%) | |

harness 有輸出時的正確率：27/28 (96%)。

| 難度 | 題數 | 正確 | silent wrong | 追問 | 拒答 |
|---|---|---|---|---|---|
| hard | 14 | 6 | 0 | 2 | 6 |
| medium | 36 | 21 | 1 | 1 | 13 |

### 漏斗：每題停在哪一步

| 停在 | 題數 |
|---|---|
| 走完全程（輸出） | 28 |
| 6 驗證（沒有程式通過） | 10 |
| 2 意圖規格（出處／卡片不適用） | 9 |
| 8 檢查員不一致 | 2 |
| 7 歧義（追問或無法區分） | 1 |

選卡正確率（選到出題用的那張卡）：46/50 (92%)；選了別張卡：4 題；選 none：0 題。

### silent wrong 逐題

| 題號 | 輸出 | 標準答案 | 選卡正確 | 卡片完整 | 卡片殺死率 | 人工檢查 |
|---|---|---|---|---|---|---|
| test-1490 | 0.036 | 3.35 | True | True | 1.0 | Average of 3.25/3.45/3.30/3.40% is 3.35%; plus 25 bp gives 3.60%. The gold solution adds 25/10000 = 0.0025 to 3.35 (a percent number), mixing units, and reports 3.35. Checked by hand 2026-10-31. |

其中 1 題經人工檢查判定為標準答案有誤（上表「人工檢查」欄），表中仍照原始評分計入 silent wrong。

## G3 約束強度：自動卡片的 VERINA 式評分

可評分 41/50 張（其餘：參考函式回傳多個值，或抽不出輸入）。

| 指標 | 自動卡片 | 手寫卡片（展示網站） | 及格線 |
|---|---|---|---|
| Complete（正確程式通過全部不變量） | 38/41 (93%) | 4/4 | ≥ 95% |
| 不變量對正確程式成立的比例（平均） | 99.3% | 100% | |
| Sound：錯誤變異被抓到的比例（中位數） | 100%（平均 92%） | 約 97% | ≥ 80% |
| 釘死答案（有 base case 且有關係式） | 42/50 (84%) | 3/4 | 報告 |
| 弱約束（只剩範圍／不等式） | 3/50 (6%) | 0/4 | ≤ 30% |

沒被抓到的變異類型：round ×7、off_by ×4、scale ×3、percent_x100 ×2、percent_input ×2、offset_small ×1

不完整的卡片（會誤殺正確程式）：

- test-1950 聯邦稅負計算：誤報 pre_brackets_format_and_values
- test-2077 即期年金每月給付金額：誤報 recurrence_years
- test-2134 零息利差 (Z-利差)：誤報 post_single_cf

## G4 追問校準

追問 3/50 (6%)。逐題（需人工判斷是否為真歧義）：

- test-1704：slots → fees, interest；選項中有正確答案：None
- test-2134：slots → cash_flows, periods；選項中有正確答案：None
- test-2097：slots → contributions, annual_limits；選項中有正確答案：None

欄位狀態統計：grounded 136、mismatch 4、missing 2

## G5 不變量的貢獻：跟「多數決」比

只看候選程式多數決（CodeT 式，不管不變量）：對 38、錯 3（有候選可投票的 41 題）。

- 多數決會錯、harness 沒有輸出錯答案：2 題 test-1593（harness abstain）, test-2021（harness abstain）
- 多數決會對、harness 卻沒輸出（驗證的代價）：11 題 test-1950（停在 inspect）, test-1560（停在 verify）, test-1422（停在 verify）, test-1828（停在 verify）, test-1448（停在 verify）, test-1678（停在 verify）, test-1709（停在 ambiguity）, test-1032（停在 verify）, test-2009（停在 verify）, test-2077（停在 verify）, test-2189（停在 verify）

