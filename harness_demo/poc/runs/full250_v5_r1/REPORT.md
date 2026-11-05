# PoC 結果：harness 在 250 題上（run `full250_v5_r1`）

## G1 泛用性、G2 安全性：三個系統在同一批題目上

| 系統 | 正確輸出 | 輸出但錯（silent wrong） | 追問 | 拒答 | 錯誤 |
|---|---|---|---|---|---|
| harness（自動卡片，單次，不回答追問） | 138/250 (55%) | 4/250 (2%) | 40/250 (16%) | 68/250 (27%) | 0 |
| VQ（論文，公式卡，含模擬使用者多輪） | 234/250 (94%) | 8/250 (3%) | — | 8/250 (3%) | |
| CoT 單次（論文） | 194/250 (78%) | 45/250 (18%) | — | 11/250 (4%) | |

harness 有輸出時的正確率：138/142 (97%)。

| 難度 | 題數 | 正確 | silent wrong | 追問 | 拒答 |
|---|---|---|---|---|---|
| hard | 70 | 22 | 1 | 21 | 26 |
| medium | 180 | 116 | 3 | 19 | 42 |

### 漏斗：每題停在哪一步

| 停在 | 題數 |
|---|---|
| 走完全程（輸出） | 142 |
| 2 意圖規格（出處／卡片不適用） | 42 |
| 6 驗證（沒有程式通過） | 37 |
| 8 檢查員不一致 | 20 |
| 4 輸入檢查 | 6 |
| 7 歧義（追問或無法區分） | 2 |
| 1 範圍閘（選卡） | 1 |

選卡正確率（選到出題用的那張卡）：213/250 (85%)；選了別張卡：36 題；選 none：1 題。

### silent wrong 逐題

| 題號 | 輸出 | 標準答案 | 選卡正確 | 卡片完整 | 卡片殺死率 | 人工檢查 |
|---|---|---|---|---|---|---|
| test-1490 | 0.036 | 3.35 | True | None | None | Average of 3.25/3.45/3.30/3.40% is 3.35%; plus 25 bp gives 3.60%. The gold solution adds 25/10000 = 0.0025 to 3.35 (a percent number), mixing units, and reports 3.35. Checked by hand 2026-10-31. |
| test-1801 | 103.833 | 108.5 | True | None | None | — |
| test-2179 | 0.835081 | 0.674 | True | None | None | — |
| test-1826 | 5.72614e+08 | 572.61 | True | None | None | — |

其中 1 題經人工檢查判定為標準答案有誤（上表「人工檢查」欄），表中仍照原始評分計入 silent wrong。
## G4 追問校準

追問 40/250 (16%)。逐題（需人工判斷是否為真歧義）：

- test-1704：slots → fees；選項中有正確答案：None
- test-2134：slots → cash_flows, periods；選項中有正確答案：None
- test-2097：slots → contributions, annual_limits；選項中有正確答案：None
- test-1458：slots → expected_return；選項中有正確答案：None
- test-1271：slots → sales_amount；選項中有正確答案：None
- test-1004：slots → stock_price_at_purchase；選項中有正確答案：None
- test-1942：slots → cash_flows；選項中有正確答案：None
- test-1980：slots → purchase_costs；選項中有正確答案：None
- test-2137：select → ；選項中有正確答案：None
- test-2093：select → ；選項中有正確答案：None
- test-2081：slots → tax_brackets；選項中有正確答案：None
- test-2057：slots → year；選項中有正確答案：None
- test-2151：convention → payment_timing；選項中有正確答案：False
- test-2019：select → ；選項中有正確答案：None
- test-2058：slots → year；選項中有正確答案：None
- test-1014：slots → x_values, y_values；選項中有正確答案：None
- test-1664：slots → rights_ratio；選項中有正確答案：None
- test-2020：slots → option_type；選項中有正確答案：None
- test-1861：select → ；選項中有正確答案：None
- test-1023：slots → tax_credit；選項中有正確答案：None
- test-2014：slots → filing_status；選項中有正確答案：None
- test-1710：slots → p3；選項中有正確答案：None
- test-2080：select → ；選項中有正確答案：None
- test-1952：slots → current_price；選項中有正確答案：None
- test-2125：slots → grant_date, exercise_date, sale_date；選項中有正確答案：None
- test-2003：slots → remaining_payments；選項中有正確答案：None
- test-1237：slots → option_type；選項中有正確答案：None
- test-1808：select → ；選項中有正確答案：None
- test-2237：select → ；選項中有正確答案：None
- test-1499：slots → salvage_value, year；選項中有正確答案：None
- test-1712：slots → holding_period_return；選項中有正確答案：None
- test-2104：select → ；選項中有正確答案：None
- test-1902：slots → rating_values；選項中有正確答案：None
- test-2088：slots → correlations；選項中有正確答案：None
- test-2052：slots → annual_interest_rate；選項中有正確答案：None
- test-1500：slots → exchange_rate；選項中有正確答案：None
- test-2187：select → ；選項中有正確答案：None
- test-2224：select → ；選項中有正確答案：None
- test-2159：slots → tax_brackets；選項中有正確答案：None
- test-1903：slots → brackets；選項中有正確答案：None

欄位狀態統計：grounded 743、mismatch 16、missing 9、default 2、ungrounded 1

## G5 不變量的貢獻：跟「多數決」比

只看候選程式多數決（CodeT 式，不管不變量）：對 182、錯 19（有候選可投票的 201 題）。

- 多數決會錯、harness 沒有輸出錯答案：15 題 test-1950（harness correct）, test-1593（harness correct）, test-2021（harness abstain）, test-1590（harness correct）, test-1706（harness abstain）, test-2079（harness abstain）, test-1777（harness abstain）, test-1958（harness abstain）, test-1588（harness abstain）, test-1810（harness abstain）, test-1577（harness correct）, test-1043（harness abstain）, test-2108（harness abstain）, test-1924（harness abstain）, test-2160（harness abstain）
- 多數決會對、harness 卻沒輸出（驗證的代價）：48 題 test-1560（停在 verify）, test-1422（停在 verify）, test-1828（停在 verify）, test-1448（停在 inspect）, test-1546（停在 inspect）, test-1032（停在 verify）, test-2218（停在 verify）, test-2009（停在 verify）, test-2236（停在 verify）, test-2077（停在 verify）, test-1595（停在 inspect）, test-1823（停在 verify）, test-2188（停在 verify）, test-1248（停在 inspect）, test-1088（停在 verify）, test-2110（停在 verify）, test-1600（停在 inspect）, test-1561（停在 verify）, test-2199（停在 inspect）, test-1264（停在 verify）, test-1578（停在 verify）, test-2151（停在 ambiguity）, test-1017（停在 inspect）, test-1969（停在 verify）, test-2211（停在 inspect）, test-1092（停在 verify）, test-2191（停在 inspect）, test-2146（停在 ambiguity）, test-2023（停在 verify）, test-1607（停在 inspect）, test-1842（停在 verify）, test-2198（停在 verify）, test-1606（停在 verify）, test-2139（停在 verify）, test-1785（停在 inspect）, test-1705（停在 verify）, test-1945（停在 verify）, test-2197（停在 inspect）, test-1470（停在 verify）, test-1787（停在 verify）, test-2231（停在 inspect）, test-1841（停在 inspect）, test-1715（停在 inspect）, test-1563（停在 verify）, test-1063（停在 verify）, test-1605（停在 verify）, test-2025（停在 verify）, test-2148（停在 verify）

