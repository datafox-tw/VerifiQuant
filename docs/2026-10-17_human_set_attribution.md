# 人工集逐題判定：庫裡有沒有適用合約（待確認，2026-10-17）

判定標準沿用 `scripts/attribute_messy_shortfalls.py`：合約宣告的用途涵蓋題目要算的量，且它的程式能用題目給的輸入算出（REQUEST-INFO 題判定「補上缺的值後能不能算」）。只是算術相同、但宣告用途不同的卡算「相近」，不算適用。
判定表與腳本：`scripts/attribute_messy_nonanswer.py`（輸出 `nonanswer_attribution.csv`）。**請逐題確認，有不同意的直接改判定，我重跑。**

## 一、停在 N 的 28 題（scope signal 對不對）

| 題號 | 人工標籤 | 題目在問 | 庫裡有適用合約？ | 適用的卡 | 理由 | VQ | R-PAL+ |
|---|---|---|---|---|---|---|---|
| mq-07 | REQUEST-INFO | difference of two rollforward lines | 沒有 | — | elementary difference | N | F |
| mq-10 | REQUEST-INFO | aggregate intrinsic value of outstanding options | 沒有 | — | near neighbor: intrinsic-value cards 1898 and 2824 price one option and take no option count | N | F |
| mq-12 | OUT-OF-SCOPE | a judgment or recommendation | 不是計算題 | — | — | N | N |
| mq-13 | REQUEST-INFO | sum of three year-end balances | 沒有 | — | elementary sum | N | F |
| mq-14 | CLARIFY | 'margin' from leased versus owned costs | 沒有 | — | ratio of two totals; Profit Margin card is net margin | N | N |
| mq-15 | OUT-OF-SCOPE | a judgment or recommendation | 不是計算題 | — | — | N | N |
| mq-16 | REQUEST-INFO | quick ratio | 沒有 | — | no quick-ratio card | N | N |
| mq-21 | OUT-OF-SCOPE | a judgment or recommendation | 不是計算題 | — | — | N | N |
| mq-24 | REQUEST-INFO | allowance as a share of gross receivables | 沒有 | — | part / whole share | N | F |
| mq-28 | CLARIFY | 10y key rate duration from +1bp bumps | 沒有 | — | Key Rate Duration card 503 hard-codes a 1% shift and needs both bumps; same verdict as mq-26 | N | I_HARD |
| mq-29 | REQUEST-INFO | change in EBITDA margin, in points | 沒有 | — | margin then difference; Profit Margin card is net margin | N | F |
| mq-35 | REQUEST-INFO | ending minus beginning balance | 沒有 | — | elementary difference | N | F |
| mq-37 | REQUEST-INFO | enterprise value | 沒有 | — | no enterprise-value card | N | N |
| mq-42 | OUT-OF-SCOPE | a judgment or recommendation | 不是計算題 | — | — | N | N |
| mq-44 | REQUEST-INFO | GDP by the expenditure approach | 沒有 | — | GDP card 1167 computes a growth rate, not the level | N | N |
| mq-45 | OUT-OF-SCOPE | a judgment or recommendation | 不是計算題 | — | — | N | N |
| mq-47 | OUT-OF-SCOPE | a judgment or recommendation | 不是計算題 | — | — | N | N |
| mq-48 | OUT-OF-SCOPE | a judgment or recommendation | 不是計算題 | — | — | N | N |
| mq-50 | OUT-OF-SCOPE | a judgment or recommendation | 不是計算題 | — | — | N | N |
| mq-51 | OUT-OF-SCOPE | a judgment or recommendation | 不是計算題 | — | — | N | N |
| mq-55 | REQUEST-INFO | total withdrawals under the 4% rule | 沒有 | — | no safe-withdrawal card; Rule 72(t) is a different payout | N | F |
| mq-57 | REQUEST-INFO | segment net revenue from an annual report | 沒有 | — | a lookup, not a formula | N | N |
| mq-63 | OUT-OF-SCOPE | a judgment or recommendation | 不是計算題 | — | — | N | number |
| mq-65 | CLARIFY | Europe's share of the store count | 沒有 | — | part / whole share | N | N |
| mq-66 | REQUEST-INFO | year with the largest provision | 沒有 | — | elementary maximum | N | F |
| mq-68 | REQUEST-INFO | growth in restructuring costs | **有** | 687 | percent change once the earlier amount is supplied | N | F |
| mq-71 | CLARIFY | Intel's return relative to the index | 沒有 | — | ratio of two cumulative values | N | number |
| mq-76 | OUT-OF-SCOPE | a judgment or recommendation | 不是計算題 | — | — | N | number |

## 二、VerifiQuant 有選到合約的 13 題（REQUEST-INFO、CLARIFY）

| 題號 | 人工標籤 | 題目在問 | 庫裡有適用合約？ | 適用的卡 | 理由 | VQ | R-PAL+ |
|---|---|---|---|---|---|---|---|
| mq-01 | CLARIFY | growth of a tax-benefit balance | **有** | 687 | selected card 2659 is unrelated | M | number |
| mq-03 | REQUEST-INFO | present value of five cash flows | **有** | 247 | selected card is applicable | F | F |
| mq-06 | CLARIFY | growth of a liability balance | **有** | 687 | selected annualized-return card 2331 is a near neighbor | number | M |
| mq-17 | REQUEST-INFO | one-year holding-period return with coupon | **有** | 477 | selected card is applicable | number | F |
| mq-25 | REQUEST-INFO | percentage drop from high to low price | **有** | 687 | selected Q/Q card 2532 is a near neighbor | F | F |
| mq-36 | CLARIFY | level annuity payment | **有** | 12 | selected card is applicable | number | number |
| mq-39 | CLARIFY | growth of a tax-benefit figure | **有** | 687 | selected card 313 is unrelated | M | M |
| mq-41 | REQUEST-INFO | year-over-year growth of a balance | **有** | 687 | selected holding-period card 1063 is a near neighbor | F | F |
| mq-43 | REQUEST-INFO | liabilities as a share of liabilities plus equity | 沒有 | — | debt-to-capitalization card 2120 is a different ratio; same verdict as mq-70 | F | F |
| mq-52 | REQUEST-INFO | real return by the exact Fisher relation | **有** | 2290 | after-tax real return with a zero tax rate | F | F |
| mq-54 | REQUEST-INFO | money-market yield on a 360-day basis | **有** | 950 | selected commercial-paper card 123 is a near neighbor | M | F |
| mq-58 | CLARIFY | change in a rate-sensitivity figure | **有** | 687 | one reading of an ambiguous ask; selected EVE card 2738 is unrelated | F | N |
| mq-80 | CLARIFY | return on a note with monthly compounding | **有** | 431 | selected YTM card 30 is a near neighbor | M | number |

## 三、照「有／無適用合約」切開後的結果

| 確實有適用合約的題目 | VerifiQuant | R-PAL+（單一 prompt） |
|---|---|---|
| REQUEST-INFO（7 題） | F 4/7；直接給數字 1（附 I_SOFT 警告）、M 1、停在 N 1 | F 7/7 |
| CLARIFY（6 題） | I_HARD 0；用 M 反問 3、F 1；直接給數字 2 | I_HARD 0；M 2；直接給數字 3；N 1 |

**scope signal 的正確率**：人工集 52 次停在 N 中，有 8 次其實庫裡有適用合約（ANSWER 類 7 次、REQUEST-INFO 類 1 次，即 mq-68），44 次正確（含 OUT-OF-SCOPE 的 11 次）。

## 四、確認後要改的地方

1. §5.4（P9，已套用的是代理指標版）：「Even where it selects a contract, it asks for the missing input on 5 of 7 such items, against 7 of 7 for R-PAL+」改成以判定為準：「Where an applicable contract exists, it asks for the missing input on 4 of 7 such items, against 7 of 7 for R-PAL+」。長度相同。
2. 附錄 E：加一張「有／無適用合約」的表（上面第三節），並寫 scope signal 52 次中 8 次錯誤。
3. 稽核：從 `nonanswer_attribution.csv` 重算 4/7、7/7、8/52，正文或附錄寫的數字對不上就報錯。
