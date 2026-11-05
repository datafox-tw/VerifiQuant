# 重複執行比較（50 題，3 次）

| 執行 | 正確 | 輸出但錯 | 追問 | 拒答 | 錯誤 |
|---|---|---|---|---|---|
| full_v4_r1 | 34 | 1 | 3 | 12 | 0 |
| full_v4_r2 | 32 | 1 | 3 | 14 | 0 |
| full_v4_r3 | 33 | 1 | 3 | 13 | 0 |
| full_v3_r1（對照） | 33 | 1 | 3 | 13 | 0 |

3 次結果完全相同：47/50 題；每次都正確 31 題，至少一次正確 34 題；至少一次輸出但錯 1 題。

### 會翻轉的題目

| 題號 | full_v4_r1 | full_v4_r2 | full_v4_r3 | 停在 |
|---|---|---|---|---|
| test-1435 | 正確 | 拒答 | 正確 | None, inspect |
| test-1575 | 正確 | 正確 | 拒答 | None, inspect |
| test-1709 | 正確 | 拒答 | 正確 | None, inspect |

### 和 full_v3_r1 不同的題目（取多數結果）

| 題號 | 之前 | 現在 | 現在停在 |
|---|---|---|---|
| test-1593 | 拒答 | 正確 | None |

### 結構輸出題：選取路徑與元素探針

| 執行 | 題號 | 路徑 | 狀態 | 探針抓到 | 結果 |
|---|---|---|---|---|---|
| full_v4_r1 | test-1597 | `[1]` | grounded | 3 | 正確（輸出） |
| full_v4_r1 | test-1921 | `[-1, 3]` | grounded | 1 | 正確（輸出） |
| full_v4_r1 | test-1546 | `[5]` | grounded | 2 | 拒答（inspect） |
| full_v4_r1 | test-1249 | `['Sales']` | grounded | 3 | 正確（輸出） |
| full_v4_r1 | test-2218 | `[0, 'interest_paid']` | grounded | — | 拒答（verify） |
| full_v4_r1 | test-2236 | `['upper_band', -1]` | grounded | 0 | 拒答（verify） |
| full_v4_r1 | test-2189 | `[0]` | grounded | 4 | 正確（輸出） |
| full_v4_r2 | test-1597 | `[1]` | grounded | 3 | 正確（輸出） |
| full_v4_r2 | test-1921 | `[-1, 3]` | grounded | 1 | 正確（輸出） |
| full_v4_r2 | test-1546 | `[5]` | grounded | 2 | 拒答（inspect） |
| full_v4_r2 | test-1249 | `['Sales']` | grounded | 3 | 正確（輸出） |
| full_v4_r2 | test-2218 | `[0, 'interest_paid']` | grounded | — | 拒答（verify） |
| full_v4_r2 | test-2236 | `['upper_band', -1]` | grounded | 0 | 拒答（verify） |
| full_v4_r2 | test-2189 | `[0]` | grounded | 4 | 正確（輸出） |
| full_v4_r3 | test-1597 | `[1]` | grounded | 3 | 正確（輸出） |
| full_v4_r3 | test-1921 | `[-1, 3]` | grounded | 1 | 正確（輸出） |
| full_v4_r3 | test-1546 | `[5]` | grounded | 2 | 拒答（inspect） |
| full_v4_r3 | test-1249 | `['Sales']` | grounded | 3 | 正確（輸出） |
| full_v4_r3 | test-2218 | `[0, 'interest_paid']` | grounded | — | 拒答（verify） |
| full_v4_r3 | test-2236 | `['upper_band', -1]` | grounded | 0 | 拒答（verify） |
| full_v4_r3 | test-2189 | `[0]` | grounded | 4 | 正確（輸出） |
