# Retrieval-matched PAL/PoT baseline（250Q）

日期：2026-07-29  
狀態：250/250 完成；未修改論文。

## 1. 結論先行

在 canonical 250Q clean set 上，加入與 VerifiQuant 完全相同 top-4 retrieval 的
PAL/PoT baseline（以下稱 R-PAL）得到：

- 原始 run：235 correct / 7 silent wrong / 8 abstain，accuracy 94.0%。
- 修正 evaluator artifacts，並統一改用 VQ canonical scorer 後：242 / 6 / 2，
  accuracy 96.8%，SWR 2.4%，coverage 99.2%，selective accuracy 97.6%。
- 相較 VQ K=3，R-PAL audited accuracy 高 3.2 percentage points、SWR 低 0.8 points；
  但 250 題 paired test 尚未達 0.05 顯著水準（accuracy McNemar p=0.134；
  SWR p=0.791）。因此應寫成「數值較高但統計上未能區分」，不能寫成顯著勝出。
- 相較 corrected plain PAL，retrieval 帶來 +5.6 points accuracy、-4.8 points SWR，
  兩者 paired differences 均顯著。

這個結果表示：目前 clean benchmark 不支持「contracts/gates 在 clean accuracy 或 SWR
上優於 retrieval+execution」的強 claim。VerifiQuant 的可防守貢獻應放在可歸因執行、
明示治理、trap/OOD resistance、可審計 gate decisions，而非 clean-set 數值 dominance。

## 2. Matched protocol

| 項目 | R-PAL 設定 |
|---|---|
| 題目 | canonical 250Q；180 medium / 70 hard |
| 模型 | Gemini 2.5 Flash；SDK default generation config |
| Retrieval | 與 canonical VQ 相同 `SQLAlchemyArtifactStore.retrieve_candidates` |
| Query | `question + "\n" + context` |
| Candidate budget | top-4（VQ canonical 相同） |
| Candidate 內容 | retrieval title、summary、public source function |
| Function library | public 3,133-function library |
| LLM calls | 每題一次；產生 self-contained `solve()` |
| 執行 | isolated Python subprocess；10 秒 timeout；只允許 `math` import |
| 明確排除 | per-question `python_solution`、gold、FIC typed schema、invariants、semantic hints、repair rules、gates |
| 評分 | 與既有 250Q 相同 1% tolerance 與 percent/decimal normalization |

Retrieval target recall@4 為 237/250 = 94.8%（rank 1/2/3/4：191/24/16/6）。
13 個 target-function miss 全部答對；樣本很小，而且 top-4 仍可能含近義函式，所以不能據此
推論 retrieval 無用。與 plain PAL 的 paired comparison 才是現有的 retrieval 增益證據。

## 3. 同範圍結果

| System | Correct | SW | Abstain | Accuracy | SWR | Coverage | Sel. Acc. |
|---|---:|---:|---:|---:|---:|---:|---:|
| VQ K=1 autonomous | 203 | 7 | 40 | 81.2% | 2.8% | 84.0% | 96.7% |
| VQ K=3 clarification-assisted | 234 | 8 | 8 | 93.6% | 3.2% | 96.8% | 96.7% |
| Plain PAL（legacy reported） | 226 | 20 | 4 | 90.4% | 8.0% | 98.4% | 91.9% |
| Plain PAL（unified VQ scorer） | 228 | 18 | 4 | 91.2% | 7.2% | 98.4% | 92.7% |
| R-PAL top-4（raw, legacy scorer） | 235 | 7 | 8 | 94.0% | 2.8% | 96.8% | 97.1% |
| **R-PAL top-4（audited, unified VQ scorer）** | **242** | **6** | **2** | **96.8%** | **2.4%** | **99.2%** | **97.6%** |

R-PAL audited 的 95% Wilson intervals：accuracy [93.81%, 98.37%]；
SWR [1.10%, 5.14%]。

### Paired comparisons（R-PAL audited 為 primary）

| Comparator | Accuracy diff | 95% paired bootstrap | McNemar p | SWR diff | 95% paired bootstrap | McNemar p |
|---|---:|---:|---:|---:|---:|---:|
| VQ K=3 | +3.2 pp | [-0.4, 6.8] | 0.1338 | -0.8 pp | [-3.6, 2.0] | 0.7905 |
| VQ K=1 | +15.6 pp | [10.8, 20.8] | 2.78e-9 | -0.4 pp | [-2.8, 2.0] | 1.0000 |
| Plain PAL unified | +5.6 pp | [2.4, 9.2] | 0.00258 | -4.8 pp | [-7.6, -2.0] | 0.00183 |

VQ K=1 是公平的 autonomous comparison；VQ K=3 是現有 headline，但含 clarification-assisted
recovery，兩者應並列而不能混稱同一 deployment setting。

## 4. Evaluator audit（不覆寫 raw）

原始 run 使用既有 regex denylist 與舊 `_num`：

1. 六個程式只是在 docstring/comment 行首寫到 `from ...`，被誤認為 import。改用 AST
   檢查並在同一 isolated runner 重播後，六題輸出全部吻合 gold。
2. `test-1842=True` 與 `test-2059=False` 的模型輸出與布林 gold 完全一致；舊 scorer
   無法解析 `True/False`，錯算成 silent wrong。
3. `test-2125` 確實使用 `from datetime import datetime`，依「只允許 math」政策維持 abstain。

此外，VQ scorer 對明示 `nearest integer/whole` 的題目採 round 後精確比較；這使
`test-1890`（139000 vs 138000）由舊 R-PAL scorer 的 correct 改為 silent wrong。
因此 raw 235/7/8 經確定性審計與共同量尺重評成 242/6/2。Audited JSONL 保留每題
`audit_action` 與 `deny_detail`，raw JSONL 未修改。

Plain PAL 的舊 log 沒保存 generated code；共同 VQ scorer 修正三個 boolean cases，
同時將一個 nearest-integer tolerance pass 改為 SW（226/20/4 → 228/18/4）。
其四個 legacy denylist abstentions 無法做同樣 replay audit。
這使 corrected plain PAL 仍是保守估計，不應假裝已完全審計。

## 5. 六個 residual SW 的人工檢視

六題全部是 retrieval hit，且都屬於模型輸出與 benchmark function/convention 的衝突；
primary table 仍嚴格依 benchmark 計為 SW，以下不擅自改分。

| Case | R-PAL / gold | 檢視 |
|---|---:|---|
| test-1490 | 3.60 / 3.35 | rates 已是 percentage points；25 bps 應加 0.25 points。gold solution 卻除以 10,000 後直接加到 3.35，scale 不一致。 |
| test-2021 | 6.67 / 6.33 | context 明稱 4% 為 after-tax debt cost；gold solution 又乘 `(1-tax)`，疑似重複扣稅。 |
| test-1890 | 139000 / 138000 | `other operating activities decreased` 的資產／負債方向未明；R-PAL 明示採 operating-asset decrease 的 +500，gold 採 -500。共同 nearest-whole 規則下仍依 gold 計 SW。 |
| test-1435 | 1.1387 / 1.1614 | context 明定 1 USD = 1.15 EUR；R-PAL 的 EUR/USD quote-direction parity 推導自洽，gold 使用反向利率比。 |
| test-1577 | 1.2409 / 1.2592 | 同為 FX quote-direction 問題；「Country C to Country B」語句仍可能被兩種方向解讀，應由 domain expert 定義 quote convention。 |
| test-1958 | 4 / 5 | R-PAL 線性插值得 4.21 年並依題意 round 到 4；gold function 回傳首次整年回收期 5，屬 payback-period convention 差異。 |

至少前三題有強烈 benchmark inconsistency 證據；其餘三題需教授/domain expert adjudication。
在正式 adjudication 前，不建議把 hypothetical 248/250、0 SW 當 headline。

## 6. 成本（描述性）

- R-PAL：2,635.5 秒，約 10.5 秒/題，一次 LLM call/題。
- Plain PAL：1,799.2 秒，約 7.2 秒/題，一次 LLM call/題。
- VQ K=3 既有 run：約 17.3 秒/題、估計 3.78 calls/題。

這些 run 不在完全相同 API 時段，wall-clock 只能描述，不能當嚴格 latency benchmark。
若論文要寫效率 claim，需記錄 token usage 並在相同時段重跑或做交錯測試。

## 7. 接下來的優先順序

### 一定要做（P0）

1. 教授/domain expert 對六個 residual SW 做 blind adjudication，並凍結 decision log。
2. 將 matched R-PAL 跑在 trap set 與 OOD-family set；這才直接檢驗 contracts/gates 的治理價值。
3. 若論文保留 clean-set superiority claim，必須改寫：目前只能說與 VQ K=3 統計上未能區分，
   不能說 VQ clean accuracy/SWR 勝過 R-PAL。
4. 所有後續 baseline 共用 boolean-aware scorer 與 AST policy，並保留 generated code。

### 應做（P1）

1. 補 token/cost accounting；R-PAL prompt 較長但 calls 較少。
2. 對 retrieval 做 controlled ablation（top-0、top-1、top-4），最好固定同一批模型設定與時段。
3. 報告 candidate recall、retrieval-hit/miss 與人工 failure taxonomy，不只報 aggregate accuracy。

### 可放附錄（P2）

1. 完整 prompt、manifest、changed-case audit table。
2. Wilson intervals、paired contingency tables、13 個 retrieval-miss cases。
3. 六個 benchmark-disagreement cases 的題目、function、模型程式與 adjudication rationale。

## 8. 可重現命令

不需新 API call 的 audit/comparison：

```bash
.venv/bin/python scripts/analyze_retrieval_pal_baseline.py
```

只驗證 retrieval/prompt、完全不呼叫模型：

```bash
PYTHONPATH=. .venv/bin/python scripts/run_retrieval_pal_baseline.py --dry-run
```
