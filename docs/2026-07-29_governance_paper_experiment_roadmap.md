# Governance-first 8-page paper：實驗位置、假設與執行順序

日期：2026-07-29  
狀態：working draft；「預期」均是假設，不是結果。

## 目前已能寫入正文的證據

- Clean paired 50Q：VQ K=1 accuracy/SWR/coverage = 74.0/0.0/74.0%；plain R-PAL = 98.0/2.0/100.0%；prompted R-PAL 三輪平均 = 95.3/0.7/96.0%。
- 有效 governance set 僅採 F/E/I/N，共 40 traps、17 個 base questions；M 不納入量化。
- FNAR：VQ K=1 7.5%；prompted R-PAL 29.2%；plain R-PAL 61.7%。
- Prompted-minus-VQ FNAR 差距 21.7 points；以 base question 分群、保留同源 perturbations 與三次重跑的 bootstrap 95% CI = [5.8, 37.0]。
- Exact diagnosis/action：VQ 82.5/82.5%；prompted R-PAL 56.7/41.7%。
- M 的既有 generator 未隔離真正的多重 intent，只能保留質性案例。

## 八頁正文中的放置位置

| 實驗 | 目前位置 | 跑完後如何更新 |
|---|---|---|
| Clean matched R-PAL | Sec. 5.1、paired clean table | 保留作為 boundary；不可因 R-PAL 較強而刪除 |
| F/E/I/N matched governance | Sec. 5.2、aggregate + operator tables | 80-trap confirmatory 結果直接替換目前 40-trap pilot |
| 三輪穩定性與 CI | Sec. 5.3 | 換成 VQ 與 R-PAL 都重跑後的 hierarchical/base-clustered interval |
| Clarification recovery | Sec. 5.4 | 加 recovery rate、wrong recovery、平均 turns、requested-field/action fidelity |
| M ambiguity | Sec. 5.5 | ICAIF 仍以質性案例呈現；只有通過獨立標註後才升格量化 |
| Retrieval near-OOD | Sec. 5.2 operator table 後 | 新增 N/near-miss paragraph；若版面不足移 supplementary |
| Prompt/model robustness | limitations 前一段或 supplementary | 報方向是否維持，不要求每個模型同幅度 |
| Independent messy-query set | limitations 前；若完成則升為主要外部效度證據 | 可將 contract-derived set 降為 mechanism unit test |

## P0：先修 confirmatory dataset，暫不呼叫外部模型

### 已實際執行的 generator audit

已用 250Q pool 與 `--per-operator 20` 在 `/tmp/vq_governance_confirmatory_20` 產生 100 題。表面上 F/E/I/N/M 各 20、`needs_review=0`，但不可直接送模型：

- 100 traps 只來自 37 個 base questions。
- E 的 20 題全部是「把某個欄位改成負值」，缺乏 zero denominator、probability/rate domain、scale、timing 等邊界多樣性。
- I 的 20 題全部是 I_SOFT；多數只是刪除 percentage/decimal phrase，沒有 I_HARD 平衡。
- N 的 20 題全部使用同一個 Heston pricing ask template，會讓拒答變成模板辨識。
- M 雖被 generator 標成 20 個 clean labels，但同樣沒有證明 context 支援至少兩個不可唯一決定的 metric，仍不能量化。

結論：增加到每類 20 題是必要但不充分；現在直接跑只會放大既有 generator bias。

### 必須先完成的資料修正

1. F：20 個不同 base questions；缺失欄位涵蓋 numerator、denominator、rate、time、balance、count。
2. E：至少四個 violation families，每類至少 4 題；negative-value 題不超過 8/20。
3. I：10 I_HARD + 10 I_SOFT；涵蓋 scale、timing、basis、direction、per-share/total，不可只測百分比。
4. N：至少五種 unsupported asks，每個 template 不超過 4 題；加入「相近但不適用」而非只有明顯跨域 ask。
5. 去重：同一 base question 最多承擔兩個 operator；統計以 base-question cluster 為單位。
6. 凍結：在任何新模型輸出前寫入 manifest、hash、operator audit 與排除理由。

完成條件：80/80 題標籤可由 contract/operator 直接驗證，0 題依賴 gold number 判斷，且兩位 reviewer 對 expected safe action 達成一致。

## P1：80-trap matched confirmatory（一定要做）

### 系統

- VQ K=1：frozen contracts、無 clarification。
- Prompted R-PAL：同 Gemini backbone、top-4、public functions、sandbox、統一 scorer；三次獨立 run。
- Plain R-PAL：建議保留為 attribution control；若成本吃緊，可只跑一次，主比較仍是 prompted R-PAL。

### 主要假設與判準

- H1：`FNAR(prompted R-PAL) - FNAR(VQ) >= 10 percentage points`。
- H1 的 base-clustered 95% CI 下界需大於 0；同源 operator 與重跑不可視為獨立題。
- H2：VQ exact action accuracy 至少高 15 points；這是「知道如何繼續」的核心次要主張。
- H3：優勢應集中在 E/I；F/N 允許相同，不能事後把失敗 operator 隱藏。
- H4：clean R-PAL 可維持較高 accuracy/coverage；這是預期 boundary，不是 VQ 失敗。

### Go/no-go

- GO：H1 通過，且 exact diagnosis/action 不低於 prompted R-PAL。
- NARROW：FNAR 點估計仍有至少 10 points，但 CI 跨 0；正文改稱 pilot evidence，confirmatory 移 future work。
- NO-GO：prompted R-PAL 在 FNAR 與 action quality 同時追平或勝過 VQ，或優勢只存在於單一模板。

## P2：Retrieval near-OOD（高優先）

設計兩個條件，各 20--30 題：

1. target FIC 從 library 移除；
2. target FIC 移除但保留語義相近、公式不適用的鄰近 functions。

指標：unsafe force-fit rate、N diagnosis accuracy、錯誤選中的 function family、structured scope explanation。預期：第二條件比第一條件更難；若 contracts 的 scope gate 有效，VQ 的 unsafe force-fit 應低於 R-PAL。這個實驗若成功，會把 N 從「prompt 可追平」提升成更貼近 retrieval failure 的證據。

## P3：Clarification recovery（高優先）

只對 VQ 正確攔截的 F/I_HARD 題提供最小充分澄清，不提供 gold number 或 worked solution。

指標：safe recovery rate、wrong recovery rate、平均 turns、是否只詢問 expected field/option、澄清後是否換錯 FIC。預期：大多數正確 exit 可在一輪恢復，且錯誤恢復不高於 clean SWR。若只會拒答、不能安全恢復，論文應刪除「make safe continuation possible」的強句。

## P4：Independent messy-query set（外部效度，最好做）

由不看 VQ contracts 的作者或領域標註者撰寫 80--100 題，雙人標註 safe action、缺失資訊與可接受澄清。至少保留 20% disagreement pool 並由第三人 adjudicate，避免只留下容易題。

預期：差距可能小於 contract-derived set，但方向應維持。若差距只存在於 contract-derived mutations，主張必須改成「contracts provide mechanism-aligned protection」，不能泛化到真實使用者問題。

## P5：M benchmark（ICAIF 可延後，ACL 再完整化）

每題 context 必須同時支持至少兩個合理 metric；問題不能含唯一 metric cue。標註需列出所有 admissible targets，而非只給一個 gold label。指標應是 option recall、premature computation rate、clarification usefulness，不宜只用 exact M label。

預期：VQ 應少做 premature computation，並提供有限選項。若 R-PAL 同樣穩定且選項品質相當，就不把 M 當作 VQ 優勢。

## 執行與外傳資料邊界

P0 只使用本地資料，已開始且不需新增外部請求。P1 之後會把新題目/context 與 top-4 public functions 傳給外部模型；這超出先前已完成的 450 次請求範圍。正式送出前需再確認：最終題數、系統組合、每個系統重跑次數與可接受的總 request 上限。
