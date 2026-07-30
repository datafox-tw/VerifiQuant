# VQ vs. matched Retrieval-PAL：三輪 governance 結果與投稿判定

日期：2026-07-29  
模型：Gemini 2.5 Flash  
檢索：與 VQ 相同的 `SQLAlchemyArtifactStore.retrieve_candidates`、top-4、public function library  
總外部請求：450（3 輪 ×〔50 plain traps + 50 prompted traps + 50 prompted clean〕）

## 結論先行

**建議繼續投稿，但只保留窄化後的 executed-governance 主張。**

R-PAL 並未全面碾壓 VQ。相反地，在這個 50 題、由 contracts 所構造的 governance stress set 上，VQ K=1 的 false numeric acceptance rate（FNAR）是 26%，低於 plain R-PAL 的三輪平均 69.3%，也低於明確加入 governance taxonomy 的 prompted R-PAL 平均 42.0%。相對 prompted R-PAL 的絕對差距為 16.0 個百分點；以 50 個獨立題目為群組、保留題內三次重跑的 bootstrap 95% CI 為 [4.0, 29.3] 個百分點。

但是，結果**不支持**以下兩個廣泛主張：

1. VQ 是更強的 clean financial calculation engine。
2. VQ 已能處理使用者不知道自己要問什麼的 open-ended intent ambiguity。

Clean set 上 R-PAL 明顯更強；而 VQ 在全部 10 個 M traps 上都輸出不安全數字。因此，論文中心應改成：

> Retrieval-PAL solves well-posed financial questions efficiently; VerifiQuant adds value when known financial requirements and boundaries can be compiled into executable qualification gates.

## 實驗控制

三個比較系統為：

1. **R-PAL plain**：相同 top-4 retrieval 與函式庫；模型直接產生並執行程式，或以非結構化文字拒答。
2. **R-PAL prompted**：相同 retrieval、函式庫、模型與一次呼叫預算；prompt 額外要求 READY/M/N/F/E/I_HARD/I_SOFT、理由、修正目標、澄清問題與可選程式碼，但沒有 deterministic contract gates。
3. **VQ K=1**：既有、凍結的單輪 funnel 結果；不使用 clarification oracle 或 K=3 結果。

R-PAL prompt 不包含 gold、reference solution、trap operator 或 expected diagnostic。計分亦已統一成 VQ 的 1% relative tolerance + 0.01 absolute tolerance，並套用 nearest-integer 與 decimal ×100 規則。

## 主要結果

三輪 R-PAL 結果以平均值呈現；括號為三輪範圍。VQ 是一個既有凍結 run，並未重跑三次，因此不能把其重複列入視為額外觀測。

| System | Trap FNAR ↓ | Structured catch ↑ | Exact diagnostic ↑ | Exact action ↑ | Clean accuracy ↑ | Clean SWR ↓ | Clean coverage ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| VQ K=1 | 26.0% | 74.0% | 66.0% | 66.0% | 74.0% | 0.0% | 74.0% |
| R-PAL plain | 69.3% (68–72) | 0.0% | N/A¹ | N/A¹ | 98.0%² | 2.0%² | 100.0%² |
| R-PAL prompted | 42.0% (38–46) | 58.0% (54–62) | 45.3% (42–50) | 33.3% (30–38) | 95.3% (94–96) | 0.7% (0–2) | 96.0% |

¹ Plain R-PAL 未被要求輸出結構化 diagnostic/action；程式中的 0 不應解釋成公平的 diagnosis ability 比較。  
² Plain R-PAL clean 是既有凍結 baseline，在三份 comparison 中重用，不是三次獨立 clean runs。

### 穩定性

- Prompted R-PAL 的三輪 FNAR：38%、42%、46%；每輪都比 VQ 高至少 12 個百分點，三輪皆通過預先凍結的 KEEP 規則。
- Prompted R-PAL 有 46/50 題（92%）在三輪中的 unsafe/safe 決策完全一致。
- Exact diagnostic label 的三輪一致率為 39/50（78%）；action correctness 一致率為 36/50（72%）。安全決策比診斷歸因穩定，但歸因仍有明顯震盪。
- Prompted clean label 有 49/50 題（98%）三輪一致：47 題三輪都正確、2 題三輪都 abstain，另 1 題只在第三輪 silent wrong。
- VQ 對 plain R-PAL 的 FNAR 優勢為 43.3 個百分點，題目群組 bootstrap 95% CI [30.0, 56.7]；對 prompted R-PAL 為 16.0 個百分點，95% CI [4.0, 29.3]。

## 優勢究竟來自哪裡

下表是每個 trap operator 的 FNAR；R-PAL 數字為三輪平均。

| Operator | VQ K=1 | R-PAL plain | R-PAL prompted | 可支持的判讀 |
|---|---:|---:|---:|---|
| F：缺必要欄位 | 0.0% | 0.0% | 0.0% | 三者都知道不能直接計算；VQ 的價值主要在結構化診斷，而非 FNAR。 |
| E：違反範圍/邊界 | 20.0% | 100.0% | 33.3% | Executed boundary gates 有實質優勢，但仍非完美。 |
| I：convention/單位歧義 | 10.0% | 100.0% | 83.3% | 本次最強的 VQ 證據；但 I 是 9 個 I_SOFT、1 個 I_HARD，不能泛化成所有 ambiguity。 |
| N：library/scope mismatch | 0.0% | 46.7% | 0.0% | Governance prompt 已可追平 VQ 的 FNAR；VQ 的 exact diagnosis 較高。 |
| M：多重/不明 intent | 100.0% | 100.0% | 93.3% | 三者幾乎都失敗；VQ 沒有優勢。必須列為限制。 |

排除目前未解的 M，只看 F/E/I/N 40 題，FNAR 為：

- VQ K=1：7.5%
- R-PAL prompted：平均 29.2%（25.0–32.5%）
- R-PAL plain：平均 61.7%（60.0–65.0%）

這個 subset 是最符合「已知要求可被編譯成 contracts」的證據，但必須明示它是事後按機制範圍解讀，不能取代預先指定的全 50 題 primary result。

## 對論文主張的影響

### 一定要保留

- **問題重構**：financial reliability 包含 query qualification，不只 answer correctness。
- **方法主張**：scope、schema、boundary 與 convention 可被編譯成 executable gates，而不只寫在 prompt 裡。
- **經驗主張**：在 contract-grounded governance perturbations 上，VQ K=1 比 matched prompted R-PAL 有較低 FNAR，且 exact diagnostic/action 較高。
- **誠實的 clean 結果**：retrieval-grounded program execution 在 well-posed inputs 上更有效率；VQ 的價值不是全面提高 answer accuracy。

### 一定要刪除或改寫

- 刪除「VQ 每個指標都勝過 R-PAL」或「VQ 是最佳金融計算器」。
- 不可宣稱 VQ 已解決 open-ended M ambiguity、使用者意圖發現或一般自然 OOD。
- 不可只報 F/E/I/N subset 而藏掉 M；全 50 題與 operator-level table 都要保留。
- 不可把 R-PAL 三次重跑當成 150 個獨立 traps，也不可把凍結的 VQ 數字假裝成三次重跑。

### 適合放限制或附錄

- 50 traps 是從 VQ contracts 產生的 mechanism-aligned stress set，可能有利於 contract-gated system；尚未等同自然分布的 messy user queries。
- I stratum 嚴重偏向 I_SOFT（9/10），需要額外 I_HARD 或人工撰寫 ambiguity set 才能做廣泛主張。
- VQ 的 M=100% FNAR 是目前方法/實作的硬限制；若論文仍把 M 列為能力，必須先修方法再重跑，而不是靠文案解釋。
- R-PAL prompted 只有一種 prompt 與一個模型；尚未證明所有 prompt-based governance 都較差。
- Clean utility 的比較來源不完全對稱：VQ 與 plain 是既有凍結結果，只有 prompted R-PAL 被重跑三次。

## 投稿 go/no-go

依照看結果前凍結的規則，三輪判定皆為 `KEEP_EXECUTED_GOVERNANCE_CLAIM`。因此：

**GO：繼續一週後的 deadline，但以窄化主張投稿。**

最安全的摘要句是：

> On 50 contract-derived governance perturbations, VerifiQuant K=1 reduced false numeric acceptance from 42.0% for a governance-prompted matched Retrieval-PAL control to 26.0% (16.0-point difference; question-clustered bootstrap 95% CI [4.0, 29.3]), while Retrieval-PAL remained substantially stronger on clean execution-ready questions. The gain concentrated in executable boundary and convention checks; neither system reliably resolved open-ended intent ambiguity.

這支持的是「**何時 executed governance 有用**」，不是「VQ 在所有金融推理上更強」。

## 可重現檔案

- Frozen protocol：`docs/2026-07-29_governance_go_no_go_protocol.md`
- R-PAL governance runner：`scripts/run_retrieval_pal_governance_eval.py`
- Single-run scorer：`scripts/analyze_governance_eval.py`
- Multi-run clustered analysis：`scripts/analyze_governance_multirun.py`
- Aggregate machine-readable result：`verifiquant/data/runs/governance_eval/governance_multirun_summary.json`
- Per-run raw and scored outputs：`verifiquant/data/runs/governance_eval/gov_run{1,2,3}_*.jsonl` and `gov_run{1,2,3}_comparison.json`
