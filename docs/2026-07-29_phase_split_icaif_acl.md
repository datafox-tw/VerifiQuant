# Phase split: ICAIF (phase 1) vs ACL (phase 2)

日期：2026-07-29
狀態：P0 完成（本地、無外部呼叫）。P1 待 egress 批准。

這份文件把工作切成兩階段，讓 8-page ICAIF 只承載守得住的窄化主張，把完整
Retrieval-PAL 比較與外部效度留給 ACL。與 `2026-07-29_governance_paper_experiment_roadmap.md`
搭配閱讀（roadmap 給 P0–P5 細節，本文件給階段邊界與現況）。

---

## 核心定位（兩階段共用）

> Retrieval-PAL solves well-posed financial questions efficiently; VerifiQuant adds
> value when known financial requirements and boundaries can be compiled into
> executable qualification gates.

- Clean set：R-PAL 與 VQ 統計上不可區分（250Q，audited）。**不主張 clean-set 數值優勢。**
- 分離點：governance-perturbed inputs（F/E/I/N traps）與 attribution。

---

## Phase 1 — ICAIF（8 pages, no supplement）

### 已鎖定（由已完成實驗支撐，不會變）

| 內容 | 來源 | 狀態 |
|---|---|---|
| Reframe：VQ = governance boundary defender，非 clean-set calculator | 250Q audited R-PAL clean | 已寫入 paper |
| 主表加入 R-PAL / plain PAL execution baselines | `retrieval_pal_baseline/` | 已寫入 |
| Gate ladder（SWR 18%→3.6%）、libsweep（abstention tracks coverage）、K-ablation、risk–coverage、RepairBench | paper_v2_250 既有 run | 既有，保留 |

### 待 P1 替換的 provisional 數字（目前來自 40-trap pilot）

- Trap table 的 operator FNAR（VQ 7.5% vs prompted R-PAL 29.2%）、CI [5.8, 37.0]、
  exact diagnosis/action。**P1 confirmatory 完成後直接替換；在那之前是 pilot evidence。**

### Phase-1 實驗範圍（只到這裡）

- **P0（完成）**：平衡 80-trap confirmatory dataset（見下方「P0 交付」）。
- **P1（待批准）**：80-trap × {VQ K=1 frozen, prompted R-PAL ×3, plain R-PAL}，統一 scorer。
  這是替換 pilot 數字、給出 base-clustered CI 的關鍵。
- （若有時間）**P2 retrieval near-OOD** 一小節，把 N 從「prompt 可追平」升成 retrieval force-fit 證據。

### Phase-1 誠實邊界（寫在 Limitations，已部分入稿）

- Clean set：R-PAL 追平 VQ。
- Governance 優勢測在 contract-derived traps（mechanism-aligned，可能利於 contract-gated）。
- **I_HARD 只有 6 題**（見 P0 發現）；I_HARD/I_SOFT FNAR 分開報，不泛化小 hard stratum。
- M：所有系統皆失敗（≥93% FNAR），列為共同 open problem，非 VQ 弱點。

---

## Phase 2 — ACL（完整版，有 supplement 空間）

| 項目 | 為何留到 ACL |
|---|---|
| 完整 R-PAL 矩陣（clean + trap + OOD，統一時段、token/cost accounting） | 需完整 latency/cost 重跑，超出 8 頁 |
| **P4 獨立 messy-query 外部效度集**（雙人標註、disagreement pool） | 需人力標註；是真正的外部效度證據 |
| **P5 M-benchmark**（每題 context 支援 ≥2 admissible metrics） | 需重做 generator 保證多重 intent；option-recall 指標 |
| **Hand/Tier-2 built I_HARD set（補足 10+ hard）** | P0 證明 phrase-deletion 只能 ground 6 hard（見下） |
| Online literal-grounding provenance guard（目前 offline prototype） | 需上線評估 false-downgrade / derived-but-legal |
| 更廣 backbone / cross-vendor replication | 需多模型預算 |

---

## P0 交付（本次完成）

**產物**
- Generator：`verifiquant/pipeline/build_trap_set_v2.py`（保留舊 `build_trap_set.py` 供 provenance）
- Dataset：`verifiquant/data/runs/governance_confirmatory/trap_set.jsonl`（80 題）
- Manifest：`verifiquant/data/runs/governance_confirmatory/trap_manifest.json`（hashes、family 分布、base-cluster、i_hard_supply、known_constraints）

**重現指令**
```bash
python3 verifiquant/pipeline/build_trap_set_v2.py \
  --questions verifiquant/data/runs/paper_v2_250/questions_250.jsonl \
  --canonical verifiquant/data/runs/paper_v2_250/results/vq_flash/output.jsonl \
  --db        verifiquant/data/runs/paper_v2_250/fic/cards.db \
  --out-dir   verifiquant/data/runs/governance_confirmatory \
  --per-operator 20
```

**修掉的三個 pilot bias**
| 之前（50-trap pilot） | 現在（80-trap confirmatory） |
|---|---|
| E 全部 negative-value（10/10） | E 4 injection families：negativity 8 / rate-out-of-unit 4 / zero-denominator 4 / magnitude 4，**每題以 contract predicate eval 驗證會觸發、clean 值不觸發** |
| I 9/10 soft，8 題同一 rule | I 6 hard / 14 soft，hard/soft 由 FIC rule suffix 決定 |
| N 單一 Heston 模板 | N 7 模板（含 near-miss），單模板 ≤4 |
| 50 traps / 17 bases，無 dedup | 80 traps / 52 bases，每 base ≤2 operators |
| 標籤靠假設 | 0 needs_review；E 標籤全部 contract-verified |

**P0 發現（會影響 ACL 規劃）：I_HARD 供給上限**
- 243 個 selected cases 中，40 個 FIC 宣告 hard ambiguity。
- 其中只有 **6** 可用 phrase-deletion ground；17 個「宣告但題目無可刪除片語」（convention 藏在 context）；
  19 個「無對應片語」（method/basis choice，如 ROA definition、cash-flow method）。
- 結論：hard, computation-changing ambiguity 在 phrase-deletable traps 中天生稀少。
  ICAIF 用 6 hard 並明示；補足到 10+ 需 hand/Tier-2 建構 → ACL。

---

## P1 confirmatory 結果（完成，GO）

80-trap balanced set，VQ K=1 vs plain/prompted R-PAL（prompted 三輪平均）。
`verifiquant/data/runs/governance_confirmatory/confirmatory_summary.json`

| Overall FNAR | VQ K=1 | R-PAL plain | R-PAL prompted |
|---|---:|---:|---:|
| Confirmatory | **11.3%** | 48.7% | 27.5% (26/30/26) |
| Pilot | 7.5% | 61.7% | 29.2% |

- Gap(prompted−VQ) = +16.3 pts，base-clustered 95% CI [5.8, 26.4]，排除 0 → **GO**（H1 通過）。
- Per-operator FNAR：VQ {F10,E10,I20,N5}；plain {F10,E90,I95,N0}；prompted {F0,E30,I80,N0}。
- I_HARD/I_SOFT：VQ 16.7/21.4；prompted 50.0/92.9（hard n=6，小樣本，分開報）。
- diag/action：VQ 75.0/73.8；prompted 65.4/55.8。
- 誠實衰減：平衡後更難（VQ 7.5→11.3，gap 21.7→16.3），主張仍成立。

## P2 retrieval near-OOD 結果（完成）

24 probes，target family 從 library 移除。force-fit = 無適用公式時仍輸出 confident number。
`verifiquant/data/runs/nearood/nearood_{vq,rpal}_run1_summary.json`

| force-fit rate | bare（target+neighbors 移除） | nearmiss（保留 neighbors） |
|---|---:|---:|
| R-PAL plain | 100% (24/24) | 100% (24/24) |
| VQ K=1 | 0% (24/24 走 N gate) | 8.3% (2/24；neighbor selection slip) |

- 這是比 trap-N 更強的 C1 scope 證據：trap-N 太明顯所有系統都拒答；near-OOD 的 ask 看似 in-domain，只有 executed scope gate 能分辨。
- 2 個 nearmiss force-fit = selection-frontier slip（§attribution）；nearmiss>bare 符合預期。

**paper 整合完成**：confirmatory 數字已換入（trap table 改 80-trap、3 系統；spine C1/C2/C3、gate-ladder I 數字更新）；near-OOD 新增 §sec:nearood；pilotnote 清空。paper 目前 10 頁（user 已放寬八頁限制，先寫完再決定框架）。

## 2026-07-30 重構：4-lens balance framing + MECE split

**新敘事（取代「beat CoT」）**：VQ 的貢獻是 *balance across lenses*，不是 clean-set accuracy。
雙重反轉（皆真實數據）：R-PAL 在 clean 平 VQ 但在 scope 最差（force-fit 100%）；CoT clean 最弱且 OOD 非選擇性。只有 VQ 四個 lens 都守得住。

**4 lenses（每個用公平的系統）**：
1. Clean 250Q：CoT/R-PAL/VQ — R-PAL≈VQ≫CoT。
2. Governance 80-trap：R-PAL/R-PAL⁺/VQ — VQ 11.3 ≪ 27.5 ≪ 48.7。
3. Near-OOD scope 24：R-PAL/VQ — R-PAL 100% vs VQ 0–8%（CoT 100% 但無 library→footnote，不公平）。
4. OOD-family selectivity 199：CoT/VQ — VQ gap 69.3 vs CoT 11.3；R-PAL-199 in flight 補第三系統。

**已改入 paper**：contribution 1（query qualification）、contribution 2（balance/4-lens）、§5 開頭 4-lens frame、§5.1→"clean lens"、§5.6 CoT selectivity 明確化、risk-coverage 標為 CoT-specific 支撐（可跳過）。near-OOD §sec:nearood 已在。paper 11 頁。

**ood_eval = 既有 refusal-permitted CoT 來源**：paper 的 "~11 points" 就是 CoT 60.3/49.0；VQ 88.9/19.6=69.3。非新實驗，只是把數字明講。

**進行中**：R-PAL on 199-OOD（bmpcyi57g）→ 補 selectivity lens 第三系統，落地後建 unified OOD abstain table（VQ 88.9 / CoT 60.3 / R-PAL ~low）+ 可選 balance scorecard（✓/✗ per lens）。

**MECE ICAIF/ACL**：
- ICAIF：clean、80-trap、near-OOD、OOD-selectivity、governance-transfer；support = gate-ladder/libsweep/attribution/hint-trigger/RepairBench+transforms。
- ACL：risk-coverage perfect-rejector 深入、12-cell 多模型 refusal grid、K-ablation recovery dynamics、binding/provenance guard、multi-agent 深入、hand-built I_HARD(>6)、messy-query(P4)、M-benchmark(P5)、R-PAL cost/latency matrix。
- MECE 邊界：K-ablation / refusal / multi-agent 為 split（headline→ICAIF，deep-dive→ACL）。

## 2026-07-30 demote-to-trade-off（完成）

OOD 誠實降級：R-PAL OOD 95% correct、selective acc 95.5%；VQ OOD selective acc 只有 63.6%（selection-frontier），SW 兩者皆 ~4%。**OOD 不是 VQ 的 advantage**。
- Advantage 收斂為：clean parity + **malformed-input (trap) safety**（VQ 11.3% vs plain 48.7% FNAR）。
- OOD = disclosed trade-off：libsweep=selective-abstention 是 architectural property（tracks coverage）；tab:oodsel=誠實三系統 trade-off 表；明講「a user who wants an answer is better served by R-PAL」「does not lower SW risk」；verifiability-value → future work/ACL。
- 已改：contribution 2、§5 lead、§6.1 oodfam、§6.2 nearood、Limitations（40→80-trap 更新 + trade-off runs against us）、Conclusion（malformed win + OOD trade-off）。
- **Section notes 已加**（% NOTE [ROLE | trim]）於每個 section/關鍵 subsection，供 8 頁 trim 導航：CORE / SUPPORT / TRADE-OFF / METHOD / ACL-candidate / MERGE。
- 完整性檢查：build OK、無 undefined ref/macro、無殘留 OOD over-claim。paper 11 頁（待 user trim 到 8）。

trim 導航（notes 內）：ACL-candidates = risk-coverage(5.2)、binding(6.4)、RepairBench(3.4)、hint-trigger 細節(6.3)、full refusal grid；MERGE = near-OOD(6.2)→oodfam(6.1)。

## 下一步（需要 user 決定）

1. **P1 egress 批准**：把 80 題 + top-4 public functions 傳給 Gemini（prompted R-PAL ×3 + plain R-PAL ×1）。
   確認：最終題數 80、系統組合、重跑次數、總 request 上限。VQ 用 frozen K=1，不外傳。
2. **Ambition**：P0+P1（守得住的 spine）或 P0+P1+P2（加 near-OOD）。
3. **Interim honesty**：是否現在就把 paper trap 數字標成 "pilot, confirmatory pending"（P1 落地後換掉）。
