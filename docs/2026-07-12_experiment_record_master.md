# VerifiQuant 實驗記錄總表（Master Record）

**Created**: 2026-07-12 · **維護原則**: 論文寫作的 single source of truth——每個數字都附出處路徑；
新 run 完成後在此登記。與 `docs/paper_draft_finnlp.md` 衝突時，**以本檔 + 原始 summary.json 為準**。

---

## 1. 資料集與 artifacts

| 資料集 | 內容 | 路徑 | 狀態 |
|---|---|---|---|
| **paper_v1 clean 50Q** | FinanceReasoning medium，難度分位分層抽樣，seed 42（Q1:12/Q2:13/Q3:12/Q4:13，難度 2.485–4.025） | `verifiquant/data/runs/paper_v1/questions_50.jsonl` + `experiment_config.yaml` | ✅ 鎖定 |
| **paper_v1 trap 50Q** | contract-grounded Tier-1，5 算子（M/N/F/E/I）×10，needs_review=0，人工抽驗過 | `verifiquant/data/runs/paper_v1/trap/trap_set.jsonl` + `trap_manifest.json`；生成器 `verifiquant/pipeline/build_trap_set.py` | ✅ 鎖定 |
| **paper_v2 250Q** | 180 medium + 70 hard，canonical 50 為子集，seed 42。難度：medium 2.48–4.14（mean 3.03）、hard 4.16–7.19（mean 4.89） | `verifiquant/data/runs/paper_v2_250/questions_250.jsonl` + `experiment_config.yaml`（含全部 question_ids） | ✅ 鎖定 |

## 2. FIC 卡片版本史

### 50Q（paper_v1）

| 版本 | db | 修補內容 | 對應結果 |
|---|---|---|---|
| V1 | `paper_v1/fic/cards.db` | 初版 | VQ Flash 43/50 (SW=2)、VQ Pro 41/50 (SW=3) |
| V2 | — | 修 3 C-error + 2 SW（E-check 語法全修） | VQ Flash 44/50 (SW=0) |
| **V3（canonical）** | `paper_v1/fic/cards_v3.db` | 再修 1 C-error（E-check 改用 `compute(inputs)`） | VQ Flash 45/50、VQ Pro 43/50 |

修卡防禦性：3 個 C-class 修正皆由 `execution_smoke_ok=false` 偵測，**無需 gold answer**。

### 250Q（paper_v2_250）— 2026-07-12 建置完成

| 項目 | 結果 | 出處 |
|---|---|---|
| 卡片生成 | core 250 / retrieval 250 / repair 1,960（diagnostic rules 1,522） | `paper_v2_250/fic/validation_report.json` |
| Execution smoke | **failed=0**；零重複、kernel dedup 零碰撞 | 同上 |
| E-check lint（standalone 複驗） | 初掃 unresolved=4（2 張卡：`fic_article_365` ×3、`fic_article_2303` ×1，statement-form try/except・import） | `fic/echeck_lint_report.json` |
| **修卡（2026-07-12，pre-run lint gate）** | 4 條 expression 改寫為等義 eval-safe 形式（ISO 日期格式 → 字串結構檢查；brackets JSON → expression 形式 dict/數值驗證），`predicate_mode` 顯式標 `validity`。修後 lint unresolved=0，5 組 sanity case 通過（好輸入 True／壞輸入 False）。**不涉及 gold answer** | `fic/card_repairs.json`；備份 `fic/core.jsonl.bak_prelint_20260712` |
| Card store | 以修復後卡片重建（舊 6/15 版含壞 check，移至 `cards.db.bak_prelint_0615`） | `fic/cards.db`（2026-07-12 09:04） |
| Cross-artifact relations | `validate_relations.py` exit 0，零錯誤 | — |

## 3. Clean Set 50Q — Canonical 主表（V3 卡片）

出處：各 `paper_v1/results/<name>/summary.json`。
⚠️ `paper_v1/paper_results_summary.json`（aggregate）**是 5/30–31 的舊檔**，缺 4 個盲審 run——別直接引用，以下表為準。

| System | Model | Correct | SW | Abstain | Sel.Acc | SWR | 附註 |
|---|---|---:|---:|---:|---:|---:|---|
| CoT single-shot | Flash | 41/50 | 9 | 0 | 82.0% | 18.0% | |
| CoT single-shot | Pro | 41/50 | 9 | 0 | 82.0% | 18.0% | |
| CoT + plain oracle（**盲審**） | Flash | 47/50 | 3 | 0 | 94.0% | 6.0% | improved 6、broken 0 |
| CoT + funnel oracle（**盲審**） | Flash | 46/50 | 4 | 0 | 92.0% | 8.0% | improved 5、broken 0 |
| CoT + plain oracle（**盲審**） | Pro | 49/50 | 1 | 0 | 98.0% | 2.0% | improved 6、broken 0 |
| CoT + funnel oracle（**盲審**） | Pro | 48/50 | 1(+1 no-ans) | 0 | 96.0% | 2.0% | improved 4、broken 0 |
| JP Morgan MAS (O-ITL) | Flash | 43/50 | 7 | 0 | 86.0% | 14.0% | 5 wrong + 2 pipeline error；ask_human 從未觸發 |
| **VerifiQuant K=3 (V3)** | **Flash** | **45/50** | **0** | **5** | **100%** | **0%** | recovery 8 (16%) |
| VerifiQuant K=3 (V3) | Pro | 43/50 | 2 | 5 | 95.6% | 4.0% | over-normalization（test-1593/1890）|

**Supplementary（V1 卡片）**：VQ Flash 43/50 (SWR 4%)、VQ Pro 41/50 (SWR 6%)。兩版本卡片下 Pro 皆低 Flash 2 題 → 真實模型效應。

**已棄用的 runs（在 results/ 目錄裡，別誤用）**：

| Run | 數字 | 棄用原因 |
|---|---|---|
| `cot_basic_oracle_flash` | 45/50, SWR 10% | GT-gated entry（oracle 收 is_correct、只在錯時進場）→ 評估外洩，由盲審版取代 |
| `cot_funnel_guided_flash` / `_pro` | 47/50 (94%) / 49/50 (98%) | 同上（GT-gated）。僅作 `prompt_comparison_all_systems.md` 的消融參照 |
| `vq_flash_v3_k1` / `_k2`（獨立 run） | 36/50 / 45/50 | 跨 run variance，改用 canonical K=3 run 的截斷（見 §5.1） |

**SFWR**：Flash 流程中 16/50 掛 I_SOFT、最終 0 錯 → SFWR-final=0%。Pro：SFWR-final=0%，但 test-1593 turn2 掛 I_SOFT→turn3 解除仍錯 → SFWR-ever=2.2% (1/45)，計入 SWR。

**診斷分布（VQ Flash V3 第 1 輪）**：37 通過 / 6 I_HARD / 3 F / 2 E / 1 N / 1 C。
最終：45 success + 5 abstain（F×3、E×1、C×1）；6 題 I_HARD 全數澄清成功。
輪數：38 題 1 輪、5 題 2 輪、7 題 3 輪；recovery 8。

## 4. Trap Set 50Q 結果

出處：`paper_v1/trap/results/<name>/trap_score.json`。

| System | Trap SWR | Caught(結構化) | Informal abstain | 附註 |
|---|---:|---:|---:|---|
| CoT single-shot Flash | **48% (24/50)** | 0 (0%) | 26 | 攔截=0 by design |
| **VQ Flash V3 K=1（funnel-only，公平度量）** | **26% (13/50)** | 37 (74%) | 0 | |
| VQ Flash V3 K=3（oracle on trap） | 96% (48/50) | 2 | 0 | **⚠️ 非能力數字**——oracle 讀 python_solution 把 redact 值填回、人為破解 trap；此 run 的存在就是「K≥2 在 trap 上=評估外洩」的證明 |

**分算子（CoT vs VQ K=1）**：

| 算子 | CoT SWR | VQ SWR | VQ confusion | 解讀 |
|---|---:|---:|---|---|
| E | 90% | **20%** | E×7, I_SOFT×2, F×1 | ★ 核心增益 |
| I | 100% | **10%** | I_SOFT_warn×9, success×1 | ★ 核心增益（I_SOFT proceed-with-warning 計 caught） |
| F | 0% | 0% | F×10 | auditability 差異（VQ 結構化、CoT 散文） |
| N | 0% | 0% | N×7, M×3 | 假平手：CoT 靠題目/context 錯配的突兀性 |
| M | 50% | 100% | I_SOFT_warn×9, success×1 | ⚠️ trap 設計限制（保留 metric context），不納入宣稱 |

## 5. Ablations

### 5.1 K-budget（canonical vq_flash_v3 run 截斷，與 §3 診斷分布自洽）

| K | Correct | SW | Abstain | SWR | 備註 |
|---|---:|---:|---:|---:|---|
| 1 | 37 (74%) | 0 | 13 | 0% | =第 1 輪通過數 |
| 2 | 42 (84%) | 0 | 8 | 0% | +5 |
| 3 | 45 (90%) | 0 | 5 | 0% | +8（recovery 16%） |

### 5.2 Run-to-run variance（V3+Flash+K=3 ×4）

canonical/var1/var2/var3 全部 **45/50、SWR=0%**（σ=0）；recovery 4–8 題（8–16%）浮動。
出處：`results/vq_flash_v3_var{1,2,3}/summary.json`。

### 5.3 Refusal prompt ablation（2026-06-09，K=1 12-cell + K=6 multi-K）

出處：`docs/results/2026-06-09_refusal_prompt_ablation.md`（完整表格與 K=6 部分見原檔）。
- gpt-5.2 給明確拒答通道後可產 6–9 個 safe refusal（L2/L3），但**瞄準很差**：over-refuse ≫ good-refuse（L3 low: 2 good/7 over）；SWR 只從 12%→7%，accuracy 88%→76%。
- gemini-2.5-flash 幾乎不受 prompt 引導（1–3 個 refusal），SWR 13–19% 無單調關係。
- 結論：**CoT 的 0% safe-refusal 是 prompt artifact，但 prompt 給了通道也達不到 selective abstention**——支持 VQ 的架構級主張。

## 6. 歷史數字（口徑不同，勿與上表混用）

| 來源 | 數字 | 差異 |
|---|---|---|
| `docs/results/2026-04-15_*.txt`（CLAUDE.md 引用的 0.90/0.88、recovery 28%/38%） | CoT 0.90、VQ 0.88 | pre-paper ad-hoc 口徑（選題、卡片、oracle 協議皆不同），已被 paper_v1 取代 |

## 7. paper_v2_250 進行中 runs

| Run | 狀態 | 啟動 | 出處 |
|---|---|---|---|
| `cot_single_shot_flash`（250Q） | ✅ **完成**（2026-07-12 09:44，27 min） | 2026-07-12 09:17 | `paper_v2_250/results/cot_single_shot_flash/` |
| `vq_flash`（K=3, 250Q） | 🔄 **執行中**（A2 接力自動啟動 09:44） | 2026-07-12 09:44 | 同上 `results/vq_flash/` |
| `cot_basic_oracle_flash`（blind, K=3, 250Q） | ⏳ 排隊中（VQ 之後同 runner 自動接） | — | config 已定義 |
| VQ Pro 250 / IMR / trap 250 / JPM 250 | ⏳ A5 擇優（優先序如左） | — | 需擴充 config |

### A1 結果：CoT single-shot Flash 250Q（2026-07-12）

overall **194/250 = 77.6%**；分層（`tier_breakdown.json`，工具 `scripts/analyze_v2_by_tier.py`，hard-min=4.15）：

| tier | N | Correct | SW | informal abstain | Coverage | Sel.Acc | SWR |
|---|---:|---:|---:|---:|---:|---:|---:|
| medium | 180 | 148 (82.2%) | 25 | 7 | 96.1% | 85.5% | **13.9%** |
| hard | 70 | 46 (65.7%) | 20 | 4 | 94.3% | 69.7% | **28.6%** |
| all | 250 | 194 | 45 | 11 | 95.6% | 81.2% | 18.0% |

**兩個關鍵訊號**：
1. **難度上移使 CoT 的 SWR 翻倍**（13.9%→28.6%）——論文 §5.2 難度分層敘事的 CoT 端已經成立；
   剩下看 VQ 是否以 abstention 吸收難度（abstain↑、SWR 平坦）。
2. **informal abstain 4.4% (11/250)**：CoT 在難題會偶爾自己標 needs_more_info 不作答，
   支持 §9-6 的三元分帳裁決（空答案≠SW）。

**Canonical-50 子集一致性**：本 run 內的 50 題子集 = 40/8/2（paper_v1 獨立 run = 41/8/1），
差 1 題,落在 CoT 已知 ±5% run variance 內 → 250Q 與 50Q 結果可比。

完成後：runner 會自動產 `paper_v2_250/paper_results_summary.json`；醒目數字先過
**medium/hard 分層**再進論文（`canonical_50_ids` 在 config 裡，可做 50Q 子集一致性檢查）。

## 8. 論文表格 ↔ 出處對照

| 論文位置 | 出處 |
|---|---|
| 主表（clean） | §3 本檔 + 各 summary.json |
| Trap 表 | §4 本檔 + trap_score.json |
| K-ablation | §5.1（截斷法，非獨立 run） |
| Variance | §5.2 |
| Prompt vs architecture 消融 | §3 盲審 4 配置 + `docs/prompt_comparison_all_systems.md` |
| 微觀案例 test-1443（CAGR 算術非決定性） | 各盲審 run 的 RUN_SUMMARY.md |
| 微觀案例 test-1593（over-normalization / I_SOFT 解除盲點） | `paper_draft_finnlp.md` §5.6（逐輪表） |
| 250Q 全部 | §7（待補） |

## 9. 已知數字衝突與裁決

1. **Recovery 16% vs 28%/38%**：16% 為 paper_v1 官方；28%/38% 為 4 月 ad-hoc 口徑，僅存查（§6）。
2. **K=1 = 37 vs 36**：37（截斷法，canonical）；36 為棄用的獨立 run。
3. **funnel-guided 94%/98% vs 盲審 92%/96%**：前者 GT-gated（棄用）；論文主表用盲審。
4. **`paper_results_summary.json`（paper_v1）過時**：缺盲審 4 run；重新聚合用
   `--aggregate-only`，但 funnel Pro 的「1 SW + 1 no-answer」聚合器會算成 SW=2，論文以人工核對的 §3 表為準。
5. **CLAUDE.md 的 0.90/0.88 baseline 表**：歷史口徑，見 §6。
6. **CoT single-shot「9 SW」中有 1 題是空答案（test-1242, `needs_more_info=True`, 無數字輸出）**：
   依論文自己的 SW 定義（有把握、無 flag 的錯誤**數值**），誠實三元分類應為 **41 correct / 8 SW / 1 informal-abstain**
   （SWR 16%、coverage 98%）。舊帳 9 SW 來自 aggregate 的粗式 `SW = total - correct`。
   **Flash 與 Pro single-shot 的空答案是同一題 test-1242**；funnel-blindreview-pro 的 no-answer 也是它。
   **裁決（2026-07-12，已套用至 ICAIF draft）**：single-shot 兩列改 41/8/(+1)、SWR 16%；
   連帶修正 funnel-Pro 的 SelAcc：48/(48+1)=**98.0%**（FinNLP draft 的 96% 是把 accuracy 誤植為 SelAcc）。
   差異不影響任何 claim（CoT SWR 仍 ≫ 0）。
   驗證工具：`scripts/analyze_v2_by_tier.py`（已對 vq_flash_v3 45/0/5、blindreview_pro 49/1/0 精確重現 canonical）。
7. **引用查證結果（2026-07-12，已修入 ICAIF draft + `docs/references_icaif.bib` + **FinNLP draft 已同步修正**（9 處：正文 7 + citation pool + 已確認表））**：
   - FinanceReasoning = **Tang et al., ACL 2025**（2025.acl-long.766, pp.15721–15749）——FinNLP draft §2.1/§5.2 的 "Liu et al." 是誤植；
   - JPM MAS = **Kundu, Sahoo, Li, Rabowsky & Varshney, EMNLP 2025 Industry Track**（2025.emnlp-industry.55, pp.812–824）——兩份 draft 的 "Yu et al." 是誤植；
   - Why MAS Fail = **Cemri et al., NeurIPS 2025 Datasets & Benchmarks（spotlight）**, arXiv:2503.13657——draft 的 "ICLR 2026" 與 "NeurIPS 2026" 均誤。
