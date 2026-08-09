# Wave 2-A · Gate ablation ladder(2026-07-21)

回應審查 P0-3 最後一塊:「增益來自 execution、contracts、還是 gates?」
Pipeline 加 `VQ_DISABLE_GATES` 開關(module-load 讀取);250Q、K=1、Flash、full 250-card lib、
累進五 rung。`scripts/run_gate_ladder.sh` + `run_vq50_pro.py`。

## 結果(unit-normalized,與論文主結果同尺度)
| Configuration | Acc | SWR | Abst |
|---|---|---|---|
| CoT single-shot(LLM 自己算,參考) | 77.6% | 18.0% | 11 |
| FIC select+bind+execute(無 gate) | 87.6% | 6.4% | 15 |
| + F/E | 86.8% | 4.4% | 22 |
| + I | 79.6% | 4.4% | 40 |
| + M/N(full funnel) | 79.2% | **4.0%** | 42 |
| + hint triggers | 80.0% | **3.6%** | 41 |

## 誠實解讀(已進論文 §results + `tab:gateladder`)
- **SWR 每階單調下降**(18→6.4→4.4→4.4→4.0→3.6),abstain 單調上升 → **gates 用 coverage 換 safety,不是換 accuracy**(正是 thesis)。
- **execution + contract selection 買到大半 SWR 下降**(18→6.4)+ 大半 accuracy。
- **F/E 明確增益**(6.4→4.4,deterministic input validation)。
- **I / M/N 在乾淨集上主要把風險轉成 abstention**(accuracy dip、SWR 幾乎不動)——
  誠實界定:它們的 silent-wrong 價值在**有注入歧義的 trap 集**(trap-I 100→10%,已在論文)
  與 **K=3 recovery**,不在乾淨 250Q。
- **hint triggers 再砍 SWR**(4.0→3.6)且小幅回補 accuracy。

## 更正(2026-08-09)
- CoT 那列原記為 `Abst = 6`,與其自己的 Acc/SWR 加不到 250(194+45+6=245)。
  重數 `paper_v2_250/results/cot_single_shot_flash/output.jsonl` 後為 **194 / 45 / 11**
  (與 `\cotAbstainAll{}` 一致)。已更正本表與 `paper/main_v2.tex` 的 `tab:v2gateladder`。
- 五個 VQ rung 已對 `runs/gate_ladder/r*_summary.json` 逐列覆核無誤:unit-norm 的
  correct = `correct + near_miss`、SWR = `(silent_wrong - near_miss)/250`,
  例如 r4 = 175+23=198 (79.2%)、(33-23)/250=4.0%、abstain 42。

## 技術註記
- 開關:`VQ_DISABLE_GATES=MN,F,E,I`(逗號分隔);MN 停用時 forced-select top candidate、
  同時解除 m_min_top_score N-floor;F/E/I 停用時跳過各自 structured exit。
- r4(full)strict 70.0%、unit-norm 79.2%,與 canonical K=1 81.2% 因 output-scale 正規化 +
  LLM run-to-run 變異而略有差(合理)。
- 開關預設關(空字串),canonical 路徑零影響。
