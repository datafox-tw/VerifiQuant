# Wave 0 結果(零 API 批次,2026-07-19)

全部離線完成,零 LLM 呼叫。四項全數完成並已寫入 `paper/main.tex`(編譯通過,11 頁)。

## 0-A RepairBench(C4 系統性驗證)✅
Script: `scripts/run_repairbench.py` · 結果: `verifiquant/data/runs/repairbench/`

| 指標 | 數值 |
|---|---|
| 可用 specs | 179/191(149 families;12 個輸入型別無法合成) |
| Valid-accept | 96.1%(172/179);**7 個 reject = 卡片 expr/invariant 不一致 = checker 抓到 contract defect** |
| 結構攻擊(call 注入 / 超大 AST) | **100% 攔截**(各 179/179,AST guard) |
| 語義突變有效攔截 | **97.7%(859/879 非等價突變)**,主力 = invariant 數值檢查 |
| 真漏網 | 20/879 = 2.3%,集中 identity(10)——**條件式 transform 的 guard 分支未被樣本踩到** |
| 等價突變 | 27(如 ×(−1) vs ÷(−1),數學同義,誠實排除) |
| 1 樣本 vs 5 樣本 | FA 60 → 47(多樣本再砍 22%) |

方法註記:攻擊模型 = 信任 contract(invariant、AST bound),對 patch expression 做六類
程式化突變(op_swap / coeff_perturb / var_swap / identity / call_inject / oversize);
「雙側同錯」盲點依構造必過,以 demonstration 呈現(對應論文 *Scope of the guarantee*)。
Actionable 後續:branch-aware sampling 可關掉 identity 漏網面。

## 0-B 統計檢定 ✅
Script: `scripts/run_stats_250q.py` · 結果: `paper_v2_250/results/stats_250q.json`

**McNemar(exact,250 paired)**
| 比較 | b / c | p |
|---|---|---|
| VQ K=3 vs CoT+oracle(accuracy) | 41/12 | **8.2e-05** |
| VQ K=3 vs CoT single(accuracy) | 52/12 | 4.6e-07 |
| **VQ K=1 vs CoT+oracle(accuracy)** | 32/34 | **0.90**(不可區分——§kablation 的 claim 有據) |
| VQ K=3 vs CoT+oracle(SW emission) | 6/35 | 4.9e-06 |
| VQ K=1 vs CoT single(SW emission) | 4/42 | 5.1e-09 |

**Bootstrap 95% CI(10k)**:VQ K=3 SWR 3.2% [1.2, 5.6] vs CoT+oracle 14.8% [10.4, 19.2](不重疊)。
注意:此 ternary 計分與論文 canonical(numbers.tex)一致;舊 `paper_results_summary.json`
的 22.4%/18.0% 是粗計分,勿混用。§kablation 的「one-sixth」已修正為 one-fifth(14.8/2.8≈5.3)。

## 0-C 成本盤點 ✅(同 script)
| 系統 | 每題 wall-clock | 每題 LLM calls |
|---|---|---|
| CoT single | 6.3s | ~1–3 |
| **VQ K=3** | **17.3s** | ≈3.8(平均 1.20 輪) |
| CoT+oracle | 27.8s | 更多 |

**賣點:帶完整 governance 的 VQ 比它贏過的 oracle loop 便宜**;contract build = 每 family 3 calls 一次性。

## 0-D 子集庫產生器 ✅(Wave 1 前置)
Script: `scripts/build_subset_libraries.py` · 輸出: `verifiquant/data/runs/libsweep/lib_{10,25,50,75,100}/cards.db`
巢狀子集(10%⊂25%⊂…⊂100%;25/62/125/188/250 families),pristine(順帶解掉 §oodfam 的
demo-mutated 庫 caveat)。Sweep 指令:
```bash
PYTHONPATH=. .venv/bin/python scripts/run_ood_family_eval.py \
  --lib-db verifiquant/data/runs/libsweep/lib_25/cards.db \
  --limit-ood 999 --limit-id 999 --no-baseline
```

## 已寫入論文(main.tex,⚠️ 未經作者逐句審閱)
1. §transforms:*RepairBench* 段 + `tab:repairbench`(97.7% / 100% 結構攻擊 / 真漏網特徵化)
2. abstract:RepairBench 一句(97.7% / 100%)
3. §kablation:McNemar p=0.90(K=1 vs CoT+oracle)+ SWR p<1e-5;one-sixth→one-fifth 修正
4. §results:顯著性句(p<1e-4 / p<1e-5)+ SWR bootstrap CI
5. §deployment:*Runtime cost* 段(17.3s vs 6.3s vs 27.8s;VQ 比 oracle loop 便宜)

## 下一步(Wave 1,隔夜 API)
lib sweep 5 點 → Pro CoT 250Q → PAL baseline → self-consistency@k。
子集庫已建好,sweep 隨時可launch。
