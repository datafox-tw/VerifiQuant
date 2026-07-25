# Wave 1 結果(隔夜 API 批次,2026-07-20/21)

tmux chain `wave1`(caffeinate -s)跑完 5 步。四項完成、PAL 修正版重跑中。

## 1-A Library-size sweep ✅(見 `2026-07-19_wave0` 已記,圖進論文 §oodfam fig:libsweep)

## 1-B VQ-Pro 50Q strict-binding on/off ✅
見 `2026-07-19_pro_binding_mitigation.md` 末段。結論:硬層零觸發、軟層 prompt 過度保守
(-12pts coverage),這組 50Q 的 OFF 基線真 SW 本就是 0 → 只量到保費、看不到賠付。
**設計修正:拆兩層 flag**(硬層外科式單開;軟層另設 flag、不預設)。

## 1-B' Pro CoT 250Q single-shot ✅ 【回應「baseline 太弱」的關鍵】
`results/cot_single_shot_pro/`

| | accuracy | SWR |
|---|---|---|
| CoT single (Flash) | 77.6% | 18.0% |
| **CoT single (Pro)** | **82.4%** | **15.2%**(medium 12.2 / hard 22.9) |
| VQ-Flash K=1 autonomous | 81.2% | **2.8%** |

**換強模型:accuracy +4.8pts(追平 VQ autonomous),但 SWR 只從 18.0→15.2%,仍是 VQ 的
5.4×,hard tier 更達 22.9%。** → thesis「等更強模型不會關 audit gap」有模型軸證據。
已進主表(tab:main 新增 Pro 區塊)+ §results 專段。

## 1-D Self-consistency@5 250Q ✅ 【回應「用真實 confidence proxy」】
`results/sc_baseline/`,k=5、vote-share 當 confidence、numeric clustering 投票。
Risk-coverage 真實點:

| vote-share thr | coverage | sel-acc | SWR |
|---|---|---|---|
| ≥0.0 | 88.8% | 91.0% | 8.0% |
| ≥0.6 | 86.8% | 92.2% | 6.8% |
| ≥0.8/1.0 | 84.0% | **92.4%** | 6.4% |

**SC@5 最佳點(92.4% @ 84% cov)恰好落在 CoT single-shot 的 perfect-rejector 上界上,
仍比 VQ 真實 K=1(96.7% @ 84%)低 4.3pts。** → 「baseline 大可用真實信心代理」的疑慮
實證關門:最強的可實現代理也追不上 VQ 真實策略。已疊上 fig:riskcov(灰方點)+ §kablation 段。

## 1-C PAL execution baseline 250Q ✅ 【最重要的 component ablation】
`results/pal_baseline/`(denylist 修正後重跑;第一版誤殺 88 docstring 已作廢)。
**acc 90.4% / SWR 8.0%**(medium 7.2 / hard 10.0,難度下幾乎不動),abstain 4、denied 4。

**Component ladder(全 250,誠實歸因):**
| System | acc | SWR | 差異來源 |
|---|---|---|---|
| CoT single (Flash) | 77.6% | 18.0% | LLM 自己算 |
| **PAL execution** | **90.4%** | **8.0%** | +deterministic exec,無 contracts/gates |
| VQ K=1 autonomous | 81.2% | 2.8%(cov 84%) | +contracts+gates+abstention |
| VQ K=3 | 93.6% | 3.2%(cov 97%) | +clarification |

**解讀(進了論文 §results 專段):**
- **執行本身買到大半 accuracy(77.6→90.4)+ 半數 SWR 下降(18→8)**——誠實承認 reviewer
  的直覺對:「不讓 LLM 算數學」貢獻很大。
- 但 PAL 的 8% 是「code 跑得完美卻算錯東西」(錯公式/錯綁定/錯慣例),且**它不知道哪 8% 錯**。
- 這殘餘正是 contracts+gates 的標的:VQ K=1 砍到 2.8% + 每個答案可歸因 + 不確定就 abstain。
- **誠實張力**:PAL raw acc(90.4%)> VQ autonomous(81.2%),因為 PAL 從不 abstain;但
  matched coverage 下 VQ selective-acc(96.7%)> PAL(91.9%),且 PAL 無 attribution/abstention/audit。
- 「execution 是 governance-by-execution 的一個 component,不是替代品。」full gate ladder = Wave 2。

## 已進論文(main.tex,11 頁,編譯通過)
- tab:main 新增 CoT single-shot (Pro) 三列 + §results「stronger model」專段
- fig:riskcov 疊 SC@5 真實點 + §kablation 論證更新
- §protocol baselines 段補列 Pro/PAL/SC@5

## 待辦
- PAL 重跑完 → 進 ablation 敘事(Wave 2-A gate ladder 的第一塊:PAL = 純執行基準)
- strict-binding 兩層 flag 拆分(小改)
- Wave 2:gate ablation ladder(需 pipeline flags)、hint-trigger holdout、paraphrase set
