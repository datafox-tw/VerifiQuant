# Reviewer 質疑 × 實驗證據對照表(Evidence Scorecard, 2026-07-21)

對應那份完整審查(P0–P3)。每條:**質疑 → 我做的實驗 → 結果 → 判定**。
判定圖例:✅ 已用實驗回應 / ⚠️ 部分(誠實承認殘餘)/ ⬜ 未做(pending/deferred)。
所有數字來源見各 `docs/2026-07-*` 與 `verifiquant/data/runs/*`;論文落點見括號。

---

## 總表

| # | 質疑 | 實驗 | 關鍵結果 | 判定 |
|---|---|---|---|---|
| P0-1 | C4 只有一個 annuity 案例 | **RepairBench**(179 specs/149 families,906 mutants,0 API) | 非等價錯 patch 攔 **97.7%**;結構攻擊 **100%**;true-FA 2.3% | ✅ 機制證實 + ⚠️ 誠實殘餘 |
| P0-2 | FIC 把 benchmark 編譯進系統 | **Library-size sweep**(10–100% 巢狀)+ **OOD family probe**(199 題) | 每個庫規模 abstain 差距 67–84pt;OOD abstain 88.9% vs ID 19.6% | ⚠️ 部分(缺第二資料集) |
| P0-3 | 只跟 Flash CoT 比、元件貢獻沒拆 | **Pro CoT 250Q + PAL + SC@5** | PAL(純執行)90.4%/8.0%;Pro CoT 82.4%/15.2%;VQ K1 81.2%/2.8% | ⚠️ 部分(缺完整 gate ladder) |
| P1-4 | Input binding 破口 | **strict_binding guard + 離線驗證 + VQ-Pro on/off** | 離線 coverage cost 6.8%,抓到發明值;硬層可攔 | ⚠️ 部分(缺 binding trap set) |
| P1-5 | M-gate 100% SWR | (依你指示延後) | — | ⬜ Deferred |
| P1-6 | Hint-trigger 像 test-set patching | (未做 held-out) | — | ⬜ Pending(Wave 2-B) |
| P2-7 | Oracle 太強、93.6% ≠ deployment | **250Q K-ablation** | K1 autonomous 81.2% 已分離揭露;93.6% 標為 recoverability | ✅(human study 仍 pending) |
| P2-8 | 缺 CI/檢定、單一標註者 | **McNemar + bootstrap CI** | K1 vs CoT+oracle acc p=0.90;SW p<1e-5;CI 不重疊 | ✅(第二標註者 pending) |
| P3-9 | 六層 pipeline / K=3 成本 | **成本盤點** | VQ K3 17.3s/題 < CoT+oracle 27.8s;1.20 輪 | ✅ |

---

## P0-1 · C4 atomic transform 系統性驗證 ✅⚠️
**質疑**:cross-verification 到底抓得到多少錯 patch?正確 repair 會不會被過嚴拒絕?
**實驗**:`scripts/run_repairbench.py`——對 250 卡庫全部 179 個可合成 result_postprocess spec
(149 families),每個配 6 類程式化錯 patch(op_swap / coeff_perturb / var_swap / identity /
call_inject / oversize),共 906 mutants;零 LLM。
**結果**:
- **非等價錯 patch 攔截 97.7%**(859/879);結構攻擊(注入/超大 AST)**100%**;
  主力 = 數值不變量檢查。
- **valid-accept 96.1%**(172/179);7 個 valid-reject = **checker 反向抓到卡片 expr/invariant
  不一致的 contract defect**(額外收穫)。
- **true-FA 僅 2.3%**(20 個),且成因單一:條件式 transform 的 guard 分支未被樣本踩到。
- point-of-use(1 樣本)vs 5 樣本:FA 60→47,量化多樣本的邊際效益。
- 27 個等價 mutant(×(−1) vs ÷(−1))誠實排除。
**佐證/反證**:**證實機制有效**(不是單一故事),同時**誠實揭露**兩個殘餘:
(a) 條件分支盲點(可用 branch-aware sampling 修);(b) code_patch 仍只有一個案例
(RepairBench 只涵蓋 result_postprocess)。→ C4 現在有量化證據撐,不必降級。
**論文**:§transforms *RepairBench* 段 + `tab:repairbench` + abstract 一句。

## P0-2 · benchmark compilation / holdout ⚠️
**質疑**:你不是測 open-world reasoning,是測「從同一 ontology 建好的函數庫能否 route 回同一 benchmark」。
**實驗**:
1. **Library-size sweep**(`scripts/build_subset_libraries.py` + `run_libsweep.sh`):
   10/25/50/75/100% 巢狀 pristine 子集庫(25/62/125/188/250 families)× 250Q。
2. **OOD family probe**:199 個不在庫的 family,同題同餵 refusal-permitted CoT。
**結果**:
- **abstention tracks coverage 是架構性質**:OOD abstain 91.6→80.6%、ID abstain 8.0→14.4%,
  **每個庫規模差距都 67–84pt**;ID accuracy 隨庫放大 10× 仍持平(81.6–84.0%);
  lib_100 端點 81.6% **獨立復刻** canonical K1 81.2%。
- **選擇性拒答**:OOD abstain 88.9% vs ID 19.6%(69.3pt);CoT 允許拒答只有 11.3pt 分離、
  且 ID 準確率崩到 45.1% → **contract-grounded refusal 有選擇性,prompted refusal 是避險**。
- 誠實 trade-off:OOD SW 隨庫密度 1.3→4.8% 緩升(selection frontier)。
**佐證/反證**:**部分反證**「只會 in-library routing」——它會選擇性地在庫外 abstain,
不是對陌生 wording 一律拒絕(sweep 的 ID acc 持平證明)。**但仍承認 coverage disclosure**
(主結果是 in-library)。缺口:**第二資料集 + natural paraphrase holdout**(Wave 2-C)。
**論文**:§oodfam + `fig:libsweep` + §protocol *Coverage disclosure*。

## P0-3 · 完整 baseline + component ablation ⚠️
**質疑**:優勢來自不讓 LLM 算數學、contracts、還是 gates?主結果會不會只是 Flash 太弱?
**實驗**:250Q 上補 **Pro CoT**、**PAL 執行式 baseline**、**SC@5**。
**結果(component ladder)**:
| System | acc | SWR |
|---|---|---|
| CoT single (Flash) | 77.6% | 18.0% |
| CoT single (**Pro**) | 82.4% | 15.2%(hard 22.9%) |
| **PAL**(純執行,無 gates) | **90.4%** | **8.0%** |
| VQ K1 autonomous | 81.2% | 2.8%(cov 84%) |
| VQ K3 | 93.6% | 3.2% |
- **強模型不關 audit gap**:Pro acc +4.8pt 但 SWR 只 18→15.2%,仍 5.4× VQ。
- **誠實承認 reviewer 的直覺對**:執行本身買到大半 accuracy(77.6→90.4)+ 半數 SWR(18→8);
  但 PAL 的 8% 是「code 完美卻算錯」且**不知道哪個錯** → 這殘餘正是 contracts+gates 標的,
  VQ 砍到 2.8% + 可歸因 + abstain。matched coverage 下 VQ sel-acc 96.7% > PAL 91.9%。
**佐證/反證**:**部分佐證 reviewer**(execution 貢獻大)+ **釐清 VQ 的邊際價值 = governance
residual,不是 raw accuracy**。缺口:**完整 gate ladder**(FIC-only→+FE→+I→+MN→+hint)= Wave 2-A。
**論文**:tab:main 加 Pro 區塊 + §results「stronger model」與「component」兩段 + `fig:riskcov` 疊 SC@5。

## P1-4 · Input provenance / binding ⚠️
**質疑**:extractor 綁錯合法值、Pro 甚至改寫 literal($2.5M→$2.875M);E-check 不驗出處。
**實驗**:`verifiquant/pipeline/strict_binding.py`(literal-grounding 守衛,opt-in flag)+
`scripts/validate_strict_binding.py` 離線 replay + VQ-Pro 50Q on/off。
**結果**:離線 coverage cost **6.8%**(調校後);**抓到發明值案例**(test-2009 `invested_amount`
全文無此數);硬層 hook 可攔真正無法溯源的值。誠實發現:軟層 prompt 對 Pro 過度保守(-12pt),
→ 已記錄「拆兩層 flag」修正。
**佐證/反證**:**設計 + 離線證據到位**,但**尚未有完整 binding trap set 的 before/after SWR**
(Wave 3)。這是論文明列的架構破口,補完可成下一個 method contribution。
**論文**:尚未寫入(等 binding trap set);機制 `strict_binding.py` 已就位。

## P1-5 · M-gate 100% SWR ⬜(依你指示延後)
**現況**:trap-M 因 generator 保留 metric context,VQ 反而 100% SWR;論文已排除 M claims、
Limitations 已列缺陷。**你的判斷:真正「無唯一 metric intent」的題目只能自寫/LLM 生成後人工審**。
**留檔設計**:模板 =「投入 X、每年賺 Y、n 年、通膨 r,會不會賺?」→ 正確 = 反問 NPV/IRR/payback;
變體 = context 同含 ROI/ROE/ROA 欄位 +「calculate the return」。demo 已有 1 題實測可觸發 abstain_m。
**佐證/反證**:⬜ 未做。短期建議:論文把 M 明確降為 exploratory component(N 的 boundary claim 不動)。

## P1-6 · Hint-trigger held-out ⬜(Wave 2-B)
**質疑**:trigger 從已見的三個 convention failure 設計,再回頭跑同一 250Q = test-set patching;
98/250 觸發偏高。
**現況**:未做 held-out。`_lexical_hint_triggers` 已是獨立函式,詞表可凍結。
**計畫**:medium 抽 dev set 設計→凍結→在剩餘 medium+hard+paraphrase 測 precision/recall +
false-positive 專項 + 4 個轉 abstention 案例覆盤。
**佐證/反證**:⬜ 未做。這是 P1 裡 CP 值最高的一塊。

## P2-7 · Oracle 太強、93.6% ≠ deployment ✅
**質疑**:K=3 oracle 看 GT computation code,遠比真實使用者強。
**實驗**:250Q K-ablation(從既有 run 精算,0 API)。
**結果**:**K=1 autonomous = 81.2% / 2.8% SWR / 84% cov** 已與 K=3 分離揭露;93.6% 明確標為
clarification-assisted **recoverability**(救回 31 題、平均 2.1 輪)。Limitations 明寫 oracle 比真人強。
**佐證/反證**:**已用實驗把 headline 誠實拆解**。缺口:realistic-user oracle(只看題面)+ human study(P2,pending)。
**論文**:§kablation + `tab:kablation250` + Limitations。

## P2-8 · 統計檢定 / variance / 標註一致性 ✅
**實驗**:`scripts/run_stats_250q.py`(0 API)。
**結果**:exact McNemar——**K1 vs CoT+oracle acc p=0.90**(佐證「indistinguishable」);
K3 acc 優勢 p<1e-4;**SW emission p<1e-5**(每設定);bootstrap 95% CI:VQ SWR 3.2%[1.2,5.6]
vs CoT+oracle 14.8%[10.4,19.2](不重疊)。
**佐證/反證**:**「statistically indistinguishable」現在有檢定**。缺口:第二標註者盲標 B/C/D。
**論文**:§results 顯著性句 + CI。

## P3-9 · 成本 / 延遲 ✅
**結果**:VQ K3 **17.3s/題**(平均 1.20 輪、~3.8 calls)< CoT+oracle **27.8s** > single **6.3s**;
contract build = 每 family 3 calls 一次性。**帶完整 governance 的 VQ 比它贏過的 oracle loop 便宜**。
**論文**:§deployment *Runtime cost* 段。

---

## 一句話總結
- **已用實驗實質回應**:C4(RepairBench)、oracle 過強(K-ablation)、統計(McNemar/CI)、成本、
  以及「abstention tracks coverage」升級成 sweep 曲線。
- **部分回應、誠實承認殘餘**:benchmark holdout(缺第二資料集)、component ablation(缺完整 gate ladder)、
  input binding(缺 trap set)。
- **未做**:M-gate 重做(延後)、hint-trigger held-out、human study、第二標註者。
- **最誠實也最有價值的新發現**:PAL 顯示 execution 買到大半 accuracy——這不削弱論文,
  反而把貢獻精準定位到 governance residual(SWR 8→2.8 + attribution + abstention),
  正好對齊「not accuracy but governability」的 thesis。

## 剩餘最小可投清單(對齊 reviewer 的三組)
1. **Wave 2-A** 完整 gate ablation ladder(P0-3 收尾,最有殺傷力)
2. **Wave 2-C** natural paraphrase holdout(P0-2 收尾)+ 理想加第二資料集
3. **Wave 2-B** hint-trigger held-out(P1-6)
4. **Wave 3** binding trap set(P1-4 收尾,可成 method contribution)

---
## ⚠️ 資料集事實澄清(2026-07-21,查證後)
**確認:250 題全部來自 FinanceReasoning 的「函式連結生成子集」,非四個上游公開資料集。**
- medium.json 1000 題:FinanceReasoning 540(**唯一有 function_id**)+ CodeFinQA 230 + CodeTAT-QA 99 + FinanceMath 104 + FinCode 27。我們的 250 全有 function_id → 只選了 FinanceReasoning 那批。
- 每題兩個 artifact:`function`(article 級參考公式,問題由它生成)+ `python_solution`(代入數字的逐題解)。**卡片建自 `function`,不用 `python_solution`。**
- 因此「independent corpus」措辭**已修正**:參考函式不獨立於問題(問題就是從它生成的),卡片依構造含有匹配問題所需公式——這正是我們要測的 **in-library** regime(routing/binding/gating),非 open-world。防守點保留:不讀逐題 worked solution/gold answer、AST dedup、OOD sweep 測庫外。
- **已撤掉**先前誤寫的「250 題橫跨四個上游資料集」(錯);改為誠實的「刻意選 function-linked 子集 + 明確揭露 in-library」。
