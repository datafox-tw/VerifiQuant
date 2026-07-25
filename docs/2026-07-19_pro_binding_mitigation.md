# Pro「亂想」對策:literal-grounding 綁定守衛(研究 + 離線驗證)

**問題**(論文 §binding 自揭):Pro 會在綁定前**重算題目明寫的值**($2.5M 被自作主張加上
利潤變 $2.875M),E-check 只驗值域、不驗出處 → 能力越強、越會鑽沒被約束的步驟。

**對策(兩層,皆掛在 `VERIFIQUANT_STRICT_BINDING=1` 之後,預設關閉、不影響 canonical):**
1. **Prompt 硬化(軟層)**:extractor prompt 加反推導規則——「逐字複製數值;禁止加總/
   縮放/調整/推導;若 schema 需要的值不在原文,標 missing 而非自行計算」。
2. **Literal-grounding 守衛(硬層,已實作)**:`verifiquant/pipeline/strict_binding.py`。
   每個數值綁定必須可由原文字面數字經**允許的正規化**得出(千分位/貨幣符號、
   scale 字 2.5M/million、percent↔decimal、±號(流出/損失)、×12/÷12 時間基底、
   小寫數字詞、頻率詞 quarterly→4 / monthly→12 …)。溯源失敗的 required 欄位
   → 降為 missing → **F gate 反問**,把潛在 silent binding slip 變成 structured ask。

## 離線驗證(零 API,replay canonical 250Q 的 stored traces)
`scripts/validate_strict_binding.py`:

| 指標 | 調校前 | 調校後 |
|---|---|---|
| correct 案例被旗(= coverage cost) | 41/234 (17.5%) | **16/234 (6.8%)** |
| silent-wrong 案例被旗 | 2/8 | **1/8**(test-2009:`invested_amount=50000` 全文無此數——正是發明值) |

殘餘誤旗特徵化:幾乎全是 **derived-but-legal** 灰區——債券 coupon 現金流由票面利率×面額
建構([50,50,1050])、rights ratio 1-for-4→0.25、年份差推 num_years。這正是 Wave 3
provenance 設計要解的「contract-declared derivation」缺口;短期行為(反問)偏保守但安全。

**預期效益**:Flash 上 6.8% coverage cost 換 provenance 保證;**Pro 上才是主戰場**
(class-C binding slips 集中處)。建議 rollout:先在 50Q 上跑 VQ-Pro 開/關對照,
確認 Pro 的 over-normalization 案例被攔,再上 250Q。

## 待套用的 hook(⚠️ sweep 跑完才動 pipeline 檔,避免中途改到正在 import 的模組)

`run_error_classification_pipeline.py`,綁定迴圈完成後(`for name, item in extracted.items()` 之後、`extraction_trace = {` 之前)插入:

```python
    # Optional literal-grounding guard (anti-over-normalization; see strict_binding.py).
    if os.environ.get("VERIFIQUANT_STRICT_BINDING", "").lower() in ("1", "true", "yes"):
        from verifiquant.pipeline.strict_binding import audit_binding
        _, _ungrounded = audit_binding(provided_inputs, q, c)
        for _n in _ungrounded:
            if _n in required_names:
                provided_inputs.pop(_n, None)
                if _n not in missing:
                    missing.append(_n)
                coercion_notes.append(
                    f"[{_n}] strict-binding: value not traceable to a literal span; requesting field."
                )
```

`_extract_inputs_with_llm` 的 prompt 尾端(同 flag 判斷)加:

```
STRICT BINDING MODE (active):
- Copy each value LITERALLY from the text. Do NOT sum, subtract, scale, average,
  annualize, or otherwise derive values, even if the schema seems to expect it.
- If the schema requires a value that is not literally present, set status="missing".
- The only allowed conversions are the declared unit conversions above (percent to
  decimal, stated scale words). Everything else belongs to the deterministic code.
```

## 檔案
- `verifiquant/pipeline/strict_binding.py`(standalone,已測)
- `scripts/validate_strict_binding.py`(離線驗證,可重跑)
- Hook:**未套用**,sweep 完成後套 + `py_compile` + flag-off 煙霧測試

---
## ✅ Hook 已套用(2026-07-20,sweep 完成後)
- Pipeline hook + strict prompt 均已進 `run_error_classification_pipeline.py`(flag: `VERIFIQUANT_STRICT_BINDING=1`,預設關)。
- 驗證:py_compile ✓;flag-off import 煙霧測試 ✓(canonical 路徑零影響);
  **flag-on live 測試 ✓**——grounded case 照常 success(2.0);derived case
  (「起始 200000、又增 100000」需推導期末值)→ `error/F` 反問而非默默推導
  (strict prompt 軟層即攔截,extractor 直接標 missing)。
- 下一步:VQ-Pro 50Q 開/關對照(Wave 1-B 前置)。

---
## VQ-Pro 50Q ON/OFF 對照結果(2026-07-20)

| | correct | SW(全為 scale near-miss) | abstain |
|---|---|---|---|
| OFF | 38 (76%) | 4(真 SW=0) | 8 (16%) |
| ON | 32 (64%) | 6(真 SW=0) | 12 (24%) |

**三個發現:**
1. **硬層 hook 一次都沒觸發(strict_fired=0)**——所有行為變化來自軟層 prompt:
   extractor 被指示「不推導、標 missing」後,把 derived-but-legal 綁定(元件加總的
   total、債券 coupon 排程)直接拒絕 → 7 個 correct→abstain。軟層**搶在**硬層前面。
2. **這組 50Q 看不到 benefit、只看得到 premium**:OFF 基線真 SW=0(over-normalization
   失敗案例出現在 K=3 framework-guided 的 cost-plus 題,不在這 50Q 的 K=1 路徑)。
   保險的賠付案例不存在時,只會量到保費(-12pts coverage)。Benefit 證據仍是離線
   replay(抓到 test-2009 發明值)。
3. 2 個 abstain→correct 是 Pro run-to-run 變異(非 guard 效果),乾淨歸因需重複跑。

**設計修正(待 chain 結束後實作):拆開兩層 flag**——
`VERIFIQUANT_STRICT_BINDING=1` 只開**硬層 hook**(外科手術式,只攔真正無法溯源的值,
離線量測 cost 6.8%);軟層 prompt 另設 `VERIFIQUANT_STRICT_BINDING_PROMPT=1`
(目前證據顯示對 Pro 過度保守,-12pts,不建議預設開)。
