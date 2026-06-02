# Oracle Prompt: Plain vs. Funnel-Guided Comparison

## Current (Plain Oracle) — 不含漏斗概念

```
_ORACLE_PROMPT_TEMPLATE = """\
You are Oracle-in-the-loop support for a pure CoT baseline.
There is no VerifiQuant framework in this loop.
You can ONLY use the logic within the ground-truth code to clarify assumptions and revise user question/context. You must NOT rely on the final numeric ground truth result.

Current question:
{current_question}

Current context:
{current_context}

Current CoT step output:
{cot_step_output_json}

Current correctness against gold (if available): {is_correct}

Ground-truth code:
{ground_truth_code}

Return JSON only."""
```

**問題**：「clarify assumptions」太泛泛，oracle 不知道該檢查哪些維度。


## Proposed: Funnel-Guided Oracle — 加入 M/N/F/E/I/C 清單

```
_ORACLE_PROMPT_TEMPLATE_FUNNEL_GUIDED = """\
You are Oracle-in-the-loop support for a CoT baseline with structured diagnostic guidance.
You can ONLY use the logic within the ground-truth code to clarify assumptions and revise user question/context. You must NOT rely on the final numeric ground truth result.

When reviewing the CoT step output for potential errors or ambiguities, systematically check the following diagnostic layers (M/N/F/E/I/C):

1. **M (Intent/Meaning)**: Does the user's question map unambiguously to a single calculation or formula? Are there multiple valid interpretations of what is being asked?
   - Example: "profit on total contract" could mean (a) profit as % of incurred costs or (b) fixed fee on total contract value.
   - If ambiguous: explicitly list the alternative interpretations.

2. **N (Scope/Domain)**: Does the question fall within a well-defined financial domain (e.g., valuation, cash flow, bond pricing)? Or does it require domain expertise outside standard finance (e.g., esoteric derivatives, regulatory frameworks not commonly automated)?
   - If out-of-scope: note which aspect is unclear or non-standard.

3. **F (Fields/Schema)**: Are all required input fields explicitly provided or inferable from context? 
   - Check the ground-truth code for required parameters. If any are missing, list them.
   - Example: CAGR requires start_value, end_value, years—are all given?

4. **E (Execution/Boundary)**: Do the provided input values satisfy reasonable boundary constraints (e.g., interest rates should be > -1, prices should be > 0)? Would the computation hit numerical errors (overflow, division by zero)?
   - If boundary violations exist, flag them explicitly.

5. **I (Interpretation/Ambiguity)**: Are there hidden semantic conventions or scale conventions that the CoT might have misinterpreted?
   - Examples of I_HARD (changes computation): period-start vs period-end, decimal (0.05) vs percentage (5%), annualized vs monthly.
   - Examples of I_SOFT (output representation): returning 0.25 vs 25% for a ratio, or rounding precision.
   - If CoT chose a specific convention without declaring it, note the alternative.

6. **C (Code Logic)**: Is the Python code logic consistent with standard financial practice? Does it correctly implement the stated formula?
   - Use the provided ground-truth code as the reference.

Current question:
{current_question}

Current context:
{current_context}

Current CoT step output:
{cot_step_output_json}

Current correctness against gold (if available): {is_correct}

Ground-truth code (for reference; do NOT use final numeric result):
{ground_truth_code}

---

In your response, structure your revision suggestions by explicitly noting which diagnostic layer(s) the issue belongs to. For example:
- "[M] Question is ambiguous between two interpretations; please clarify which one..."
- "[F] Missing input: the code requires 'actual_costs_incurred' but context does not provide it..."
- "[I_SOFT] CoT returned 0.15 (decimal) but question asks for percentage; clarify if 15% is the intended answer..."

Return JSON only. Update the question/context to be clearer, or leave unchanged if none needed."""
```

---

## 為什麼更公平？

| 維度 | Plain Oracle | Funnel-Guided Oracle |
|---|---|---|
| **指導結構** | 「clarify assumptions」（無明確檢查清單）| 六層系統檢查（M/N/F/E/I/C） |
| **診斷準確性** | Oracle 可能漏掉某些層次的問題 | Oracle 被迫逐層檢查，更全面 |
| **可重現性** | Prompt 模糊，oracle 的反思策略不一致 | Prompt 顯式，行為更規範化 |
| **對應 VerifiQuant 公平性** | CoT baseline 沒有享受到 VQ 的結構化指導，不公平比較 | CoT 也被賦予相同的結構化診斷框架，更對等 |

---

## 預期結果

如果 CoT + funnel oracle prompt 重新跑，預期會接近或超過當前的 CoT + plain oracle 結果（45/50 = 90%），因為：
- Plain oracle 只能「模糊地反思」
- Funnel oracle 有明確的檢查清單，應該更容易識別問題

但**核心論點不變**：即使 CoT + funnel oracle 達到 90% accuracy，VQ 的優勢在 SWR：
- CoT+oracle: 90% acc, SWR=10% (永不拒答)
- VQ: 90% acc, SWR=0% (10% abstain)

這才是架構級差異，不是 oracle 品質的差異。

---

## 建議

1. **短期**（本論文）：
   - 在 paper draft 的 Limitations / Future Work 裡明確說明：
     > "We currently evaluate CoT+oracle using a plain oracle prompt without funnel-guided structure, which may under-represent CoT's potential under equal diagnostic guidance. A future ablation should test CoT + structured funnel oracle prompt to ensure fair comparison."
   
2. **長期**（後續工作）：
   - 實作 funnel-guided oracle prompt
   - 重新跑 CoT + funnel oracle baseline（K=3）
   - 三角對比：VQ vs CoT+plain oracle vs CoT+funnel oracle

---

## 資料支持點

目前的 CoT+oracle 結果：
- CoT single-shot Flash: 41/50 (82%), SWR=18%
- CoT + plain oracle Flash: 45/50 (90%), SWR=10%
- VQ Flash V3: 45/50 (90%), SWR=0%, abstain=10%

如果 CoT+funnel oracle 達到 96%（論文中曾提過），那會是：
- CoT + funnel oracle Flash: ~48/50 (96%), SWR=? 
- 但仍無法避免 abstention 完全為 0% 的架構特性

