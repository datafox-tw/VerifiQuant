# Funnel-Guided Oracle Prompt - Final Version
# 用於 run_cot_self_improve_pipeline.py 的新增版本

_ORACLE_PROMPT_TEMPLATE_FUNNEL_GUIDED = """\
You are Oracle-in-the-loop support for a CoT baseline with structured diagnostic guidance.
You can ONLY use the logic within the ground-truth code to clarify assumptions and revise user question/context. You must NOT rely on the final numeric ground truth result.

When reviewing the CoT step output for potential errors or ambiguities, systematically check the following diagnostic layers (M/N/F/E/I/C):

1. **M (Intent/Meaning)**: Does the user's question map unambiguously to a single calculation or formula? Are there multiple valid interpretations of what is being asked?
   - Example: "profit on total contract" could mean (a) profit as % of incurred costs or (b) fixed fee on total contract value.
   - If ambiguous: explicitly list the alternative interpretations.

2. **N (Scope/Domain)**: Does the question fall within a well-defined financial domain (e.g., valuation, cash flow, bond pricing)? Or does it require domain expertise outside standard finance?
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

---

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

INSTRUCTIONS FOR YOUR RESPONSE:

1. Work through the six diagnostic layers (M → N → F → E → I → C) in order.
2. For each layer, decide: Is there a potential issue that needs clarification?
3. If you identify issues across multiple layers, prioritize them: M (fundamental ambiguity) > N (scope) > F (missing fields) > E (boundary) > I (convention) > C (logic).
4. Formulate your revision suggestions to address the highest-priority issues.
5. Structure your output by clearly noting which diagnostic layer(s) the issue belongs to.

Examples of good output:
- "[M] Question is ambiguous between interpretations X and Y; please clarify which one."
- "[F] Missing input: code requires 'actual_costs_incurred' but context doesn't provide it."
- "[I_SOFT] CoT returned 0.15 (decimal) but question asks for 'percentage'; clarify if 15% is intended."
- "[M→I] The term 'cost-plus' is ambiguous (cost-plus on actual costs vs. cost-plus as fixed fee), AND this ambiguity affects the interpretation of the percentage input. Both need clarification."

If no issues found in a layer, you may note "[Layer]: OK" or skip it.

Return JSON only."""


# 新增的 schema （結構同舊版，不變）
def _schema_oracle_rewrite() -> Any:
    return genai_types.Schema(
        type=genai_types.Type.OBJECT,
        properties={
            "updated_question": genai_types.Schema(type=genai_types.Type.STRING),
            "updated_context": genai_types.Schema(type=genai_types.Type.STRING),
            "notes": genai_types.Schema(type=genai_types.Type.STRING),
        },
        required=["updated_question", "updated_context", "notes"],
    )


# 新的 oracle function
def _oracle_rewrite_for_cot_funnel_guided(
    *,
    client: Any,
    model: str,
    row: Dict[str, Any],
    current_question: str,
    current_context: str,
    cot_step_output: Dict[str, Any],
    is_correct: Optional[bool],
) -> Dict[str, str]:
    """
    Oracle with explicit M/N/F/E/I/C layer guidance for CoT baseline.
    Ensures fair comparison with VerifiQuant's structured diagnostics.
    """
    prompt = _ORACLE_PROMPT_TEMPLATE_FUNNEL_GUIDED.format(
        current_question=current_question,
        current_context=current_context,
        cot_step_output_json=json.dumps(cot_step_output, ensure_ascii=False, indent=2),
        is_correct=is_correct,
        ground_truth_code=row.get("code", row.get("python_solution", "")),
    )
    out = _llm_json(client, model=model, prompt=prompt, schema=_schema_oracle_rewrite())
    return {
        "updated_question": str(out.get("updated_question", "") or current_question).strip() or current_question,
        "updated_context": str(out.get("updated_context", "") or current_context).strip() or current_context,
        "notes": str(out.get("notes", "")).strip(),
    }


# 修改後的主迴圈（核心改動）
def _cot_oracle_loop_funnel_guided(
    *,
    client: Any,
    model: str,
    oracle_model: str,
    row: Dict[str, Any],
    max_turns: int = 3,
) -> Dict[str, Any]:
    """
    CoT self-improve loop with funnel-guided oracle.
    Same as _cot_oracle_loop but uses funnel-guided oracle prompt.
    """
    question = row["question"]
    context = row["context"]
    history = []
    final_answer = None
    final_correct = None
    final_abs_error = None

    for turn in range(1, max_turns + 1):
        step = _cot_step(
            client=client,
            model=model,
            question=question,
            context=context,
        )

        ans = _parse_number(step.get("answer"))
        gold_num = _parse_number(_gold_value(row))
        abs_err, is_correct = _answer_match(question, ans, gold_num)
        final_answer = ans
        final_correct = is_correct
        final_abs_error = abs_err

        history.append(
            {
                "turn": turn,
                "question": question,
                "context": context,
                "cot_step": step,
                "parsed_answer": ans,
                "is_correct": is_correct,
                "abs_error": abs_err,
            }
        )

        print(f"  ⤳ CoT Turn {turn}: is_correct={is_correct}")

        needs_more = bool(step.get("needs_more_info"))
        should_iterate = needs_more or (is_correct is False)
        if not should_iterate or turn >= max_turns:
            break

        # 使用 funnel-guided oracle
        rewrite = _oracle_rewrite_for_cot_funnel_guided(
            client=client,
            model=oracle_model,
            row=row,
            current_question=question,
            current_context=context,
            cot_step_output=step,
            is_correct=is_correct,
        )
        new_q = rewrite["updated_question"] or str(step.get("revised_question", "")).strip() or question
        new_c = rewrite["updated_context"] or str(step.get("revised_context", "")).strip() or context
        if new_q == question and new_c == context:
            break
        question, context = new_q, new_c

    return {
        "rounds": len(history),
        "final_answer": final_answer,
        "final_is_correct": final_correct,
        "final_abs_error": final_abs_error,
        "history": history,
    }


# 使用方式
if __name__ == "__main__":
    # 在 main() 的 argparse 裡加上：
    # parser.add_argument("--oracle-mode", choices=["plain", "funnel-guided"], default="plain")

    # 在 _cot_oracle_loop 的呼叫點改為：
    # if args.oracle_mode == "funnel-guided":
    #     result = _cot_oracle_loop_funnel_guided(...)
    # else:
    #     result = _cot_oracle_loop(...)  # plain version

    pass
