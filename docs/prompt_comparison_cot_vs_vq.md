# Prompt Comparison: CoT vs VerifiQuant (Flash Pro)

> 來源檔案：
> - `verifiquant/pipeline/run_cot_self_improve_pipeline.py`
> - `verifiquant/pipeline/run_framework_guided_self_improve_pipeline.py`

---

## 架構概覽

| 維度 | CoT Pipeline | VerifiQuant (FG) Pipeline |
|---|---|---|
| **主角色** | CoT Solver + Oracle | VerifiQuant ErrorClassificationAPI + Oracle |
| **LLM 呼叫數/turn** | 2（solver + oracle） | 1 oracle（framework 本身做診斷） |
| **診斷邏輯所在** | Oracle prompt 裡描述 | `run_error_classification_pipeline.py` 執行 |
| **Oracle 輸入** | 問題 + CoT 步驟輸出 + GT code | 問題 + framework diagnostic JSON + GT code |
| **Oracle 輸出** | 改寫後的 question/context | 改寫後的 question/context |

---

## Prompt 1：CoT Solver（CoT pipeline 專屬）

```
You are a financial CoT solver with self-improvement hints.
Solve using the current question/context and return a tentative answer.
If information is missing/ambiguous, set needs_more_info=true and list missing items.
IMPORTANT: `answer` must be a numeric string only (for example: "10185.19"),
or empty string if you cannot provide one.

Question: {question}
Context: {context}

Return JSON only.
```

**特點：**
- VQ pipeline **沒有這個 prompt**（框架直接呼叫 `ErrorClassificationAPI.diagnose_row()`）
- 要求純數字答案字串
- 沒有任何診斷層引導

---

## Prompt 2：Oracle — Plain（CoT pipeline）

```
You are Oracle-in-the-loop support for a pure CoT baseline.
There is no VerifiQuant framework in this loop.
You can ONLY use the logic within the ground-truth code to clarify assumptions
and revise user question/context. You must NOT rely on the final numeric ground truth result.

Current question: {current_question}
Current context: {current_context}
Current CoT step output: {cot_step_output_json}
Ground-truth code: {ground_truth_code}

Return JSON only.
```

**特點：**
- 明確宣告「no VerifiQuant framework」
- 輸入是 CoT 的**中間輸出 JSON**（solver 的回傳結果）
- Oracle 完全不知道診斷層，只憑 GT code 判斷

---

## Prompt 3：Oracle — Funnel-Guided（CoT pipeline，`--oracle-mode funnel-guided`）

Plain Oracle 的強化版，在問題前插入 M/N/F/E/I/C 六層系統性檢查框架：

```
You are Oracle-in-the-loop support for a CoT baseline with structured diagnostic guidance.
There is no VerifiQuant execution framework in this loop; you are only providing
an oracle rewrite for the CoT baseline.
You can ONLY use the logic within the ground-truth code to clarify assumptions
and revise user question/context. You must NOT rely on the final numeric ground truth result.

When reviewing the CoT step output for potential errors or ambiguities,
systematically check the following diagnostic layers (M/N/F/E/I/C):

1. M (Intent/Meaning): Does the user's question map unambiguously to a single
   calculation or formula? Are there multiple valid interpretations?
   - Example: "profit on total contract" could mean profit as % of incurred costs
     or a fixed fee on total contract value.

2. N (Scope/Domain): Does the question fall within a well-defined financial domain?
   Or does it require expertise outside standard finance?

3. F (Fields/Schema): Are all required input fields explicitly provided or inferable?
   - Check the ground-truth code for required parameters. If any are missing, list them.
   - Example: CAGR requires start_value, end_value, and years.

4. E (Execution/Boundary): Do inputs satisfy reasonable boundary constraints
   (interest rates > -1, prices > 0)? Would computation hit numerical errors (div/0)?

5. I (Interpretation/Ambiguity): Are there hidden semantic or scale conventions
   the CoT might have misinterpreted?
   - I_HARD examples: period-start vs period-end, decimal (0.05) vs percentage (5%),
     annualized vs monthly.
   - I_SOFT examples: returning 0.25 vs 25% for a ratio, or rounding precision.

6. C (Code Logic): Is the Python code logic consistent with standard financial practice?
   Does it correctly implement the stated formula? Use GT code as reference.

[...same variable block as Plain Oracle...]

Instructions:
1. Work through the six diagnostic layers in priority order: M > N > F > E > I > C.
2. Identify the highest-priority issue.
3. Revise question/context only enough to address that issue.
4. In notes, clearly mark the relevant layer: [M], [F], [I_HARD], [I_SOFT], or [C].
5. If no issue is found, leave question/context unchanged and explain checked layer(s).

Return JSON only.
```

**特點：**
- 仍然**宣告「no VerifiQuant execution framework」**
- M/N/F/E/I/C 以**描述文字**注入 Oracle prompt — 診斷知識在 prompt 裡，不在系統裡
- 強制優先順序 M > N > F > E > I > C
- 輸入仍是 CoT 的**中間輸出**，不是 framework 的 structured diagnostic

---

## Prompt 4：Oracle — VQ Framework-Guided（VQ pipeline 專屬）

```
You are an Oracle-in-the-loop rewriter for VerifiQuant framework-guided iteration.
You can ONLY use the logic within the ground-truth code to clarify assumptions
and revise user question/context. You must NOT rely on the final numeric ground truth result.

Goal:
- Given framework diagnostic output, rewrite question/context so the next run can pass gates.
- Keep semantics faithful to original task and the logic implied by the provided code.
- Do NOT output the final numeric answer directly in question/context.

Current question: {current_question}
Current context: {current_context}

Framework diagnostic:
{json.dumps(diagnostic, ensure_ascii=False, indent=2)}

Ground-truth code: {row["code"]}

Return JSON only.
```

**特點：**
- 輸入是 `ErrorClassificationAPI.diagnose_row()` 回傳的**structured diagnostic JSON**
  （含 `status`, `diagnostic_type`, `findings`, `requested_fields`, `clarification_request` 等）
- Oracle **不需要自己做診斷**，診斷已由 framework 完成；任務是「讓下一輪能通過 gates」
- Prompt 最短——診斷複雜度移到系統層，不在 prompt 裡
- 進入 Oracle 的條件由 `_framework_oracle_loop()` 控制：
  - `status != "success"` 且 `diagnostic_type in {M, N, F, E, I, I_HARD, I_SOFT}`
  - 特殊攔截：`I_SOFT_MISMATCH`（成功執行但結果不符 GT，且有 soft warnings）

---

## 核心差異對照

| 差異點 | CoT Plain | CoT Funnel-Guided | VQ Framework-Guided |
|---|---|---|---|
| 診斷知識來源 | 無 | Prompt 文字描述 | Framework 執行結果 JSON |
| 層次優先順序 | 無 | Prompt 裡硬編 M>N>F>E>I>C | Framework 已決定（`diagnostic_type`） |
| Oracle 的輸入 | CoT solver 輸出 JSON | CoT solver 輸出 JSON | `DiagnosticReport` dict |
| Oracle 的任務 | 自由改寫 | 按層次改寫 | 讓下一輪通過 gates |
| I_SOFT 攔截 | 無 | 無（只能靠 Oracle 自己識別） | `has_i_soft + is_correct==False` 顯式攔截 |
| Prompt 長度 | 最短 | 最長（~400 字描述層次） | 中等（邏輯在系統層） |
| 需要 GT code | 是 | 是 | 是（但 diagnostic 已消化過） |

---

## 執行模式選擇（CLI）

```bash
# CoT plain oracle
python3 preprocessing/run_cot_self_improve_pipeline.py \
  --oracle-mode plain --max-turns 3

# CoT funnel-guided oracle
python3 preprocessing/run_cot_self_improve_pipeline.py \
  --oracle-mode funnel-guided --max-turns 3

# VerifiQuant framework-guided
python3 preprocessing/run_framework_guided_self_improve_pipeline.py \
  --max-turns 3
```

---

## 設計意圖小結

- **CoT Plain**：最乾淨的 baseline，Oracle 沒有任何 VQ 知識。
- **CoT Funnel-Guided**：消融實驗用途——把 M/N/F/E/I/C *知識* 給 Oracle，但**不給系統**，驗證「光靠 prompt 描述診斷層有沒有效果」。
- **VQ Framework-Guided**：完整系統——診斷由 deterministic code 做，Oracle 只負責問題改寫；分工最清晰，也最可預測。
