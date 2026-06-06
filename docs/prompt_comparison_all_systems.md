# Prompt & System Comparison: CoT vs VerifiQuant

> 來源：
> - `verifiquant/pipeline/run_cot_self_improve_pipeline.py`
> - `verifiquant/pipeline/run_framework_guided_self_improve_pipeline.py`
> 
> 實驗結果：`verifiquant/data/runs/paper_v1/results/`

---

## 一、實驗結果全覽（50Q, paper_v1）

| System | Model | Acc | Correct | Improved |
|---|---|---|---|---|
| CoT single-shot | Flash | 82% | 41/50 | — |
| CoT single-shot | Pro | 82% | 41/50 | — |
| CoT + plain oracle | Flash | 90% | 45/50 | 3 |
| CoT + plain blind-review | Flash | 94% | 47/50 | 6 |
| CoT + plain blind-review | Pro | 98% | 49/50 | 6 |
| **CoT + funnel-guided oracle** | **Flash** | **94%** | **47/50** | **4** |
| **CoT + funnel-guided oracle** | **Pro** | **98%** | **49/50** | **6** |
| CoT + funnel blind-review | Flash | 92% | 46/50 | 5 |
| CoT + funnel blind-review | Pro | 96% | 48/50 | 4 |
| VQ (Flash V3) | Flash | 90% | 45/50 | — |
| VQ (Pro V3) | Pro | 86% | 43/50 | — |

**核心觀察**：CoT + funnel-guided oracle 在 Flash 上達到 94%，高於 VQ Flash V3 (90%)。  
但 SWR 架構差異不變：CoT 系列 abstain=0%（永不拒答），VQ abstain~10%（不確定時拒答）。

---

## 二、Oracle 模式分類

### 2.1 Blind-Review vs Guided 的差異

| | **Blind-Review** | **Guided Oracle** |
|---|---|---|
| 進入條件 | 每輪必進（無條件） | 僅在上一輪答錯時進 |
| Oracle 是否知道答對答錯 | 不知道（雙盲） | 知道 (`is_correct` 傳入） |
| 設計用途 | 模擬 LLM 自省，無外部信號 | 帶回饋的自改善 |

---

## 三、Prompt 全文對照

### 3.1 Prompt A：CoT Solver（CoT pipeline 專屬，VQ 無此步驟）

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

**特點**：VQ pipeline 直接呼叫 `ErrorClassificationAPI.diagnose_row()`，不需要這個 solver prompt。

---

### 3.2 Prompt B：Plain Oracle（CoT pipeline, `--oracle-mode plain`）

```
You are Oracle-in-the-loop support for a pure CoT baseline.
There is no VerifiQuant framework in this loop.
You can ONLY use the logic within the ground-truth code to clarify assumptions
and revise user question/context. You must NOT rely on the final numeric ground truth result.

Current question: {current_question}
Current context: {current_context}
Current CoT step output: {cot_step_output_json}
Current correctness against gold (if available): {is_correct}
Ground-truth code: {ground_truth_code}

Return JSON only.
```

**問題**：「clarify assumptions」太泛泛，Oracle 不知道該檢查哪些維度。

---

### 3.3 Prompt C：Funnel-Guided Oracle（CoT pipeline, `--oracle-mode funnel-guided`）

Plain Oracle 加入 M/N/F/E/I/C 六層結構化檢查框架：

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
   - If ambiguous, explicitly list the alternative interpretations.

2. N (Scope/Domain): Does the question fall within a well-defined financial domain
   (valuation, cash flow, bond pricing)? Or does it require expertise outside
   standard finance?
   - If out-of-scope, note which aspect is unclear or non-standard.

3. F (Fields/Schema): Are all required input fields explicitly provided or inferable?
   - Check the ground-truth code for required parameters. If any are missing, list them.
   - Example: CAGR requires start_value, end_value, and years. Are all given?

4. E (Execution/Boundary): Do inputs satisfy reasonable boundary constraints
   (interest rates > -1, prices > 0)? Would computation hit numerical errors (div/0)?
   - If boundary violations exist, flag them explicitly.

5. I (Interpretation/Ambiguity): Are there hidden semantic or scale conventions
   the CoT might have misinterpreted?
   - I_HARD (changes computation): period-start vs period-end, decimal (0.05) vs
     percentage (5%), annualized vs monthly.
   - I_SOFT (output representation): returning 0.25 vs 25% for a ratio, rounding.
   - If CoT chose a convention without declaring it, note the alternative.

6. C (Code Logic): Is the Python code logic consistent with standard financial practice?
   Does it correctly implement the stated formula? Use GT code as reference.

Current question: {current_question}
Current context: {current_context}
Current CoT step output: {cot_step_output_json}
Current correctness against gold (if available): {is_correct}
Ground-truth code (for reference; do NOT use final numeric result): {ground_truth_code}

Instructions:
1. Work through the six diagnostic layers in priority order: M > N > F > E > I > C.
2. Identify the highest-priority issue that should be clarified or corrected.
3. Revise the question/context only enough to address that issue while preserving the original task.
4. In notes, clearly mark the relevant layer: [M], [F], [I_HARD], [I_SOFT], or [C].
5. If no issue is found, leave question/context unchanged and explain checked layer(s) briefly.

Return JSON only.
```

**與 Plain Oracle 的差異**：
- 明確宣告「no VerifiQuant execution framework」（對稱設計）
- M/N/F/E/I/C 知識以**描述文字**注入 prompt — 診斷知識在 prompt 裡，不在系統裡
- 強制優先順序 M > N > F > E > I > C
- 輸入仍是 CoT 的**中間輸出**，不是 framework 的 structured diagnostic

---

### 3.4 Prompt D：VQ Framework Oracle（VQ pipeline 專屬）

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

**與 CoT Oracle 的差異**：
- 輸入是 `ErrorClassificationAPI.diagnose_row()` 回傳的 **structured diagnostic JSON**
  （含 `status`, `diagnostic_type`, `findings`, `requested_fields`, `clarification_request`）
- Oracle **不需要自己做診斷**，任務是「讓下一輪能通過 gates」
- Prompt 最短——診斷複雜度移到系統層，不在 prompt 裡
- 進入條件由 `_framework_oracle_loop()` 控制：
  `diagnostic_type in {M, N, F, E, I, I_HARD, I_SOFT}` 或 `I_SOFT_MISMATCH` 特殊攔截

---

## 四、核心差異對照表

| 維度 | CoT Plain | CoT Funnel-Guided | VQ Framework-Guided |
|---|---|---|---|
| 診斷知識來源 | 無 | Prompt 文字描述 | Framework 執行結果 JSON |
| 層次優先順序 | 無 | Prompt 裡硬編 M>N>F>E>I>C | Framework 已決定（`diagnostic_type`） |
| Oracle 輸入 | CoT solver 輸出 JSON | CoT solver 輸出 JSON | `DiagnosticReport` dict |
| Oracle 的任務 | 自由改寫 | 按層次改寫 | 讓下一輪通過 gates |
| I_SOFT 攔截 | 無 | 無（Oracle 自己識別） | `has_i_soft + is_correct==False` 顯式攔截 |
| Prompt 長度 | 最短 | 最長（~400 字描述層次） | 中等（邏輯在系統層） |
| LLM 呼叫數/turn | 2（solver + oracle） | 2（solver + oracle） | 1 oracle（framework 做診斷） |

---

## 五、設計意圖（消融分析視角）

| 系統 | 設計目的 |
|---|---|
| **CoT Plain Oracle** | 最乾淨的 baseline：Oracle 沒有任何 VQ 知識 |
| **CoT Funnel-Guided Oracle** | 消融實驗：把 M/N/F/E/I/C *知識* 給 Oracle，但**不給系統**。驗證「光靠 prompt 描述診斷層有沒有效果」。結果：Flash 94%（≥ VQ Flash 90%），但 SWR 架構差異不變 |
| **VQ Framework-Guided** | 完整系統：診斷由 deterministic code 做，Oracle 只負責問題改寫；分工最清晰，SWR=0%，abstain≈10% |

**核心論點**：CoT + funnel oracle 在 accuracy 上可達甚至超過 VQ，但無法實現 abstention。  
VQ 的架構優勢不在準確率，而在**可預測的失敗行為**（SWR=0%）。

---

## 六、CLI 執行命令

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
