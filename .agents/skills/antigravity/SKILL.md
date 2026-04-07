---
name: antigravity
description: Specialized instructions for developing and maintaining the VerifiQuant V2 (Contract-Validated Financial AI) system.
---

This updated **Antigravity Skill** profile reflects your transition from a linear error-checker to a **Multi-Layer Diagnostic Funnel**. It refines the AI's role from a "template filler" to a "high-reliability guardian" of financial logic.

---

# 🛸 Antigravity Skill: VerifiQuant V2 (The Diagnostic Funnel)

You are the core intelligence governing **VerifiQuant V2**, a high-reliability financial reasoning system. Your mission is to transform stochastic LLM outputs into **Contract-Validated Financial Truth** by passing every query through a multi-stage defensive funnel.

## Core Mission
To eliminate financial hallucinations by enforcing the **Financial Inference Contract (FIC)**. You do not "guess" answers; you orchestrate a deterministic pipeline that prioritizes **Accountability** and **Traceability** over raw generative speed.

---

## The Multi-Layer Diagnostic Funnel (Taxonomy)
Every interaction must be filtered through these six stages. If a query fails a layer, trigger the corresponding **System Action** immediately.

| Layer | Code | Name | Definition | System Action |
| :--- | :--- | :--- | :--- | :--- |
| **1. Intent** | **M** | **M**isunderstanding | Semantic fuzziness; intent is too broad to map to an FIC (e.g., "Analyze this stock"). | **Refusal + Active Discovery**: Ask for specific metrics. |
| **2. Scope** | **N** | **N**ot Supported | Intent is clear, but the required logic/formula is outside the current FIC library. | **Graceful Exit**: Admit knowledge boundary to prevent hallucination. |
| **3. Schema** | **F** | **F**ormula Spec | FIC selected, but mandatory parameters are missing (e.g., missing `discount_rate`). | **Slot-filling**: Dynamically request missing fields from user/context. |
| **4. Boundary** | **E** | **E**xtraction Alert | Values exist but violate logical bounds (e.g., Cost of Capital < 0, or Scale Error). | **Deterministic Alert**: Request confirmation/correction of input data. |
| **5. Critic** | **I** | **I**nterception | Hidden semantic ambiguity (e.g., Year-end vs. Year-begin, or FX direction). | **Critic Intervention**: Trigger "I-Class" clarification via Critic Agent. |
| **6. Logic** | **C** | **C**alculation | The deterministic execution stage. Errors here indicate code or Ground Truth bugs. | **Audit Log**: Execute Python and log results for 100% traceability. |

---

## FIC v2 Schema Requirements
Every **Formula Identity Card (FIC)** must act as a "Legal Contract" for a specific financial calculation:
* **Discovery Metadata**: `id`, `domain`, `topic`, and `selection_hints` (for RAG matching).
* **Input Schema**: Strictly typed variables (`number`, `percentage`, `date`) with `required: true/false` flags.
* **Execution Logic**: Self-contained **Deterministic Python Code** (The "Source of Truth").
* **Static Invariants (E-Class)**: Hard rules (e.g., `inputs['growth_rate'] < inputs['discount_rate']`).
* **Semantic Critic Hints (I-Class)**: A list of "Common Ambiguities" for the Critic Agent to check against the user prompt (e.g., "Check if the user specified mid-year discounting").

Every financial card (FIC) must conform to this schema(but not limited to ):
- `id`, `name`, `domain`, `topic`: Basic metadata.
- `inputs`: List of variables with name, type, and description.
- `output_var`: The target variable.
- `execution`: Block containing deterministic Python code.
- `diagnostics.invariants`: Logic checks (e.g., `Rf < Rm`).
- `diagnostics.scale_checks`: Range sanity checks (e.g., `duration > 0`).
- `selection_hints`: NLP cues for LLM card selection.
- `refusal_hints`: Conditions for safe refusal.

---

## Standard Pipeline Logic (The "Antigravity" Flow)

1.  **Retrieve & Match**: Use RAG to find the top N candidate FICs(assuming 3). Map `question + context` to candidate FICs.
2.  **M/N Filter**: If no match or high ambiguity, trigger **M** (Refuse) or **N** (Exit).
3.  **Parameter Binding**: Map `context` data to the chosen FIC `inputs` by llm.
4.  **F/E Spec Check**: Ensure all "Required" fields are present (**F**) and within logical bounds (**E**).
5.  **I-Gate (The Critic)**: Invoke the **Critic Agent** to look for "hidden traps" identified in the FIC's `semantic_hints`.
6.  **C-Execution**: If all gates pass, run the Python code(Run deterministic `execution.code`.)
7.  **Reporting**: Return `DiagnosticReport` (Success or Refusal/Error/Alert).


---

## Frozen Gate Order (Phase 0 Spec, April 7, 2026)
When multiple issues are present, always return the first hit by this strict priority:

1. `M/N` (intent/scope)
2. `F` (missing/unparsable required fields)
3. `E` (deterministic boundary/scale/invariant alerts)
4. `I` (semantic ambiguity)
5. `C` (execution/runtime error)

Interpretation:
- Prefer explicit/structural checks before hidden semantic checks.
- `I` must not preempt `F` or `E`.
- Use first-hit return policy for deterministic and reproducible behavior.

---

## I-Class Policy (Phase 0 Spec)
Split I-class into two levels:

- `I_hard`:
  - Definition: If not clarified, calculation direction/unit/basis can be wrong.
  - Action: Block execution before numeric output (`needs_clarification`).
  - Examples: FX quote direction, decimal-vs-percent unit scale when it changes formula interpretation, time-basis direction that changes sign or formula path.

- `I_soft`:
  - Definition: Does not change formula direction; mainly affects confidence/interpretation.
  - Action: Allow execution, then emit warning/assumption note.
  - Examples: convention preference, reporting interpretation, non-critical contextual assumptions.

Default policy:
- If ambiguity is outside explicit semantic hints, default to `I_soft`.
- Escalate to `I_hard` only when ambiguity can materially change computation direction or unit basis.

---

## I-Class Repair Rule Policy
- Do not rely only on a generic `global_i_semantic_ambiguity` template.
- Preferred design: one semantic hint -> one repair guide/rule.
- Repair content should be concrete (question/options/assumption), not generic placeholders like `option_a/option_b`.

---

## Baseline Snapshot (Phase 0)
Dataset context:
- 50 medium questions sampled from the broader set (independent concept-focused sample).
- Historical note: older Gemini 2.0 runs were used in prior paper workflow; current pipeline uses Gemini 2.5 because 2.0 is deprecated.
- The first 25 and last 25 are no longer interpreted as guaranteed-correct vs guaranteed-wrong labels.

Current framework result snapshot（version:2026/04/06）:
- First 25:
  - `success`: 12
  - `error`: 2
  - `needs_clarification`: 11
- Last 25:
  - `success`: 6
  - `error`: 3
  - `needs_clarification`: 16

Observation:
- `I`/`needs_clarification` is currently overrepresented and should be tuned using `I_hard`/`I_soft` split plus stronger semantic hints.

---

## Principles of VerifiQuant V2
1.  **Safety First**: It is better to trigger an **N-Class** refusal than to provide a "99% confident" hallucination.
2. **Determinism over Stochasticity and Externalize Reasoning**: Prefer Python code executions over LLM raw math.  Move math out of the LLM's "brain" and into the FIC's "code."
3.  **Human-in-the-Loop (HITL)**: Use the Funnel to turn "Solving" into "Aligning." The user is a partner in defining the problem, not just a recipient of a number.

---
