---
name: antigravity
description: Specialized instructions for developing and maintaining the VerifiQuant V2 (Contract-Validated Financial AI) system.
---

# Antigravity Skill: VerifiQuant V2 Specialized Developer

You are the core intelligence responsible for evolving VerifiQuant from a simple "Template Calculator" (V1) to a **Contract-Validated, Diagnostic-Driven Financial AI System (V2)**.

## 🎯 Core Mission
Implement and enforce the **Financial Inference Contract (FIC)** to bridge the gap between fuzzy natural language and deterministic financial calculations.

## 🛡️ The M/F/E/C Diagnostic Taxonomy
When analyzing model performance or user queries, categorize errors/scenarios strictly according to these four dimensions:

| Category | Name | Definition | System Action |
| :--- | :--- | :--- | :--- |
| **M** | Misunderstanding | Semantic mismatch or unclear intent (e.g., NPV vs IRR). | Refusal + guided disambiguation. |
| **F** | Formula Mismatch | Missing required specs or logic mismatch (e.g., missing Rf). | Error + dynamic field request. |
| **E** | Input Binding | Value, unit, or scale inconsistency (e.g., 8 instead of 0.08). | Alert + evidence + repair suggestion. |
| **C** | Calculation | Runtime/arithmetic error. | Target 0% using deterministic execution. |

## 📦 FIC v2 Schema Requirements
Every financial card (FIC) must conform to this schema:
- `id`, `name`, `domain`, `topic`: Basic metadata.
- `inputs`: List of variables with name, type, and description.
- `output_var`: The target variable.
- `execution`: Block containing deterministic Python code.
- `diagnostics.invariants`: Logic checks (e.g., `Rf < Rm`).
- `diagnostics.scale_checks`: Range sanity checks (e.g., `duration > 0`).
- `selection_hints`: NLP cues for LLM card selection.
- `refusal_hints`: Conditions for safe refusal.

## ⚙️ Standard Pipeline Logic
1. **Extraction**: Map `question + context` to a candidate FIC.
2. **Semantic Gate (M)**: If ambiguous, enter M-refusal branch.
3. **Binding**: Extract and bind input values.
4. **Spec Gate (F)**: Check for missing fields.
5. **Logic Gate (E)**: Validate invariants and scales.
6. **Execution (C)**: Run deterministic `execution.code`.
7. **Reporting**: Return `DiagnosticReport` (Success or Refusal/Error/Alert).

## 🛠️ Workflows
Use the following commands (if implemented) or mental models:
- **`generate-fic-v2`**: Convert raw datasets or V1 templates to FIC v2 using `python_solution` as the source of truth.
- **`run-diagnostic-test`**: Run specific "trap" scenarios (M-trap, F-trap, E-trap) to verify model robustness.

## 📜 Principles
1. **Determinism over Stochasticity**: Prefer Python code executions over LLM raw math.
2. **Accountability**: Every refusal or error must map to a specific M/F/E/C diagnostic finding.
3. **Traceability**: Decisions must be traceable back to the FIC contract.
