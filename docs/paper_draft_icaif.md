# VerifiQuant: Governance-by-Execution for Financial LLM Workflows

**Status**: Full working draft (English), 2026-07-12
**Target**: ICAIF 2026 (deadline 08/02) · ACM sigconf, 8 pages
**Number sources**: `docs/2026-07-12_experiment_record_master.md` (single source of truth)

---

## 📋 TODO Master List（要改哪些——投稿前逐項打勾）

| # | 位置 | 動作 | 依賴 |
|---|---|---|---|
| T1 | Abstract + §5.1 主表 + §5.2 | 50Q 數字換成 250Q（全文搜 `[TODO-V2]`） | A2/A4 跑完（~07/26） |
| T2 | §5.2 | 新增 medium/hard 分層表 + Fig 2（難度上移時 abstention/SWR 行為） | A4 |
| T3 | Abstract/§5/§8 | 若 250Q VQ 出現 SW：套用 fallback 措辭（"every residual failure attributed to a named funnel layer"），並在 §7 逐題歸因 | A4 結果 |
| T4 | §5.1 | VQ Pro 250 若跑完（A5-①），over-normalization 由質性升級為統計驗證，§6.4 加一句 | A5-① |
| T5 | §4.1 | 250Q 卡片良率段落：修卡數量、`card_repairs.json` 紀律說明 | A3 |
| ~~T6~~ | Fig 1 | ✅ **完成 (07/12)**：`docs/figures/fig1_pipeline.svg`（1240×560，LLM=橘/deterministic=藍/exit=紅/repair=紫虛線，C1–C6 badge 對應 Table 1）。LaTeX 移植時轉 PDF：`rsvg-convert -f pdf` 或 Inkscape（併入 T8） | — |
| ~~T7~~ | References | ✅ **完成 (07/12)**：全清單逐筆查證（Tang ✓ / **Yu→Kundu 誤植修正** / Cemri=NeurIPS'25 D&B spotlight）；BibTeX 產出 `docs/references_icaif.bib`，內文引用同步 | — |
| T8 | 全文 | LaTeX sigconf 移植；確認 ICAIF 頁數含不含 references、double-blind、appendix 政策 | CFP |
| T9 | §3.3 | O-ITL 在 ICAIF 版壓縮後是否仍夠 reviewer 理解 K=3 的意義——advisor pass 確認 | advisor |
| T10 | 匿名化 | repo/demo URL、致謝（若 double-blind） | T8 |
| ~~T11~~ | §5.1 Table 2 | ✅ **完成 (07/12)**：Flash/Pro single-shot 改 41/8/(+1) SWR 16%、funnel-Pro SelAcc 修正 96→98%（=48/49）、†/‡ footnote、§1/Abstract/§6.1 措辭同步。三個配置的空答案皆為同一題 test-1242 | — |

---

# Abstract

Large language models are increasingly embedded in financial workflows, yet they are governed — if at all — by prompts and post-hoc review. We argue that the central deployment risk is not accuracy but *governability*: a production financial AI system must declare its scope boundaries, refuse structurally when it cannot verify its own reasoning, expose replayable decision traces, and mathematically verify any repair it applies. We present **VerifiQuant**, a framework that compiles these governance controls into the reasoning pipeline itself. Each financial formula family is represented as a **Financial Inference Contract (FIC)** — retrieval metadata, deterministic executable code, invariants, and declared semantic ambiguities — and every query passes through a six-layer diagnostic funnel (**M/N/F/E/I/C**) in which each layer is a legitimate, auditable exit. Semantic repairs are restricted to **verifiable atomic transforms** whose numerical effect is cross-checked against a declared algebraic identity before acceptance. On a 50-question FinanceReasoning subset **[TODO-V2: 250-question (180 medium + 70 hard)]**, VerifiQuant achieves 90% accuracy with **zero silent-wrong answers** (selective accuracy 100%, abstention 10%), while the strongest CoT+oracle baseline reaches 98% accuracy yet cannot eliminate silent wrongs (SWR ≥ 2%) and provides no structured abstention. On contract-grounded adversarial traps, VerifiQuant reduces silent-wrong rates from 90% to 20% (boundary violations) and from 100% to 10% (semantic ambiguity). Critically, injecting the same diagnostic taxonomy into oracle prompts fails to reproduce these guarantees — under blind review it *degrades* accuracy — indicating that governance must be executed by architecture, not described in prompts.

**[TODO-T3]** 若 250Q 出現 SW，"zero silent-wrong answers" → "eliminates silent-wrong answers on the medium tier and reduces them to [x]% overall, with every residual failure attributed to a named funnel layer"。

---

# 1. Introduction

Financial institutions are deploying large language models in workflows that end in numbers: valuation memos, portfolio analytics, client-facing calculators, regulatory reports. Benchmark accuracy on financial question answering has climbed steadily, and it is tempting to treat the remaining gap as an accuracy problem — one more model generation away from solved. We argue this framing misses what actually blocks deployment. A financial LLM workflow fails in production not primarily because it computes the wrong number, but because *nobody can govern how it computes any number*. Three failure modes, all invisible to headline accuracy, illustrate the gap:

- **Silent wrong.** The model outputs a confident, well-formatted, wrong number, with no signal distinguishing it from a correct one. In our experiments, a chain-of-thought (CoT) baseline on medium-difficulty financial questions produces confident wrong answers on 16% of cases — and on adversarially perturbed inputs, up to 100% for some perturbation classes (§5.3).
- **Audit gap.** When a loss occurs, the institution must reconstruct *which* formula was applied, *which* inputs were bound, and *why* the system did not stop. CoT traces cannot serve this role: they are sampled, non-reproducible, and — as a substantial faithfulness literature shows — not causally connected to the computation that produced the answer [Turpin et al. 2023; Lanham et al. 2023]. Notably, this disconnect *worsens* with model capability, so waiting for stronger models does not close the audit gap.
- **Boundary failure.** Asked a question outside its competence — an exotic derivative it has no procedure for — the system answers anyway, often by silently substituting a different question it *can* answer (§6.3 shows a baseline pricing a "Heston stochastic volatility" request by computing an unrelated manufacturing ratio).

Regulatory frameworks for high-risk AI — Basel model-risk guidance, the EU AI Act's logging provisions, FINRA's supervisory-control expectations — converge on a common demand: systems must declare their boundaries, log reproducibly, expose checkpoints for human intervention, and constrain internal modification (§2). Today these demands are met, when at all, by *prompt instructions* and *post-hoc review* — governance described in natural language and hoped for at runtime.

**Our position is that governance controls must be executed, not described.** We present VerifiQuant, a framework built around three mechanisms:

1. **Financial Inference Contracts (FICs).** Each financial formula family is compiled offline into a machine-readable contract: retrieval metadata, typed inputs, deterministic executable Python, numerical invariants and scale checks, and — crucially — *declared semantic ambiguities* (period-start vs. period-end timing, percent vs. decimal conventions, exchange-rate direction). The LLM never performs arithmetic; it selects contracts, binds fields, and flags ambiguities, all within contract-declared bounds.
2. **A six-layer diagnostic funnel (M/N/F/E/I/C).** Every query passes gates for intent mapping (M), scope (N), schema completeness (F), boundary validity (E), semantic ambiguity (I — split into blocking I_HARD and warning I_SOFT), and finally deterministic computation (C). Each gate is a *legitimate, structured exit*: refusal, slot-filling request, alert, or clarification, each carrying machine-readable diagnostics rather than prose apologies.
3. **Verifiable atomic transforms.** When clarification changes the required computation, the system does not let the LLM rewrite code freely. Repairs are restricted to bounded transforms — a result post-processing expression or a single-site code patch — accepted only if their numerical effect matches a declared algebraic identity under cross-verification.

**Contributions.**
- We reframe financial LLM reliability as a *governance* problem and derive six executable controls from regulatory requirements (§2, Table 1), mapping each to an architectural mechanism and an experimental test.
- We present the FIC + diagnostic-funnel architecture and verifiable atomic transforms (§3), which make refusal, clarification, and repair first-class, auditable outcomes.
- On FinanceReasoning [Tang et al. 2025], we show that VerifiQuant achieves 90% accuracy with SWR = 0% and selective accuracy 100%, while CoT+oracle baselines reach up to 98% accuracy but architecturally cannot eliminate silent wrongs (§5.1) **[TODO-V2]**. On contract-grounded traps, the funnel cuts silent-wrong rates by 4.5× overall and by an order of magnitude on boundary and ambiguity perturbations (§5.3).
- We show that governance does not transfer through prompts: giving a CoT oracle the *complete description* of our diagnostic taxonomy fails to reproduce any guarantee, and under blind review *reduces* accuracy (§6.1). Reliability comes from the executed architecture, not from the knowledge being available in context.

---

# 2. The Governance Problem

## 2.1 What deployment actually requires

We survey three regulatory regimes that financial AI deployments must answer to, and extract the architectural properties they jointly demand. Basel model-risk management (SR 11-7 and successors) requires that a model *identify its own failure regimes* and enforce output boundaries rather than extrapolate. The EU AI Act (§12–13) requires event logging sufficient to *trace any output back to its inputs*, reproducibly. FINRA supervisory guidance requires *explicit human checkpoints* where the system cannot autonomously resolve ambiguity, and that internal modifications be bounded and reviewable.

Table 1 restates these demands as six concrete controls, each paired with the VerifiQuant mechanism that executes it and the experiment that tests it. The rest of the paper is organized around this table: §3 walks the mechanism column; §5–§6 walk the evidence column.

**Table 1: Governance controls → executing mechanism → experimental evidence.**

| # | Governance control | Executing mechanism (§3) | Evidence (§5–6) | Regulatory anchor |
|---|---|---|---|---|
| C1 | Boundary declaration | M/N structured refusal | Trap-N/M; MAS baseline lacks it → SWR 14% | Basel output boundaries |
| C2 | Input & boundary validation | Deterministic E-checks | Trap-E: SWR 90%→20% | Model risk mgmt |
| C3 | Ambiguity disclosure | I_HARD block / I_SOFT warning | Trap-I: SWR 100%→10%; SFWR 0% | FINRA checkpoints |
| C4 | Verified repair | Atomic transforms (AST + numeric cross-verify) | Worked example §3.4; 0 unverified repairs | Bounded modification |
| C5 | Replayable trace | DiagnosticReport + deterministic execution | 4 reruns: accuracy/SWR variance = 0 | EU AI Act §12–13 |
| C6 | Abstention as control | Selective-prediction exits at every gate | 98%-accuracy CoT keeps SWR ≥ 2%; VQ reaches 0% | Accountability |

## 2.2 Why existing approaches do not provide these controls

**Execution-based reasoning** (PAL [Gao et al. 2023], Program-of-Thoughts [Chen et al. 2023], and financial derivatives such as FINDER) delegates arithmetic to an interpreter, addressing C-class errors. But a syntactically perfect program can encode the wrong financial semantics — an end-of-period annuity where the question implies beginning-of-period — and the interpreter neither detects this nor produces structured attribution. Controls C1–C4 are absent.

**Neuro-symbolic translation** (Logic-LM [Pan et al. 2023], LINC [Olausson et al. 2023], SATLM [Ye et al. 2023]) obtains formal guarantees by translating natural language into solver input, but the translation step itself is the dominant error source, and recent analysis argues specification synthesis is undecidable in general [Zhou et al. 2026]. Formal guarantees conditioned on unreliable translation do not yield C5-grade traces. VerifiQuant moves the formalization burden *offline* — contracts are built and validated at build time — so runtime only selects and binds within a pre-verified space.

**Multi-agent and self-reflection systems** rely on natural-language consensus between agents; failures propagate silently across agent boundaries [Cemri et al. 2025]. We reimplement a nine-node financial multi-agent system published by J.P. Morgan researchers [Kundu et al. 2025] and find that, even with oracle support, it never triggers its own `ask_human` node and delivers SWR 14% (§5.1) — an architecture with a human-checkpoint *node* but no executed *control* that routes to it.

**Prompted refusal** is the cheapest candidate control and deserves its own test: we show in §6.1–§6.2 that neither describing the funnel to an oracle nor explicitly instructing a CoT model to refuse produces selective abstention — refusals appear, but on the wrong questions.

---

# 3. Method: VerifiQuant

## 3.1 Financial Inference Contracts

An FIC packages one formula family as three artifacts, keyed by `fic_id`:

- **Retrieval card**: selection hints, applicability and non-applicability conditions, scope boundaries, keywords — used by BM25 + an LLM selector to decide *whether this contract applies*.
- **Core card**: typed `inputs`, deterministic `execution.code` (plain Python), `invariants` and `scale_checks` (E-checks), and `semantic_hints` declaring known ambiguities (timing, annualization basis, percent/decimal, FX direction) with their resolution options.
- **Repair card**: maps each diagnostic outcome to one of eight structured actions (request missing fields, present clarification options, declare scope boundary, confirm unit conversion, …), each declaring its legal next-step transitions.

The LLM's role is *mapping into contracts* — selecting, binding, flagging — never computing. All arithmetic and all rule evaluation is deterministic contract code. E-check expressions are lint-gated at build time against the exact evaluation namespace used at runtime, so a contract that builds is a contract that runs. **[TODO-T5: 250Q 良率與修卡紀律一句]**

## 3.2 The M/N/F/E/I/C diagnostic funnel

| Layer | Gate question | Structured exit |
|---|---|---|
| **M** (intent) | Does the question map to exactly one contract family? | Refusal + disambiguation request |
| **N** (scope) | Intent clear, but is any contract applicable? | Scope-boundary refusal |
| **F** (schema) | Are all required inputs present? | Slot-filling request (`requested_fields`) |
| **E** (boundary) | Do bound inputs satisfy invariants/scale checks? | Deterministic alert |
| **I_HARD** (ambiguity, blocking) | Would an unresolved ambiguity *change the computation*? | Block + clarification, then verified transform |
| **I_SOFT** (ambiguity, warning) | Ambiguity affects interpretation but not computation path | Proceed + explicit warning |
| **C** (compute) | — | Deterministic execution + audit log |

Every case yields a `DiagnosticReport = (exit_layer, fic_id, binding, invariant_trace, action)` — replayable, layer-attributable, and machine-readable. Abstention is not a fallback; it is the designed output of five of the seven gates (control C6).

## 3.3 Oracle-in-the-loop evaluation protocol

To measure *recoverability* — how far the framework can get when intent is clear but the answer is unknown — we use a constrained oracle agent that simulates an ideal clarifying user. Per turn, the oracle reads the current DiagnosticReport and the question's ground-truth *computation code* (the intent specification), but **never the numeric answer**; it may only rewrite question/context, not inject values. Iteration is capped at K=3. The same GT-blind discipline is applied to all CoT+oracle baselines: the oracle reviews *every* turn (blind review), receives no correctness signal, and never sees the numeric ground truth — otherwise "intervene only when wrong" itself leaks the label. Across all four CoT oracle configurations, `broken_count = 0` (blind review never corrupted a correct answer), confirming the protocol is non-destructive.

## 3.4 Verifiable atomic transforms

When an I_HARD clarification changes the required computation, free-form code regeneration would reopen the door the funnel just closed. VerifiQuant permits exactly two repair forms:

- **`result_postprocess`**: a pure-arithmetic expression over the original result and bound inputs (e.g., `result × (1 + i)`), bounded by an AST node-count limit, a safe-name whitelist, and a numerical invariant check.
- **`code_patch`**: a single-site textual replacement in the contract code (e.g., `start=1 → start=0`), accepted only if (i) the pattern occurs exactly once, (ii) the shallow AST diff is within a declared blast radius, (iii) no new imports or dangerous calls appear, and (iv) **cross-verification**: the patched code's output equals the declared algebraic identity applied to the original output, on multiple sampled inputs.

*Worked example (annuity timing).* A contract computes an end-of-period annuity (result 579.98). The user clarifies payments occur at period start. The system applies the pre-declared post-process `result × … ` **and** the code patch `start=1 → start=0`, reruns, and compares: both yield 577.10, so the repair is accepted and logged with its invariant. A repair that cannot be independently confirmed by the algebraic identity is rejected — a code change is admissible only when its numerical effect is provable without trusting the LLM that proposed it (control C4).

## 3.5 Integration

```
q → BM25 retrieval → LLM selector (M/N) → LLM extractor (F)
  → deterministic E-checks → LLM critic (I_HARD / I_SOFT)
  → verified transform if triggered → deterministic C execution
  → DiagnosticReport
```

---

# 4. Experimental Protocol

## 4.1 Datasets

**Clean set.** FinanceReasoning [Tang et al. 2025] medium subset, 50 questions, stratified by difficulty quartile (seed 42, question IDs committed). **[TODO-V2: 250 questions — 180 medium + 70 hard (difficulty 4.16–7.19), superset of the canonical 50, same stratified protocol. 加 hard-tier 動機一句：測 governance 行為在難度 shift 下的穩定性。]** **[TODO-T5: FIC build yield — smoke-test pass rate, lint repairs (4 expressions / 2 cards, statement-form E-checks), all detectable without gold answers.]**

**Trap set (contract-grounded).** Trap labels that are *guessed* from question text are themselves unreliable. We instead derive each trap from a clean question plus its FIC by a deterministic operator, so the label *is* the operator: **F** removes a required input; **E** injects a boundary-violating value (negative revenue) anchored to a declared E-check predicate; **I** deletes the disambiguating phrase whose absence re-opens a declared semantic hint (hard/soft per the contract's repair-rule suffix); **N** swaps in an out-of-scope ask (exotic pricing) while keeping the original context; **M** replaces the question with an under-specified intent. 50 traps (10 per operator), zero `needs_review`, manually spot-checked. *Honest scoping:* N is defined relative to the contract library (a principled OOD check for VQ, an incidental mismatch cue for CoT), and the Tier-1 M operator retains the metric-bearing context, which limits what M results can claim (§5.3, §7).

## 4.2 Baselines

(1) CoT single-shot, Gemini 2.5 Flash and Pro; (2) CoT + GT-blind oracle, plain and funnel-guided prompts × Flash/Pro (K=3, blind review every turn); (3) J.P. Morgan MAS reimplementation (nine-node LangGraph, oracle-in-the-loop conditions); (4) VerifiQuant, Flash (canonical) and Pro, K=3. Primary comparison is Flash-to-Flash. **[TODO-V2: 250Q 只跑 single-shot / blind-oracle Flash / VQ Flash (+Pro if A5-①)；完整 grid 與 MAS 保留在 50Q，表註說明。]**

## 4.3 Metrics

Following selective prediction [El-Yaniv & Wiener 2010; Geifman & El-Yaniv 2017], every case is Correct, Silent Wrong (confident unflagged wrong number), or Safe Refusal. We report **Coverage**, **Selective Accuracy** = Correct/(Correct+SW), and **SWR** = SW/N — the key deployment-risk metric. For VerifiQuant we additionally report **SFWR** (soft-flagged wrong rate: delivered wrong *with* an I_SOFT warning) in final-flag and ever-flagged variants. On traps, where no legitimate numeric answer exists, the correct behavior is interception: we report **Trap SWR** (answered confidently with no flag) and the structured-catch rate.

---

# 5. Results

## 5.1 Clean set: accuracy is not the differentiator — failure mode is

**Table 2: Main results (50Q clean set, canonical V3 contracts).** **[TODO-V2: 換 250Q 主表；50Q 表移 supplementary 或並列]**

| System | Model | Correct | SW | Refusal | Coverage | Sel. Acc | SWR |
|---|---|---:|---:|---:|---:|---:|---:|
| CoT single-shot | Flash | 41 | 8 | 0 (+1†) | 98% | 83.7% | 16.0% |
| CoT single-shot | Pro | 41 | 8 | 0 (+1†) | 98% | 83.7% | 16.0% |
| CoT + plain oracle (blind) | Flash | 47 | 3 | 0 | 100% | 94.0% | 6.0% |
| CoT + funnel oracle (blind) | Flash | 46 | 4 | 0 | 100% | 92.0% | 8.0% |
| CoT + plain oracle (blind) | Pro | 49 | 1 | 0 | 100% | 98.0% | 2.0% |
| CoT + funnel oracle (blind) | Pro | 48 | 1 | 0 (+1†) | 98% | 98.0% | 2.0% |
| JPM MAS (O-ITL) | Flash | 43 | 7‡ | 0 | 100% | 86.0% | 14.0% |
| **VerifiQuant (K=3)** | **Flash** | **45** | **0** | **5** | **90%** | **100%** | **0%** |
| VerifiQuant (K=3) | Pro | 43 | 2 | 5 | 90% | 95.6% | 4.0% |

† Informal no-answer (the model declares `needs_more_info` and outputs no number) — the *same question* (test-1242) in all three configurations. Per §4.3, a no-answer is not a silent wrong; it is also not a structured refusal: it carries no diagnostic type, no requested fields, no route to remediation. ‡ Includes 2 pipeline errors delivered as final outputs.

Three observations. **(1) CoT accuracy can exceed VerifiQuant's — and it does not matter.** The strongest leak-free CoT configuration (plain oracle, Pro) reaches 98%, above VQ's 90%. But it retains one silent wrong and offers zero abstentions; across all six CoT configurations, SWR never falls below 2%, and the only abstentions that ever appear are informal no-answers (one question) carrying no diagnostics — prose shrugs, not routable refusals. VerifiQuant spends 10% abstention to buy SWR = 0% — a different point on the risk-coverage frontier that no CoT configuration can reach, at any accuracy (control C6). **(2) The residual CoT error is architectural, not informational.** The surviving silent wrong (test-1443, a CAGR computation) fails in 3 of 4 oracle configurations for a reason no oracle can fix: LLM arithmetic on the fractional power 2.4^(1/5) is non-deterministic, oscillating across 19.11–19.16 between turns — the same model computes 19.14 on turn 2 and drifts to 19.13 on turn 3. VerifiQuant executes this step in Python (deterministically 19.14 every time) and flags the residual percent/decimal convention with an I_SOFT warning. **(3) A multi-agent system with a human-escalation node never uses it.** The MAS baseline finished all 50 questions without once triggering `ask_human`, delivering SWR 14% — possessing a checkpoint is not the same as executing one (C1).

**Recovery and safety of iteration.** Truncating the canonical VQ run at K=1/2/3 yields 74% → 84% → 90% accuracy with SWR = 0% at *every* K: oracle turns convert abstentions to correct answers but never convert them to silent wrongs, because clarification cannot cross a deterministic gate. Oracle budget buys accuracy; the funnel owns safety — the two contributions are orthogonal.

**Reproducibility (C5).** Four independent runs of the canonical configuration: accuracy and SWR identical in all four (45/50, 0%); only the recovery path varies (4–8 oracle triggers). The deterministic gates pin the outcome set; LLM sampling variance is confined to *which* questions need clarification, not *what* is finally delivered.

## 5.2 Difficulty shift **[TODO-V2 / T2: 本節全新，等 250Q]**

> 預留：medium (180) vs hard (70) 分層表 + Fig 2。Punchline 目標：abstention 隨難度單調上升、SWR 平坦（governance 行為在 distribution shift 下穩定）。若 hard SWR > 0 → 逐題歸因表 + §7 誠實段（T3）。

## 5.3 Trap resistance

**Table 3: Trap SWR by operator (50 contract-grounded traps; VQ evaluated funnel-only at K=1** — with any oracle, the GT computation code re-injects redacted values and artificially defuses the traps; we verified this empirically: the same VQ system at K=3 "solves" 48/50 traps, which is precisely why K≥2 trap numbers are evaluation leakage, not capability.)

| Operator | CoT SWR | VQ SWR | VQ interception | Reading |
|---|---:|---:|---|---|
| E (illegal boundary value) | 90% | **20%** | `alert/E` | CoT computes straight through a −$3M expense |
| I (deleted disambiguator) | 100% | **10%** | I_SOFT warning + declared transform | CoT silently picks one reading |
| F (removed required input) | 0% | 0% | `error/F` + `requested_fields` | Parity in SWR; VQ's refusal is machine-readable |
| N (out-of-scope ask) | 0% | 0% | `refusal/N` | *False parity*: CoT refuses because the mismatch is conspicuous, not because it knows its scope |
| M (under-specified intent) | 50% | 100% | (not caught) | Tier-1 operator flaw: metric context retained; excluded from claims |
| **Overall** | **48%** | **26%** | structured catch 74% vs 0% | |

The differentiated rows are E and I — model-agnostic failure modes where the contract's deterministic checks beat prompt reasoning by 4.5–10×. On I-traps, CoT's failures split into *representational divergence* (outputs 100× off, e.g., Kelly criterion 25.0 vs 0.25, with no convention declared) and *silent convention choice* (same number as VQ, but no flag — indistinguishable downstream from a wrong number). F and N are parity in SWR but not in governance: VQ's exits carry `diagnostic_type`, `requested_fields`, and gate actions that route to HITL; CoT's refusals are prose. M is reported as a limitation of the trap generator, not evidence either way (§7).

---

# 6. Governance Analysis: why prompts are not controls

## 6.1 Injecting the taxonomy into prompts does not transfer the guarantees

The funnel-guided oracle receives the *complete* M/N/F/E/I/C taxonomy — layer definitions, priority order, worked examples — as prompt text; the plain oracle receives none of it. Under GT-blind review, funnel-guided **underperforms** plain (Flash 92% vs 94%; Pro 96% vs 98%). When the oracle cannot see correctness, instructions to "systematically check six layers" cause over-rewriting and noise injection. Diagnostic knowledge in context neither produces structured abstention (neither variant emits a single diagnosable refusal; SWR ≥ 2%) nor even reliably helps accuracy. The guarantee lives in the *executed* gates, not in the description of the gates. Prompt-described governance is a *plausible* control — it reads like diligence — with no mechanism behind it.

## 6.2 Prompted refusal is non-selective

A natural objection: CoT never refuses because we never let it. We ran a 12-cell ablation (GPT-5.2 × 2 reasoning efforts + Gemini 2.5 Flash, × 4 refusal-encouragement levels from "must answer" to "run an explicit M/F/E/I self-check before answering"). With an explicit refusal channel, GPT-5.2 does refuse (6–9 refusals at the strictest levels) — but on the *wrong questions*: at every cell, refusals of would-have-been-correct answers outnumber rescued wrongs (e.g., 2 good vs 7 over-refusals), so SWR falls only 12%→7% while accuracy collapses 88%→76%. Gemini is nearly inert to the same instructions (1–3 refusals, SWR wandering 13–19%). Prompted refusal exists, but it is not *selective* — it lacks the information that a deterministic gate has: which specific check failed. Abstention without attribution is just lost coverage.

## 6.3 Case studies (from the deployed demo system)

- **E — boundary violation.** Context contains total expenses of −$3,000,000. The raw-LLM control silently drops the sign and reports a working ratio of 88.46%, fluent and confident. VerifiQuant's E-gate fires the contract's non-negativity check and exits with a deterministic alert naming the offending field. A calculator tool cannot help the baseline here: it guarantees the parentheses, not the premise (C2).
- **N — out-of-scope with plausible context.** Asked to price a scenario "using a Heston stochastic volatility model" over a manufacturing context, the control answers 21.5 — confidently computing an unrelated throughput quantity, i.e., answering a different question than asked. VerifiQuant finds no applicable contract and exits `refusal/N` with the scope boundary named (C1).
- **I_HARD — ambiguity with verified repair.** An annuity question is silent on payment timing. VerifiQuant blocks, presents the two contract-declared readings, and on clarification applies the atomic transform of §3.4 — result adjusted 579.98 → 577.10 with the invariant and both verification paths logged in the diagnostic report. The user gets the corrected number *and* the proof that only the declared adjustment changed (C3+C4).

## 6.4 Where the guarantee ends: input binding

We disclose the sharpest boundary we know of. With Pro as the underlying model, VerifiQuant produced two silent wrongs sharing one pattern: **over-normalization above the funnel**. In one case (a cost-plus contract), the model *recomputed* a literally-given input (contract value $2.5M → $2.875M, adding profit the contract code would add again) before binding. The resulting value is positive, plausible, and passes every E-check — the funnel validates *bound inputs and outputs*, not whether a given value was silently rewritten during binding. The same case exposes an I_SOFT lifecycle flaw: the oracle resolved the *flagged* dimension, the flag was cleared, and a flagged-wrong degraded into a silent wrong. Notably, the *stronger* model fails here and the weaker one does not (Flash binds the literal value; Flash SWR 0% vs Pro 4%): capability increases exploitation of whichever step is left unconstrained. This is the core thesis in miniature — reliability tracks the constraint set, not model scale — and it yields a concrete roadmap: input-provenance checks that force every bound value to trace to a literal context span. **[TODO-T4: A5-① IMR/VQ Pro 250 若完成，加統計驗證一句]**

---

# 7. Limitations

**The funnel's guarantee is conditional on faithful input binding** (§6.4); provenance checking is designed but not yet implemented. **Contracts are built by an LLM at build time**; build-time validation (execution smoke tests, E-check linting, cross-artifact relation checks) catches the mechanical failure classes without consulting gold answers, but semantic contract errors can survive — our 50Q history required repairing 5 of 50 cards across versions, all detectable from execution signals alone. **[TODO-T5: 250Q 修卡數據]** **The M operator of the trap generator does not isolate intent ambiguity** (retained metric context lets the selector proceed); M results are excluded from claims pending a redesign. **Abstention has a price**: 10% coverage loss on the clean set, which we argue is the correct trade in high-stakes settings but is not free. **Scale**: results are on 50 clean + 50 trap questions at medium difficulty **[TODO-V2: 250Q + hard tier 後改寫；若 hard SWR>0 → 此處放逐題歸因]**; the oracle simulates an intent-explicit, answer-blind user, not real user behavior.

---

# 8. Conclusion

Financial LLM deployment is blocked less by what models get wrong than by what systems cannot govern. We showed that six governance controls demanded by regulators — boundary declaration, input validation, ambiguity disclosure, verified repair, replayable traces, and selective abstention — can be compiled into the reasoning pipeline as executable mechanisms, and that none of them transfers through prompts: describing the diagnostic funnel to an oracle degrades it, and prompted refusal misfires. A governed system will sometimes say less — VerifiQuant answers 90% of questions where a CoT stack answers 100% at up to 98% accuracy — but everything it says is attributable, replayable, and free of silent wrongs **[TODO-T3]**. In high-stakes finance, an auditable refusal is worth more than an unauditable answer.

---

# References（✅ 已查證，2026-07-12；BibTeX 完整版：`docs/references_icaif.bib`）

- Tang et al. 2025. FinanceReasoning: Benchmarking Financial Numerical Reasoning More Credible, Comprehensive and Challenging. **ACL 2025**, pp. 15721–15749. doi:10.18653/v1/2025.acl-long.766 ✅（FinNLP draft 的 "Liu et al." 為誤植）
- Kundu, Sahoo, Li, Rabowsky & Varshney. 2025. A Multi-Agent Framework for Quantitative Finance: An Application to Portfolio Management Analytics. **EMNLP 2025 Industry Track**, pp. 812–824 ✅（兩份 draft 的 "Yu et al." 為誤植）
- Cemri et al. 2025. Why Do Multi-Agent LLM Systems Fail? **NeurIPS 2025 Datasets & Benchmarks（spotlight）**; arXiv:2503.13657 ✅（draft 的 "ICLR 2026"/"NeurIPS 2026" 均誤）
- Turpin, Michael, Perez & Bowman. 2023. Language Models Don't Always Say What They Think. **NeurIPS 2023** ✅
- Lanham et al. 2023. Measuring Faithfulness in Chain-of-Thought Reasoning. arXiv:2307.13702 ✅
- Gao et al. 2023. PAL: Program-aided Language Models. **ICML 2023** ✅
- Chen, Ma, Wang & Cohen. 2023. Program of Thoughts Prompting. **TMLR** ✅
- Pan, Albalak, Wang & Wang. 2023. Logic-LM. **Findings of EMNLP 2023**, pp. 3806–3824 ✅
- Olausson et al. 2023. LINC. **EMNLP 2023**, pp. 5153–5176 ✅
- Ye, Chen, Dillig & Durrett. 2023. SatLM. **NeurIPS 2023** ✅
- Zhou et al. 2026. FormalJudge. arXiv:2602.11136（under review）
- El-Yaniv & Wiener. 2010. On the Foundations of Noise-free Selective Classification. **JMLR 11**, pp. 1605–1641 ✅
- Geifman & El-Yaniv. 2017. Selective Classification for Deep Neural Networks. **NeurIPS 2017** ✅
- Fed/OCC. 2011. Supervisory Guidance on Model Risk Management. **SR Letter 11-7**
- European Parliament & Council. 2024. **Regulation (EU) 2024/1689**（AI Act）, Art. 12–13
- FINRA. 2020. Artificial Intelligence (AI) in the Securities Industry. Report

（Mosier & Skitka 1996 已移除——本版正文未引用 automation bias；若 §6 加回該論點再補。）

---

## 附錄（草稿內部用，投稿版視 CFP 決定去留）

- Fig 1 規格：pipeline 流程圖，七個閘門各標 C1–C6 控制編號（T6）。
- Fig 2 規格：x = 難度分位（medium Q1–Q4, hard），y = correct/abstain/SW stacked 佔比（T2）。
- 頁數預算 sanity check：Abstract 0.15 / §1 0.8 / §2 0.8 / §3 1.7 / §4 0.9 / §5 1.2 / §6 1.0 / §7 0.5 / §8 0.25 / Ref ~0.9 ≈ 8.2 頁 → §3.3 或 §6.3 有 0.2 頁壓縮空間。
