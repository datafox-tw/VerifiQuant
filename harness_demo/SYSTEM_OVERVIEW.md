# A Verifiable Harness for LLM-Generated Financial Code: System Overview

*Demo in `harness_demo/`. Status as of 2026-10-29.*

## 1. Purpose

The harness lets an LLM write financial code while keeping the decision to trust that code outside the model. The LLM proposes specifications and programs. Deterministic components decide whether a program is accepted, whether the user must be asked a question, or whether the system should refuse. No number in the final answer comes from the model's own arithmetic or its self-explanation. Every number comes from an executed program that passed a set of checks written in advance.

The demo uses deliberately simple finance problems (annuity future value, mortgage payment, CAGR, NPV) so that the verification logic stays visible.

## 2. Design principles

1. **A card is an invariant set, not a formula.** Each card covers one *type* of financial question. It lists the inputs that must be known, the conventions that must be decided, and the invariants any correct program must satisfy. It never stores the formula itself. The LLM writes the formula.
2. **The LLM proposes, the harness decides.** The model extracts the specification, writes candidate programs, and writes an independent inspector program. Every acceptance decision is deterministic.
3. **No ground truth is needed.** Invariants are chosen so that they constrain the answer without knowing it. For the annuity card, a base case ("one year of deposits") plus a recurrence ("one more year") determines the answer for every horizon by induction.
4. **Self-reports are claims, not evidence.** Each program declares the conventions it assumes. The harness tests that declaration against the program's actual behaviour.
5. **Three exits only.** The harness can emit an answer, ask the user, or abstain. Each exit has a checkable trigger.

## 3. Pipeline

| # | Step | What happens | LLM involved? |
|---|---|---|---|
| 1 | Scope gate | Match the question to a card. No card, or a tie, means abstain. | Only as fallback in live mode |
| 2 | Intent specification | Freeze the question into named slots, conventions and a function signature. Each slot value must come with a verbatim quote from the question. The harness re-parses the number from the quote and rejects any value it cannot reproduce. Missing slots trigger a question to the user. | Yes (extraction only) |
| 3 | Contract | List the card's invariants: preconditions, postconditions, metamorphic relations, and convention-pinning boundary checks. | No |
| 4 | Input precheck | Run the preconditions on the bound inputs (for example, a rate must be a decimal in [0, 1)). | No |
| 5 | Generate and execute | Sample several candidate programs, run each in a sandboxed subprocess, and cluster them by output. If every candidate fails, regenerate once with the failed invariants as feedback. | Yes |
| 6 | Verify | Run every invariant on every candidate. Compare each candidate's declared convention with the convention its behaviour actually satisfies. | No |
| 7 | Bug vs. ambiguity | Discard candidates that failed verification. If the card's conventions are not fixed by the question, ask the user and show each reading's verified answer. This happens even when all surviving candidates agree. | No (regenerates if no survivor matches the user's choice) |
| 8 | Independent inspection | An inspector program sees only the frozen specification (not the question, not the candidates) and recomputes the answer by a different method, such as simulation or bisection. | Yes |
| 9 | Traceability | Link every slot, convention and fixed assumption to the code lines that implement it. Flag uncovered items. | No |
| 10 | Decision and audit | Emit, ask, or abstain. Write an audit record of inputs and their sources, conventions and who decided them, the program hash, every invariant result, and the reasons each rejected candidate was rejected. | No |

## 4. Invariant cards

| Card | Signature | Invariants (pre / post / meta / boundary) | Fixed assumptions shown to the user |
|---|---|---|---|
| Annuity future value | `solve(pmt, annual_rate, years, freq)` | 4 / 3 / 2 / 2 | Period rate = annual rate ÷ deposits per year. Convention to decide: deposit timing (begin or end). |
| Mortgage payment | `solve(principal, annual_rate, years)` | 3 / 3 / 2 / 0 | Monthly rate = annual rate ÷ 12. Number of payments = years × 12. |
| CAGR | `solve(start_value, end_value, years)` | 3 / 2 / 2 / 0 | Annual compounding. |
| NPV | `solve(initial, cash_flow, years, rate)` | 3 / 1 / 4 / 0 | Outlay at t = 0 is not discounted. Inflows at the end of years 1 to n. Invest if NPV > 0. |

Each card has one invariant set that pins the answer completely:

- **Annuity:** the one-year base case plus the "one more year" recurrence.
- **Mortgage:** paying the computed amount for 12 × years months leaves a zero balance.
- **CAGR:** start × (1 + g)^years must equal the end value.
- **NPV:** three relations together fix every term. A zero rate gives the plain sum. One extra unit invested today lowers NPV by exactly one. One extra year adds exactly the discounted cash flow.

## 5. Demo cases

Scripted mode has six cases. The candidate programs are hand-written stand-ins that reproduce common LLM errors. Execution and verification are real.

| Case | What it shows |
|---|---|
| Yearly savings, timing unstated | Six candidates. Four are bugs and get rejected: double percent conversion, division by zero at a 0% rate, an off-by-one loop, and a program that claims "end" but computes "begin". Two valid readings remain, so the harness asks. |
| Same question, "at the start of each year" | The question fixes the convention, so the harness does not ask. |
| Mortgage payment | Catches using the annual rate as a monthly rate, and using 20 periods instead of 240. No ambiguity, so no question. |
| CAGR | Catches simple averaging and continuous compounding. Replaces mental arithmetic that drifted between 19.11% and 19.16% in an earlier VerifiQuant experiment (test-1443). |
| NPV without a discount rate | The extractor invents 10% with no quote, so the value is rejected and the user is asked. At 10% the project is rejected. At 8% it is accepted. |
| Tomorrow's stock price | No card fits, so the harness abstains. |

Live mode sends a free-text question to Gemini 2.5 Flash for extraction, four candidate programs, the inspector, and a no-harness baseline. A run takes about five seconds. Observations from runs on 2026-10-29:

- **The direct answer changed on every run, and was wrong each time.** Asked about saving 5,000 a month at 2% for 10 years, Gemini answered directly with 661,170.59, then 663,026.34, then 662,997. The verified answer is 663,598.30.
- **The mortgage answer was also off.** Gemini answered 19,782.75 directly. The verified program gives 19,756.04.
- **All four candidates silently chose end-of-period deposits.** Their outputs agreed, so a disagreement-based detector would not ask. The card still flags timing as undecided.
- **After the user chose "begin", the harness regenerated.** Three of the four new programs failed the zero-rate invariant and were rejected.
- **The first version of the annuity card was too narrow.** It assumed the rate and count were per period, so a monthly question with an annual rate did not fit. The harness abstained instead of answering, and the card was then extended with a deposit-frequency slot.

## 6. Concepts and the papers behind them

Each row is one idea the harness uses, the papers that discuss it, and where the harness departs from them. Two concepts are the harness's own design and have no direct source.

| Concept | Step(s) | Papers | How the harness differs |
|---|---|---|---|
| Abstain when out of scope | 1 | Selective classification (El-Yaniv & Wiener 2010; Geifman & El-Yaniv 2017); abstention survey (Wen et al. 2025) | The trigger is a checkable condition (no card exists), not a confidence score. |
| Intent specification as an intermediate layer | 2 | Astrogator (Councilman et al. 2025); Why Do Multi-Agent LLM Systems Fail? (Cemri et al. 2025) | Astrogator's query language targets Ansible. Here the language is financial slots and conventions, and every component reads the same frozen spec. |
| Values must be grounded in a source quote | 2 | ALCE (Gao et al. 2023); FinanceReasoning (Tang et al. 2025) | ALCE judges citation support with a model. Here the quote must appear verbatim and the number is re-parsed deterministically. |
| Ask instead of guessing missing information | 2, 4 | FinanceReasoning; abstention survey | FinanceReasoning assumes complete information and leaves clarification as future work. The harness names the missing field and asks. |
| Pre- and postconditions as contracts | 3, 4, 6 | Design by Contract (Meyer 1992); nl2postcond (Endres et al. 2024) | Contracts are written once per question type by a domain expert, not generated by the LLM, so they cannot share the program's mistake. |
| Metamorphic relations instead of an oracle | 3, 6 | Metamorphic testing (Chen et al. 1998; Segura et al. 2016); the oracle problem (Barr et al. 2015) | Financial identities (linearity, monotonicity, the zero-rate case) are written as metamorphic relations. |
| Base case plus recurrence determines the answer *(own design)* | 3, 6 | Closest: nl2postcond's notion of postcondition completeness | The harness aims for a mathematically unique answer, not a high rate of distinguishing bugs. |
| Programs compute, the model does not | 5 | PAL (Gao et al. 2023); PoT (Chen et al. 2023); FinanceReasoning | PAL and PoT fix arithmetic but not misreadings. The harness adds verification and clarification afterwards. |
| Sample many programs and cluster by behaviour | 5 | AlphaCode (Li et al. 2022); CodeT (Chen et al. 2023) | They cluster to pick the likeliest program. Here clustering only detects disagreement, and agreement is never treated as correctness. |
| Regenerate with execution feedback | 5, 7 | Self-Debugging (Chen et al. 2024) | The feedback is the list of failed card invariants, and the harness retries only once. |
| Remove bugs before treating disagreement as ambiguity *(own design)* | 6, 7 | Contrast: ClarifyGPT (Mu et al. 2024); CodeT | ClarifyGPT asks whenever outputs differ, so a single buggy program can trigger a question. The harness first removes programs that fail verification. Only the remaining readings reach the user, and the card's convention relation (for example, begin = end × (1 + r)) is shown as a consistency check. |
| Ask about undecided conventions even when candidates agree | 7 | ClarifyGPT; Astrogator | Disagreement-based detection misses shared guesses. The card lists conventions in advance, similar to Astrogator surfacing what the program filled in on its own. |
| Rules run outside the model, never in the prompt | 6 | VERAFI (Akinfaderin & Subramanian 2025); Logic-LM, LINC, SatLM (2023); FormalJudge (Zhou et al. 2026) | VERAFI places rules in the prompt as a soft constraint. Solver approaches translate each question at run time. Here invariants are numeric checks written once per type and executed as a hard gate. |
| Test self-reports against behaviour | 6, 9 | Turpin et al. 2023; Lanham et al. 2023; Learning to Seek Evidence (Huang et al. 2025) | These test model explanations through interventions. Here the "explanation" is a one-line convention declaration, tested deterministically by boundary invariants. |
| Independent recomputation from the spec only | 8 | AIRepr (Zeng et al. 2025); Selection-Inference (Creswell et al. 2023); Knight & Leveson 1986 | Knight and Leveson show that independently built versions still fail together. Agreement is therefore an extra check after verification, not proof. |
| Specification-to-code trace links | 9 | Verifiable Literate Programming (Yuan et al. 2026) | Links are the program's own tags. The numeric claims behind them were already checked in step 6. |
| Only three exits: emit, ask, abstain | 10 | Learning to Seek Evidence; conformal abstention (Abbasi Yadkori et al. 2024) | Exits are triggered by rules, not learned policies or statistical thresholds. A conformal layer could be added on top. |
| The audit log records only what was executed | 10 | Turpin et al.; EU AI Act Art. 12; SR 26-2 | The log contains no chain of thought. SR 26-2 places generative and agentic AI outside its scope, which leaves the gap this harness addresses. |

## 7. One paper at a time

The demo's third mode removes every harness step except one paper's mechanism and runs it on the same scripted candidates (`lenses.py`). Programs and checks really execute. The bug labels are used only to score the outcomes, and the harness itself never reads them.

| Used alone | Savings, timing unstated | Savings, start of year | Mortgage | CAGR | NPV, no rate | Stock price |
|---|---|---|---|---|---|---|
| PAL / PoT | silent wrong | silent wrong | silent wrong | silent wrong | silent wrong | answers anyway |
| CodeT / AlphaCode | assumed | silent wrong | correct | correct | assumed | answers anyway |
| ClarifyGPT | noisy question | noisy question | noisy question | noisy question | noisy question | answers anyway |
| AIRepr | assumed | silent wrong | correct | correct | assumed | answers anyway |
| Full harness | asks | correct | correct | correct | asks | abstains |

*Assumed* means the number is right but a convention or input was assumed silently. A *noisy question* offers buggy answers as options, or asks when nothing was ambiguous. The PAL row shows the worst program that could be sampled.

**VERINA (Ye et al., arXiv 2505.23135) is used to grade the cards.** A postcondition is sound if every output it accepts is correct, and complete if it accepts every correct output. Positive programs are those labelled correct. Negative programs are the labelled bugs plus mutants of correct programs (× 1.001, + 1 yuan or + 1 bp, rounded). Each runs on three input sets.

| Card | Post completeness | Post soundness | Pre completeness | Pre soundness |
|---|---|---|---|---|
| Annuity future value | 2 / 2 | 10 / 10 | 3 / 3 | 4 / 4 |
| Mortgage payment | 2 / 2 | 8 / 8 | 3 / 3 | 3 / 3 |
| CAGR | 2 / 2 | 8 / 8 | 3 / 3 | 3 / 3 |
| NPV | 1 / 1 | 5 / 6 | 3 / 3 | 3 / 3 |

- **The NPV card lets rounding through.** An answer rounded to the whole yuan passes every invariant on all three inputs. Whether that counts as a bug is a tolerance decision for the card author.
- **The question's own input is not enough.** The rounded annuity programs pass on the question's numbers and are rejected only on other inputs.
- **Some invariants caught nothing**, such as "a higher rate does not lower the balance" and "a longer loan does not raise the payment".
- **Design changes VERINA suggests:** grade every card before use; also give the invariants to the generator (VERINA finds a ground-truth spec improves code generation) while still executing them outside the model; keep tests rather than proofs, since VERINA's proof success rate is below 5%.

## 8. Limitations

- **Only four question types.** Card selection uses keywords, so a rephrased question may be refused.
- **Invariants are hand-written.** Weak invariants could let a wrong program through. Step 7 abstains when verified programs still disagree, but it cannot catch a case where every program is wrong in the same way and no invariant covers that error.
- **The sandbox is for demonstration only.** It is a subprocess with restricted builtins, `math` as the only import, and CPU and time limits. It is not production isolation.
- **Scripted candidates are hand-written.** They reproduce error types; they are not recorded model outputs. The page labels them as such.

## 9. References

"Checked" records how the bibliographic details were confirmed. *Web* means the arXiv, ACL Anthology or publisher page was checked on 2026-10-29. *Bib* means the details were already verified in `docs/references_icaif.bib`. *Read* means the primary text was read earlier. *Unchecked* means the details are from memory. Claims attributed to papers come from abstracts or the author's reading notes, so check figures against the source before quoting them.

| Paper | Venue | Checked | Link |
|---|---|---|---|
| Tang et al., FinanceReasoning | ACL 2025 | Web | [arXiv 2506.05828](https://arxiv.org/abs/2506.05828) |
| Councilman et al., Towards Formal Verification of LLM-Generated Code from Natural Language Prompts (Astrogator) | arXiv 2025 | Web | [arXiv 2507.13290](https://arxiv.org/abs/2507.13290) |
| Cemri et al., Why Do Multi-Agent LLM Systems Fail? | NeurIPS 2025 D&B | Bib | [arXiv 2503.13657](https://arxiv.org/abs/2503.13657) |
| Gao, Yen, Yu & Chen, Enabling LLMs to Generate Text with Citations (ALCE) | EMNLP 2023 | Web | [arXiv 2305.14627](https://arxiv.org/abs/2305.14627) |
| Wen et al., Know Your Limits: A Survey of Abstention in LLMs | TACL 2025 | Web | [ACL Anthology](https://aclanthology.org/2025.tacl-1.26) |
| El-Yaniv & Wiener, On the Foundations of Noise-free Selective Classification | JMLR 2010 | Bib | [JMLR](https://jmlr.org/papers/v11/el-yaniv10a.html) |
| Geifman & El-Yaniv, Selective Classification for Deep Neural Networks | NeurIPS 2017 | Bib | [arXiv 1705.08500](https://arxiv.org/abs/1705.08500) |
| Abbasi Yadkori et al., Mitigating LLM Hallucinations via Conformal Abstention | arXiv 2024 | Read | [arXiv 2405.01563](https://arxiv.org/abs/2405.01563) |
| Meyer, Applying "Design by Contract" | IEEE Computer 1992 | Unchecked | [DOI](https://doi.org/10.1109/2.161279) |
| Endres et al., Can LLMs Transform NL Intent into Formal Method Postconditions? (nl2postcond) | FSE 2024 | Web | [arXiv 2310.01831](https://arxiv.org/abs/2310.01831) |
| Chen, Cheung & Yiu, Metamorphic Testing | HKUST tech report 1998 | Web | [arXiv 2002.12543](https://arxiv.org/abs/2002.12543) |
| Segura et al., A Survey on Metamorphic Testing | IEEE TSE 2016 | Web | [White Rose](https://eprints.whiterose.ac.uk/110335/) |
| Barr et al., The Oracle Problem in Software Testing: A Survey | IEEE TSE 2015 | Read | n/a |
| Gao et al., PAL: Program-aided Language Models | ICML 2023 | Bib | [arXiv 2211.10435](https://arxiv.org/abs/2211.10435) |
| Chen et al., Program of Thoughts Prompting | TMLR 2023 | Bib | [arXiv 2211.12588](https://arxiv.org/abs/2211.12588) |
| Li et al., Competition-level Code Generation with AlphaCode | Science 2022 | Web | [arXiv 2203.07814](https://arxiv.org/abs/2203.07814) |
| Chen et al., CodeT: Code Generation with Generated Tests | ICLR 2023 | Web | [arXiv 2207.10397](https://arxiv.org/abs/2207.10397) |
| Mu et al., ClarifyGPT | FSE 2024 | Web | [arXiv 2310.10996](https://arxiv.org/abs/2310.10996) |
| Chen, Lin, Schärli & Zhou, Teaching LLMs to Self-Debug | ICLR 2024 | Web | [arXiv 2304.05128](https://arxiv.org/abs/2304.05128) |
| Akinfaderin & Subramanian, VERAFI | arXiv 2025 | Web | [arXiv 2512.14744](https://arxiv.org/abs/2512.14744) |
| Pan et al., Logic-LM | Findings of EMNLP 2023 | Bib | [arXiv 2305.12295](https://arxiv.org/abs/2305.12295) |
| Olausson et al., LINC | EMNLP 2023 | Bib | [arXiv 2310.15164](https://arxiv.org/abs/2310.15164) |
| Ye et al., SatLM | NeurIPS 2023 | Bib | [arXiv 2305.09656](https://arxiv.org/abs/2305.09656) |
| Zhou et al., FormalJudge | arXiv 2026 | Web | [arXiv 2602.11136](https://arxiv.org/abs/2602.11136) |
| Turpin et al., Language Models Don't Always Say What They Think | NeurIPS 2023 | Bib | [arXiv 2305.04388](https://arxiv.org/abs/2305.04388) |
| Lanham et al., Measuring Faithfulness in Chain-of-Thought Reasoning | arXiv 2023 | Bib | [arXiv 2307.13702](https://arxiv.org/abs/2307.13702) |
| Huang, Lin, Zhong & Liu, Learning to Seek Evidence | arXiv 2025 | Web | [arXiv 2511.01425](https://arxiv.org/abs/2511.01425) |
| Zeng et al., AIRepr | Findings of EMNLP 2025 | Web | [ACL Anthology](https://aclanthology.org/2025.findings-emnlp.539) |
| Creswell, Shanahan & Higgins, Selection-Inference | ICLR 2023 | Web | [arXiv 2205.09712](https://arxiv.org/abs/2205.09712) |
| Knight & Leveson, An Experimental Evaluation of the Assumption of Independence in Multiversion Programming | IEEE TSE 1986 | Web | [DOI](https://doi.org/10.1109/TSE.1986.6312924) |
| Yuan et al., Guiding Human Validation of LLM-Generated Code via Verifiable Literate Programming | arXiv 2026 | Web | [arXiv 2607.02333](https://arxiv.org/abs/2607.02333) |
| Ye, Yan, He, Kasriel, Yang & Song, VERINA: Benchmarking Verifiable Code Generation | arXiv 2025; ICLR 2026 poster site | Web | [arXiv 2505.23135](https://arxiv.org/abs/2505.23135) |
| Regulation (EU) 2024/1689 (AI Act), Article 12 | 2024 | Read | [EUR-Lex](https://eur-lex.europa.eu/eli/reg/2024/1689/oj) |
| Federal Reserve, OCC & FDIC, SR 26-2 | 2026 | Read | [PDF](https://www.federalreserve.gov/supervisionreg/srletters/SR2602.pdf) |
