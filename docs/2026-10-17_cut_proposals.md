# ACM 版壓到九面：刪減提案（完整前後文）

> **已套用（10/17）**：A1–A10（A6a）、B1、B4、B6、B7（正文不提 depth 細節）、B8、C1，加上 §4.5 短版（B9 試過後還原）。實測還能再加 5 行正文。ACM 版實測 9 頁。下方的「改前」是套用前的版本。

由 `scripts/build_cut_proposals.py` 產生。目前 ACM 版 10 頁。
每一項都實際套用到一份副本、用 pdfLaTeX 編譯，數全文少了幾行文字，所以「實測省」是編譯出來的結果，不是估計。
行數可以相加；能不能進九頁要看分頁，所以下面另外逐步累加、每一步都實際編譯看頁數。

**改前、改後都是整段**，數字巨集換成實際數值、引用換成實際編號；差異欄用 ~~刪除線~~ 表示刪、**粗體** 表示新增。

## 〇、已經改進論文的出處更正（改前＝更正前的存檔，改後＝目前版本）

### §3.1 建庫（三段都寫出來）

**改前（整段）**

> An FIC packages one formula family into four essentials keyed by `fic_id`: (i) *applicability and non-applicability* conditions plus retrieval metadata that decide whether the contract applies; (ii) *typed inputs*; (iii) *deterministic compute* code in plain Python; and (iv) *numerical invariants and declared semantic ambiguities* (period timing, percent-versus-decimal, unit basis), each ambiguity carrying resolution options and a hard/soft level. For concreteness, the `Loan` card types `principal`, `annual_interest_rate`, and `loan_term_years`, carries a boundary check that fires when the rate reaches 1.0, and declares the decimal-versus-percentage reading of that same rate as a hard ambiguity. A rate entered as 6 is therefore intercepted deterministically, while the same confusion below 1.0 surfaces as a clarification. The LLM's role is *mapping into contracts*, that is, selecting, binding, and flagging. It never performs the financial computation. Contracts are governed at build time: compute code passes execution smoke tests, invariant expressions are lint-gated against the exact runtime evaluation namespace, and a pre-declared output-type policy is enforced by audit. The 250-card library used here is generated fully automatically (three staged LLM calls per family plus deterministic normalization) and never reads any per-question worked solution or gold answer. No contract was hand-tuned for these experiments. Build-time corrections are automatic lint and type-policy rules, and the prompts that generate the library are released with the code.https://github.com/datafox-tw/icaif-verifiquant At runtime the funnel checks inputs against each contract but does not independently verify that the contract itself is correct (§6).

**改後（整段）**

> An FIC packages one formula family into four essentials keyed by `fic_id`: (i) *applicability and non-applicability* conditions plus retrieval metadata that decide whether the contract applies; (ii) *typed inputs*; (iii) *deterministic compute* code in plain Python; and (iv) *numerical invariants and declared semantic ambiguities* (period timing, percent-versus-decimal, unit basis), each ambiguity carrying resolution options and a hard/soft level. For concreteness, the `Loan` card types `principal`, `annual_interest_rate`, and `loan_term_years`, carries a boundary check that fires when the rate reaches 1.0, and declares the decimal-versus-percentage reading of that same rate as a hard ambiguity. A rate entered as 6 is therefore intercepted deterministically, while the same confusion below 1.0 surfaces as a clarification. The LLM's role is *mapping into contracts*, that is, selecting, binding, and flagging. It never performs the financial computation. Contracts are governed at build time: compute code passes execution smoke tests, invariant expressions are lint-gated against the exact runtime evaluation namespace, and a pre-declared output-type policy is enforced by audit. The 250-card library used here is generated fully automatically by three LLM calls per family plus deterministic normalization. The first writes the core card from the family's reference function and the question it was sampled from, with that question's context and worked solution. The second and third derive retrieval metadata and repair rules from the core card. No call reads a gold answer. No contract was hand-tuned for these experiments. Build-time corrections are automatic lint and type-policy rules, and the prompts that generate the library are released with the code.https://github.com/datafox-tw/icaif-verifiquant At runtime the funnel checks inputs against each contract but does not independently verify that the contract itself is correct (§6).

**差異**

> An FIC packages one formula family into four essentials keyed by `fic_id`: (i) *applicability and non-applicability* conditions plus retrieval metadata that decide whether the contract applies; (ii) *typed inputs*; (iii) *deterministic compute* code in plain Python; and (iv) *numerical invariants and declared semantic ambiguities* (period timing, percent-versus-decimal, unit basis), each ambiguity carrying resolution options and a hard/soft level. For concreteness, the `Loan` card types `principal`, `annual_interest_rate`, and `loan_term_years`, carries a boundary check that fires when the rate reaches 1.0, and declares the decimal-versus-percentage reading of that same rate as a hard ambiguity. A rate entered as 6 is therefore intercepted deterministically, while the same confusion below 1.0 surfaces as a clarification. The LLM's role is *mapping into contracts*, that is, selecting, binding, and flagging. It never performs the financial computation. Contracts are governed at build time: compute code passes execution smoke tests, invariant expressions are lint-gated against the exact runtime evaluation namespace, and a pre-declared output-type policy is enforced by audit. The 250-card library used here is generated fully automatically ~~(three staged~~ **by three** LLM calls per family plus deterministic ~~normalization)~~ **normalization. The first writes the core card from the family's reference function** and ~~never~~ **the question it was sampled from, with that question's context and worked solution. The second and third derive retrieval metadata and repair rules from the core card. No call** reads ~~any per-question worked solution or~~ **a** gold answer. No contract was hand-tuned for these experiments. Build-time corrections are automatic lint and type-policy rules, and the prompts that generate the library are released with the code.https://github.com/datafox-tw/icaif-verifiquant At runtime the funnel checks inputs against each contract but does not independently verify that the contract itself is correct (§6).


### §4.1 建庫與卡片重複題目數值（簡短版）

**改前（整段）**

> We evaluate on 250 FinanceReasoning [tang2025financereasoning] questions (180 medium, 70 hard), a difficulty-stratified random sample (seed 42) of the function-linked subset in which each question carries a formula-level reference implementation. The contract library is built from those reference functions. The pipeline never reads the per-question worked solution or numeric gold answer, and it deduplicates formula kernels by AST hashing. By construction a card therefore contains the formula its matching question needs: the evaluation is *in-library*, and it measures routing, binding, and gating, not unseen-formula derivation. This is a structural constraint of the only benchmark that ships per-question formula provenance. The out-of-library analysis of §5 uses a second, fixed split of the same 250 questions: the library is rebuilt from 50 formula families, which leaves 199 questions whose family is absent (OOD) and 51 whose family is present (ID). The comparison there is matched in the same way as the clean set. Both systems run the same backbone on the same 199 questions with the same question and context, and both retrieve top-4 from the *same* 50-family library, so the baseline is not given a larger library or a different prompt. Because the split is predetermined by formula family rather than difficulty, both systems receive the same unchanged questions and differ only in whether the funnel runs.

**改後（整段）**

> We evaluate on 250 FinanceReasoning [tang2025financereasoning] questions (180 medium, 70 hard), a difficulty-stratified random sample (seed 42) of the function-linked subset in which each question carries a formula-level reference implementation. The contract library is built from those reference functions and their questions (§3), and the pipeline deduplicates formula kernels by AST hashing. By construction a card therefore contains the formula its matching question needs, and about a third of the cards also repeat a value from that question (Appendix K of the extended version). The evaluation is *in-library*, and it measures routing, binding, and gating, not unseen-formula derivation. This is a structural constraint of the only benchmark that ships per-question formula provenance. The out-of-library analysis of §5 uses a second, fixed split of the same 250 questions: the library is rebuilt from 50 formula families, which leaves 199 questions whose family is absent (OOD) and 51 whose family is present (ID). The comparison there is matched in the same way as the clean set. Both systems run the same backbone on the same 199 questions with the same question and context, and both retrieve top-4 from the *same* 50-family library, so the baseline is not given a larger library or a different prompt. Because the split is predetermined by formula family rather than difficulty, both systems receive the same unchanged questions and differ only in whether the funnel runs.

**差異**

> We evaluate on 250 FinanceReasoning [tang2025financereasoning] questions (180 medium, 70 hard), a difficulty-stratified random sample (seed 42) of the function-linked subset in which each question carries a formula-level reference implementation. The contract library is built from those reference ~~functions. The~~ **functions and their questions (§3), and the** pipeline ~~never reads the per-question worked solution or numeric gold answer, and it~~ deduplicates formula kernels by AST hashing. By construction a card therefore contains the formula its matching question ~~needs:~~ **needs, and about a third of** the **cards also repeat a value from that question (Appendix K of the extended version). The** evaluation is *in-library*, and it measures routing, binding, and gating, not unseen-formula derivation. This is a structural constraint of the only benchmark that ships per-question formula provenance. The out-of-library analysis of §5 uses a second, fixed split of the same 250 questions: the library is rebuilt from 50 formula families, which leaves 199 questions whose family is absent (OOD) and 51 whose family is present (ID). The comparison there is matched in the same way as the clean set. Both systems run the same backbone on the same 199 questions with the same question and context, and both retrieve top-4 from the *same* 50-family library, so the baseline is not given a larger library or a different prompt. Because the split is predetermined by formula family rather than difficulty, both systems receive the same unchanged questions and differ only in whether the funnel runs.


### §4.5 人工集的 18 題

**改前（整段）**

> The F/E/I/N traps are contract-derived, so a skeptic can ask whether their labels merely re-encode our own checks (§4). We therefore add a set whose correct handling is decided by people who see neither our contracts nor our systems. An independent GPT-5.2 author rewrites FinanceReasoning topics into 80 naturalistic queries without access to our contracts, taxonomy, gold answers, or system outputs. It differs from the Gemini solver that every evaluated system uses, so a query's messiness does not track what the solver finds easy. The sources lie outside the 250 of §4, because rewriting those would guarantee a contract for every item, and the set is therefore *library-agnostic*: it mixes in-library and out-of-library questions in a proportion we do not set. Items are stratified over four latent user intents so that enough of them call for something other than a number, and the stratum is hidden from annotators. Five finance-trained annotators, blind to the source, the stratum, and every system output, each assign one action by a top-down rule: OUT-OF-SCOPE if the ask is not to compute one number, REQUEST-INFO if a required value is missing, CLARIFY if an interpretation choice would change the number, and ANSWER otherwise. These actions map onto funnel outcomes (compute, F, I, and N/M), so a confident number on a non-ANSWER item is a false acceptance defined by human judgment rather than by our taxonomy. Majority labels are frozen before any system is scored. The full construction, label definitions, and scoring rules are in Appendix E of the extended version.

**改後（整段）**

> The F/E/I/N traps are contract-derived, so a skeptic can ask whether their labels merely re-encode our own checks (§4). We therefore add a set whose correct handling is decided by people who see neither our contracts nor our systems. GPT-5.2 authored the 80 queries from FinanceReasoning topics without access to our contracts, taxonomy, gold answers, or system outputs. An audit found that the 18 meant to lack an input simply left the context empty, so we revised those 18 by hand with a Codex agent, seeded by the GPT-5.2 versions, to give a near-complete context with exactly one indispensable number missing. That revision ran in our repository and is not blind to our contracts, whereas the annotators who label the revised set see neither contracts nor systems. Neither author is the Gemini solver that every evaluated system uses, so a query's messiness does not track what the solver finds easy. The sources lie outside the 250 of §4, because rewriting those would guarantee a contract for every item, and the set is therefore *library-agnostic*: it mixes in-library and out-of-library questions in a proportion we do not set. Items are stratified over four latent user intents so that enough of them call for something other than a number, and the stratum is hidden from annotators. Five finance-trained annotators, blind to the source, the stratum, and every system output, each assign one action by a top-down rule: OUT-OF-SCOPE if the ask is not to compute one number, REQUEST-INFO if a required value is missing, CLARIFY if an interpretation choice would change the number, and ANSWER otherwise. These actions map onto funnel outcomes (compute, F, I, and N/M), so a confident number on a non-ANSWER item is a false acceptance defined by human judgment rather than by our taxonomy. Majority labels are frozen before any system is scored. The full construction, label definitions, and scoring rules are in Appendix E of the extended version.

**差異**

> The F/E/I/N traps are contract-derived, so a skeptic can ask whether their labels merely re-encode our own checks (§4). We therefore add a set whose correct handling is decided by people who see neither our contracts nor our systems. ~~An independent~~ GPT-5.2 ~~author rewrites~~ **authored the 80 queries from** FinanceReasoning topics ~~into 80 naturalistic queries~~ without access to our contracts, taxonomy, gold answers, or system outputs. ~~It differs from~~ **An audit found that the 18 meant to lack an input simply left the context empty, so we revised those 18 by hand with a Codex agent, seeded by the GPT-5.2 versions, to give a near-complete context with exactly one indispensable number missing. That revision ran in our repository and is not blind to our contracts, whereas the annotators who label the revised set see neither contracts nor systems. Neither author is** the Gemini solver that every evaluated system uses, so a query's messiness does not track what the solver finds easy. The sources lie outside the 250 of §4, because rewriting those would guarantee a contract for every item, and the set is therefore *library-agnostic*: it mixes in-library and out-of-library questions in a proportion we do not set. Items are stratified over four latent user intents so that enough of them call for something other than a number, and the stratum is hidden from annotators. Five finance-trained annotators, blind to the source, the stratum, and every system output, each assign one action by a top-down rule: OUT-OF-SCOPE if the ask is not to compute one number, REQUEST-INFO if a required value is missing, CLARIFY if an interpretation choice would change the number, and ANSWER otherwise. These actions map onto funnel outcomes (compute, F, I, and N/M), so a confident number on a non-ANSWER item is a false acceptance defined by human judgment rather than by our taxonomy. Majority labels are frozen before any system is scored. The full construction, label definitions, and scoring rules are in Appendix E of the extended version.


### §7 限制

**改前（整段）**

> **Limitations and next steps.** This work is a first step. VerifiQuant reduces and localizes silent wrongs without eliminating them, and their attribution uses gold answers unavailable in deployment. The evidence rests on one formula-provenance benchmark and one backbone family. The trap advantage also rests on contract-derived inputs, because the library-agnostic human set reproduces the ordering but does not separate the two governed systems. The staged controls use one prompt wording per stage, and a retrieval-matched selective baseline and a calibrated conformal procedure remain untested. Utility, finally, is bounded by a library built from a random sample of 250 questions, so on library-agnostic queries the system mostly offers a scope signal. We see three directions. First, a larger library, evaluated together with contract selection, since the sweep shows near-neighbor silent wrongs rising as the library densifies. Second, letting the model synthesize the computation within the conditions and conventions a contract declares, so that elementary calculations need no dedicated contract, while every constant and input keeps a source and the output can still be traced back to each decision. Third, better retrieval and contract naming for queries whose contract exists but is not found. A study with real users, beyond the scripted user of §6, also remains open.

**改後（整段）**

> **Limitations and next steps.** This work is a first step. VerifiQuant reduces and localizes silent wrongs without eliminating them, and their attribution uses gold answers unavailable in deployment. The evidence rests on one formula-provenance benchmark and one backbone family. The trap advantage also rests on contract-derived inputs and on contracts written with their source questions in view, because the library-agnostic human set reproduces the ordering but does not separate the two governed systems. The staged controls use one prompt wording per stage, and a retrieval-matched selective baseline and a calibrated conformal procedure remain untested. Utility, finally, is bounded by a library built from a random sample of 250 questions, so on library-agnostic queries the system mostly offers a scope signal. We see three directions. First, a larger library, evaluated together with contract selection, since the sweep shows near-neighbor silent wrongs rising as the library densifies. Second, letting the model synthesize the computation within the conditions and conventions a contract declares, so that elementary calculations need no dedicated contract, while every constant and input keeps a source and the output can still be traced back to each decision. Third, better retrieval and contract naming for queries whose contract exists but is not found. A study with real users, beyond the scripted user of §6, also remains open.

**差異**

> **Limitations and next steps.** This work is a first step. VerifiQuant reduces and localizes silent wrongs without eliminating them, and their attribution uses gold answers unavailable in deployment. The evidence rests on one formula-provenance benchmark and one backbone family. The trap advantage also rests on contract-derived ~~inputs,~~ **inputs and on contracts written with their source questions in view,** because the library-agnostic human set reproduces the ordering but does not separate the two governed systems. The staged controls use one prompt wording per stage, and a retrieval-matched selective baseline and a calibrated conformal procedure remain untested. Utility, finally, is bounded by a library built from a random sample of 250 questions, so on library-agnostic queries the system mostly offers a scope signal. We see three directions. First, a larger library, evaluated together with contract selection, since the sweep shows near-neighbor silent wrongs rising as the library densifies. Second, letting the model synthesize the computation within the conditions and conventions a contract declares, so that elementary calculations need no dedicated contract, while every constant and input keeps a source and the output can still be traced back to each decision. Third, better retrieval and contract naming for queries whose contract exists but is not found. A study with real users, beyond the scripted user of §6, also remains open.


---

## 總覽

| 編號 | 位置 | 實測省（行） | 資訊去向 |
|---|---|---|---|
| A1 | §3.3 修復段（尾行 wild.） | 1 | 只刪連接詞與同義詞，兩句界線說法與 879 分母都保留。 |
| A2 | §2 範圍訊號定義（尾行 to CoT.） | 1 | 意思不變。 |
| A3 | Table 6 圖說（尾行 result.） | 1 | oracle 上限的標註保留（校準規則）；「not an autonomous result」在 Table 2 圖說與… |
| A4 | §5.2 Pro 那句（尾行 version).） | 1 | 數字全保留，意思不變。 |
| A5 | §3.2 N/M 段（尾行 is resolved.） | 1 | 「fitted」已涵蓋 trained 與 tuned。 |
| A6a | §5 開頭，修尾行（尾行 human set.） | 1 | 意思不變。 |
| A6b | §5 開頭，整段刪（A6a 的替代方案） | 5 | 小節標題 RQ1–RQ4 已說明順序與庫內、邊界、人工集的區分。 |
| A7 | 貢獻三（尾行 vouches for.） | 1 | 意思不變。 |
| A8 | §5.3 gate 消融（尾行 not accuracy.） | 1 | backbone 在 §5.1 已講（Every other system comparison holds the b… |
| A9 | §4.5 第二段（尾行 extended version.） | 1 | 意思不變。 |
| A10 | 結論（尾行 binding constraint.） | 1 | 四類檢查在摘要與貢獻一已列。 |
| B1 | 導論「兩個系統服務不同目標」整段 | 7 | 1.5 倍延遲來自 §5.3（12.2 秒對 8.2 秒），少 9 點覆蓋率在 Table 2 與 §5.3（92.9%… |
| B2 | §2 自洽性細節 | 3 | 掃描端點（覆蓋率、SWR 的起訖）與「不含檢索、不是 conformal」移到附錄 H。 |
| B3 | §3.2 Audit trail | 5 | AI Act 第 12 條那句與完整的報告欄位清單移到附錄 G（法規背景，對應 C5）。可重放的界線說法完整保留。提醒：… |
| B4 | §4.1 OOD split 的配對說明 | 5 | 同段三句講同一件事，合併成一句，無資訊損失。 |
| B5 | §4.2 容忍度 | 4 | 「比 benchmark 排行榜慣例寬鬆」那句移到附錄 H。 |
| B6 | §5.1 RQ1 刪兩句 | 5 | 配對檢定的資料來源移到附錄 D；最後一句和前一句「Standing on the clean set therefore… |
| B7 | §5.3 延遲但書 | 4 | 「含供應商端變異與重試」「未計 token」併入附錄 D（附錄 D 原本已有 depth 3 循序、depth 4 並行… |
| B8 | §5.4 最後一段 | 4 | 「量兩次才能講出邊界的價格」在貢獻三（we put a price on the boundary）與 §6 已講；保留… |
| B9 | §6 Where the claim holds 最後一句 | 2 | in-library by construction 在 §4.1，庫外取捨在 RQ3。 |
| C1 | 附錄引用縮短成「App. X」，第一次出現時加註腳說明延伸版在哪裡 | 6 | 14 處「Appendix X of the extended version」改成「App. X」，延伸版位置在第一次… |
| C2 | 正文完全不提附錄（C1 的替代方案） | 11 | 14 處附錄引用全部刪除。附錄內容仍在 arXiv 版，但 ACM 讀者不會知道。 |

**逐步累加（每一步都實際編譯）**：先套建議組合 A1, A2, A3, A4, A5, A6a, A7, A8, A9, A10, B1, B4, B6，再依「對內容影響由小到大」加入其他項目。

| 累加到 | 累計省（行） | ACM 頁數 |
|---|---|---|
| 建議組合 | 27 | 10 |
| + B9 | 29 | 10 |
| + B8 | 33 | 10 |
| + B7 | 39 | 9 |
| + B5 | 43 | 9 |
| + C1 | 45 | 9 |
| + B2 | 49 | 9 |
| + B3 | 54 | 9 |

---

### A1　§3.3 修復段（尾行 wild.）　（實測省 1 行）

**改前（整段）**

> When a hard ambiguity changes the required computation, free-form regeneration would reopen the door the funnel just closed. Repairs hence use AST-bounded transforms. Before acceptance, the numerical effect must match a contract-declared algebraic relation at the point of use on multiple sampled inputs. This is bounded numerical validation, not formal equivalence. It catches single-sided errors, but a wrong invariant paired with a compensating patch would pass. We assume a non-adversarial patch generator, not an attacker fitting patches to the sampled bindings. We audit the 179 synthesizable transform specs in the 250-card library with six generated patch-error classes (Appendix F of the extended version). Of 906 mutants, 27 are arithmetically equivalent and excluded. The verifier rejects 97.7% of the remaining 879 mutants. The other 20 are false accepts (2.3%), and both structural attacks are rejected at 100%. This controlled distribution probes verifier mechanisms. It does not estimate the catch rate for LLM-generated patches in the wild.

**改後（整段）**

> When a hard ambiguity changes the required computation, free-form regeneration would reopen the door the funnel just closed. Repairs use AST-bounded transforms. Before acceptance, the numerical effect must match a contract-declared algebraic relation at the point of use on multiple sampled inputs. This is bounded numerical validation, not formal equivalence. It catches single-sided errors, but a wrong invariant paired with a compensating patch would pass. We assume a non-adversarial patch generator, not an attacker fitting patches to the sampled bindings. We audit the 179 synthesizable transform specs in the 250-card library with six generated patch-error classes (Appendix F of the extended version). Of 906 mutants, 27 are arithmetically equivalent and excluded. The verifier rejects 97.7% of the remaining 879 mutants. The other 20 are false accepts (2.3%), and both structural attacks are rejected at 100%. This controlled set probes verifier mechanisms. It does not estimate the catch rate for LLM-generated patches in the wild.

**差異**

> When a hard ambiguity changes the required computation, free-form regeneration would reopen the door the funnel just closed. Repairs ~~hence~~ use AST-bounded transforms. Before acceptance, the numerical effect must match a contract-declared algebraic relation at the point of use on multiple sampled inputs. This is bounded numerical validation, not formal equivalence. It catches single-sided errors, but a wrong invariant paired with a compensating patch would pass. We assume a non-adversarial patch generator, not an attacker fitting patches to the sampled bindings. We audit the 179 synthesizable transform specs in the 250-card library with six generated patch-error classes (Appendix F of the extended version). Of 906 mutants, 27 are arithmetically equivalent and excluded. The verifier rejects 97.7% of the remaining 879 mutants. The other 20 are false accepts (2.3%), and both structural attacks are rejected at 100%. This controlled ~~distribution~~ **set** probes verifier mechanisms. It does not estimate the catch rate for LLM-generated patches in the wild.

**資訊去向**：只刪連接詞與同義詞，兩句界線說法與 879 分母都保留。


---

### A2　§2 範圍訊號定義（尾行 to CoT.）　（實測省 1 行）

**改前（整段）**

> In the trap set T, no legitimate *unqualified single* numeric answer exists: a required field is missing, a declared boundary is violated, a material convention remains unresolved, or the ask is outside the evaluated task scope. An un-intercepted number is therefore a false numeric acceptance by construction, not an ordinary accuracy error, and the rate mathrmFNAR(S)=tfrac1|T|sum_tin T e_S(t)(1-w_S(t)) needs no gold answer. Here w_S(t)=1 only when t is a soft-ambiguity trap that S labels I_SOFT and answers with a warning. For such a trap the ambiguity changes how the number is read, not how it is computed, so a warned number is the correct handling. We also report two attribution scores. *Exact diagnostic* mathrmDiag(S) is the fraction of traps whose emitted label hatd_S(t) equals the generating class d(t) (F, E, I_HARD/I_SOFT, or N). *Action* mathrmAct(S) is stricter than merely withholding a number: it also requires the matching grounded response, namely requesting the removed F field, naming the targeted E rule or field, asking a non-numeric I_HARD clarification or issuing an I_SOFT warning, or refusing an N request as out of scope. Finally, we write s_S(i)=1 when S stops on item i with a structured statement that its library holds no applicable contract. We call this a *scope signal*. It is defined only for systems that carry a library, so it does not apply to CoT.

**改後（整段）**

> In the trap set T, no legitimate *unqualified single* numeric answer exists: a required field is missing, a declared boundary is violated, a material convention remains unresolved, or the ask is outside the evaluated task scope. An un-intercepted number is therefore a false numeric acceptance by construction, not an ordinary accuracy error, and the rate mathrmFNAR(S)=tfrac1|T|sum_tin T e_S(t)(1-w_S(t)) needs no gold answer. Here w_S(t)=1 only when t is a soft-ambiguity trap that S labels I_SOFT and answers with a warning. For such a trap the ambiguity changes how the number is read, not how it is computed, so a warned number is the correct handling. We also report two attribution scores. *Exact diagnostic* mathrmDiag(S) is the fraction of traps whose emitted label hatd_S(t) equals the generating class d(t) (F, E, I_HARD/I_SOFT, or N). *Action* mathrmAct(S) is stricter than merely withholding a number: it also requires the matching grounded response, namely requesting the removed F field, naming the targeted E rule or field, asking a non-numeric I_HARD clarification or issuing an I_SOFT warning, or refusing an N request as out of scope. Finally, we write s_S(i)=1 when S stops on item i with a structured statement that its library holds no applicable contract. We call this a *scope signal*. It is defined only for systems that carry a library, so not for CoT.

**差異**

> In the trap set T, no legitimate *unqualified single* numeric answer exists: a required field is missing, a declared boundary is violated, a material convention remains unresolved, or the ask is outside the evaluated task scope. An un-intercepted number is therefore a false numeric acceptance by construction, not an ordinary accuracy error, and the rate mathrmFNAR(S)=tfrac1|T|sum_tin T e_S(t)(1-w_S(t)) needs no gold answer. Here w_S(t)=1 only when t is a soft-ambiguity trap that S labels I_SOFT and answers with a warning. For such a trap the ambiguity changes how the number is read, not how it is computed, so a warned number is the correct handling. We also report two attribution scores. *Exact diagnostic* mathrmDiag(S) is the fraction of traps whose emitted label hatd_S(t) equals the generating class d(t) (F, E, I_HARD/I_SOFT, or N). *Action* mathrmAct(S) is stricter than merely withholding a number: it also requires the matching grounded response, namely requesting the removed F field, naming the targeted E rule or field, asking a non-numeric I_HARD clarification or issuing an I_SOFT warning, or refusing an N request as out of scope. Finally, we write s_S(i)=1 when S stops on item i with a structured statement that its library holds no applicable contract. We call this a *scope signal*. It is defined only for systems that carry a library, so ~~it does~~ not ~~apply to~~ **for** CoT.

**資訊去向**：意思不變。


---

### A3　Table 6 圖說（尾行 result.）　（實測省 1 行）

**改前（整段）**

> Clean questions under clarification budgets. Every K>1 row is an intent-oracle upper bound, not an autonomous result.

**改後（整段）**

> Clean questions under clarification budgets. Every K>1 row is an intent-oracle upper bound.

**差異**

> Clean questions under clarification budgets. Every K>1 row is an intent-oracle upper ~~bound, not an autonomous result.~~ **bound.**

**資訊去向**：oracle 上限的標註保留（校準規則）；「not an autonomous result」在 Table 2 圖說與 §4.4 都有。


---

### A4　§5.2 Pro 那句（尾行 version).）　（實測省 1 行）

**改前（整段）**

> **Staging versus execution.** The staged controls of §4.2 separate the two coupled differences. Staged lowers the trap FNAR of R-PAL+ from 27.5% to 15.8%, but only by abstaining more widely, and its clean coverage falls to 67.1%. Staged+Spec lowers trap FNAR further to 8.8%, level with VerifiQuant (difference -1.7 points, 95% CI [-8.3, 4.8]). Clean coverage, however, reaches only 74.8% against 84.0% (exact McNemar p<.01 in each of three runs), and most of that gap is model-written programs that fail to run, which executing the contract's own code avoids. On malformed inputs the advantage therefore comes from deciding in stages over declared contract content. Execution adds no measurable reduction in FNAR there. What it buys is clean coverage and a record that replays as stored. Repeating the trap comparison on Gemini 2.5 Pro narrows the gap to 11.7 points (R-PAL+ 22.1%, VerifiQuant 10.4%, 95% CI [1.7,21.5]), and nearly all of what remains lies in I (Appendix C of the extended version).

**改後（整段）**

> **Staging versus execution.** The staged controls of §4.2 separate the two coupled differences. Staged lowers the trap FNAR of R-PAL+ from 27.5% to 15.8%, but only by abstaining more widely, and its clean coverage falls to 67.1%. Staged+Spec lowers trap FNAR further to 8.8%, level with VerifiQuant (difference -1.7 points, 95% CI [-8.3, 4.8]). Clean coverage, however, reaches only 74.8% against 84.0% (exact McNemar p<.01 in each of three runs), and most of that gap is model-written programs that fail to run, which executing the contract's own code avoids. On malformed inputs the advantage therefore comes from deciding in stages over declared contract content. Execution adds no measurable reduction in FNAR there. What it buys is clean coverage and a record that replays as stored. On Gemini 2.5 Pro the gap narrows to 11.7 points (R-PAL+ 22.1%, VerifiQuant 10.4%, 95% CI [1.7,21.5]), nearly all of it in I (Appendix C of the extended version).

**差異**

> **Staging versus execution.** The staged controls of §4.2 separate the two coupled differences. Staged lowers the trap FNAR of R-PAL+ from 27.5% to 15.8%, but only by abstaining more widely, and its clean coverage falls to 67.1%. Staged+Spec lowers trap FNAR further to 8.8%, level with VerifiQuant (difference -1.7 points, 95% CI [-8.3, 4.8]). Clean coverage, however, reaches only 74.8% against 84.0% (exact McNemar p<.01 in each of three runs), and most of that gap is model-written programs that fail to run, which executing the contract's own code avoids. On malformed inputs the advantage therefore comes from deciding in stages over declared contract content. Execution adds no measurable reduction in FNAR there. What it buys is clean coverage and a record that replays as stored. ~~Repeating the trap comparison on~~ **On** Gemini 2.5 Pro ~~narrows~~ the gap **narrows** to 11.7 points (R-PAL+ 22.1%, VerifiQuant 10.4%, 95% CI [1.7,21.5]), ~~and~~ nearly all of ~~what remains lies~~ **it** in I (Appendix C of the extended version).

**資訊去向**：數字全保留，意思不變。


---

### A5　§3.2 N/M 段（尾行 is resolved.）　（實測省 1 行）

**改前（整段）**

> N and M decide, in two stages, whether exactly one applicable contract exists. The first stage is deterministic. Candidates come from a purely lexical retriever: SQLite FTS5 BM25 over each card's title, summary, keywords, and embedding text, merged with a keyword-overlap ranking and filtered by the contract's declared selection hints. Nothing in the retriever is trained, tuned, or fitted to the evaluation sets, and the same retriever and top-4 depth serve all three compared systems. Two deterministic conditions end at N: no candidate is retrieved, or the top retrieval score falls below a fixed threshold. Otherwise an LLM *selector* reads the top-4 candidates and makes a three-way decision. It selects one contract, or it stops at M with a disambiguation request when the user names only a broad goal that several contracts could serve, or it stops at N when no candidate supports the requested quantity. The N gate therefore involves an LLM step. In our runs every N refusal came from the selector, all 47 on the traps over three runs and all 52 on the human-labeled set. Depth is not the binding constraint here. The gold contract of a clean question lies in the top 4 for 94.8% of the 250, and widening to 16 adds only 4.0 points, so we fix k=4 (Appendix D of the extended version). Dense and hybrid retrievers remain untested. F and E come next. The LLM binds values from the text to the contract's typed inputs, and the two gates that follow are deterministic Python. F checks that every required input is present and otherwise requests the missing fields. E runs the contract's boundary and scale checks and otherwise names the violated check. Both gates inspect the values the LLM bound, so a wrong value that still lies within bounds passes, which is the binding frontier of §6. I asks whether a convention the contract declares in advance, such as percent versus decimal or the start versus the end of a period, is left open by the question. The LLM decides only which declared conventions remain unresolved. A convention that changes the computation (I_HARD) stops the query and presents the declared options. One that changes only the interpretation (I_SOFT) lets computation proceed with a warning. The blocking level is declared by the contract, not chosen at runtime. C runs the contract's deterministic code, so no number is produced by the model. Thus, abstention is a designed outcome rather than a fallback. Five funnel conditions end in a structured exit that states where the query stopped, why, and what is needed to continue, and I_SOFT is the only case that continues with a warning. Across the whole funnel the LLM acts in exactly three places: selecting a contract, binding inputs, and judging whether a declared ambiguity is resolved.

**改後（整段）**

> N and M decide, in two stages, whether exactly one applicable contract exists. The first stage is deterministic. Candidates come from a purely lexical retriever: SQLite FTS5 BM25 over each card's title, summary, keywords, and embedding text, merged with a keyword-overlap ranking and filtered by the contract's declared selection hints. Nothing in the retriever is fitted to the evaluation sets, and the same retriever and top-4 depth serve all three compared systems. Two deterministic conditions end at N: no candidate is retrieved, or the top retrieval score falls below a fixed threshold. Otherwise an LLM *selector* reads the top-4 candidates and makes a three-way decision. It selects one contract, or it stops at M with a disambiguation request when the user names only a broad goal that several contracts could serve, or it stops at N when no candidate supports the requested quantity. The N gate therefore involves an LLM step. In our runs every N refusal came from the selector, all 47 on the traps over three runs and all 52 on the human-labeled set. Depth is not the binding constraint here. The gold contract of a clean question lies in the top 4 for 94.8% of the 250, and widening to 16 adds only 4.0 points, so we fix k=4 (Appendix D of the extended version). Dense and hybrid retrievers remain untested. F and E come next. The LLM binds values from the text to the contract's typed inputs, and the two gates that follow are deterministic Python. F checks that every required input is present and otherwise requests the missing fields. E runs the contract's boundary and scale checks and otherwise names the violated check. Both gates inspect the values the LLM bound, so a wrong value that still lies within bounds passes, which is the binding frontier of §6. I asks whether a convention the contract declares in advance, such as percent versus decimal or the start versus the end of a period, is left open by the question. The LLM decides only which declared conventions remain unresolved. A convention that changes the computation (I_HARD) stops the query and presents the declared options. One that changes only the interpretation (I_SOFT) lets computation proceed with a warning. The blocking level is declared by the contract, not chosen at runtime. C runs the contract's deterministic code, so no number is produced by the model. Thus, abstention is a designed outcome rather than a fallback. Five funnel conditions end in a structured exit that states where the query stopped, why, and what is needed to continue, and I_SOFT is the only case that continues with a warning. Across the whole funnel the LLM acts in exactly three places: selecting a contract, binding inputs, and judging whether a declared ambiguity is resolved.

**差異**

> N and M decide, in two stages, whether exactly one applicable contract exists. The first stage is deterministic. Candidates come from a purely lexical retriever: SQLite FTS5 BM25 over each card's title, summary, keywords, and embedding text, merged with a keyword-overlap ranking and filtered by the contract's declared selection hints. Nothing in the retriever is ~~trained, tuned, or~~ fitted to the evaluation sets, and the same retriever and top-4 depth serve all three compared systems. Two deterministic conditions end at N: no candidate is retrieved, or the top retrieval score falls below a fixed threshold. Otherwise an LLM *selector* reads the top-4 candidates and makes a three-way decision. It selects one contract, or it stops at M with a disambiguation request when the user names only a broad goal that several contracts could serve, or it stops at N when no candidate supports the requested quantity. The N gate therefore involves an LLM step. In our runs every N refusal came from the selector, all 47 on the traps over three runs and all 52 on the human-labeled set. Depth is not the binding constraint here. The gold contract of a clean question lies in the top 4 for 94.8% of the 250, and widening to 16 adds only 4.0 points, so we fix k=4 (Appendix D of the extended version). Dense and hybrid retrievers remain untested. F and E come next. The LLM binds values from the text to the contract's typed inputs, and the two gates that follow are deterministic Python. F checks that every required input is present and otherwise requests the missing fields. E runs the contract's boundary and scale checks and otherwise names the violated check. Both gates inspect the values the LLM bound, so a wrong value that still lies within bounds passes, which is the binding frontier of §6. I asks whether a convention the contract declares in advance, such as percent versus decimal or the start versus the end of a period, is left open by the question. The LLM decides only which declared conventions remain unresolved. A convention that changes the computation (I_HARD) stops the query and presents the declared options. One that changes only the interpretation (I_SOFT) lets computation proceed with a warning. The blocking level is declared by the contract, not chosen at runtime. C runs the contract's deterministic code, so no number is produced by the model. Thus, abstention is a designed outcome rather than a fallback. Five funnel conditions end in a structured exit that states where the query stopped, why, and what is needed to continue, and I_SOFT is the only case that continues with a warning. Across the whole funnel the LLM acts in exactly three places: selecting a contract, binding inputs, and judging whether a declared ambiguity is resolved.

**資訊去向**：「fitted」已涵蓋 trained 與 tuned。


---

### A6a　§5 開頭，修尾行（尾行 human set.）　（實測省 1 行）

**改前（整段）**

> The results are read by library status. RQ1 and RQ2 stay in-library, RQ3 moves to the boundary, and RQ4 uses the library-agnostic human set.

**改後（整段）**

> RQ1 and RQ2 stay in-library, RQ3 moves to the library boundary, and RQ4 uses the library-agnostic human set.

**差異**

> ~~The results are read by library status.~~ RQ1 and RQ2 stay in-library, RQ3 moves to the **library** boundary, and RQ4 uses the library-agnostic human set.

**資訊去向**：意思不變。


---

### A6b　§5 開頭，整段刪（A6a 的替代方案）　（實測省 5 行）

**改前（整段）**

> The results are read by library status. RQ1 and RQ2 stay in-library, RQ3 moves to the boundary, and RQ4 uses the library-agnostic human set.

**改後（整段）**

> （整段刪除）

**差異**

> ~~The results are read by library status. RQ1 and RQ2 stay in-library, RQ3 moves to the boundary, and RQ4 uses the library-agnostic human set.~~

**資訊去向**：小節標題 RQ1–RQ4 已說明順序與庫內、邊界、人工集的區分。


---

### A7　貢獻三（尾行 vouches for.）　（實測省 1 行）

**改前（整段）**

> **Contributions.** *First*, VerifiQuant executes scope, schema, boundary, and ambiguity checks as contract-bound code, and a matched comparison shows that its advantage on malformed inputs does not come from the taxonomy itself. Two staged controls then locate the gain. It comes from deciding in stages over what each contract declares, while executing that code is what preserves clean coverage. On clean, well-posed questions the baselines match or beat autonomous VerifiQuant, so clean accuracy is not what we contribute. *Second*, Financial Inference Contracts and an executable diagnostic funnel turn refusal, clarification, and computation into structured outcomes that can be traced back to the check that produced them. *Third*, we put a price on the boundary of this guarantee with library-coverage sweeps, an out-of-library split, and 80 queries labeled by five finance-trained annotators blind to our systems. Their labels reproduce the trap ordering, but the set is drawn without reference to our library, so it mainly measures a *scope signal*: the system's statement that the library it was given holds no applicable contract, made instead of a guess. Declining buys no accuracy, only a boundary between answers a contract vouches for and answers nothing vouches for.

**改後（整段）**

> **Contributions.** *First*, VerifiQuant executes scope, schema, boundary, and ambiguity checks as contract-bound code, and a matched comparison shows that its advantage on malformed inputs does not come from the taxonomy itself. Two staged controls then locate the gain. It comes from deciding in stages over what each contract declares, while executing that code is what preserves clean coverage. On clean, well-posed questions the baselines match or beat autonomous VerifiQuant, so clean accuracy is not what we contribute. *Second*, Financial Inference Contracts and an executable diagnostic funnel turn refusal, clarification, and computation into structured outcomes that can be traced back to the check that produced them. *Third*, we put a price on the boundary of this guarantee with library-coverage sweeps, an out-of-library split, and 80 queries labeled by five finance-trained annotators blind to our systems. Their labels reproduce the trap ordering, but the set is drawn without reference to our library, so it mainly measures a *scope signal*: the system's statement that its library holds no applicable contract, made instead of a guess. Declining buys no accuracy, only a boundary between answers a contract vouches for and answers nothing vouches for.

**差異**

> **Contributions.** *First*, VerifiQuant executes scope, schema, boundary, and ambiguity checks as contract-bound code, and a matched comparison shows that its advantage on malformed inputs does not come from the taxonomy itself. Two staged controls then locate the gain. It comes from deciding in stages over what each contract declares, while executing that code is what preserves clean coverage. On clean, well-posed questions the baselines match or beat autonomous VerifiQuant, so clean accuracy is not what we contribute. *Second*, Financial Inference Contracts and an executable diagnostic funnel turn refusal, clarification, and computation into structured outcomes that can be traced back to the check that produced them. *Third*, we put a price on the boundary of this guarantee with library-coverage sweeps, an out-of-library split, and 80 queries labeled by five finance-trained annotators blind to our systems. Their labels reproduce the trap ordering, but the set is drawn without reference to our library, so it mainly measures a *scope signal*: the system's statement that ~~the~~ **its** library ~~it was given~~ holds no applicable contract, made instead of a guess. Declining buys no accuracy, only a boundary between answers a contract vouches for and answers nothing vouches for.

**資訊去向**：意思不變。


---

### A8　§5.3 gate 消融（尾行 not accuracy.）　（實測省 1 行）

**改前（整段）**

> A gate ablation adds controls cumulatively on the same backbone (Table 4). It is a dedicated autonomous K=1 run over all 250 clean questions with Gemini 2.5 Flash and the full contract library, using the same output-scale-normalized accuracy and SWR definitions as the main results. This gate-disable run is distinct from the canonical K=1 history truncation in Table 6: its full-funnel rung has 4.4% SWR (11/250), compared with the canonical 2.8% (7/250), so the claim concerns the within-run trend rather than equality of the absolute endpoints. Silent-wrong rate falls or holds at every rung, from CoT's 18.0% to 4.4% at the full funnel, while accuracy trades off as the I and M/N gates convert ambiguous cases into abstention. Execution buys most of the initial drop, and the later gates move answerable-but-unqualified cases into abstention. The pattern is monotone in the metric that matters: *gates buy safety with coverage, not accuracy*.

**改後（整段）**

> A gate ablation adds controls cumulatively on the same backbone (Table 4). It is a dedicated autonomous K=1 run over all 250 clean questions with the full contract library and the same scoring as the main results. This gate-disable run is distinct from the canonical K=1 history truncation in Table 6: its full-funnel rung has 4.4% SWR (11/250), compared with the canonical 2.8% (7/250), so the claim concerns the within-run trend rather than equality of the absolute endpoints. Silent-wrong rate falls or holds at every rung, from CoT's 18.0% to 4.4% at the full funnel, while accuracy trades off as the I and M/N gates convert ambiguous cases into abstention. Execution buys most of the initial drop, and the later gates move answerable-but-unqualified cases into abstention. The pattern is monotone in the metric that matters: *gates buy safety with coverage, not accuracy*.

**差異**

> A gate ablation adds controls cumulatively on the same backbone (Table 4). It is a dedicated autonomous K=1 run over all 250 clean questions with ~~Gemini 2.5 Flash and~~ the full contract ~~library, using~~ **library and** the same ~~output-scale-normalized accuracy and SWR definitions~~ **scoring** as the main results. This gate-disable run is distinct from the canonical K=1 history truncation in Table 6: its full-funnel rung has 4.4% SWR (11/250), compared with the canonical 2.8% (7/250), so the claim concerns the within-run trend rather than equality of the absolute endpoints. Silent-wrong rate falls or holds at every rung, from CoT's 18.0% to 4.4% at the full funnel, while accuracy trades off as the I and M/N gates convert ambiguous cases into abstention. Execution buys most of the initial drop, and the later gates move answerable-but-unqualified cases into abstention. The pattern is monotone in the metric that matters: *gates buy safety with coverage, not accuracy*.

**資訊去向**：backbone 在 §5.1 已講（Every other system comparison holds the backbone fixed at Flash），計分定義在 §4.2。


---

### A9　§4.5 第二段（尾行 extended version.）　（實測省 1 行）

**改前（整段）**

> The F/E/I/N traps are contract-derived, so a skeptic can ask whether their labels merely re-encode our own checks (§4). We therefore add a set whose correct handling is decided by people who see neither our contracts nor our systems. GPT-5.2 authored the 80 queries from FinanceReasoning topics without access to our contracts, taxonomy, gold answers, or system outputs. An audit found that the 18 meant to lack an input simply left the context empty, so we revised those 18 by hand with a Codex agent, seeded by the GPT-5.2 versions, to give a near-complete context with exactly one indispensable number missing. That revision ran in our repository and is not blind to our contracts, whereas the annotators who label the revised set see neither contracts nor systems. Neither author is the Gemini solver that every evaluated system uses, so a query's messiness does not track what the solver finds easy. The sources lie outside the 250 of §4, because rewriting those would guarantee a contract for every item, and the set is therefore *library-agnostic*: it mixes in-library and out-of-library questions in a proportion we do not set. Items are stratified over four latent user intents so that enough of them call for something other than a number, and the stratum is hidden from annotators. Five finance-trained annotators, blind to the source, the stratum, and every system output, each assign one action by a top-down rule: OUT-OF-SCOPE if the ask is not to compute one number, REQUEST-INFO if a required value is missing, CLARIFY if an interpretation choice would change the number, and ANSWER otherwise. These actions map onto funnel outcomes (compute, F, I, and N/M), so a confident number on a non-ANSWER item is a false acceptance defined by human judgment rather than by our taxonomy. Majority labels are frozen before any system is scored. The full construction, label definitions, and scoring rules are in Appendix E of the extended version.

**改後（整段）**

> The F/E/I/N traps are contract-derived, so a skeptic can ask whether their labels merely re-encode our own checks (§4). We therefore add a set whose correct handling is decided by people who see neither our contracts nor our systems. GPT-5.2 authored the 80 queries from FinanceReasoning topics without access to our contracts, taxonomy, gold answers, or system outputs. An audit found that the 18 meant to lack an input simply left the context empty, so we revised those 18 by hand with a Codex agent, seeded by the GPT-5.2 versions, to give a near-complete context with exactly one indispensable number missing. That revision ran in our repository and is not blind to our contracts, whereas the annotators who label the revised set see neither contracts nor systems. Neither author is the Gemini solver that every evaluated system uses, so a query's messiness does not track what the solver finds easy. The sources lie outside the 250 of §4, because rewriting those would guarantee a contract for every item, and the set is therefore *library-agnostic*: it mixes in-library and out-of-library questions in a proportion we do not set. Items are stratified over four latent user intents so that enough of them call for something other than a number, and the stratum is hidden from annotators. Five finance-trained annotators, blind to the source, the stratum, and every system output, each assign one action by a top-down rule: OUT-OF-SCOPE if the ask is not to compute one number, REQUEST-INFO if a required value is missing, CLARIFY if an interpretation choice would change the number, and ANSWER otherwise. These actions map onto funnel outcomes (compute, F, I, and N/M), so a confident number on a non-ANSWER item is a false acceptance defined by human judgment rather than by our taxonomy. Majority labels are frozen before any system is scored. Full details are in Appendix E of the extended version.

**差異**

> The F/E/I/N traps are contract-derived, so a skeptic can ask whether their labels merely re-encode our own checks (§4). We therefore add a set whose correct handling is decided by people who see neither our contracts nor our systems. GPT-5.2 authored the 80 queries from FinanceReasoning topics without access to our contracts, taxonomy, gold answers, or system outputs. An audit found that the 18 meant to lack an input simply left the context empty, so we revised those 18 by hand with a Codex agent, seeded by the GPT-5.2 versions, to give a near-complete context with exactly one indispensable number missing. That revision ran in our repository and is not blind to our contracts, whereas the annotators who label the revised set see neither contracts nor systems. Neither author is the Gemini solver that every evaluated system uses, so a query's messiness does not track what the solver finds easy. The sources lie outside the 250 of §4, because rewriting those would guarantee a contract for every item, and the set is therefore *library-agnostic*: it mixes in-library and out-of-library questions in a proportion we do not set. Items are stratified over four latent user intents so that enough of them call for something other than a number, and the stratum is hidden from annotators. Five finance-trained annotators, blind to the source, the stratum, and every system output, each assign one action by a top-down rule: OUT-OF-SCOPE if the ask is not to compute one number, REQUEST-INFO if a required value is missing, CLARIFY if an interpretation choice would change the number, and ANSWER otherwise. These actions map onto funnel outcomes (compute, F, I, and N/M), so a confident number on a non-ANSWER item is a false acceptance defined by human judgment rather than by our taxonomy. Majority labels are frozen before any system is scored. ~~The full construction, label definitions, and scoring rules~~ **Full details** are in Appendix E of the extended version.

**資訊去向**：意思不變。


---

### A10　結論（尾行 binding constraint.）　（實測省 1 行）

**改前（整段）**

> For formula-grounded financial workflows, model accuracy alone is insufficient: deployment also requires governing when a numeric answer is justified. Within a declared contract library, VerifiQuant compiles scope, schema, boundary, and ambiguity checks into an executable funnel that makes refusal, clarification, and repair attributable outcomes. Its value emerges not on well-posed in-library questions, where matched execution already suffices, but on inputs that should not yield a number. At the edge of the library it offers a scope signal rather than accuracy: out of library it answers 7.0% of questions correctly where matched execution answers 95.0%, and it does not lower the silent-wrong rate. Whether that exchange is worth making is a property of the deployment rather than of the method. It pays wherever an unflagged wrong number costs more than a declined one, as in valuation, regulatory reporting, and client-facing calculation, and it does not pay where coverage is the binding constraint.

**改後（整段）**

> For formula-grounded financial workflows, model accuracy alone is insufficient: deployment also requires governing when a numeric answer is justified. Within a declared contract library, VerifiQuant compiles its checks into an executable funnel that makes refusal, clarification, and repair attributable outcomes. Its value emerges not on well-posed in-library questions, where matched execution already suffices, but on inputs that should not yield a number. At the edge of the library it offers a scope signal rather than accuracy: out of library it answers 7.0% of questions correctly where matched execution answers 95.0%, and it does not lower the silent-wrong rate. Whether that exchange is worth making is a property of the deployment rather than of the method. It pays wherever an unflagged wrong number costs more than a declined one, as in valuation, regulatory reporting, and client-facing calculation, and it does not pay where coverage is the binding constraint.

**差異**

> For formula-grounded financial workflows, model accuracy alone is insufficient: deployment also requires governing when a numeric answer is justified. Within a declared contract library, VerifiQuant compiles ~~scope, schema, boundary, and ambiguity~~ **its** checks into an executable funnel that makes refusal, clarification, and repair attributable outcomes. Its value emerges not on well-posed in-library questions, where matched execution already suffices, but on inputs that should not yield a number. At the edge of the library it offers a scope signal rather than accuracy: out of library it answers 7.0% of questions correctly where matched execution answers 95.0%, and it does not lower the silent-wrong rate. Whether that exchange is worth making is a property of the deployment rather than of the method. It pays wherever an unflagged wrong number costs more than a declined one, as in valuation, regulatory reporting, and client-facing calculation, and it does not pay where coverage is the binding constraint.

**資訊去向**：四類檢查在摘要與貢獻一已列。


---

### B1　導論「兩個系統服務不同目標」整段　（實測省 7 行）

**改前（整段）**

> The two systems therefore serve different engineering objectives. Where throughput is the goal, prompting the taxonomy is cheaper and retains more coverage. Where the goal is a defensible non-answer, one whose reason names the check that fired and replays from the stored binding, executing the checks is what buys it, at roughly 1.5× the latency of the taxonomy-prompted baseline and nine fewer points of clean coverage (§5).

**改後（整段）**

> （整段刪除）

**差異**

> ~~The two systems therefore serve different engineering objectives. Where throughput is the goal, prompting the taxonomy is cheaper and retains more coverage. Where the goal is a defensible non-answer, one whose reason names the check that fired and replays from the stored binding, executing the checks is what buys it, at roughly 1.5× the latency of the taxonomy-prompted baseline and nine fewer points of clean coverage (§5).~~

**資訊去向**：1.5 倍延遲來自 §5.3（12.2 秒對 8.2 秒），少 9 點覆蓋率在 Table 2 與 §5.3（92.9% 對 84.0%），「依部署目標取捨」在結論（Whether that exchange is worth making is a property of the deployment）。刪掉後也解決這段以 therefore 接在法規句後、語意接不上的問題。


---

### B2　§2 自洽性細節　（實測省 3 行）

**改前（整段）**

> **Deciding whether to answer.** Answer-or-abstain methods [elyaniv2010foundations,geifman2017selective], and more recently conformal and calibration-based procedures [quach2024conformal,kadavath2022language], threshold a confidence score and can carry statistical error bounds, but they abstain on a probability rather than on a named cause. A score-based rule also buys little here. Sweeping a self-consistency threshold (k=5, vote share as the score) over the same 250 clean questions of §4 moves coverage only from 88.8% to 84.0% and SWR from 8.0% to 6.4%, because 210 of the 250 are already unanimous and the score has little resolution left to spend. At the threshold matching VerifiQuant's 84.0% coverage it holds 92.4% selective accuracy against 96.7%. That reference point carries no retrieval and is not a conformal procedure. VerifiQuant instead abstains on *contract-declared checks* (scope, schema, boundary, ambiguity), so each exit names which check failed. Model-risk, auditability, and oversight requirements [fed2026sr262,pra2023ss123,euaiact2024,finra2020ai] motivate these controls but define targets, not a compliance claim, and multi-agent financial systems [kundu2025multiagent] add natural-language consensus in which failures propagate silently across agents [cemri2025mas].

**改後（整段）**

> **Deciding whether to answer.** Answer-or-abstain methods [elyaniv2010foundations,geifman2017selective], and more recently conformal and calibration-based procedures [quach2024conformal,kadavath2022language], threshold a confidence score and can carry statistical error bounds, but they abstain on a probability rather than on a named cause. A score-based rule also buys little here. On the same 250 clean questions, a five-sample self-consistency threshold set to VerifiQuant's 84.0% coverage holds 92.4% selective accuracy against 96.7%, because 210 of the 250 are already unanimous (Appendix H of the extended version). VerifiQuant instead abstains on *contract-declared checks* (scope, schema, boundary, ambiguity), so each exit names which check failed. Model-risk, auditability, and oversight requirements [fed2026sr262,pra2023ss123,euaiact2024,finra2020ai] motivate these controls but define targets, not a compliance claim, and multi-agent financial systems [kundu2025multiagent] add natural-language consensus in which failures propagate silently across agents [cemri2025mas].

**差異**

> **Deciding whether to answer.** Answer-or-abstain methods [elyaniv2010foundations,geifman2017selective], and more recently conformal and calibration-based procedures [quach2024conformal,kadavath2022language], threshold a confidence score and can carry statistical error bounds, but they abstain on a probability rather than on a named cause. A score-based rule also buys little here. ~~Sweeping a self-consistency threshold (k=5, vote share as the score) over~~ **On** the same 250 clean ~~questions of §4 moves~~ **questions, a five-sample self-consistency threshold set to VerifiQuant's 84.0%** coverage ~~only from 88.8% to 84.0% and SWR from 8.0% to 6.4%,~~ **holds 92.4% selective accuracy against 96.7%,** because 210 of the 250 are already unanimous ~~and~~ **(Appendix H of** the ~~score has little resolution left to spend. At the threshold matching VerifiQuant's 84.0% coverage it holds 92.4% selective accuracy against 96.7%. That reference point carries no retrieval and is not a conformal procedure.~~ **extended version).** VerifiQuant instead abstains on *contract-declared checks* (scope, schema, boundary, ambiguity), so each exit names which check failed. Model-risk, auditability, and oversight requirements [fed2026sr262,pra2023ss123,euaiact2024,finra2020ai] motivate these controls but define targets, not a compliance claim, and multi-agent financial systems [kundu2025multiagent] add natural-language consensus in which failures propagate silently across agents [cemri2025mas].

**附錄新增**

> **Self-consistency reference point.** Sweeping a self-consistency threshold (five samples, vote share as the score) over the 250 clean questions moves coverage only from 88.8% to 84.0% and SWR from 8.0% to 6.4%, because 210 of the 250 are already unanimous and the score has little resolution left to spend. That reference point carries no retrieval and is not a conformal procedure.

**資訊去向**：掃描端點（覆蓋率、SWR 的起訖）與「不含檢索、不是 conformal」移到附錄 H。


---

### B3　§3.2 Audit trail　（實測省 5 行）

**改前（整段）**

> **Audit trail.** Of the design targets in 1, a traceable record is the one most directly written into regulation: the EU AI Act requires high-risk systems to log events automatically over their lifetime for traceability [euaiact2024]. VerifiQuant is auditable at two levels. At the system level, each contract is a reviewable, versioned document that states the formula, input types, checks, and declared conventions, and it passes lint and smoke gates before entering the library. At the query level, every output carries a *DiagnosticReport*. When a number is emitted, the report records the selected contract and its version, the retrieved candidates and their scores, the selector's decision, the bound values, the checks passed, and the ambiguities handled. When the query stops, it records the gate, the rule and evidence behind the stop, and what the user should supply. Deterministic execution and invariant evaluation replay exactly from the stored binding. LLM-mediated selection and binding are logged and can be re-executed as recorded, but a fresh rerun from raw text is *not* guaranteed to take the identical path. A model's own chain of thought cannot stand in for such a record, because it need not reflect the computation that produced the answer [turpin2023language].

**改後（整段）**

> **Audit trail.** VerifiQuant is auditable at two levels. At the system level, each contract is a reviewable, versioned document that states the formula, input types, checks, and declared conventions, and it passes lint and smoke gates before entering the library. At the query level, every output carries a *DiagnosticReport* that records the contract and its version, the retrieved candidates and the selector's decision, the bound values, and each check passed or failed, with the evidence behind a stop and what the user should supply. Deterministic execution and invariant evaluation replay exactly from the stored binding. LLM-mediated selection and binding are logged and can be re-executed as recorded, but a fresh rerun from raw text is *not* guaranteed to take the identical path. A model's own chain of thought cannot stand in for such a record, because it need not reflect the computation that produced the answer [turpin2023language].

**差異**

> **Audit trail.** ~~Of the design targets in 1, a traceable record is the one most directly written into regulation: the EU AI Act requires high-risk systems to log events automatically over their lifetime for traceability [euaiact2024].~~ VerifiQuant is auditable at two levels. At the system level, each contract is a reviewable, versioned document that states the formula, input types, checks, and declared conventions, and it passes lint and smoke gates before entering the library. At the query level, every output carries a ~~*DiagnosticReport*. When a number is emitted, the report~~ ***DiagnosticReport* that** records the ~~selected~~ contract and its version, the retrieved candidates and ~~their scores,~~ the selector's decision, the bound values, **and each check passed or failed, with** the ~~checks passed, and the ambiguities handled. When the query stops, it records the gate, the rule and~~ evidence behind ~~the stop,~~ **a stop** and what the user should supply. Deterministic execution and invariant evaluation replay exactly from the stored binding. LLM-mediated selection and binding are logged and can be re-executed as recorded, but a fresh rerun from raw text is *not* guaranteed to take the identical path. A model's own chain of thought cannot stand in for such a record, because it need not reflect the computation that produced the answer [turpin2023language].

**附錄新增**

> Of the six targets, a traceable record (C5) is the one most directly written into regulation: the EU AI Act requires high-risk systems to log events automatically over their lifetime for traceability [euaiact2024]. When a number is emitted, the report records the selected contract and its version, the retrieved candidates and their scores, the selector's decision, the bound values, the checks passed, and the ambiguities handled. When the query stops, it records the gate, the rule and evidence behind the stop, and what the user should supply.

**資訊去向**：AI Act 第 12 條那句與完整的報告欄位清單移到附錄 G（法規背景，對應 C5）。可重放的界線說法完整保留。提醒：這段對應設計目標 C5，審稿人可能在意。


---

### B4　§4.1 OOD split 的配對說明　（實測省 5 行）

**改前（整段）**

> We evaluate on 250 FinanceReasoning [tang2025financereasoning] questions (180 medium, 70 hard), a difficulty-stratified random sample (seed 42) of the function-linked subset in which each question carries a formula-level reference implementation. The contract library is built from those reference functions and their questions (§3), and the pipeline deduplicates formula kernels by AST hashing. By construction a card therefore contains the formula its matching question needs, and about a third of the cards also repeat a value from that question (Appendix K of the extended version). The evaluation is *in-library*, and it measures routing, binding, and gating, not unseen-formula derivation. This is a structural constraint of the only benchmark that ships per-question formula provenance. The out-of-library analysis of §5 uses a second, fixed split of the same 250 questions: the library is rebuilt from 50 formula families, which leaves 199 questions whose family is absent (OOD) and 51 whose family is present (ID). The comparison there is matched in the same way as the clean set. Both systems run the same backbone on the same 199 questions with the same question and context, and both retrieve top-4 from the *same* 50-family library, so the baseline is not given a larger library or a different prompt. Because the split is predetermined by formula family rather than difficulty, both systems receive the same unchanged questions and differ only in whether the funnel runs.

**改後（整段）**

> We evaluate on 250 FinanceReasoning [tang2025financereasoning] questions (180 medium, 70 hard), a difficulty-stratified random sample (seed 42) of the function-linked subset in which each question carries a formula-level reference implementation. The contract library is built from those reference functions and their questions (§3), and the pipeline deduplicates formula kernels by AST hashing. By construction a card therefore contains the formula its matching question needs, and about a third of the cards also repeat a value from that question (Appendix K of the extended version). The evaluation is *in-library*, and it measures routing, binding, and gating, not unseen-formula derivation. This is a structural constraint of the only benchmark that ships per-question formula provenance. The out-of-library analysis of §5 uses a second, fixed split of the same 250 questions: the library is rebuilt from 50 formula families, which leaves 199 questions whose family is absent (OOD) and 51 whose family is present (ID). Both systems run the same backbone on the same 199 unchanged questions and retrieve top-4 from the *same* 50-family library, so they differ only in whether the funnel runs.

**差異**

> We evaluate on 250 FinanceReasoning [tang2025financereasoning] questions (180 medium, 70 hard), a difficulty-stratified random sample (seed 42) of the function-linked subset in which each question carries a formula-level reference implementation. The contract library is built from those reference functions and their questions (§3), and the pipeline deduplicates formula kernels by AST hashing. By construction a card therefore contains the formula its matching question needs, and about a third of the cards also repeat a value from that question (Appendix K of the extended version). The evaluation is *in-library*, and it measures routing, binding, and gating, not unseen-formula derivation. This is a structural constraint of the only benchmark that ships per-question formula provenance. The out-of-library analysis of §5 uses a second, fixed split of the same 250 questions: the library is rebuilt from 50 formula families, which leaves 199 questions whose family is absent (OOD) and 51 whose family is present (ID). ~~The comparison there is matched in the same way as the clean set.~~ Both systems run the same backbone on the same 199 **unchanged** questions ~~with the same question~~ and ~~context, and both~~ retrieve top-4 from the *same* 50-family library, so ~~the baseline is not given a larger library or a different prompt. Because the split is predetermined by formula family rather than difficulty, both systems receive the same unchanged questions and~~ **they** differ only in whether the funnel runs.

**資訊去向**：同段三句講同一件事，合併成一句，無資訊損失。


---

### B5　§4.2 容忍度　（實測省 4 行）

**改前（整段）**

> We compare systems on the clean set. *CoT* is a single-shot generative baseline (the LLM computes). *PAL* writes Python executed deterministically, with no retrieval and no contracts. R-PAL, a retrieval-augmented PAL/PoT baseline, is the load-bearing comparison: it receives VerifiQuant's *identical* top-4 contract retrieval as candidate context, then writes a self-contained program executed deterministically. R-PAL shares VerifiQuant's backbone (Gemini 2.5 Flash), question and context, top-4 retrieval, deterministic execution, and scoring (1% output-scale-normalized tolerance). VerifiQuant differs in two coupled ways: it runs executable governance gates, and it reaches them through separate selection, binding, and ambiguity calls where R-PAL makes one. Any R-PAL program that fails to parse, raises, times out, or returns ``cannot solve'' is counted as abstention, not silent wrong. The 1% tolerance is a general-purpose threshold rather than the stricter convention the benchmark applies to its own leaderboard, and one scorer applies it to every system, so it cannot favor any of them. It also does not touch the headline: false numeric acceptance counts whether a system emitted a number on an input that admits none, so the trap results in §5 are independent of the tolerance entirely. R-PAL+ is identical to R-PAL in backbone, retrieval context, execution environment, and scoring. Its prompt additionally supplies the complete M/N/F/E/I governance taxonomy and requires an explicit execute, refuse, or clarify decision before program generation. It has no executable gates, so it separates *describing* the governance controls from *executing* them, although it still makes a single call where VerifiQuant makes several. Two staged controls pull the coupled differences apart. *Staged R-PAL+* (Staged) splits R-PAL+ into the same five stages, from scope to program, and leaves every check to the model. *Spec-prompted R-PAL+* (Staged+Spec) keeps those stages and also hands each one the contract's declared specification as text, which is still never executed (Appendix A of the extended version).

**改後（整段）**

> We compare systems on the clean set. *CoT* is a single-shot generative baseline (the LLM computes). *PAL* writes Python executed deterministically, with no retrieval and no contracts. R-PAL, a retrieval-augmented PAL/PoT baseline, is the load-bearing comparison: it receives VerifiQuant's *identical* top-4 contract retrieval as candidate context, then writes a self-contained program executed deterministically. R-PAL shares VerifiQuant's backbone (Gemini 2.5 Flash), question and context, top-4 retrieval, deterministic execution, and scoring (1% output-scale-normalized tolerance). VerifiQuant differs in two coupled ways: it runs executable governance gates, and it reaches them through separate selection, binding, and ambiguity calls where R-PAL makes one. Any R-PAL program that fails to parse, raises, times out, or returns ``cannot solve'' is counted as abstention, not silent wrong. One scorer applies the 1% tolerance to every system, and FNAR does not depend on it, because it counts only whether a number was emitted. R-PAL+ is identical to R-PAL in backbone, retrieval context, execution environment, and scoring. Its prompt additionally supplies the complete M/N/F/E/I governance taxonomy and requires an explicit execute, refuse, or clarify decision before program generation. It has no executable gates, so it separates *describing* the governance controls from *executing* them, although it still makes a single call where VerifiQuant makes several. Two staged controls pull the coupled differences apart. *Staged R-PAL+* (Staged) splits R-PAL+ into the same five stages, from scope to program, and leaves every check to the model. *Spec-prompted R-PAL+* (Staged+Spec) keeps those stages and also hands each one the contract's declared specification as text, which is still never executed (Appendix A of the extended version).

**差異**

> We compare systems on the clean set. *CoT* is a single-shot generative baseline (the LLM computes). *PAL* writes Python executed deterministically, with no retrieval and no contracts. R-PAL, a retrieval-augmented PAL/PoT baseline, is the load-bearing comparison: it receives VerifiQuant's *identical* top-4 contract retrieval as candidate context, then writes a self-contained program executed deterministically. R-PAL shares VerifiQuant's backbone (Gemini 2.5 Flash), question and context, top-4 retrieval, deterministic execution, and scoring (1% output-scale-normalized tolerance). VerifiQuant differs in two coupled ways: it runs executable governance gates, and it reaches them through separate selection, binding, and ambiguity calls where R-PAL makes one. Any R-PAL program that fails to parse, raises, times out, or returns ``cannot solve'' is counted as abstention, not silent wrong. ~~The~~ **One scorer applies the** 1% tolerance ~~is a general-purpose threshold rather than the stricter convention the benchmark applies to its own leaderboard, and one scorer applies it~~ to every system, ~~so it cannot favor any of them. It also~~ **and FNAR** does not ~~touch the headline: false numeric acceptance~~ **depend on it, because it** counts **only** whether a ~~system emitted a~~ number ~~on an input that admits none, so the trap results in §5 are independent of the tolerance entirely.~~ **was emitted.** R-PAL+ is identical to R-PAL in backbone, retrieval context, execution environment, and scoring. Its prompt additionally supplies the complete M/N/F/E/I governance taxonomy and requires an explicit execute, refuse, or clarify decision before program generation. It has no executable gates, so it separates *describing* the governance controls from *executing* them, although it still makes a single call where VerifiQuant makes several. Two staged controls pull the coupled differences apart. *Staged R-PAL+* (Staged) splits R-PAL+ into the same five stages, from scope to program, and leaves every check to the model. *Spec-prompted R-PAL+* (Staged+Spec) keeps those stages and also hands each one the contract's declared specification as text, which is still never executed (Appendix A of the extended version).

**附錄新增**

> The 1% tolerance of §4.2 is a general-purpose threshold rather than the stricter convention the benchmark applies to its own leaderboard.

**資訊去向**：「比 benchmark 排行榜慣例寬鬆」那句移到附錄 H。


---

### B6　§5.1 RQ1 刪兩句　（實測省 5 行）

**改前（整段）**

> No, and this expected result precisely delineates the boundary of our claim. Panel A of Table 2 reports both VerifiQuant operating modes. Its R-PAL+ row is a mean of three runs, and the other rows are single audited runs. The primary deployable setting, autonomous K=1, reaches 81.2% accuracy and 2.8% SWR at 84.0% coverage. R-PAL answers nearly all questions and reaches 96.8% accuracy and 2.4% SWR. VerifiQuant's clarification-assisted K=3 result (93.6% accuracy, 3.2% SWR) is an idealized recoverability upper bound rather than the autonomous headline. R-PAL and that upper bound are not statistically distinguishable (accuracy Delta=+3.2 points, 95% CI [-0.4, 6.8], exact McNemar p=.134; SWR Delta=-0.8 points, 95% CI [-3.6, 2.0], p=.791). These paired tests use per-question binary outcomes from one audited R-PAL run on all clean questions and the canonical VerifiQuant K=3 run on the same case IDs, not averages over repeats. Moreover, the clean-accuracy axis is *saturated*: once the right formula is retrieved and executed deterministically, well-posed in-library questions leave little headroom for any governance layer to claim. Scaling the backbone does not remove the failure mode either: replacing Flash with Gemini 2.5 Pro raises single-shot CoT accuracy from 77.6% to 82.4%, but its silent-wrong rate falls only from 18.0% to 15.2%. Every other system comparison holds the backbone fixed at Flash, and §5.2 repeats the trap comparison on Pro. R-PAL+ behaves the same way (92.0% accuracy at 0.9% SWR), yet it accepts more than twice as many false numbers as VerifiQuant on the traps of RQ2. Standing on the clean set therefore does not predict governed behavior. Clean scalar accuracy is the wrong axis on which to measure governance, and instead, the systems must be separated where an answer should not be emitted.

**改後（整段）**

> No, and this expected result precisely delineates the boundary of our claim. Panel A of Table 2 reports both VerifiQuant operating modes. Its R-PAL+ row is a mean of three runs, and the other rows are single audited runs. The primary deployable setting, autonomous K=1, reaches 81.2% accuracy and 2.8% SWR at 84.0% coverage. R-PAL answers nearly all questions and reaches 96.8% accuracy and 2.4% SWR. VerifiQuant's clarification-assisted K=3 result (93.6% accuracy, 3.2% SWR) is an idealized recoverability upper bound rather than the autonomous headline. R-PAL and that upper bound are not statistically distinguishable (accuracy Delta=+3.2 points, 95% CI [-0.4, 6.8], exact McNemar p=.134; SWR Delta=-0.8 points, 95% CI [-3.6, 2.0], p=.791). Moreover, the clean-accuracy axis is *saturated*: once the right formula is retrieved and executed deterministically, well-posed in-library questions leave little headroom for any governance layer to claim. Scaling the backbone does not remove the failure mode either: replacing Flash with Gemini 2.5 Pro raises single-shot CoT accuracy from 77.6% to 82.4%, but its silent-wrong rate falls only from 18.0% to 15.2%. Every other system comparison holds the backbone fixed at Flash, and §5.2 repeats the trap comparison on Pro. R-PAL+ behaves the same way (92.0% accuracy at 0.9% SWR), yet it accepts more than twice as many false numbers as VerifiQuant on the traps of RQ2. Standing on the clean set therefore does not predict governed behavior.

**差異**

> No, and this expected result precisely delineates the boundary of our claim. Panel A of Table 2 reports both VerifiQuant operating modes. Its R-PAL+ row is a mean of three runs, and the other rows are single audited runs. The primary deployable setting, autonomous K=1, reaches 81.2% accuracy and 2.8% SWR at 84.0% coverage. R-PAL answers nearly all questions and reaches 96.8% accuracy and 2.4% SWR. VerifiQuant's clarification-assisted K=3 result (93.6% accuracy, 3.2% SWR) is an idealized recoverability upper bound rather than the autonomous headline. R-PAL and that upper bound are not statistically distinguishable (accuracy Delta=+3.2 points, 95% CI [-0.4, 6.8], exact McNemar p=.134; SWR Delta=-0.8 points, 95% CI [-3.6, 2.0], p=.791). ~~These paired tests use per-question binary outcomes from one audited R-PAL run on all clean questions and the canonical VerifiQuant K=3 run on the same case IDs, not averages over repeats.~~ Moreover, the clean-accuracy axis is *saturated*: once the right formula is retrieved and executed deterministically, well-posed in-library questions leave little headroom for any governance layer to claim. Scaling the backbone does not remove the failure mode either: replacing Flash with Gemini 2.5 Pro raises single-shot CoT accuracy from 77.6% to 82.4%, but its silent-wrong rate falls only from 18.0% to 15.2%. Every other system comparison holds the backbone fixed at Flash, and §5.2 repeats the trap comparison on Pro. R-PAL+ behaves the same way (92.0% accuracy at 0.9% SWR), yet it accepts more than twice as many false numbers as VerifiQuant on the traps of RQ2. Standing on the clean set therefore does not predict governed behavior. ~~Clean scalar accuracy is the wrong axis on which to measure governance, and instead, the systems must be separated where an answer should not be emitted.~~

**附錄新增**

> The paired tests of §5 use per-question binary outcomes from one audited R-PAL run on all clean questions and the canonical VerifiQuant K=3 run on the same case IDs, not averages over repeats.

**資訊去向**：配對檢定的資料來源移到附錄 D；最後一句和前一句「Standing on the clean set therefore does not predict governed behavior」同義。


---

### B7　§5.3 延遲但書　（實測省 4 行）

**改前（整段）**

> **What the funnel costs.** The same ladder prices the controls, since every rung runs the identical 250 questions on one backbone in one environment (Table 4). Selection, binding, and execution alone take 6.6 s per question, and the *deterministic* F/E gates add only 0.2 s because they are executed Python rather than model calls. The cost sits in the ambiguity gate, which introduces a third model call and raises the figure to 12.7 s. The full funnel runs at 12.2 s against 10.5 s for plain R-PAL and 8.2 s for R-PAL+ on the same set, so executed governance costs roughly 1.2× a retrieval-and-execution baseline and under a third of five-sample self-consistency (38.1 s). These are wall-clock means that include provider-side variance and retries, and we do not report token accounting, so they bound the order of magnitude rather than the marginal cost of a deployment. They were measured on an earlier sequential run of the same ladder at retrieval depth 3. The depth-4 rerun behind the table ran its rungs concurrently, so its wall-clock times are not comparable (Appendix D of the extended version).

**改後（整段）**

> **What the funnel costs.** The same ladder prices the controls, since every rung runs the identical 250 questions on one backbone in one environment (Table 4). Selection, binding, and execution alone take 6.6 s per question, and the *deterministic* F/E gates add only 0.2 s because they are executed Python rather than model calls. The cost sits in the ambiguity gate, which introduces a third model call and raises the figure to 12.7 s. The full funnel runs at 12.2 s against 10.5 s for plain R-PAL and 8.2 s for R-PAL+ on the same set, so executed governance costs roughly 1.2× a retrieval-and-execution baseline and under a third of five-sample self-consistency (38.1 s). These wall-clock means come from an earlier sequential run of the ladder and bound the order of magnitude rather than the marginal cost of a deployment (Appendix D of the extended version).

**差異**

> **What the funnel costs.** The same ladder prices the controls, since every rung runs the identical 250 questions on one backbone in one environment (Table 4). Selection, binding, and execution alone take 6.6 s per question, and the *deterministic* F/E gates add only 0.2 s because they are executed Python rather than model calls. The cost sits in the ambiguity gate, which introduces a third model call and raises the figure to 12.7 s. The full funnel runs at 12.2 s against 10.5 s for plain R-PAL and 8.2 s for R-PAL+ on the same set, so executed governance costs roughly 1.2× a retrieval-and-execution baseline and under a third of five-sample self-consistency (38.1 s). These ~~are~~ wall-clock means ~~that include provider-side variance~~ **come from an earlier sequential run of the ladder** and ~~retries, and we do not report token accounting, so they~~ bound the order of magnitude rather than the marginal cost of a ~~deployment. They were measured on an earlier sequential run of the same ladder at retrieval depth 3. The depth-4 rerun behind the table ran its rungs concurrently, so its wall-clock times are not comparable~~ **deployment** (Appendix D of the extended version).

**附錄新增**

> They are wall-clock means that include provider-side variance and retries, and we do not report token accounting.

**資訊去向**：「含供應商端變異與重試」「未計 token」併入附錄 D（附錄 D 原本已有 depth 3 循序、depth 4 並行的說明）。


---

### B8　§5.4 最後一段　（實測省 4 行）

**改前（整段）**

> The same boundary explains why VerifiQuant's Match (32.5%) and answered count (11/80) are low. It answers only 7 of the 34 items the panel judged answerable (20.6%). In 24 of the 27 shortfalls the query exits at N. We checked each of the 24 against all 250 contracts, counting a contract as applicable when its declared purpose covers the asked quantity and its code computes it from the given inputs. For 17 the library holds no applicable contract: 10 ask for a named financial formula and 7 for an elementary calculation such as a share, a ratio, or a difference. Declining is then the designed behavior, as when a LIFO-costing question finds only a FIFO-costing card and the selector rejects it. For the other 7 a contract exists but ranks outside the top 4, six of them a generic percent-change card, which is a genuine retrieval loss that a wider k does not recover. The library comes from a random sample of 250 questions and was not built to cover common requests, so which human-set items happen to be covered is a property of that sample. A gold answer derived for all 34 answerable items shows the cost plainly. R-PAL and R-PAL+ are correct on every item they answer, and the only item any system misses is a single forward-rate question, missed by both CoT and VerifiQuant. Abstention here buys no accuracy. This is the same exchange as the out-of-library split of §5, now under naturalistic phrasing rather than a controlled ablation: parametric knowledge buys coverage without a signal, and contract grounding buys a signal without coverage. Measuring it twice, once by ablating the library and once by sampling queries without reference to it, lets us state the price of the boundary rather than only its benefit.

**改後（整段）**

> The same boundary explains why VerifiQuant's Match (32.5%) and answered count (11/80) are low. It answers only 7 of the 34 items the panel judged answerable (20.6%). In 24 of the 27 shortfalls the query exits at N. We checked each of the 24 against all 250 contracts, counting a contract as applicable when its declared purpose covers the asked quantity and its code computes it from the given inputs. For 17 the library holds no applicable contract: 10 ask for a named financial formula and 7 for an elementary calculation such as a share, a ratio, or a difference. Declining is then the designed behavior, as when a LIFO-costing question finds only a FIFO-costing card and the selector rejects it. For the other 7 a contract exists but ranks outside the top 4, six of them a generic percent-change card, which is a genuine retrieval loss that a wider k does not recover. The library comes from a random sample of 250 questions and was not built to cover common requests, so which human-set items happen to be covered is a property of that sample. A gold answer derived for all 34 answerable items shows the cost plainly. R-PAL and R-PAL+ are correct on every item they answer, and the only item any system misses is a single forward-rate question, missed by both CoT and VerifiQuant. Abstention here buys no accuracy. As on the out-of-library split, parametric knowledge buys coverage without a signal, and contract grounding buys a signal without coverage.

**差異**

> The same boundary explains why VerifiQuant's Match (32.5%) and answered count (11/80) are low. It answers only 7 of the 34 items the panel judged answerable (20.6%). In 24 of the 27 shortfalls the query exits at N. We checked each of the 24 against all 250 contracts, counting a contract as applicable when its declared purpose covers the asked quantity and its code computes it from the given inputs. For 17 the library holds no applicable contract: 10 ask for a named financial formula and 7 for an elementary calculation such as a share, a ratio, or a difference. Declining is then the designed behavior, as when a LIFO-costing question finds only a FIFO-costing card and the selector rejects it. For the other 7 a contract exists but ranks outside the top 4, six of them a generic percent-change card, which is a genuine retrieval loss that a wider k does not recover. The library comes from a random sample of 250 questions and was not built to cover common requests, so which human-set items happen to be covered is a property of that sample. A gold answer derived for all 34 answerable items shows the cost plainly. R-PAL and R-PAL+ are correct on every item they answer, and the only item any system misses is a single forward-rate question, missed by both CoT and VerifiQuant. Abstention here buys no accuracy. ~~This is the same exchange as~~ **As on** the out-of-library ~~split of §5, now under naturalistic phrasing rather than a controlled ablation:~~ **split,** parametric knowledge buys coverage without a signal, and contract grounding buys a signal without coverage. ~~Measuring it twice, once by ablating the library and once by sampling queries without reference to it, lets us state the price of the boundary rather than only its benefit.~~

**資訊去向**：「量兩次才能講出邊界的價格」在貢獻三（we put a price on the boundary）與 §6 已講；保留金句。


---

### B9　§6 Where the claim holds 最後一句　（實測省 2 行）

**改前（整段）**

> **Where the claim holds.** The empirical claim requires three conditions: the target formula family exists in the library; the contract's specification and invariants are sufficiently correct; and the question is formula-grounded deterministic numerical reasoning. Open-world numerical reasoning and cross-dataset generalization fall outside the scope of this claim. Where these conditions fail, VerifiQuant provides at most a scope signal, not a correct answer. The clean-set result is in-library by construction, and out-of-library behavior is exactly the trade-off examined in RQ3.

**改後（整段）**

> **Where the claim holds.** The empirical claim requires three conditions: the target formula family exists in the library; the contract's specification and invariants are sufficiently correct; and the question is formula-grounded deterministic numerical reasoning. Open-world numerical reasoning and cross-dataset generalization fall outside the scope of this claim. Where these conditions fail, VerifiQuant provides at most a scope signal, not a correct answer.

**差異**

> **Where the claim holds.** The empirical claim requires three conditions: the target formula family exists in the library; the contract's specification and invariants are sufficiently correct; and the question is formula-grounded deterministic numerical reasoning. Open-world numerical reasoning and cross-dataset generalization fall outside the scope of this claim. Where these conditions fail, VerifiQuant provides at most a scope signal, not a correct answer. ~~The clean-set result is in-library by construction, and out-of-library behavior is exactly the trade-off examined in RQ3.~~

**資訊去向**：in-library by construction 在 §4.1，庫外取捨在 RQ3。


---

### C1　附錄引用縮短成「App. X」，第一次出現時加註腳說明延伸版在哪裡　（實測省 6 行）

**巨集定義**：正文全部附錄引用由「Appendix X of the extended version」改印成「App. X」。逐句改前改後：

1. 改前：The major supervisory frameworks do not yet say how an LLM inside a numeric financial workflow should be governed [fed2026sr262,pra2023ss123,euaiact2024], so we treat the requirements they share as design targets, marked C1–C6 in Figure 1 and detailed in Appendix G of the extended version.
   改後：The major supervisory frameworks do not yet say how an LLM inside a numeric financial workflow should be governed [fed2026sr262,pra2023ss123,euaiact2024], so we treat the requirements they share as design targets, marked C1–C6 in Figure 1 and detailed in App. G.

2. 改前：The gold contract of a clean question lies in the top 4 for 94.8% of the 250, and widening to 16 adds only 4.0 points, so we fix k=4 (Appendix D of the extended version).
   改後：The gold contract of a clean question lies in the top 4 for 94.8% of the 250, and widening to 16 adds only 4.0 points, so we fix k=4 (App. D).

3. 改前：We audit the 179 synthesizable transform specs in the 250-card library with six generated patch-error classes (Appendix F of the extended version).
   改後：We audit the 179 synthesizable transform specs in the 250-card library with six generated patch-error classes (App. F).

4. 改前：Each evaluation answers a different question, summarized in Appendix H of the extended version.
   改後：Each evaluation answers a different question, summarized in App. H.

5. 改前：By construction a card therefore contains the formula its matching question needs, and about a third of the cards also repeat a value from that question (Appendix K of the extended version).
   改後：By construction a card therefore contains the formula its matching question needs, and about a third of the cards also repeat a value from that question (App. K).

6. 改前：*Spec-prompted R-PAL+* (Staged+Spec) keeps those stages and also hands each one the contract's declared specification as text, which is still never executed (Appendix A of the extended version).
   改後：*Spec-prompted R-PAL+* (Staged+Spec) keeps those stages and also hands each one the contract's declared specification as text, which is still never executed (App. A).

7. 改前：The full construction, label definitions, and scoring rules are in Appendix E of the extended version.
   改後：The full construction, label definitions, and scoring rules are in App. E.

8. 改前：We release the 250-card contract library, the staged prompts that build it, the 80-trap set, and the human evaluation with its queries, per-annotator labels, frozen majority, agreement report, and scoring scripts, in the repository of §3, and print every prompt in Appendix K of the extended version.
   改後：We release the 250-card contract library, the staged prompts that build it, the 80-trap set, and the human evaluation with its queries, per-annotator labels, frozen majority, agreement report, and scoring scripts, in the repository of §3, and print every prompt in App. K.

9. 改前：When one E trap sets a stock's previous price to 2.5, for example, both baselines report a 6,500%$ weekly gain in every run, while VerifiQuant's boundary check stops it in every run (Appendix J of the extended version).
   改後：When one E trap sets a stock's previous price to 2.5, for example, both baselines report a 6,500%$ weekly gain in every run, while VerifiQuant's boundary check stops it in every run (App. J).

10. 改前：Repeating the trap comparison on Gemini 2.5 Pro narrows the gap to 11.7 points (R-PAL+ 22.1%, VerifiQuant 10.4%, 95% CI [1.7,21.5]), and nearly all of what remains lies in I (Appendix C of the extended version).
   改後：Repeating the trap comparison on Gemini 2.5 Pro narrows the gap to 11.7 points (R-PAL+ 22.1%, VerifiQuant 10.4%, 95% CI [1.7,21.5]), and nearly all of what remains lies in I (App. C).

11. 改前：The depth-4 rerun behind the table ran its rungs concurrently, so its wall-clock times are not comparable (Appendix D of the extended version).
   改後：The depth-4 rerun behind the table ran its rungs concurrently, so its wall-clock times are not comparable (App. D).

12. 改前：Writing mathrmAbs=1-mathrmCov for the abstention rate on a subset, VerifiQuant's abstention is selective at every size (Appendix I of the extended version).
   改後：Writing mathrmAbs=1-mathrmCov for the abstention rate on a subset, VerifiQuant's abstention is selective at every size (App. I).

13. 改前：Match is agreement with the human majority action, binary for CoT and plain R-PAL, which never say why they decline, and four-way otherwise (Appendix E of the extended version).
   改後：Match is agreement with the human majority action, binary for CoT and plain R-PAL, which never say why they decline, and four-way otherwise (App. E).

14. 改前：VerifiQuant then reaches a correct answer on 73.1% and R-PAL+ on 69.2%, with no wrong number after the reply, so asking pays but is not specific to our funnel (Appendix B of the extended version).
   改後：VerifiQuant then reaches a correct answer on 73.1% and R-PAL+ on 69.2%, with no wrong number after the reply, so asking pays but is not specific to our funnel (App. B).


**改前（整段）**

> Testing this claim requires holding everything else fixed. A system with contracts also carries retrieval and deterministic execution, so beating a plain generative baseline would not say which component earned the gain. We therefore build R-PAL, a retrieval-augmented program-aided baseline in the PAL/PoT family: it receives the *same* top-4 retrieved contracts VerifiQuant does, writes a self-contained program, and executes it deterministically, but runs no gates. On clean questions R-PAL matches or beats autonomous VerifiQuant (§5). On inputs that support no legitimate numeric answer, executed governance intercepts what execution alone force-fits. We then supply that baseline, as prompt text, with the same taxonomy of intent, scope, input, boundary, and ambiguity checks (M/N/F/E/I, Table 1). Its false numeric acceptance rate (FNAR), the share of inputs admitting no valid number on which a system still commits one (§2), stays at 27.5%, whereas executing the checks as contract-bound code brings it to 10.4%. The difference is not only in noticing a problem. The prompted baseline names the right diagnosis on 65.4% of these inputs but takes the matching action on only 55.8%, while VerifiQuant moves from 76.7% to 75.4%. A prompt can point at the problem without governing what happens next. Clean-set accuracy is therefore not our contribution. What we contribute is *governed behavior on malformed inputs*. Regulation also shaped the design. The major supervisory frameworks do not yet say how an LLM inside a numeric financial workflow should be governed [fed2026sr262,pra2023ss123,euaiact2024], so we treat the requirements they share as design targets, marked C1–C6 in Figure 1 and detailed in Appendix G of the extended version. We do not claim regulatory compliance.

**改後（整段）**

> Testing this claim requires holding everything else fixed. A system with contracts also carries retrieval and deterministic execution, so beating a plain generative baseline would not say which component earned the gain. We therefore build R-PAL, a retrieval-augmented program-aided baseline in the PAL/PoT family: it receives the *same* top-4 retrieved contracts VerifiQuant does, writes a self-contained program, and executes it deterministically, but runs no gates. On clean questions R-PAL matches or beats autonomous VerifiQuant (§5). On inputs that support no legitimate numeric answer, executed governance intercepts what execution alone force-fits. We then supply that baseline, as prompt text, with the same taxonomy of intent, scope, input, boundary, and ambiguity checks (M/N/F/E/I, Table 1). Its false numeric acceptance rate (FNAR), the share of inputs admitting no valid number on which a system still commits one (§2), stays at 27.5%, whereas executing the checks as contract-bound code brings it to 10.4%. The difference is not only in noticing a problem. The prompted baseline names the right diagnosis on 65.4% of these inputs but takes the matching action on only 55.8%, while VerifiQuant moves from 76.7% to 75.4%. A prompt can point at the problem without governing what happens next. Clean-set accuracy is therefore not our contribution. What we contribute is *governed behavior on malformed inputs*. Regulation also shaped the design. The major supervisory frameworks do not yet say how an LLM inside a numeric financial workflow should be governed [fed2026sr262,pra2023ss123,euaiact2024], so we treat the requirements they share as design targets, marked C1–C6 in Figure 1 and detailed in Appendix G of the extended versionAppendices A–K are in the extended version, released with the code at https://github.com/datafox-tw/icaif-verifiquant.. We do not claim regulatory compliance.

**差異**

> Testing this claim requires holding everything else fixed. A system with contracts also carries retrieval and deterministic execution, so beating a plain generative baseline would not say which component earned the gain. We therefore build R-PAL, a retrieval-augmented program-aided baseline in the PAL/PoT family: it receives the *same* top-4 retrieved contracts VerifiQuant does, writes a self-contained program, and executes it deterministically, but runs no gates. On clean questions R-PAL matches or beats autonomous VerifiQuant (§5). On inputs that support no legitimate numeric answer, executed governance intercepts what execution alone force-fits. We then supply that baseline, as prompt text, with the same taxonomy of intent, scope, input, boundary, and ambiguity checks (M/N/F/E/I, Table 1). Its false numeric acceptance rate (FNAR), the share of inputs admitting no valid number on which a system still commits one (§2), stays at 27.5%, whereas executing the checks as contract-bound code brings it to 10.4%. The difference is not only in noticing a problem. The prompted baseline names the right diagnosis on 65.4% of these inputs but takes the matching action on only 55.8%, while VerifiQuant moves from 76.7% to 75.4%. A prompt can point at the problem without governing what happens next. Clean-set accuracy is therefore not our contribution. What we contribute is *governed behavior on malformed inputs*. Regulation also shaped the design. The major supervisory frameworks do not yet say how an LLM inside a numeric financial workflow should be governed [fed2026sr262,pra2023ss123,euaiact2024], so we treat the requirements they share as design targets, marked C1–C6 in Figure 1 and detailed in Appendix G of the extended ~~version.~~ **versionAppendices A–K are in the extended version, released with the code at https://github.com/datafox-tw/icaif-verifiquant..** We do not claim regulatory compliance.

**資訊去向**：14 處「Appendix X of the extended version」改成「App. X」，延伸版位置在第一次出現時的註腳說明一次。延伸版 PDF 需要放進 repo（或日後補上 arXiv 連結）。


---

### C2　正文完全不提附錄（C1 的替代方案）　（實測省 11 行）

每一處附錄引用所在的句子：

1. 改前：The major supervisory frameworks do not yet say how an LLM inside a numeric financial workflow should be governed [fed2026sr262,pra2023ss123,euaiact2024], so we treat the requirements they share as design targets, marked C1–C6 in Figure 1 and detailed in Appendix G of the extended version.
   改後：The major supervisory frameworks do not yet say how an LLM inside a numeric financial workflow should be governed [fed2026sr262,pra2023ss123,euaiact2024], so we treat the requirements they share as design targets, marked C1–C6 in Figure 1.

2. 改前：The gold contract of a clean question lies in the top 4 for 94.8% of the 250, and widening to 16 adds only 4.0 points, so we fix k=4 (Appendix D of the extended version).
   改後：The gold contract of a clean question lies in the top 4 for 94.8% of the 250, and widening to 16 adds only 4.0 points, so we fix k=4.

3. 改前：We audit the 179 synthesizable transform specs in the 250-card library with six generated patch-error classes (Appendix F of the extended version).
   改後：We audit the 179 synthesizable transform specs in the 250-card library with six generated patch-error classes.

4. 改前：Each evaluation answers a different question, summarized in Appendix H of the extended version.
   改後：Each evaluation answers a different question.

5. 改前：By construction a card therefore contains the formula its matching question needs, and about a third of the cards also repeat a value from that question (Appendix K of the extended version).
   改後：By construction a card therefore contains the formula its matching question needs, and about a third of the cards also repeat a value from that question.

6. 改前：*Spec-prompted R-PAL+* (Staged+Spec) keeps those stages and also hands each one the contract's declared specification as text, which is still never executed (Appendix A of the extended version).
   改後：*Spec-prompted R-PAL+* (Staged+Spec) keeps those stages and also hands each one the contract's declared specification as text, which is still never executed.

7. 改前：The full construction, label definitions, and scoring rules are in Appendix E of the extended version.
   改後：

8. 改前：We release the 250-card contract library, the staged prompts that build it, the 80-trap set, and the human evaluation with its queries, per-annotator labels, frozen majority, agreement report, and scoring scripts, in the repository of §3, and print every prompt in Appendix K of the extended version.
   改後：We release the 250-card contract library, the staged prompts that build it, the 80-trap set, and the human evaluation with its queries, per-annotator labels, frozen majority, agreement report, and scoring scripts, in the repository of §3.

9. 改前：When one E trap sets a stock's previous price to 2.5, for example, both baselines report a 6,500%$ weekly gain in every run, while VerifiQuant's boundary check stops it in every run (Appendix J of the extended version).
   改後：When one E trap sets a stock's previous price to 2.5, for example, both baselines report a 6,500%$ weekly gain in every run, while VerifiQuant's boundary check stops it in every run.

10. 改前：Repeating the trap comparison on Gemini 2.5 Pro narrows the gap to 11.7 points (R-PAL+ 22.1%, VerifiQuant 10.4%, 95% CI [1.7,21.5]), and nearly all of what remains lies in I (Appendix C of the extended version).
   改後：Repeating the trap comparison on Gemini 2.5 Pro narrows the gap to 11.7 points (R-PAL+ 22.1%, VerifiQuant 10.4%, 95% CI [1.7,21.5]), and nearly all of what remains lies in I.

11. 改前：The depth-4 rerun behind the table ran its rungs concurrently, so its wall-clock times are not comparable (Appendix D of the extended version).
   改後：The depth-4 rerun behind the table ran its rungs concurrently, so its wall-clock times are not comparable.

12. 改前：Writing mathrmAbs=1-mathrmCov for the abstention rate on a subset, VerifiQuant's abstention is selective at every size (Appendix I of the extended version).
   改後：Writing mathrmAbs=1-mathrmCov for the abstention rate on a subset, VerifiQuant's abstention is selective at every size.

13. 改前：Match is agreement with the human majority action, binary for CoT and plain R-PAL, which never say why they decline, and four-way otherwise (Appendix E of the extended version).
   改後：Match is agreement with the human majority action, binary for CoT and plain R-PAL, which never say why they decline, and four-way otherwise.

14. 改前：VerifiQuant then reaches a correct answer on 73.1% and R-PAL+ on 69.2%, with no wrong number after the reply, so asking pays but is not specific to our funnel (Appendix B of the extended version).
   改後：VerifiQuant then reaches a correct answer on 73.1% and R-PAL+ on 69.2%, with no wrong number after the reply, so asking pays but is not specific to our funnel.

**資訊去向**：14 處附錄引用全部刪除。附錄內容仍在 arXiv 版，但 ACM 讀者不會知道。


---
