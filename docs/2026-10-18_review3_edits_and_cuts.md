# 第三輪審稿意見：已套用的修改與刪減選項（2026-10-18）

所有行數都是實際編譯 ACM 版（pdfLaTeX）後數 PDF 文字行得到的。來源檔：`paper/main_camera_ready_9p.tex`。修改前備份：`paper/archive/2026-10-17_main_9p_before_review3.tex`、`paper/archive/2026-10-17_appendix_before_review3.tex`。

## 現況

**決定（2026-10-18）：採用 R1（S1 + O1–O6 + B5），本輪新句不動。ACM 9 頁，剩約 1 行；稽核 303 項全過。套用前備份：`paper/archive/2026-10-18_main_9p_before_R1.tex`、`paper/archive/2026-10-18_appendix_before_R1.tex`。**

- 審稿意見 1–6 全部套用，稽核 303 項全過（新增 10 項，對修改前版本做反向對照時，新增檢查全數失敗，其餘照過）。
- 這些修改共多出 **14 行**，ACM 版從 9 頁變成 **10 頁**。
- 內容遺失檢查：唯一被標出的是舊的 Fisher p=.74，已連同 McNemar 的不一致題數補進附錄 E。

## 一、已套用的修改（改前 → 改後）

### 審稿 1：問題定義（p.2）補上兩題 F trap 的例外（+3 行）

改前：

> In the trap set $T$, no legitimate \emph{unqualified single} numeric answer exists: a required field is missing, a declared boundary is violated, a material convention remains unresolved, or the ask is outside the evaluated task scope.

改後：

> Each trap in $T$ is built so that no legitimate \emph{unqualified single} numeric answer exists: a required field is missing, a declared boundary is violated, a material convention remains unresolved, or the ask is outside the evaluated task scope. Two F traps miss this, because the field they remove has a declared default (\S\ref{sec:v2rq2}).

### 審稿 1：§4.3「出口由建構決定」加上例外（+1 行）

改前：

> Natural queries rarely come with a known correct handling, so the operators supply one: each trap's expected exit is known by construction.

改後：

> Natural queries rarely come with a known correct handling, so the operators supply one: each trap's intended exit is fixed by construction, apart from the two F traps of \S\ref{sec:v2rq2}.

### 審稿 1：Table 2 Panel B 標題（0 行）

改前：

> \emph{Panel B: inputs with no valid numeric answer (F/E/I/N)}

改後：

> \emph{Panel B: traps built to admit no valid number (F/E/I/N)}

### 審稿 2：§4.1 刪 only，寫明兩系統共用什麼、差在哪（+1 行）

改前：

> retrieve top-4 from the \emph{same} 50-family library, so they differ only in whether the funnel runs.

改後：

> retrieve top-4 from the \emph{same} 50-family library. They differ in what follows retrieval: the funnel and the contract's code for VerifiQuant, a model-written program for R-PAL.

### 審稿 2：§6 建庫缺陷句限定範圍（+2 行）

改前：

> These defects are caught before runtime by the contract audits and the pre-declared type-policy check of \S\ref{sec:v2method}.

改後：

> The audits and the pre-declared type-policy check of \S\ref{sec:v2method} caught these three before runtime, but they do not guarantee that every specification error is found: one contract marks a conditionally required input optional (\extref{app:reruns}).

### 審稿 3+4：§5.4 改 exact McNemar，人工集主張收斂（+1 行）

改前：

> so only the separation from plain R-PAL is individually significant (Fisher exact $p<10^{-4}$, against $p{=}.74$ for VerifiQuant versus R-PAL$^{+}$). What RQ4 establishes is therefore not a second powered comparison of the two governed systems, but external validity for the direction: a panel blind to every system, labeling queries our operators never touched, reproduces the Panel~B ordering, so that ordering is not an artifact of our own trap construction.

改後：

> so only the separation from plain R-PAL is individually significant (exact McNemar on the same items, $p<10^{-6}$, against $p{=}.73$ for VerifiQuant versus R-PAL$^{+}$). What RQ4 establishes is therefore not a second powered comparison of the two governed systems. A panel blind to every system, labeling queries our operators never touched, reproduces the numeric ordering of Panel~B. It does not validate the F and I checks on natural phrasing, because that low rate rests mainly on scope refusal, as we show next.

### 審稿 5：Match 計分方式寫進 Table 5 前的正文（+3 行）

改前：

> Match is agreement with the human majority action, binary for CoT and plain R-PAL, which never say why they decline, and four-way otherwise (\extref{app:human}).

改後：

> Match is agreement with the human majority action. It is binary for CoT and plain R-PAL, which never say why they decline, and four-way otherwise, so Match is comparable only within each pair. An M refusal is scored \textsc{out-of-scope}, and scoring it as \textsc{clarify} instead raises VerifiQuant's Match from $32.5\%$ to $35.0\%$ (\extref{app:human}).

### 審稿 6：6,500% 例子點名違反的合約檢查（+1 行）

改前：

> while VerifiQuant's boundary check stops it in every run (\extref{app:cases}).

改後：

> while the contract's E check \texttt{price\_coherence\_\allowbreak extreme\_change}, which fires on a price change above $200\%$, stops VerifiQuant in every run (\extref{app:cases}).

### 附錄 H：同一句「differ only」也一併改掉（arXiv 版，不佔頁數）

> Because the split is predetermined by formula family rather than difficulty, both systems receive the same unchanged questions. They differ in what follows retrieval: VerifiQuant runs the funnel and the selected contract's code, whereas R-PAL writes its own program with the retrieved contracts as context.

### 附錄 E：配對檢定細節（arXiv 版，不佔頁數）

> Every system answers the same 46 non-\textsc{answer} items, so the tests of \S\ref{sec:v2rq4} are paired. Against plain R-PAL, VerifiQuant alone emits a number on 0 items and R-PAL alone on 22 (exact McNemar $p<10^{-6}$). Against R-PAL$^{+}$ the counts are 3 and 5 ($p{=}.73$), and against CoT 1 and 7 ($p{=}.07$). An unpaired Fisher test on the same rates gives the same reading ($p<10^{-5}$ and $p{=}.74$).

McNemar 的結論和原本 Fisher 相同：只有對 plain R-PAL 顯著，對 R-PAL+ 不顯著。

## 二、刪減候選（每項單獨實測）

| 編號 | 內容 | 單獨省 | 資訊去向 |
|---|---|---|---|
| T1 | p.2 開頭改回原句型（加 by design），because 改 as | 1 行 | 無資訊損失（只改措辭） |
| T2 | Match 計分說明精簡 | 0 行 | 無資訊損失；實測省 0 行，不建議 |
| T3 | §6 稽核限定句精簡 | 1 行 | 無資訊損失 |
| T4 | 6,500% 例子的檢查名稱改用括號 | 1 行 | 無資訊損失，檢查名稱與門檻都保留 |
| S1 | §5.4 刪掉和新句重複的引導句與結論前半 | 2 行 | 「低 FNAR 主要來自範圍拒答」已由本輪新句說明（It does not validate … rests mainly on scope refusal），刪的是重複 |
| O1 | §3.2 刪 Across the whole funnel | 1 行 | 無資訊損失（「整個 funnel 中」由「exactly three places」已隱含） |
| O2 | §4 開頭 summarized in 改括號 | 1 行 | 無資訊損失（仍指向附錄 H） |
| O3 | §4.2 Staged+Spec 定義精簡 | 1 行 | 無資訊損失（Staged+Spec 沿用 Staged 的五個階段，由 also 表達） |
| O4 | §5.3 monotone 句精簡 | 1 行 | 無資訊損失 |
| O6 | §6 scripted user 句精簡 | 1 行 | 無資訊損失（「兩個系統都受益」即「不是我們 funnel 特有」） |
| B5 | §4.2 容忍度縮成一句（排行榜慣例那句移附錄 H） | 4 行 | 「1% 是通用門檻、比 benchmark 排行榜慣例寬鬆」那句移到附錄 H 開頭；「同一個 scorer、FNAR 不受影響」保留在正文 |

T 開頭是精簡本輪自己新加的句子，S1 是 §5.4 因本輪新句而出現的重複，O 開頭是最後一行只剩一兩個字的段落，B5 是之前提過但還沒用的提案。

## 三、組合實測

| 組合 | 省下行數 | ACM 頁數 | 9 頁內剩餘 |
|---|---|---|---|
| T1+T3+T4+S1+B5 | 約 9 | 10 | — |
| T 全部 + S1 + O 全部（不含 B5） | 10 | 10 | — |
| **R1：S1 + O1–O6 + B5**（本輪新句不動） | 13 | **9** | 1 行 |
| **R2：T1+T3+T4 + S1 + O1–O6 + B5** | 16 | **9** | 2 行 |

結論：不用 B5 的話，目前的小刪減加起來回不到 9 頁；B5 加上 O 組幾乎是必要的。R2 多留 2 行，給之後要補的人工集判定（P9 從 5/7 改 4/7）之類的小修改。

## 四、各刪減候選的完整段落（改前 → 改後）

### T1：p.2 開頭改回原句型（加 by design），because 改 as（省 1 行）

改前：

> Each trap in $T$ is built so that no legitimate \emph{unqualified single} numeric answer exists: a required field is missing, a declared boundary is violated, a material convention remains unresolved, or the ask is outside the evaluated task scope. Two F traps miss this, because the field they remove has a declared default (\S\ref{sec:v2rq2}).

改後：

> In the trap set $T$, no legitimate \emph{unqualified single} numeric answer exists by design: a required field is missing, a declared boundary is violated, a material convention remains unresolved, or the ask is outside the evaluated task scope. Two F traps miss this, because the field they remove has a declared default (\S\ref{sec:v2rq2}).

改前：

> Each trap in $T$ is built so that no legitimate \emph{unqualified single} numeric answer exists: a required field is missing, a declared boundary is violated, a material convention remains unresolved, or the ask is outside the evaluated task scope. Two F traps miss this, because the field they remove has a declared default (\S\ref{sec:v2rq2}).

改後：

> Each trap in $T$ is built so that no legitimate \emph{unqualified single} numeric answer exists: a required field is missing, a declared boundary is violated, a material convention remains unresolved, or the ask is outside the evaluated task scope. Two F traps miss this, as the field they remove has a declared default (\S\ref{sec:v2rq2}).

### T2：Match 計分說明精簡（省 0 行）

改前：

> Human-FNAR is the share of the 46 non-\textsc{answer} items on which a system emits a number, $\mathrm{HumanFNAR}(S)=|\{i\in\mathrm{RCO}: e_S(i){=}1\}|/|\mathrm{RCO}|$ with $\mathrm{RCO}=\{i:\mathrm{GT}(i)\neq\text{A}\}$. A number that carries an I\textsubscript{SOFT} warning counts as emitted, which is stricter than the trap metric, because a \textsc{clarify} label asks for the question before any computation. Human-FNAR is the headline because it needs only whether a number was emitted, which every system exposes, and no gold answer. Answer coverage is the share of the 34 \textsc{answer} items a system answers. Match is agreement with the human majority action. It is binary for CoT and plain R-PAL, which never say why they decline, and four-way otherwise, so Match is comparable only within each pair. An M refusal is scored \textsc{out-of-scope}, and scoring it as \textsc{clarify} instead raises VerifiQuant's Match from $32.5\%$ to $35.0\%$ (\extref{app:human}).

改後：

> Human-FNAR is the share of the 46 non-\textsc{answer} items on which a system emits a number, $\mathrm{HumanFNAR}(S)=|\{i\in\mathrm{RCO}: e_S(i){=}1\}|/|\mathrm{RCO}|$ with $\mathrm{RCO}=\{i:\mathrm{GT}(i)\neq\text{A}\}$. A number that carries an I\textsubscript{SOFT} warning counts as emitted, which is stricter than the trap metric, because a \textsc{clarify} label asks for the question before any computation. Human-FNAR is the headline because it needs only whether a number was emitted, which every system exposes, and no gold answer. Answer coverage is the share of the 34 \textsc{answer} items a system answers. Match is agreement with the human majority action, binary for CoT and plain R-PAL, which never say why they decline, and four-way otherwise, so it compares only within each pair. Scoring an M refusal as \textsc{clarify} instead of \textsc{out-of-scope} raises VerifiQuant's Match from $32.5\%$ to $35.0\%$ (\extref{app:human}).

### T3：§6 稽核限定句精簡（省 1 行）

改前：

> Distinct from these are \emph{build-time defects}, such as a contract that omits an ambiguity or encodes a wrong invariant, and, in this library, three indicator cards that returned a series instead of a scalar. The audits and the pre-declared type-policy check of \S\ref{sec:v2method} caught these three before runtime, but they do not guarantee that every specification error is found: one contract marks a conditionally required input optional (\extref{app:reruns}).

改後：

> Distinct from these are \emph{build-time defects}, such as a contract that omits an ambiguity or encodes a wrong invariant, and, in this library, three indicator cards that returned a series instead of a scalar. The audits and the pre-declared type-policy check of \S\ref{sec:v2method} caught these three before runtime, but they cannot catch every specification error: one contract marks a conditionally required input optional (\extref{app:reruns}).

### T4：6,500% 例子的檢查名稱改用括號（省 1 行）

改前：

> The FNAR advantage concentrates on E and I, where correctness depends on an executable boundary or an unresolved convention rather than on an obviously missing field (Table~\ref{tab:v2trapop}). There, plain R-PAL force-fits almost always, and even the taxonomy-prompted variant stays high, whereas VerifiQuant's contract checks hold. When one E trap sets a stock's previous price to \$2.5, for example, both baselines report a $6{,}500\%$ weekly gain in every run, while the contract's E check \texttt{price\_coherence\_\allowbreak extreme\_change}, which fires on a price change above $200\%$, stops VerifiQuant in every run (\extref{app:cases}).

改後：

> The FNAR advantage concentrates on E and I, where correctness depends on an executable boundary or an unresolved convention rather than on an obviously missing field (Table~\ref{tab:v2trapop}). There, plain R-PAL force-fits almost always, and even the taxonomy-prompted variant stays high, whereas VerifiQuant's contract checks hold. When one E trap sets a stock's previous price to \$2.5, for example, both baselines report a $6{,}500\%$ weekly gain in every run, while VerifiQuant's E check \texttt{price\_coherence\_\allowbreak extreme\_change} (a price change above $200\%$) stops it in every run (\extref{app:cases}).

### S1：§5.4 刪掉和新句重複的引導句與結論前半（省 2 行）

改前：

> The ordering says how the systems rank. What the low rate means depends on where VerifiQuant stops. Of the 42 non-answer items on which it emits no number, 28 exit at N, that is, with a scope signal. Of the 20 items the panel labels \textsc{request-info}, only 5 exit at F. Even where it selects a contract, it asks for the missing input on 5 of 7 such items, against 7 of 7 for R-PAL$^{+}$ (\extref{app:human}). The low human-defined FNAR therefore comes mostly from the library boundary, and this set provides no evidence that VerifiQuant detects missing information or ambiguity better than a single prompt.

改後：

> Of the 42 non-answer items on which it emits no number, 28 exit at N, that is, with a scope signal. Of the 20 items the panel labels \textsc{request-info}, only 5 exit at F. Even where it selects a contract, it asks for the missing input on 5 of 7 such items, against 7 of 7 for R-PAL$^{+}$ (\extref{app:human}). The low human-defined FNAR therefore comes mostly from the library boundary, and this set provides no evidence that VerifiQuant detects missing information or ambiguity better than a single prompt.

改前：

> The ordering says how the systems rank. What the low rate means depends on where VerifiQuant stops. Of the 42 non-answer items on which it emits no number, 28 exit at N, that is, with a scope signal. Of the 20 items the panel labels \textsc{request-info}, only 5 exit at F. Even where it selects a contract, it asks for the missing input on 5 of 7 such items, against 7 of 7 for R-PAL$^{+}$ (\extref{app:human}). The low human-defined FNAR therefore comes mostly from the library boundary, and this set provides no evidence that VerifiQuant detects missing information or ambiguity better than a single prompt.

改後：

> The ordering says how the systems rank. What the low rate means depends on where VerifiQuant stops. Of the 42 non-answer items on which it emits no number, 28 exit at N, that is, with a scope signal. Of the 20 items the panel labels \textsc{request-info}, only 5 exit at F. Even where it selects a contract, it asks for the missing input on 5 of 7 such items, against 7 of 7 for R-PAL$^{+}$ (\extref{app:human}). This set therefore provides no evidence that VerifiQuant detects missing information or ambiguity better than a single prompt.

### O1：§3.2 刪 Across the whole funnel（省 1 行）

改前：

> Thus, abstention is a designed outcome rather than a fallback. Five funnel conditions end in a structured exit that states where the query stopped, why, and what is needed to continue, and I\textsubscript{SOFT} is the only case that continues with a warning. Across the whole funnel the LLM acts in exactly three places: selecting a contract, binding inputs, and judging whether a declared ambiguity is resolved.

改後：

> Thus, abstention is a designed outcome rather than a fallback. Five funnel conditions end in a structured exit that states where the query stopped, why, and what is needed to continue, and I\textsubscript{SOFT} is the only case that continues with a warning. The LLM acts in exactly three places: selecting a contract, binding inputs, and judging whether a declared ambiguity is resolved.

### O2：§4 開頭 summarized in 改括號（省 1 行）

改前：

> Each evaluation answers a different question, summarized in \extref{app:evals}. The two in-library sets test the checks themselves. The out-of-library split and the library-agnostic human set test only the scope signal.

改後：

> Each evaluation answers a different question (\extref{app:evals}). The two in-library sets test the checks themselves. The out-of-library split and the library-agnostic human set test only the scope signal.

### O3：§4.2 Staged+Spec 定義精簡（省 1 行）

改前：

> Two staged controls pull the coupled differences apart. \emph{Staged R-PAL$^{+}$} (Staged) splits R-PAL$^{+}$ into the same five stages, from scope to program, and leaves every check to the model. \emph{Spec-prompted R-PAL$^{+}$} (Staged+Spec) keeps those stages and also hands each one the contract's declared specification as text, which is still never executed (\extref{app:staged}).

改後：

> Two staged controls pull the coupled differences apart. \emph{Staged R-PAL$^{+}$} (Staged) splits R-PAL$^{+}$ into the same five stages, from scope to program, and leaves every check to the model. \emph{Spec-prompted R-PAL$^{+}$} (Staged+Spec) also hands each stage the contract's declared specification as text, which is never executed (\extref{app:staged}).

### O4：§5.3 monotone 句精簡（省 1 行）

改前：

> The pattern is monotone in the metric that matters: \emph{gates buy safety with coverage, not accuracy}.

改後：

> The pattern is monotone: \emph{gates buy safety with coverage, not accuracy}.

### O6：§6 scripted user 句精簡（省 1 行）

改前：

> On the 26 traps that remove a required input or a hard convention, a scripted user restates only what was removed. VerifiQuant then reaches a correct answer on $73.1\%$ and R-PAL$^{+}$ on $69.2\%$, with no wrong number after the reply, so asking pays but is not specific to our funnel (\extref{app:scripted}).

改後：

> On the 26 traps that remove a required input or a hard convention, a scripted user restates only what was removed. VerifiQuant then reaches a correct answer on $73.1\%$ and R-PAL$^{+}$ on $69.2\%$, with no wrong number after the reply, so asking pays for both systems (\extref{app:scripted}).

### B5：§4.2 容忍度縮成一句（排行榜慣例那句移附錄 H）（省 4 行）

改前：

> The $1\%$ tolerance is a general-purpose threshold rather than the stricter convention the benchmark applies to its own leaderboard, and one scorer applies it to every system, so it cannot favor any of them. It also does not touch the headline: false numeric acceptance counts whether a system emitted a number on an input that admits none, so the trap results in \S\ref{sec:v2res} are independent of the tolerance entirely.

改後：

> One scorer applies the $1\%$ tolerance to every system, and FNAR does not depend on it, because it counts only whether a number was emitted.

附錄新增：

> The $1\%$ tolerance of \S\ref{sec:v2baselines} is a general-purpose threshold rather than the stricter convention the benchmark applies to its own leaderboard.

## 需要你決定

1. 採用 R1（只刪舊文字、本輪新句不動，剩 1 行）或 R2（連新句一起精簡，剩 2 行），或其他組合。
2. 本輪套用的 8 處正文修改是否都 OK（尤其 §4.1 的「what differs」寫法、§6 舉 F-1271 那張卡當例子）。
