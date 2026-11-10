# ICAIF '26 平板／筆電 deck 講稿（18 張）

Deck：https://claude.ai/artifact/PbMzGGY7r5e8ay1U479vsc （目前是 private，要給別人看要先從 Share 開）

這份是配合 18 張 slides 的逐頁講稿，和 `poster/talk_scripts.md`（對著海報講）分開。數字全部和海報 final 版、`paper/main_camera_ready_9p.tex` 一致。講稿是英文（現場要講的），每頁附：大約時長、手要指哪裡、轉場句、這頁最常被問的問題。語速以每分鐘約 140 字估。

使用情境是一對一或兩三個人圍著平板，所以每頁都設計成可以停下來回答問題，再從下一頁的轉場句接回來。

---

## 三條路線

| 路線 | 頁碼 | 時間 | 什麼時候用 |
|---|---|---|---|
| 1 分鐘 | 3 → 6 → 10 → 18 | 約 1 分 | 對方只是路過，或在排隊等別人 |
| 3 分鐘 | 1 → 3 → 6 → 7 → 8 → 10 → 12 → 13 → 18 | 約 3–4 分 | 對方停下來、有興趣但沒時間 |
| 完整 | 1 → 18 | 約 9–10 分 | 對方坐下來、或是同領域的人 |

依對方背景跳頁（先問一句 “Are you more on the industry side or the research side?”）：

| 對方 | 先講 | 重點加講 |
|---|---|---|
| 銀行／fintech 實務 | 3（trap）→ 7（合約表） | 9 DiagnosticReport、12 代價、18 demo |
| ML／NLP 研究者 | 10（主結果） | 13 Staged 對照組、14–15 驗證器、16 庫外 |
| 風控／監理 | 4（三種結果） | 9 可重播紀錄、17 邊界（不宣稱合規） |

---

## 1 分鐘版（約 140 字）

**[3]** When an LLM sits inside a financial workflow, the dangerous failure is a confident, well-formatted, wrong number. In this trap we changed one price to an impossible value. Two baselines that see the same contracts and run code both report a 6,500% weekly gain. VerifiQuant stops and names the check that failed.

**[6]** The idea is a spreadsheet. Each formula is written down once as a contract. The LLM only fills it in. It never computes the answer.

**[10]** With the same checks written into a prompt, 27.5% of trap questions still get a number. Run in stages, 10.4%.

**[18]** The price is coverage, not accuracy. The demo replays every trap. The QR code is right here.

---

## 完整版逐頁

### 1 · Cover（約 15 秒）

指：標題，然後右下 QR。

> Hi, I'm Yu-Chi, from National Taiwan University and Academia Sinica. This is VerifiQuant. The one-line version is: keep the spreadsheet, and let the LLM fill it in. Can I show you one example first?

### 2 · A number is not an answer（約 25 秒）

指：上面那句橘字 “wrong number”，再左右兩張卡。

> LLMs in finance now sit in workflows that end in a number. Valuation memos, client calculators, regulatory reports. They rarely fail by crashing. They fail with a clean, wrong number, and nothing marks it as wrong. Program-aided reasoning fixes the arithmetic, because an interpreter runs the code. But it does not decide whether the question justifies a number at all.

轉場：*Here is what that looks like in practice.*

### 3 · One impossible price, three systems（約 40 秒）

指：左邊 $2.5 → 右邊三列由上往下。

> This is a real trap from our evaluation. It asks for the weekly price rate of change. We changed the opening price from 150 dollars to 2.5. R-PAL sees the same retrieved contracts and runs code. It reports 6,500%. R-PAL plus also gets every one of our checks described in its prompt. It still reports 6,500%, in all three runs. VerifiQuant stops at gate E in all three runs, and it shows you the check: a weekly price change above 200%. Same model, same contracts. The difference is whether the check is executed or only described.

被問「這是不是挑出來的例子？」：*It is one example, chosen because it is easy to read. The aggregate numbers come on slide 10.*

### 4 · When is emitting a number legitimate?（約 20 秒）

指：三張卡由左到右，最後指底下深藍條。

> So the question we ask is not how accurate the system is. It is when a number may leave at all. Every response is one of three things: a correct number, a silent wrong, or a named exit that says why. The goal is to turn silent wrongs into named exits, while keeping as many correct numbers as possible.

### 5 · Five ways a question fails（約 30 秒）

指：左邊字母，一列一列往下，例子念一兩個就好。

> There are five reasons a question may not justify a number, and each gets a letter. N, nothing in the library covers it. M, the intent is unclear, like “the return” without saying holding-period or annualized. F, a required input is missing. E, a value breaks a declared check, like that 2.5-dollar price. I, a convention is left open, like whether payments come at the start or end of each period. These letters are also the gates you will see in a moment.

轉場：*So how do we catch them? We borrow from how analysts already work.*

### 6 · Keep the spreadsheet. Let the LLM fill it in.（約 40 秒）

指：左欄第一句，然後右欄三個橘色號碼，最後深藍條。

> An analyst does not re-derive a loan formula for each client. They open a vetted sheet, type in the inputs, and the sheet refuses impossible values. We write each formula family once as such a sheet. We call it a Financial Inference Contract. The LLM gets three jobs, and only three. Pick the sheet. Fill the input cells. Flag which declared conventions the text leaves open. It never computes the answer, and it never freely rewrites the formula.

被問「真的接 Excel 嗎？」：*No, it is an analogy. Each contract is a stored card. We draw it as a sheet because that is how analysts think.*

被問「合約誰寫的？」：*They are generated offline, three LLM calls per formula family, then lint and smoke tests. None was hand-tuned. There are 250.*

### 7 · Four blocks, four gates（約 45 秒）

指：左上的 query，然後表格四種顏色由上往下，最後左下橘框。

> This is the real loan contract. The orange rows are what the LLM fills: principal, rate, term. The blue rows are data validation, run as code. Principal must be positive, and the rate must stay below 1.0. Then come declared conventions. Rate scale is hard, because it changes the computation, so we ask. Payment timing is soft, because it changes how you read the number, so we answer and warn. The green row is locked Python code, which gives 1,498.88. Now suppose the LLM binds “6%” as 6 instead of 0.06. The rate check fails, and the query ends at an alert that names the check. No number leaves.

### 8 · A funnel decides before anything is computed（約 40 秒）

指：七個方塊由左到右，橘色和藍色交替。

> At run time, the query goes through this funnel, left to right. Retrieval is BM25, top four contracts. Then three LLM steps, in orange, alternate with code steps, in blue. Select picks one contract or none. Bind fills the inputs. Code then checks that every required input is present and every declared boundary holds. The ambiguity step asks whether a convention is still open. Only a query that passes every gate reaches the contract’s own code.

被問「Bind 錯了怎麼辦？」：*Bind has no exit of its own. The next two gates check what was bound, which is exactly the 6-versus-0.06 case.*

### 9 · Every stop says why（約 30 秒）

指：六張卡掃過，停在 I-HARD，再指底下深藍條。

> Every stop is a named exit. N refuses. M asks which one you meant. F asks for the missing field by name. E names the failed check and the value. For conventions, a soft one returns the number with a warning, and a hard one asks the user to pick an option, then applies a verified edit. I will come back to that. And every path, number or not, writes a DiagnosticReport that replays from the stored binding.

措辭注意：說 “replays from the stored binding”，不要說 “traced back to the source document”。

轉場：*So does it work? Here is the headline.*

### 10 · Execution is not governance（約 40 秒）

指：27.5% → 10.4%，再指下面 CI。

> We derived 80 traps from the contracts, 20 each of missing input, impossible value, open convention, and out of scope. We count how often a system still gives an unqualified number. R-PAL plus has our whole checklist in its prompt, and it still answers 27.5% of them. VerifiQuant runs the same checks in stages and answers 10.4%. That is 17 points lower, with a 95% interval from about 7 to 27. Same model, Gemini 2.5 Flash, same top-four contracts, three-run means.

被問「trap 是從你們自己的合約生出來的，當然抓得到？」：*Yes, so this is a stress test of the mechanism, not a real-world rate. What it shows is that staged checks beat the same checklist in a prompt. The blind human-labeled set on slide 16 keeps the same ordering.*

被問 FNAR 定義：*Any trap that gets an unqualified number. A soft-ambiguity trap answered with the correct warning does not count, because that is the right handling.*

### 11 · Boundary and ambiguity traps（約 30 秒）

指：兩列綠底，再指最下面 All，最後左下框。

> Where do the 17 points come from? Boundary and ambiguity. On impossible values, R-PAL plus accepts 30% and we accept 8.3. On open conventions, 80 versus 21.7. Those are the cases where the right answer depends on an executed check or an unstated convention. On missing inputs and out-of-scope asks, a prompt already refuses, and we are no better. Plain R-PAL, which only runs code, answers 48.8% of all traps.

被問「為什麼不直接把檢查寫進 prompt？」：*That is R-PAL plus. It names the problem 65% of the time but takes the matching action only 56%. We go from 77 to 75.*

### 12 · Safety costs coverage, not accuracy（約 35 秒）

指：表格 R-PAL 列，再指綠底列，最後灰色 K = 3 列和右下小字。

> This is not an accuracy result. On 250 well-posed questions, R-PAL reaches 96.8%. We reach 81.2%, because we decline about 16% of them. Our silent-wrong rate is close to R-PAL’s, 2.8 versus 2.4. The grey row is an upper bound, not a deployment number. An oracle that knows the intended computation, but never the answer, may clarify up to twice. Then we reach 93.6%.

措辭注意：K = 3 一定要說 “upper bound”。被問速度：*12.2 seconds per question versus 10.5 for plain R-PAL, about 1.2 times.*

### 13 · Staging cuts FNAR, code keeps coverage（約 40 秒）

指：右下綠星 ideal，再指兩個空心點，最後右邊深藍框。

> This puts safety and utility on one picture. Coverage on clean questions runs across, false acceptance on traps runs up. The ideal is the lower right. The two hollow points answer a fair question: is the gain just from splitting the prompt into stages? Staging alone gets to 15.8%, but only by refusing much more. Staging plus the contract text ties us on traps, at 8.8%, but covers only 75%. So staging over the declared contract lowers false acceptance, and running the contract’s own code keeps coverage. Priced against R-PAL plus, each trap number we avoid costs 2.0 correct clean answers.

轉場：*One more technical piece. Remember the hard convention on slide 9?*

### 14 · Verified edits, not regeneration（約 40 秒）

指：左卡 result_expr → 1,498.88 → 1,491.42，中間 AST，右邊三個勾。

> When a user picks a non-default convention, the computation has to change. If we let the LLM rewrite the code, we reopen the door the funnel just closed. So every edit is declared on the contract at build time. For the loan, paying at the start of each period means dividing by one plus the monthly rate. The invariant says: multiply back, and you recover the old payment. The payment moves from 1,498.88 to 1,491.42. Before it runs, three checks. Bounded: 13 AST nodes, the declared limit. Safe: whitelisted names only. Matches: the invariant holds on five sampled inputs.

被問「能改程式碼，不只是轉換輸出嗎？」：*In a proof of concept, yes. For NPV timing, a one-node patch changes start=1 to start=0, and it is accepted only if its output matches the declared relation. The library itself uses output transforms.*

### 15 · Wrong patches are rejected, within bounds（約 25 秒）

指：97.7%，再指右邊兩張卡，最後深藍條。

> To test the verifier, we generated 906 faulty patches on purpose. 27 turned out equivalent to the original. Of the other 879, it rejects 97.7%, including every structural attack. 20 get through. This is bounded numerical validation, not a formal proof. Most of the 20 sit in branches that none of the sampled inputs reached.

措辭注意：分母一定說 879。

### 16 · A scope signal, not more answers（約 40 秒）

指：左卡 88.9%，再指右卡三條 bar。

> Outside the library, we ran two separate tests. On the left, we rebuilt the library from 50 formula families, so 199 questions have no matching contract. VerifiQuant abstains on 89% of them. R-PAL abstains on half a percent. It is right 95% of the time from memory, but with no signal when it is wrong. So we are not more accurate there. What we offer is a scope signal. On the right, five finance-trained annotators labeled 80 natural queries blind. On the 46 that call for no number, the ordering holds, but only the gap to plain R-PAL is significant.

被問「庫外 silent wrong 有比較低嗎？」：*No, 4.0 versus 4.5%. Out of library we offer a scope signal, not safer numbers.*

措辭注意：說 “out-of-library within the same benchmark”，不要說 open-world generalization。

### 17 · Where the claim holds, and where it fails（約 25 秒）

指：四列由上往下。

> To be clear about the boundary. The claim holds for single-number, formula-grounded questions inside the library, with a correct contract. The silent wrongs that remain trace to the three LLM steps: selection, binding, and recalling a convention. We check schema and bounds, not where a value came from. And this is not a compliance claim. Traceable records are a design target.

被問「換大模型呢？」：*Gemini 2.5 Pro narrows the gap to 11.7 points, mostly in ambiguity.*

被問「擴充合約庫會怎樣？」：*As the library grows tenfold, in-library accuracy stays between 83 and 87%. Out-of-library silent wrongs rise from 2.2 to 4.8%, not monotonically, because near neighbors get more tempting.*

### 18 · Takeaway（約 20 秒）

指：標題，然後 QR。

> So: keep the spreadsheet, and let the LLM fill it in. It pays wherever an unflagged wrong number costs more than a declined question, as in valuation, reporting, and client-facing calculation. If coverage is all you need, R-PAL is the better choice. The live demo replays every trap from the paper. This QR code goes there, and this one goes to the paper.

---

## 收尾和交換聯絡方式

- 對方有興趣：*Can I send you the paper? What’s the best email?* 或請對方掃 18 頁中間的 Code & data QR。
- 對方是業界：*If you have a workflow where a wrong number is expensive, I’d love to hear what the checks would look like.*
- 對方在挑錯：先同意邊界（17 頁），再回到 10 和 13 的對照組，不要辯成 accuracy。

## 現場注意

- 平板橫放。手指點著數字講，每頁停一下讓對方看完再講。
- 被問到 slide 沒有的數字，先查 `poster/talk_scripts.md` 的「備用數字」表。
- 會場網路可能不穩，出發前從 deck 的 Share › Export 下載一份 PDF 放在平板上。
