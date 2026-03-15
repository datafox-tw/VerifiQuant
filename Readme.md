# “We verify before we compute. We compute only what we can verify.”



NSFCA — Neuro-Symbolic Financial Computation Agent

                 ┌──────────────────────────────────┐
                 │ Natural Language Query            │
                 └───────────────────┬──────────────┘
                                     ↓
                ┌─────────────────────────────────────┐
                │ Task Interpreter (LLM)               │
                │ → variables, formula, window, freq   │
                └───────────────────┬──────────────────┘
                                    ↓
       ┌────────────────────────────────────────────────────────┐
       │ Data Layer: PDF Parser + RAG over Financial Docs       │
       └───────────────────┬────────────────────────────────────┘
                           ↓
       ┌────────────────────────────────────────────────────────┐
       │ Symbolic Math Engine (SymPy) + Rule-based Formula Set  │
       │ → deterministic computation                             │
       └───────────────────┬────────────────────────────────────┘
                           ↓
       ┌────────────────────────────────────────────────────────┐
       │ Verification Layer                                      │
       │ → financial invariants, math invariants, consistency    │
       └───────────────────┬────────────────────────────────────┘
                           ↓
       ┌────────────────────────────────────────────────────────┐
       │ Confidence Scoring + Safe Refusal                      │
       └───────────────────┬────────────────────────────────────┘
                           ↓
       ┌────────────────────────────────────────────────────────┐
       │ Final Output (auditable pipeline)                      │
       │ → source tables, formula, symbolic trace, answer       │
       └────────────────────────────────────────────────────────┘


------

# Current Best: JPMorgan 2025 Multi-Agent Framework
1. 優點
- 多代理（Reflection, Insights, Query Refiner, Data Summarizer）
- 能產生可執行 code
- 多步推理

2. 缺點（研究機會）

- Pass@1 = 0.46 → 太低，不足以使用於真正的投信/資產管理
- 缺乏：Symbolic math 保障 correctness、Verification Layer、Confidence gating、Auditability pipeline
- LLM 仍會使用錯變數、忽略 missing values、用錯 futures contract（FinChain 同樣指出）

3. 結論：
## Multi-agent 進步了，但還遠遠不夠可靠。

------
你的系統會做到：
1. Natural Language → Formalized Task Plan
- Extract required variables
- Extract formula
- Extract scope / frequency / aggregation rules
2. PDF / 10-K Parser + RAG
- 只拉 relevant table / ratio / line item
- 提供 primary source for auditing
- 防止 hallucination
3. Symbolic Math & Rule-based Financial Engine
- 像是：Sharpe, Sortino, Convexity, Factor exposure
- All deterministic
- All verifiable
4. Verification Layer
- Range checks
- Dimensional analysis
- Financial invariants (balance, weight =1, etc.)
- Statistical invariants (e.g. std ≥ 0, corr ∈ [-1,1])

5. Confidence Scoring + Safe Refusal
- 低於 threshold → 不給答案 → escalate to human
- 像醫療 AI 的 safety gate
6. Fully Auditable Output
- Data provenance
- Formula
- Full symbolic computation trace
- Code (optional)
- Final numeric answer

# Research Questions

1. Can a hybrid neuro-symbolic agent achieve >90% correctness on deterministic financial analytics tasks?
2. Does verification reduce critical errors in financial computations?
3. Can confidence gating reduce hallucination-driven wrong answers to near-zero?
4. **Do portfolio managers feel comfortable using such a system?**


python verifiquant.py \
    --question """John Doe can invest $107641 in Amazon Web Services Expansion. The project will save $24450 per year for 4 years, after which it can be sold for $54788 but will incur an environmental cleanup cost of $28699. Given a discount rate of 7.58%, what is the NPV?""" \
    --domain "Investment Analysis" --topic "Net Present Value"

以上是１２月的
現在以下是3月的資料處理測試
 source .venv/bin/activate 
python3 preprocessing/dataset_case_to_fic_v2.py  \
  --input verifiquant_v2/data/unique_investment_backup_10npv.jsonl \
  --output verifiquant_v2/data/unique_investment_backup_10npv.fic_v2.json \
  --dump-stage1-core verifiquant_v2/data/unique_investment_backup_10npv.stage1_core.json


# 修正版題目擴展by llm就直接用原始題目進行擴展即可
python3 preprocessing/expand_cases_v2.py \
  --input verifiquant_v2/data/unique_investment_backup_10npv.jsonl \
  --output verifiquant_v2/data/unique_investment_backup_10npv.expanded_40.jsonl \
  --model gemini-2.5-flash

小修改之後重新生成 以下是prompt for codex
To better analyze the capabilities and limitations
of LRMs on difficult problems in our dataset, we
conduct a thorough and comprehensive error analysis. This analysis is based on 80 DeepSeek-R1
failure cases with PoT, with stratified sampling (20
Easy /20 Medium /40 Hard). We summarize four
types of error in the current LRMs on challenging domain-specific reasoning problems, some of
which involve compound errors. The detailed error
distribution is shown in Table 4. More details of
error cases are provided in Appendix B.
• Misunderstanding of Problem: The model incorrectly interprets the question and context due
to a lack of financial knowledge.
• Formula Application Errors: Owing to inexperience in financial reasoning, the model uses an
incorrect formula that does not correspond to the
specified conditions of the problem
• Numerical Extraction Errors: The model extracts incorrect variables, especially when processing structured tabular data, despite the fact
that the reasoning process and the selected formula are correct.
第一種是因為題目模糊導致的錯誤（應該要透過搜尋卡片，檢查卡片與問題之間關聯性是否足夠來進行回絕，ex. 我只問說「這專案會不會賺錢」但是不說用npv irr還是payback計算
第二種是因為題目公式不清楚或者是input不足所導致的錯誤，例如想要算npv卻沒有給年份
第三種是因為題目公式選擇正確但是抓出來的數字錯誤所以出現的數字詭異（可以靠改context達成，可以假設原本的context內容都沒有問題）

這段定義很清楚，而且和你現在的系統設計是對齊的。你可以用下面這版當正式實驗描述。

## 三類錯誤（對應你現在要做的）
1. **M: Misunderstanding of Problem（語意誤解）**
- 成因：題目意圖模糊，無法判斷該用哪個公式族。
- 例子：「這專案會不會賺錢？」（沒說 NPV/IRR/Payback）
- 系統預期行為：**Refusal**（拒答）+ 要求澄清問題意圖。
- 驗證重點：卡片檢索關聯不足時是否拒答，而不是硬選一張卡。

2. **F: Formula/Spec Mismatch（公式或規格缺失）**
- 成因：公式方向雖可能正確，但必要條件或輸入不完整。
- 例子：要算 NPV 但缺 `discount_rate`、`n`（或 period 定義）。
- 系統預期行為：**Error**（缺規格）+ 回傳缺失欄位清單（requested fields）。
- 驗證重點：是否能準確指出「缺什麼」，而不是亂補值。

3. **E: Numerical Extraction/Binding Error（數值抽取/綁定錯誤）**
- 成因：公式選對，但抽值錯、單位錯、欄位綁錯，導致數值荒謬。
- 例子：`8` 當 `0.08`、年/月頻率錯配、把 A 欄抓成 B 欄。
- 系統預期行為：**Alert**（異常警報）+ 顯示觸發規則（invariant/scale check）。
- 驗證重點：是否攔截「有數字但不合理」的情況。

---

你可以在論文/簡報中加一句總結：

**M 是「不知道該算什麼」；F 是「知道要算什麼但資料不夠」；E 是「資料看起來夠但綁錯或尺度錯」。**

如果你要，我可以再幫你把這三類轉成你 `expand_cases_v2` 的自動評測標註規則（expected_status / expected_diagnostic_type）對照表。

今天花了：10*2(生成結構）+10（錯誤的expand question)+10(修正expand question prompt)