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
