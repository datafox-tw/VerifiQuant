"""Concept map: each idea the harness uses, the real papers that discuss it, and where it lives in code.

PAPERS hold bibliographic facts and what each paper itself reports. CONCEPTS hold the link:
what that paper says about this one idea, how the harness does it, and where the two differ.

`read`    the paper appears in the user's own reading notes.
`checked` how the bibliographic facts were confirmed:
          "1029" arXiv / ACL Anthology / publisher pages checked on 2026-10-29
          "bib"  already verified in docs/references_icaif.bib (2026-07-12)
          "memo" primary text read earlier (memory notes, 2026-09-30)
          "none" from memory only; check before citing
`said` is paraphrased from the abstract or the user's notes, never quoted at length.
"""

PAPERS = {
    "financereasoning": {
        "short": "FinanceReasoning",
        "cite": "Tang, E, Ma, et al. ACL 2025 (Long), pp. 15721–15749",
        "links": [["ACL Anthology", "https://doi.org/10.18653/v1/2025.acl-long.766"], ["arXiv 2506.05828", "https://arxiv.org/abs/2506.05828"]],
        "read": True, "checked": "1029",
        "said": "2,238 題財金數值推理題，每題附可執行的 Python 解答；評估的是前提全部給定的完美資訊情境，歧義消除與主動澄清列為未來方向（你的筆記）。",
    },
    "astrogator": {
        "short": "Astrogator",
        "cite": "Councilman, Fu, Gupta, Wang, Grove, Wang & Adve. arXiv 2507.13290, 2025",
        "links": [["arXiv 2507.13290", "https://arxiv.org/abs/2507.13290"]],
        "read": True, "checked": "1029",
        "said": "以形式化查詢語言表達使用者意圖，驗證器檢查 LLM 產生的 Ansible 程式是否符合查詢；21 個任務中驗證了 83% 的正確程式、標出 92% 的錯誤程式。",
    },
    "cemri": {
        "short": "Why MAS Fail",
        "cite": "Cemri, Pan, Yang, et al. NeurIPS 2025 Datasets & Benchmarks (Spotlight)",
        "links": [["arXiv 2503.13657", "https://arxiv.org/abs/2503.13657"]],
        "read": True, "checked": "bib",
        "said": "整理多代理 LLM 系統的失敗模式；常見問題是代理之間以為彼此理解，對共享狀態的解讀卻不同（你的筆記）。",
    },
    "alce": {
        "short": "ALCE",
        "cite": "Gao, Yen, Yu & Chen. EMNLP 2023, pp. 6465–6488",
        "links": [["ACL Anthology", "https://doi.org/10.18653/v1/2023.emnlp-main.398"], ["arXiv 2305.14627", "https://arxiv.org/abs/2305.14627"]],
        "read": False, "checked": "1029",
        "said": "要求 LLM 生成附引用的答案，並自動評估引用品質；在 ELI5 上，最好的模型仍有約一半的答案缺乏完整的引用支持。",
    },
    "wen_abstention": {
        "short": "Abstention survey",
        "cite": "Wen, Yao, Feng, Xu, Tsvetkov, Howe & Wang. TACL 13 (2025), pp. 529–556",
        "links": [["ACL Anthology", "https://aclanthology.org/2025.tacl-1.26"], ["arXiv 2407.18418", "https://arxiv.org/abs/2407.18418"]],
        "read": False, "checked": "1029",
        "said": "從問題、模型、人類價值三個角度整理 LLM 拒答（abstention）的方法、資料集與評估指標。",
    },
    "selective": {
        "short": "Selective classification",
        "cite": "El-Yaniv & Wiener, JMLR 11 (2010) 1605–1641；Geifman & El-Yaniv, NeurIPS 2017",
        "links": [["JMLR 2010", "https://jmlr.org/papers/v11/el-yaniv10a.html"], ["arXiv 1705.08500", "https://arxiv.org/abs/1705.08500"]],
        "read": True, "checked": "bib",
        "said": "允許分類器在沒把握時不回答，以覆蓋率換取錯誤率（risk–coverage trade-off）。",
    },
    "conformal_abstention": {
        "short": "Conformal abstention",
        "cite": "Abbasi Yadkori et al. arXiv 2405.01563, 2024",
        "links": [["arXiv 2405.01563", "https://arxiv.org/abs/2405.01563"]],
        "read": True, "checked": "memo",
        "said": "用 conformal prediction 決定何時拒答，對幻覺率給出統計上的保證。",
    },
    "dbc": {
        "short": "Design by Contract",
        "cite": "Meyer. Applying \"Design by Contract\". IEEE Computer 25(10), 1992",
        "links": [["IEEE", "https://doi.org/10.1109/2.161279"]],
        "read": True, "checked": "none",
        "said": "以前置條件、後置條件、類別不變量明訂呼叫方與實作方的責任，違約即是錯誤。",
    },
    "nl2postcond": {
        "short": "nl2postcond",
        "cite": "Endres, Fakhoury, Chakraborty & Lahiri. FSE 2024",
        "links": [["arXiv 2310.01831", "https://arxiv.org/abs/2310.01831"]],
        "read": False, "checked": "1029",
        "said": "讓 LLM 把自然語言意圖轉成後置條件；GPT-4 在 EvalPlus 最多 96% 的題目產出正確後置條件，平均能分辨 81% 的錯誤程式，並抓到 Defects4J 的 64 個歷史 bug。提出後置條件的正確性與完整性指標。",
    },
    "metamorphic": {
        "short": "Metamorphic testing",
        "cite": "Chen, Cheung & Yiu, HKUST-CS98-01 (1998)；Segura, Fraser, Sánchez & Ruiz-Cortés, IEEE TSE 42(9) 2016",
        "links": [["Segura 2016 (accepted PDF)", "https://eprints.whiterose.ac.uk/110335/"], ["Chen 1998 (arXiv 重刊)", "https://arxiv.org/abs/2002.12543"]],
        "read": False, "checked": "1029",
        "said": "沒有標準答案（oracle problem）時，改檢查輸入之間的變換是否帶來預期的輸出關係；Segura 等人整理了 1998 年以來的研究與應用。",
    },
    "oracle_problem": {
        "short": "Oracle problem",
        "cite": "Barr, Harman, McMinn, Shahbaz & Yoo. IEEE TSE 41(5), 2015",
        "links": [],
        "read": False, "checked": "memo",
        "said": "綜述「怎麼知道程式輸出是對的」這個測試根本難題，以及各類替代 oracle（含變形關係、規格）。",
    },
    "pal": {
        "short": "PAL",
        "cite": "Gao, Madaan, Zhou, Alon, Liu, Yang, Callan & Neubig. ICML 2023",
        "links": [["arXiv 2211.10435", "https://arxiv.org/abs/2211.10435"]],
        "read": True, "checked": "bib",
        "said": "LLM 只負責把題目寫成程式，計算交給 Python 直譯器執行。",
    },
    "pot": {
        "short": "PoT",
        "cite": "Chen, Ma, Wang & Cohen. TMLR 2023",
        "links": [["arXiv 2211.12588", "https://arxiv.org/abs/2211.12588"]],
        "read": True, "checked": "bib",
        "said": "把推理和計算分開：模型寫程式表達計算步驟，再由直譯器算出數值，於財金與數學資料集上優於 CoT。",
    },
    "alphacode": {
        "short": "AlphaCode",
        "cite": "Li et al. Science 378(6624):1092–1097, 2022",
        "links": [["arXiv 2203.07814", "https://arxiv.org/abs/2203.07814"]],
        "read": False, "checked": "1029",
        "said": "大量抽樣候選程式，先過濾再分群，最多只提交 10 份；Codeforces 模擬平均排名前 54.3%。分群依據是程式在模型生成的輸入上的輸出行為（此細節依記憶，請核對原文）。",
    },
    "codet": {
        "short": "CodeT",
        "cite": "Chen, Zhang, Nguyen, Zan, Lin, Lou & Chen. ICLR 2023",
        "links": [["arXiv 2207.10397", "https://arxiv.org/abs/2207.10397"]],
        "read": False, "checked": "1029",
        "said": "用同一個模型生成測試，依「與測試一致」且「與其他候選一致」的雙重執行一致性挑選程式；HumanEval pass@1 提升到 65.8%。",
    },
    "clarifygpt": {
        "short": "ClarifyGPT",
        "cite": "Mu, Shi, et al. FSE 2024 (PACMSE, Article 103)",
        "links": [["arXiv 2310.10996", "https://arxiv.org/abs/2310.10996"]],
        "read": False, "checked": "1029",
        "said": "抽樣多份程式在生成的輸入上執行；輸出不一致就判定需求需要澄清，再比較分歧的程式產生澄清問題。GPT-4 在 MBPP-sanitized 的 Pass@1 由 70.96% 升至 80.80%。",
    },
    "selfdebug": {
        "short": "Self-Debugging",
        "cite": "Chen, Lin, Schärli & Zhou. ICLR 2024",
        "links": [["ICLR 2024", "https://iclr.cc/virtual/2024/poster/18880"], ["arXiv 2304.05128", "https://arxiv.org/abs/2304.05128"]],
        "read": False, "checked": "1029",
        "said": "讓模型看自己程式的執行結果再修正；有單元測試時最多提升 12%，且比單純多抽 10 份以上候選更有效率。",
    },
    "verafi": {
        "short": "VERAFI",
        "cite": "Akinfaderin & Subramanian (AWS). arXiv 2512.14744, 2025",
        "links": [["arXiv 2512.14744", "https://arxiv.org/abs/2512.14744"]],
        "read": True, "checked": "1029",
        "said": "把 GAAP / SEC 與數學驗證規則轉成 SMT-lib 政策，放進代理的上下文；FinanceBench 事實正確率由 52.4% 升到 94.7%，其中政策層本身貢獻 4.3 個百分點。你讀原文後的判斷：規則只放在 prompt 裡請模型「內部驗證」。",
    },
    "logic_lm": {
        "short": "Logic-LM / LINC / SatLM",
        "cite": "Pan et al., Findings EMNLP 2023；Olausson et al., EMNLP 2023；Ye et al., NeurIPS 2023",
        "links": [["Logic-LM", "https://arxiv.org/abs/2305.12295"], ["LINC", "https://arxiv.org/abs/2310.15164"], ["SatLM", "https://arxiv.org/abs/2305.09656"]],
        "read": True, "checked": "bib",
        "said": "把自然語言翻成 ASP、一階邏輯或 SMT 交給求解器。你的筆記：Logic-LM 在 AR-LSAT 僅 43.04%；LINC 的錯誤分析常見語法錯誤與語意偏移；SatLM 一個符號錯就 UNSAT 或崩潰。",
    },
    "formaljudge": {
        "short": "FormalJudge",
        "cite": "Zhou, Sheng, Lou, Yang & Fu. arXiv 2602.11136, 2026",
        "links": [["arXiv 2602.11136", "https://arxiv.org/abs/2602.11136"]],
        "read": True, "checked": "1029",
        "said": "LLM 當規格編譯器，把意圖拆成原子約束，再用 Dafny 與 Z3 檢查；作者指出自然語言轉形式規格是採用形式驗證的關鍵瓶頸。",
    },
    "turpin": {
        "short": "Turpin et al.",
        "cite": "Turpin, Michael, Perez & Bowman. NeurIPS 2023",
        "links": [["arXiv 2305.04388", "https://arxiv.org/abs/2305.04388"]],
        "read": True, "checked": "bib",
        "said": "在提示中加入偏誤特徵會左右模型答案，但 CoT 解釋完全不提這些特徵：解釋可以看似合理卻不忠實。",
    },
    "lanham": {
        "short": "Lanham et al.",
        "cite": "Lanham et al. arXiv 2307.13702, 2023",
        "links": [["arXiv 2307.13702", "https://arxiv.org/abs/2307.13702"]],
        "read": True, "checked": "bib",
        "said": "以截斷、插入錯誤等介入測量 CoT 對答案的因果影響；多數任務上模型越大，推理越不忠實（你的筆記）。",
    },
    "seek_evidence": {
        "short": "Learning to Seek Evidence",
        "cite": "Huang, Lin, Zhong & Liu (Fudan). arXiv 2511.01425, 2025",
        "links": [["arXiv 2511.01425", "https://arxiv.org/abs/2511.01425"]],
        "read": True, "checked": "1029",
        "said": "醫療影像代理只能從受限動作中選（你的筆記：取證、宣告、棄權、停止），以強化學習決定何時取外部證據；遮住它選用的證據後表現變差（ΔBrier +0.029），用來證明證據真的驅動決策。",
    },
    "airepr": {
        "short": "AIRepr",
        "cite": "Zeng, Jin, Wang, Zheng & Li. Findings of EMNLP 2025",
        "links": [["ACL Anthology", "https://aclanthology.org/2025.findings-emnlp.539"], ["arXiv 2502.16395", "https://arxiv.org/abs/2502.16395"]],
        "read": True, "checked": "1029",
        "said": "分析師 LLM 產出工作流程與程式，檢查員 LLM 只依工作流程重寫並比對結論；15 組分析師–檢查員、1,032 個任務，越可重現的工作流程越常是正確的。",
    },
    "selection_inference": {
        "short": "Selection-Inference",
        "cite": "Creswell, Shanahan & Higgins. ICLR 2023",
        "links": [["arXiv 2205.09712", "https://arxiv.org/abs/2205.09712"]],
        "read": True, "checked": "1029",
        "said": "把推理拆成「選事實」與「推論」兩個模組交替進行，推論模組只看被選出的事實；7B 模型 5-shot 在 10 個邏輯任務上比基線提升超過 100%，並勝過 280B 基線。",
    },
    "knight_leveson": {
        "short": "Knight & Leveson",
        "cite": "Knight & Leveson. IEEE TSE 12(1):96–109, 1986",
        "links": [["IEEE", "https://doi.org/10.1109/TSE.1986.6312924"]],
        "read": False, "checked": "1029",
        "said": "27 個依同一規格獨立開發的版本跑一百萬筆測試，同時出錯的次數遠多於「獨立失敗」假設的預期：N 版本程式的可靠度不能假設錯誤獨立。",
    },
    "vlp": {
        "short": "Verifiable Literate Programming",
        "cite": "Yuan, Lu, Wu, Jin & Wu (HKU). arXiv 2607.02333, 2026",
        "links": [["arXiv 2607.02333", "https://arxiv.org/abs/2607.02333"]],
        "read": True, "checked": "1029",
        "said": "以自然語言風格的文件層對應程式，LLM 標出可疑的文件行讓人審，再以 model checking 驗證使用者確認過的性質；作者的 bug 研究指出失敗多來自需求不完整與細微語意偏移。",
    },
    "verina": {
        "short": "VERINA",
        "cite": "Ye, Yan, He, Kasriel, Yang & Song. arXiv 2505.23135 (ICLR 2026 poster page lists it)",
        "links": [["arXiv 2505.23135", "https://arxiv.org/abs/2505.23135"], ["Dataset / code", "https://github.com/sunblaze-ucb/verina"]],
        "read": False, "checked": "1029",
        "said": "189 個 Lean 任務，分別評估程式、規格、證明生成與其組合。規格以對照標準規格的 soundness / completeness 評分，證明不出來時退回正例與反例測試。v2 摘要：o4-mini 程式正確率 61.4%、規格 51.0%、證明僅 3.6%；提供標準規格能提升程式生成。",
    },
    "eu_ai_act": {
        "short": "EU AI Act Art. 12",
        "cite": "Regulation (EU) 2024/1689, Article 12 (record-keeping)",
        "links": [["EUR-Lex", "https://eur-lex.europa.eu/eli/reg/2024/1689/oj"]],
        "read": True, "checked": "memo",
        "said": "高風險 AI 系統須在生命週期內自動記錄事件以利追溯；部署者須保存紀錄至少六個月（Art. 26(6)）。",
    },
    "sr26_2": {
        "short": "SR 26-2",
        "cite": "Federal Reserve / OCC / FDIC, SR 26-2 (2026-04-17)",
        "links": [["Fed PDF", "https://www.federalreserve.gov/supervisionreg/srletters/SR2602.pdf"]],
        "read": True, "checked": "memo",
        "said": "取代 SR 11-7；腳註 3 明言生成式與代理式 AI 不在此指引範圍內，由銀行自身治理決定控制方式。",
    },
}

# Each concept: where it lives in the harness, and what each linked paper says about exactly this idea.
CONCEPTS = [
    {
        "id": "scope_gate", "name": "範圍外就拒答", "steps": ["scope"],
        "where": "harness.py 步驟 1；cards.retrieve()",
        "here": "沒有對應的問題類型卡就直接拒答，不硬湊一張卡。拒答的依據是可檢查的條件（沒有卡），不是模型自評信心。",
        "links": [
            {"paper": "selective", "says": "選擇性預測的基本框架：用覆蓋率換錯誤率。", "gap": "依模型分數決定是否回答；這裡依「有沒有可驗證的卡」決定。"},
            {"paper": "wen_abstention", "says": "把拒答整理成問題、模型、價值三個面向；「問題本身無法回答」屬於問題面。", "gap": "survey 不提供具體機制；這裡是問題面拒答的一個可執行版本。"},
        ],
    },
    {
        "id": "intent_spec", "name": "意圖規格（中介層）", "steps": ["intent"],
        "where": "harness.py _bind_slots() / _bind_conventions()；cards.py 的 slots、conventions",
        "here": "問題先被凍結成欄位、約定與函式簽名，之後所有元件只讀這份規格。",
        "links": [
            {"paper": "astrogator", "says": "在自然語言與程式之間加一層使用者讀得懂的形式化查詢語言，並把程式自行補上的部分攤給使用者。", "gap": "Astrogator 針對 Ansible；這裡把查詢語言換成財金欄位與約定。"},
            {"paper": "cemri", "says": "多代理系統常因各方對共享狀態解讀不同而靜默出錯。", "gap": "這裡只有一份凍結規格，生成者與檢查員讀同一份。"},
        ],
    },
    {
        "id": "quote_grounding", "name": "數值要有原文出處", "steps": ["intent"],
        "where": "harness.py _bind_slots()；units.parse()",
        "here": "LLM 只能指出「原文哪幾個字是這個欄位」。引用必須逐字出現在原文，數值由程式從引用重新解析，對不上就不採用。",
        "links": [
            {"paper": "alce", "says": "要求生成內容附引用並評估引用是否真的支持內容；最好的模型仍常缺乏完整支持。", "gap": "ALCE 用模型判斷引用是否支持；這裡是逐字比對加確定性解析，數值錯一位就退回。"},
            {"paper": "financereasoning", "says": "誤讀題目數值（取數錯）是 LRM 的主要錯誤類型之一。", "gap": "FinanceReasoning 只做錯誤分類；這裡在計算前就把取數錯擋下。"},
        ],
    },
    {
        "id": "ask_not_guess", "name": "缺資訊就問，不猜", "steps": ["intent", "precheck"],
        "where": "harness.py 步驟 2、4 的 pending = slots",
        "here": "必要欄位沒有原文出處（包含 LLM 自己補的值）就停下來問使用者；補上的值在稽核紀錄標為「使用者提供」。",
        "links": [
            {"paper": "financereasoning", "says": "評估假設前提完整，歧義消除與主動澄清列為範圍外。", "gap": "這裡處理的正是被排除的那一塊。"},
            {"paper": "wen_abstention", "says": "把「資訊不足的問題」列為拒答的正當理由之一。", "gap": "這裡不只拒答，而是指名缺哪個欄位。"},
        ],
    },
    {
        "id": "contracts", "name": "前置／後置條件（合約）", "steps": ["contract", "precheck", "verify"],
        "where": "cards.py invariants（kind = pre / post）；sandbox.eval_pre()",
        "here": "每種問題類型的合約由人寫一次、重複使用；在模型外面真的執行，而不是請模型自己遵守。",
        "links": [
            {"paper": "dbc", "says": "以前置、後置條件明訂責任，違約即錯誤。", "gap": "原本由寫程式的人寫合約；這裡由卡片作者寫，程式由 LLM 寫。"},
            {"paper": "nl2postcond", "says": "LLM 可以從自然語言產生後置條件，用來分辨錯誤程式。", "gap": "LLM 產生的條件可能和程式犯同一個錯，所以這裡改由專家撰寫。"},
        ],
    },
    {
        "id": "metamorphic", "name": "變形關係（不需要標準答案）", "steps": ["contract", "verify"],
        "where": "cards.py invariants（kind = meta），例如本金加倍、利率歸零、多存一年",
        "here": "不問「答案是多少」，而問「輸入這樣變，輸出應該怎樣變」。全部在沙箱裡對候選程式實際執行。",
        "links": [
            {"paper": "metamorphic", "says": "在沒有 oracle 時，用輸入變換與輸出關係來測試程式。", "gap": "原本用於軟體測試；這裡把財金恆等式（線性、單調、歸零）寫成變形關係。"},
            {"paper": "oracle_problem", "says": "測試的根本困難是不知道正確輸出；變形關係是替代 oracle 之一。", "gap": "這裡正好處於沒有 ground truth 的情境。"},
        ],
    },
    {
        "id": "induction_complete", "name": "base case + 遞迴 = 唯一決定答案", "steps": ["contract", "verify"], "own": True,
        "where": "cards.py：annuity_fv 的 base_end / base_begin + step_year；loan_pmt 的 amortizes；cagr 的 round_trip",
        "here": "選一組強到能唯一決定答案的不變量。通過它們就等於算對，不需要標準答案。這是本系統自己的設計，下列文獻只提供相關的「完整性」概念。",
        "links": [
            {"paper": "nl2postcond", "says": "定義後置條件的「完整性」：能分辨多少錯誤程式。", "gap": "這裡追求的是數學上唯一決定，而非統計上分辨率高。"},
        ],
    },
    {
        "id": "program_aided", "name": "數字由程式算，不由模型心算", "steps": ["generate"],
        "where": "harness.py 步驟 5；sandbox.run()",
        "here": "最終答案一律來自在沙箱執行的程式，模型不輸出任何數字。",
        "links": [
            {"paper": "pal", "says": "LLM 寫程式，直譯器算答案。", "gap": "PAL 不檢查程式是否讀對題意；這裡在後面加上不變量與澄清。"},
            {"paper": "pot", "says": "把推理與計算分開，數值推理優於 CoT。", "gap": "同上：解決算術錯誤，不解決語意偏移（期初當期末）。"},
            {"paper": "financereasoning", "says": "財金數值題的最佳設定使用 PoT。", "gap": "仍以單一數字準確率評估。"},
        ],
    },
    {
        "id": "sample_cluster", "name": "多候選，依執行行為分群", "steps": ["generate"],
        "where": "harness.py _clusters()；LiveProvider.candidates()（溫度 1.0 抽 4 份）",
        "here": "抽多份程式在同一組輸入上執行，答案相同的歸成一群。",
        "links": [
            {"paper": "alphacode", "says": "大量抽樣後依程式行為分群，從各群挑代表提交。", "gap": "AlphaCode 用分群挑「最可能對的」；這裡用分群偵測「有沒有分歧」。"},
            {"paper": "codet", "says": "以候選之間與生成測試之間的執行一致性挑選程式。", "gap": "一致不等於正確：大家一起錯時 CodeT 會選錯，所以這裡分群之後還要過不變量。"},
        ],
    },
    {
        "id": "exec_feedback", "name": "帶著執行失敗回饋重新生成", "steps": ["generate", "ambiguity"],
        "where": "harness.py 步驟 5 的 feedback；步驟 7 約定不符時的重新生成",
        "here": "第一輪全部失敗時，把「哪條不變量沒過」交回模型重寫一次；只重試一次，仍失敗就拒答。",
        "links": [
            {"paper": "selfdebug", "says": "讓模型看執行結果自我修正，比單純多抽候選更有效率。", "gap": "Self-Debugging 的回饋是測試結果；這裡的回饋是卡片不變量，且重試次數有上限。"},
        ],
    },
    {
        "id": "bug_vs_ambiguity", "name": "先淘汰 bug，剩下的分歧才是歧義", "steps": ["verify", "ambiguity"], "own": True,
        "where": "harness.py 步驟 6–7：只有通過不變量的候選才參與歧義判定",
        "here": "分歧可能來自寫錯，也可能來自讀法不同。先用不變量淘汰寫錯的，剩下且能用約定關係解釋的分歧才拿去問使用者。",
        "links": [
            {"paper": "clarifygpt", "says": "輸出不一致即判定需求有歧義，產生澄清問題。", "gap": "ClarifyGPT 不分 bug 與歧義：一份寫錯的程式也會觸發提問。"},
            {"paper": "codet", "says": "以一致性挑選程式。", "gap": "同上，一致性本身不區分錯誤與讀法。"},
        ],
    },
    {
        "id": "ask_on_convention", "name": "約定沒寫明就問，即使候選全部一致", "steps": ["ambiguity"],
        "where": "harness.py 步驟 7：卡片列出的 convention 只要原文沒決定就問",
        "here": "即時模式實測：4 份 Gemini 程式全都默默選了「期末」，答案一致；只看分歧的方法不會問，卡片仍會問。",
        "links": [
            {"paper": "clarifygpt", "says": "以候選分歧作為歧義訊號。", "gap": "所有候選一起猜同一個約定時偵測不到。"},
            {"paper": "astrogator", "says": "驗證器要把程式自行補上、查詢沒說的部分（p ∖ s）交給使用者確認。", "gap": "這裡由卡片事先列出這類部分，而非從程式反推。"},
        ],
    },
    {
        "id": "rules_outside_model", "name": "規則在模型外執行，不放進 prompt", "steps": ["verify"],
        "where": "_runner.py 逐條 eval 不變量；候選程式完全看不到不變量",
        "here": "不變量從不出現在生成 prompt 裡，只在模型外面對程式執行。",
        "links": [
            {"paper": "verafi", "says": "把驗證規則（SMT-lib）放進上下文，請模型內部驗證。", "gap": "規則在 prompt 裡是軟約束，模型可以不遵守；這裡是硬閘門。"},
            {"paper": "logic_lm", "says": "把自然語言翻成邏輯交給求解器。", "gap": "翻譯錯一個符號就失敗；這裡不在執行時翻譯，不變量事先寫成可執行的數值檢查。"},
            {"paper": "formaljudge", "says": "自然語言轉形式規格是形式驗證落地的關鍵瓶頸。", "gap": "這裡用「每種問題類型寫一次」繞開逐題翻譯。"},
        ],
    },
    {
        "id": "claim_vs_behavior", "name": "程式的自我宣稱要用行為驗證", "steps": ["verify", "trace"],
        "where": "harness.py _run_candidates()：declared ASSUMPTIONS 對照 boundary 不變量推出的 behaves_as",
        "here": "程式宣告自己用哪個約定，但只有約定釘子（boundary 不變量）通過才算數。示範中候選 F 宣稱期末、實際算期初，被淘汰。",
        "links": [
            {"paper": "turpin", "says": "模型的解釋可以看似合理卻不反映真正影響答案的因素。", "gap": "那是 CoT 文字；這裡把「自述」縮成一個可測的宣告，再用執行驗證。"},
            {"paper": "lanham", "says": "用介入實驗測量推理文字對答案的因果影響。", "gap": "同樣的思路：不信自述，改做介入；這裡的介入是改輸入看輸出。"},
            {"paper": "seek_evidence", "says": "遮住代理宣稱使用的證據，看表現是否變差，以檢驗證據是否真的驅動決策。", "gap": "他們在模型層面做因果檢驗；這裡在程式層面做，而且是確定性的。"},
        ],
    },
    {
        "id": "independent_check", "name": "只看規格的獨立重算", "steps": ["inspect"],
        "where": "harness.py 步驟 8；cases.INSPECTORS / LiveProvider.inspector()",
        "here": "檢查員看不到原題與候選程式，只拿凍結規格，用不同方法（模擬、二分搜尋）重算。",
        "links": [
            {"paper": "airepr", "says": "檢查員只依工作流程重寫程式，結論一致才交給人審。", "gap": "AIRepr 以重現性為主要訊號；這裡放在不變量之後當第二道。"},
            {"paper": "selection_inference", "says": "推論模組看不到原題，只看選出的事實，迫使推理可追溯。", "gap": "這裡的「選出的事實」就是凍結規格。"},
            {"paper": "knight_leveson", "says": "獨立開發的版本仍會在同一處一起出錯。", "gap": "所以獨立重算一致不能當作正確的證明，只是額外一道檢查。"},
        ],
    },
    {
        "id": "trace_links", "name": "規格項目 ↔ 程式行追溯", "steps": ["trace"],
        "where": "harness.py _spec_tags()；候選程式的 # @spec 註記",
        "here": "每個欄位、約定、固定假設都要連到實作它的程式行，沒被實作的會標紅。",
        "links": [
            {"paper": "vlp", "says": "以自然語言文件層對應程式，標出可疑的對應讓人審，再驗證使用者確認的性質。", "gap": "VLP 的對應由 LLM 判斷是否吻合；這裡的對應是程式自己的宣告，數值部分已由不變量驗證。"},
        ],
    },
    {
        "id": "three_exits", "name": "只有三種出口：輸出／追問／拒答", "steps": ["decision"],
        "where": "harness.py finish()：action ∈ {EMIT, ASK, ABSTAIN}",
        "here": "harness 不能自由回話，只能從三個動作中選一個，且每個動作都有可檢查的觸發條件。",
        "links": [
            {"paper": "seek_evidence", "says": "代理只能從受限動作中選，推理成為可稽核的動作序列。", "gap": "他們的觸發依學到的策略與機率；這裡依確定性條件。"},
            {"paper": "conformal_abstention", "says": "以統計方法決定何時拒答並保證幻覺率。", "gap": "這裡沒有統計保證，拒答條件是規則；兩者可以互補。"},
        ],
    },
    {
        "id": "audit_log", "name": "稽核紀錄只記執行過的東西", "steps": ["decision"],
        "where": "harness.py audit：輸入與來源、約定與決定者、程式 sha256、各不變量結果、淘汰理由",
        "here": "紀錄裡沒有模型的自我解釋，只有被執行、可重跑的東西。",
        "links": [
            {"paper": "turpin", "says": "CoT 解釋可能不忠實。", "gap": "所以 CoT 不能當稽核紀錄。"},
            {"paper": "eu_ai_act", "says": "高風險 AI 系統須自動記錄事件以利追溯。", "gap": "法規只要求記錄；這裡示範記什麼才可重現。"},
            {"paper": "sr26_2", "says": "生成式與代理式 AI 不在模型風險指引範圍內，交由銀行自行治理。", "gap": "這留下了一個空白，正是這類 harness 想填的位置。"},
        ],
    },
]

STEP_CONCEPTS = {}
for _c in CONCEPTS:
    for _s in _c["steps"]:
        STEP_CONCEPTS.setdefault(_s, []).append(_c["id"])
