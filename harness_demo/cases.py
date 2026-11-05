"""Demo cases for scripted mode.

The candidate programs here are hand-written to reproduce error types that LLMs
commonly make (each has a `bug` note shown only after verification). They are
not recorded model outputs. Live mode replaces them with real Gemini output.
"""

FV_INTENT_SLOTS = {
    "pmt": {"quote": "10 萬元", "value": 100000},
    "annual_rate": {"quote": "年利率 3%", "value": 0.03},
    "years": {"quote": "5 年", "value": 5},
    "freq": {"quote": "每年", "value": 1},
}

FV_CANDIDATES = [
    {"id": "A", "bug": None, "code": '''ASSUMPTIONS = {"timing": "end"}

def solve(pmt, annual_rate, years, freq):
    r = annual_rate / freq            # @spec annual_rate, freq, period_rate
    fv = 0.0
    for _ in range(years * freq):     # @spec years, freq
        fv = fv * (1 + r) + pmt       # @spec pmt, timing
    return fv
'''},
    {"id": "B", "bug": None, "code": '''ASSUMPTIONS = {"timing": "begin"}

def solve(pmt, annual_rate, years, freq):
    r = annual_rate / freq                     # @spec annual_rate, freq, period_rate
    n = years * freq                           # @spec years, freq
    if r == 0:
        return pmt * n
    ordinary = pmt * ((1 + r) ** n - 1) / r    # @spec pmt
    return ordinary * (1 + r)                  # @spec timing
'''},
    {"id": "C", "bug": "利率已經是 0.03，程式又除一次 100（百分比重複換算）。", "code": '''ASSUMPTIONS = {"timing": "end"}

def solve(pmt, annual_rate, years, freq):
    r = annual_rate / 100 / freq           # @spec annual_rate, freq, period_rate
    n = years * freq                       # @spec years, freq
    if r == 0:
        return pmt * n
    return pmt * ((1 + r) ** n - 1) / r    # @spec pmt, timing
'''},
    {"id": "D", "bug": "公式正確，但利率為 0 時除以零。題目的數字跑得出來，邊界輸入會崩潰。", "code": '''ASSUMPTIONS = {"timing": "end"}

def solve(pmt, annual_rate, years, freq):
    r = annual_rate / freq                     # @spec annual_rate, freq, period_rate
    n = years * freq                           # @spec years, freq
    return pmt * ((1 + r) ** n - 1) / r        # @spec pmt, timing
'''},
    {"id": "E", "bug": "迴圈少跑一期（range(1, n)），漏掉最後一筆存款。", "code": '''ASSUMPTIONS = {"timing": "end"}

def solve(pmt, annual_rate, years, freq):
    r = annual_rate / freq                 # @spec annual_rate, freq, period_rate
    n = years * freq                       # @spec years, freq
    fv = 0.0
    for t in range(1, n):
        fv += pmt * (1 + r) ** (n - t)     # @spec pmt, timing
    return fv
'''},
    {"id": "F", "bug": "宣稱是期末年金，實際算的是期初年金。程式的自我說明不可信，要用行為驗證。", "code": '''ASSUMPTIONS = {"timing": "end"}

def solve(pmt, annual_rate, years, freq):
    r = annual_rate / freq                     # @spec annual_rate, freq, period_rate
    n = years * freq                           # @spec years, freq
    total = 0.0
    for t in range(n):
        total += pmt * (1 + r) ** (n - t)      # @spec pmt, timing
    return total
'''},
]

LOAN_CANDIDATES = [
    {"id": "A", "bug": None, "code": '''ASSUMPTIONS = {}

def solve(principal, annual_rate, years):
    r = annual_rate / 12          # @spec annual_rate, monthly_rate
    n = years * 12                # @spec years, months
    if r == 0:
        return principal / n
    return principal * r / (1 - (1 + r) ** -n)   # @spec principal
'''},
    {"id": "B", "bug": "忘了把年利率除以 12，把 2.1% 當成月利率。", "code": '''ASSUMPTIONS = {}

def solve(principal, annual_rate, years):
    n = years * 12                                   # @spec years, months
    if annual_rate == 0:
        return principal / n
    return principal * annual_rate / (1 - (1 + annual_rate) ** -n)   # @spec principal, annual_rate
'''},
    {"id": "C", "bug": "期數用了年數 20，而不是 240 個月。", "code": '''ASSUMPTIONS = {}

def solve(principal, annual_rate, years):
    r = annual_rate / 12                            # @spec annual_rate, monthly_rate
    if r == 0:
        return principal / years
    return principal * r / (1 - (1 + r) ** -years)  # @spec principal, years
'''},
    {"id": "D", "bug": None, "code": '''ASSUMPTIONS = {}

def solve(principal, annual_rate, years):
    r = annual_rate / 12                       # @spec annual_rate, monthly_rate
    factor = 0.0
    for m in range(1, years * 12 + 1):        # @spec years, months
        factor += 1 / (1 + r) ** m
    return principal / factor                 # @spec principal
'''},
]

CAGR_CANDIDATES = [
    {"id": "A", "bug": None, "code": '''ASSUMPTIONS = {}

def solve(start_value, end_value, years):
    growth = end_value / start_value     # @spec start_value, end_value
    return growth ** (1 / years) - 1     # @spec years, compounding
'''},
    {"id": "B", "bug": "用總報酬除以年數（單利平均），不是複利年化。", "code": '''ASSUMPTIONS = {}

def solve(start_value, end_value, years):
    total_return = end_value / start_value - 1   # @spec start_value, end_value
    return total_return / years                  # @spec years, compounding
'''},
    {"id": "C", "bug": "算成連續複利報酬 ln(期末/期初)/年數，和 CAGR 的定義不同。", "code": '''import math
ASSUMPTIONS = {}

def solve(start_value, end_value, years):
    return math.log(end_value / start_value) / years   # @spec start_value, end_value, years, compounding
'''},
    {"id": "D", "bug": None, "code": '''import math
ASSUMPTIONS = {}

def solve(start_value, end_value, years):
    log_growth = math.log(end_value / start_value)   # @spec start_value, end_value
    return math.exp(log_growth / years) - 1          # @spec years, compounding
'''},
]

NPV_CANDIDATES = [
    {"id": "A", "bug": None, "code": '''ASSUMPTIONS = {}

def solve(initial, cash_flow, years, rate):
    pv = 0.0
    for t in range(1, years + 1):            # @spec years, flow_timing
        pv += cash_flow / (1 + rate) ** t    # @spec cash_flow, rate
    return pv - initial                      # @spec initial, t0
'''},
    {"id": "B", "bug": "Excel NPV() 的經典陷阱：把今天的投入也折現了一年。", "code": '''ASSUMPTIONS = {}

def solve(initial, cash_flow, years, rate):
    flows = [-initial] + [cash_flow] * years                            # @spec initial, cash_flow, years
    return sum(cf / (1 + rate) ** (t + 1) for t, cf in enumerate(flows))  # @spec rate, t0, flow_timing
'''},
    {"id": "C", "bug": "回收從 t = 0 開始算，第一筆回收沒有折現。", "code": '''ASSUMPTIONS = {}

def solve(initial, cash_flow, years, rate):
    pv = sum(cash_flow / (1 + rate) ** t for t in range(years))   # @spec cash_flow, rate, years, flow_timing
    return pv - initial                                           # @spec initial, t0
'''},
    {"id": "D", "bug": "沒有用使用者給的折現率，自己寫死 10%。", "code": '''ASSUMPTIONS = {}

def solve(initial, cash_flow, years, rate):
    r = 0.10                                                          # @spec rate
    pv = sum(cash_flow / (1 + r) ** t for t in range(1, years + 1))   # @spec cash_flow, years, flow_timing
    return pv - initial                                               # @spec initial, t0
'''},
]

INSPECTORS = {
    "annuity_fv": '''TIMING = "{timing}"

def solve(pmt, annual_rate, years, freq):
    # inspector: simulate the account balance period by period
    r = annual_rate / freq
    balance = 0.0
    for _ in range(years * freq):
        if TIMING == "begin":
            balance = (balance + pmt) * (1 + r)
        else:
            balance = balance * (1 + r) + pmt
    return balance
''',
    "loan_pmt": '''def solve(principal, annual_rate, years):
    # inspector: search for the payment that leaves a zero balance (bisection)
    r, months = annual_rate / 12, years * 12
    lo, hi = 0.0, principal
    for _ in range(200):
        mid = (lo + hi) / 2
        balance = principal
        for _ in range(months):
            balance = balance * (1 + r) - mid
        lo, hi = (mid, hi) if balance > 0 else (lo, mid)
    return (lo + hi) / 2
''',
    "cagr": '''def solve(start_value, end_value, years):
    # inspector: search for g with start * (1+g)^years == end (bisection)
    lo, hi = -0.99, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if start_value * (1 + mid) ** years < end_value else (lo, mid)
    return (lo + hi) / 2
''',
    "npv": '''def solve(initial, cash_flow, years, rate):
    # inspector: write out the cash-flow table, then discount each row
    table = [(0, -initial)] + [(t, cash_flow) for t in range(1, years + 1)]
    return sum(amount * (1 + rate) ** -t for t, amount in table)
''',
}

CASES = [
    {
        "id": "fv_ambiguous",
        "title": "定期定額存款",
        "tag": "歧義 → 追問",
        "question": "我每年存 10 萬元到帳戶，年利率 3%，連續存 5 年，5 年後帳戶裡會有多少錢？",
        "story": "題目沒說錢是年初還是年底存。兩種讀法都合理，答案差 1.6 萬元。harness 要分出「寫錯的程式」和「讀法不同的程式」，只對後者發問。",
        "intent": {"slots": FV_INTENT_SLOTS, "conventions": {"timing": None}},
        "candidates": FV_CANDIDATES,
        "baseline": {"kind": "illustration", "text": "直接套期末年金公式：100,000 × ((1.03)^5 − 1) / 0.03 ≈ 530,914 元。回答裡沒有提到期初或期末的假設。"},
    },
    {
        "id": "fv_explicit",
        "title": "定期定額（題目寫明年初）",
        "tag": "不必問就不問",
        "question": "我每年年初存 10 萬元到帳戶，年利率 3%，連續存 5 年，5 年後帳戶裡會有多少錢？",
        "story": "同一題，但寫明「每年年初」。約定已由原文決定，harness 不該再問；宣稱期末的候選直接被淘汰。",
        "intent": {"slots": FV_INTENT_SLOTS | {"freq": {"quote": "每年年初", "value": 1}}, "conventions": {"timing": {"quote": "每年年初", "value": "begin"}}},
        "candidates": FV_CANDIDATES,
        "baseline": {"kind": "illustration", "text": "多數情況會算對期初年金 ≈ 546,841 元，但沒有任何檢查證據可以留存。"},
    },
    {
        "id": "loan",
        "title": "房貸月付額",
        "tag": "bug 被不變量攔下",
        "question": "我向銀行貸款 1,000 萬元買房，年利率 2.1%，貸款 20 年，本息平均攤還，每月要繳多少？",
        "story": "沒有歧義，但 LLM 常把年利率當月利率、或把 20 年當 20 期。不變量「逐月攤還後餘額歸零」不需要標準答案就能抓到。",
        "intent": {"slots": {
            "principal": {"quote": "1,000 萬元", "value": 10000000},
            "annual_rate": {"quote": "2.1%", "value": 0.021},
            "years": {"quote": "20 年", "value": 20},
        }, "conventions": {}},
        "candidates": LOAN_CANDIDATES,
        "baseline": {"kind": "illustration", "text": "常見錯誤類型：把 2.1% 直接當月利率，算出每月約 21 萬元，比正確值大 4 倍以上，回答仍然語氣肯定。"},
    },
    {
        "id": "cagr",
        "title": "年化報酬率",
        "tag": "程式取代心算",
        "question": "一檔股票 5 年前買進價格 100 元，現在漲到 240 元，年化報酬率（CAGR）是多少？",
        "story": "2.4 開 5 次方根，模型心算會在 19.11%–19.16% 之間飄。改由程式算，再用「反推回 240 元」驗證，不需要標準答案。",
        "intent": {"slots": {
            "start_value": {"quote": "100 元", "value": 100},
            "end_value": {"quote": "240 元", "value": 240},
            "years": {"quote": "5 年", "value": 5},
        }, "conventions": {}},
        "candidates": CAGR_CANDIDATES,
        "baseline": {"kind": "measured", "text": "VerifiQuant 實驗 test-1443（同樣是 2.4^(1/5)，標準答案 19.14%）：CoT 答案在 19.11%–19.16% 之間擲骰子；Gemini Pro 第 2 輪算出 19.14%，第 3 輪又漂回 19.13%，連自己都重現不了。（來源：MEETING_BRIEF_advisor.md）"},
    },
    {
        "id": "npv_missing",
        "title": "專案值不值得投資",
        "tag": "缺資訊 → 不猜",
        "question": "公司考慮一個專案：今天投入 100 萬元，之後 3 年每年年底回收 40 萬元。這個專案值得投資嗎？",
        "story": "題目沒給折現率。LLM 的抽取結果自己補了 10%，但原文找不到出處，harness 不採用，改問使用者。折現率 8% 和 10% 會得到相反的決策。",
        "intent": {"slots": {
            "initial": {"quote": "100 萬元", "value": 1000000},
            "cash_flow": {"quote": "40 萬元", "value": 400000},
            "years": {"quote": "3 年", "value": 3},
            "rate": {"quote": None, "value": 0.10},
        }, "conventions": {}},
        "candidates": NPV_CANDIDATES,
        "baseline": {"kind": "illustration", "text": "自行假設折現率 10%：NPV ≈ −5,259 元，結論「不值得投資」。若公司的資金成本其實是 8%，NPV ≈ +30,839 元，結論相反。"},
    },
    {
        "id": "out_of_scope",
        "title": "預測明天股價",
        "tag": "超出範圍 → 拒答",
        "question": "台積電明天收盤會漲還是跌？請給我一個預測價格。",
        "story": "這不是有確定答案、可以寫成程式驗證的問題。卡片庫裡沒有對應的問題類型，harness 直接拒答，而不是硬湊一張卡。",
        "intent": None,
        "candidates": [],
        "baseline": {"kind": "illustration", "text": "LLM 通常會給出一段看似合理的分析與一個具體價格，語氣篤定。"},
    },
]

CASES_BY_ID = {c["id"]: c for c in CASES}
