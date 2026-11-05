"""Invariant cards: one card per kind of financial question.

A card does not hold "the" formula. It holds what must be known (slots and
conventions) and what must be true of any correct program (invariants). The
LLM writes the program; the card only judges it.

Invariant kinds
  pre       checked on the inputs alone (input contract, Design by Contract)
  post      a property of f at a chosen point (postcondition)
  meta      a relation between two runs of f (metamorphic relation)
  boundary  pins a convention; only applies when the candidate declares it

Every `check` is a Python expression evaluated in the sandbox with
  f       the candidate's solve()
  close   close(a, b, rel=1e-6)
  math    the math module
  result  f(**inputs)
  and every input slot as a bare name.
"""

from __future__ import annotations

CARDS = {
    "annuity_fv": {
        "id": "annuity_fv",
        "title": "定期定額終值（年金終值）",
        "topic": "貨幣時間價值",
        "keywords": ["每年存", "每月存", "每季存", "年初存", "年底存", "月初存", "月底存", "定期定額", "存款", "終值", "帳戶裡會有", "帳戶裡有"],
        "signature": "solve(pmt, annual_rate, years, freq)",
        "output": {"name": "fv", "label": "終值", "unit": "元"},
        "slots": [
            {"name": "pmt", "label": "每次存入金額", "kind": "money", "unit": "元"},
            {"name": "annual_rate", "label": "年利率", "kind": "rate", "unit": "小數"},
            {"name": "years", "label": "存款年數", "kind": "count", "unit": "年"},
            {"name": "freq", "label": "每年存入次數", "kind": "freq", "unit": "次/年"},
        ],
        "conventions": [
            {
                "name": "timing",
                "label": "存入時點",
                "question": "每期的錢是在期初存入，還是期末存入？",
                "options": [
                    {"value": "end", "label": "期末存入（一般年金）"},
                    {"value": "begin", "label": "期初存入（期初年金）"},
                ],
                "cues": {"begin": ["年初", "期初", "月初"], "end": ["年底", "年末", "期末", "月底"]},
                # how the two readings must relate if both programs are otherwise right
                "relation": {"from": "end", "to": "begin", "text": "期初版 = 期末版 × (1 + 每期利率)",
                             "check": "close(v_begin, v_end * (1 + annual_rate / freq))"},
            }
        ],
        "fixed": [
            {"name": "period_rate", "text": "每期利率 = 年利率 ÷ 每年存入次數（複利頻率與存入頻率相同）"},
        ],
        "invariants": [
            {"id": "pre_pmt", "kind": "pre", "text": "每次存入金額 > 0", "check": "pmt > 0", "source": "dbc"},
            {"id": "pre_rate", "kind": "pre", "text": "年利率是小數且 0 ≤ rate < 1（3% 要寫成 0.03）",
             "check": "0 <= annual_rate < 1", "source": "dbc"},
            {"id": "pre_years", "kind": "pre", "text": "年數是正整數", "check": "years == int(years) and years >= 1",
             "source": "dbc"},
            {"id": "pre_freq", "kind": "pre", "text": "每年存入次數是 1、2、4 或 12", "check": "freq in (1, 2, 4, 12)",
             "source": "dbc"},
            {"id": "zero_rate", "kind": "post", "text": "利率為 0 時，終值 = 每次金額 × 總次數",
             "check": "close(f(pmt=pmt, annual_rate=0, years=years, freq=freq), pmt * years * freq)",
             "source": "metamorphic"},
            {"id": "lower_bound", "kind": "post", "text": "利率 ≥ 0 時，終值 ≥ 存入本金總額",
             "check": "result >= pmt * years * freq - 1e-9", "source": "dbc"},
            {"id": "step_year", "kind": "post",
             "text": "多存一年：FV(年數+1) = FV(年數)·(1+每期利率)^次數 + FV(1 年)",
             "check": "close(f(pmt=pmt, annual_rate=annual_rate, years=years + 1, freq=freq), "
                      "result * (1 + annual_rate / freq) ** freq + f(pmt=pmt, annual_rate=annual_rate, years=1, freq=freq))",
             "source": "induction"},
            {"id": "linear_pmt", "kind": "meta", "text": "每次金額加倍，終值也剛好加倍",
             "check": "close(f(pmt=2 * pmt, annual_rate=annual_rate, years=years, freq=freq), 2 * result)",
             "source": "metamorphic"},
            {"id": "monotone_rate", "kind": "meta", "text": "利率調高，終值不會變少",
             "check": "f(pmt=pmt, annual_rate=annual_rate + 0.01, years=years, freq=freq) >= result - 1e-9",
             "source": "metamorphic"},
            {"id": "base_end", "kind": "boundary", "when": {"timing": "end"},
             "text": "（期末）只存 1 年：每筆滾到年底，最後一筆沒有利息",
             "check": "close(f(pmt=pmt, annual_rate=annual_rate, years=1, freq=freq), "
                      "sum(pmt * (1 + annual_rate / freq) ** k for k in range(freq)))",
             "source": "induction"},
            {"id": "base_begin", "kind": "boundary", "when": {"timing": "begin"},
             "text": "（期初）只存 1 年：每筆都至少滾一期利息",
             "check": "close(f(pmt=pmt, annual_rate=annual_rate, years=1, freq=freq), "
                      "sum(pmt * (1 + annual_rate / freq) ** k for k in range(1, freq + 1)))",
             "source": "induction"},
        ],
        "completeness_note": "「只存 1 年」的 base case + 「多存一年」的遞迴關係兩條同時成立，依數學歸納法，對所有年數的答案就被唯一決定。"
                             "也就是說，不需要標準答案（oracle），通過這兩條就等於算對。",
    },
    "loan_pmt": {
        "id": "loan_pmt",
        "title": "本息平均攤還月付額",
        "topic": "貸款",
        "keywords": ["貸款", "房貸", "月付", "每月要繳", "攤還", "車貸"],
        "signature": "solve(principal, annual_rate, years)",
        "output": {"name": "payment", "label": "每月應繳", "unit": "元"},
        "slots": [
            {"name": "principal", "label": "貸款本金", "kind": "money", "unit": "元"},
            {"name": "annual_rate", "label": "年利率", "kind": "rate", "unit": "小數"},
            {"name": "years", "label": "貸款年限", "kind": "count", "unit": "年"},
        ],
        "conventions": [],
        "fixed": [
            {"name": "monthly_rate", "text": "月利率 = 年利率 ÷ 12（台灣房貸以名目年利率報價）"},
            {"name": "months", "text": "繳款期數 = 年限 × 12，每月月底繳一次"},
        ],
        "invariants": [
            {"id": "pre_principal", "kind": "pre", "text": "本金 > 0", "check": "principal > 0", "source": "dbc"},
            {"id": "pre_rate", "kind": "pre", "text": "年利率是小數且 0 ≤ rate < 1",
             "check": "0 <= annual_rate < 1", "source": "dbc"},
            {"id": "pre_years", "kind": "pre", "text": "年限是正整數", "check": "years == int(years) and years >= 1",
             "source": "dbc"},
            {"id": "zero_rate", "kind": "post", "text": "利率為 0 時，月付 = 本金 ÷ 總月數",
             "check": "close(f(principal=principal, annual_rate=0, years=years), principal / (12 * years))",
             "source": "metamorphic"},
            {"id": "covers_interest", "kind": "post", "text": "月付一定大於第一個月的利息（否則永遠還不完）",
             "check": "result > principal * annual_rate / 12", "source": "dbc"},
            {"id": "amortizes", "kind": "post",
             "text": "照這個月付逐月攤還 12×年限 期，最後餘額必須剛好歸零",
             "check": "abs(amortize(principal, annual_rate / 12, 12 * years, result)) < 0.01 * principal / (12 * years) + 1e-6",
             "source": "simulation"},
            {"id": "linear_principal", "kind": "meta", "text": "本金加倍，月付也剛好加倍",
             "check": "close(f(principal=2 * principal, annual_rate=annual_rate, years=years), 2 * result)",
             "source": "metamorphic"},
            {"id": "longer_cheaper", "kind": "meta", "text": "年限拉長，月付不會變多",
             "check": "f(principal=principal, annual_rate=annual_rate, years=years + 5) <= result + 1e-9",
             "source": "metamorphic"},
        ],
        "completeness_note": "「逐月攤還後餘額歸零」這條本身就唯一決定了月付額，等於用模擬取代標準答案。",
    },
    "cagr": {
        "id": "cagr",
        "title": "年化報酬率（CAGR）",
        "topic": "投資報酬",
        "keywords": ["年化", "CAGR", "複合成長", "年均報酬", "漲到"],
        "signature": "solve(start_value, end_value, years)",
        "output": {"name": "cagr", "label": "年化報酬率", "unit": "小數"},
        "slots": [
            {"name": "start_value", "label": "期初價值", "kind": "money", "unit": "元"},
            {"name": "end_value", "label": "期末價值", "kind": "money", "unit": "元"},
            {"name": "years", "label": "持有年數", "kind": "count", "unit": "年"},
        ],
        "conventions": [],
        "fixed": [{"name": "compounding", "text": "CAGR 定義為每年複利一次的幾何平均成長率"}],
        "invariants": [
            {"id": "pre_start", "kind": "pre", "text": "期初價值 > 0", "check": "start_value > 0", "source": "dbc"},
            {"id": "pre_end", "kind": "pre", "text": "期末價值 > 0", "check": "end_value > 0", "source": "dbc"},
            {"id": "pre_years", "kind": "pre", "text": "年數 > 0", "check": "years > 0", "source": "dbc"},
            {"id": "round_trip", "kind": "post",
             "text": "反推回去：期初 × (1+CAGR)^年數 必須等於期末",
             "check": "close(start_value * (1 + result) ** years, end_value)", "source": "round_trip"},
            {"id": "below_simple", "kind": "post",
             "text": "有賺錢時，複利年化 ≤ 單利平均（總報酬 ÷ 年數）",
             "check": "end_value <= start_value or result <= (end_value / start_value - 1) / years + 1e-12",
             "source": "dbc"},
            {"id": "scale_free", "kind": "meta", "text": "期初與期末同乘 10 倍，CAGR 不變",
             "check": "close(f(start_value=10 * start_value, end_value=10 * end_value, years=years), result)",
             "source": "metamorphic"},
            {"id": "flat_zero", "kind": "meta", "text": "期末 = 期初時，CAGR = 0",
             "check": "abs(f(start_value=start_value, end_value=start_value, years=years)) < 1e-12",
             "source": "metamorphic"},
        ],
        "completeness_note": "「反推回期末值」這條唯一決定了 CAGR：只有一個 g 讓 (1+g)^n 等於期末 ÷ 期初。",
    },
    "npv": {
        "id": "npv",
        "title": "淨現值（NPV）投資決策",
        "topic": "資本預算",
        "keywords": ["值得投資", "淨現值", "NPV", "折現", "專案", "回收"],
        "signature": "solve(initial, cash_flow, years, rate)",
        "output": {"name": "npv", "label": "淨現值", "unit": "元"},
        "slots": [
            {"name": "initial", "label": "今天投入金額", "kind": "money", "unit": "元"},
            {"name": "cash_flow", "label": "每年回收金額", "kind": "money", "unit": "元"},
            {"name": "years", "label": "回收年數", "kind": "count", "unit": "年"},
            {"name": "rate", "label": "折現率", "kind": "rate", "unit": "小數"},
        ],
        "conventions": [],
        "fixed": [
            {"name": "t0", "text": "今天的投入發生在 t = 0，不折現"},
            {"name": "flow_timing", "text": "每年回收發生在第 1, 2, …, years 年年底"},
            {"name": "decision", "text": "決策規則：NPV > 0 才值得投資", "by": "harness"},
        ],
        "invariants": [
            {"id": "pre_initial", "kind": "pre", "text": "投入金額 ≥ 0", "check": "initial >= 0", "source": "dbc"},
            {"id": "pre_rate", "kind": "pre", "text": "折現率是小數且 0 ≤ rate < 1", "check": "0 <= rate < 1",
             "source": "dbc"},
            {"id": "pre_years", "kind": "pre", "text": "年數是正整數", "check": "years == int(years) and years >= 1",
             "source": "dbc"},
            {"id": "zero_rate", "kind": "post", "text": "折現率為 0 時，NPV = 回收總額 − 投入",
             "check": "close(f(initial=initial, cash_flow=cash_flow, years=years, rate=0), cash_flow * years - initial)",
             "source": "metamorphic"},
            {"id": "t0_undiscounted", "kind": "meta", "text": "今天多投入 1 元，NPV 剛好少 1 元（t=0 不折現）",
             "check": "close(f(initial=initial + 1, cash_flow=cash_flow, years=years, rate=rate), result - 1)",
             "source": "metamorphic"},
            {"id": "last_flow_timing", "kind": "meta",
             "text": "多回收一年，NPV 剛好增加 cash_flow ÷ (1+rate)^(years+1)",
             "check": "close(f(initial=initial, cash_flow=cash_flow, years=years + 1, rate=rate) - result, cash_flow / (1 + rate) ** (years + 1))",
             "source": "metamorphic"},
            {"id": "monotone_rate", "kind": "meta", "text": "折現率調高，NPV 不會變大",
             "check": "f(initial=initial, cash_flow=cash_flow, years=years, rate=rate + 0.01) <= result + 1e-9",
             "source": "metamorphic"},
            {"id": "uses_rate", "kind": "meta", "text": "折現率真的有被用到（換一個折現率，答案要跟著變）",
             "check": "abs(f(initial=initial, cash_flow=cash_flow, years=years, rate=rate + 0.02) - result) > 1e-9",
             "source": "metamorphic"},
        ],
        "completeness_note": "t=0 不折現 + 每多一年的增量 + 折現率為 0 的總和，三條合起來把 NPV 的每一項都釘死。",
    },
}


def get_card(card_id: str) -> dict:
    return CARDS[card_id]


def retrieve(question: str) -> tuple[str | None, list[dict]]:
    """Keyword retrieval. Returns (best card id or None, score table)."""
    table = []
    for cid, card in CARDS.items():
        hits = [k for k in card["keywords"] if k.lower() in question.lower()]
        table.append({"card": cid, "title": card["title"], "hits": hits, "score": len(hits)})
    table.sort(key=lambda r: -r["score"])
    best = table[0]
    if best["score"] < 2 or (len(table) > 1 and table[1]["score"] == best["score"]):
        return None, table
    return best["card"], table
