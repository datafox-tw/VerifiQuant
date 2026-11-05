"""Single-paper lenses: apply one paper's mechanism alone to the finance cases.

Each lens strips the harness down to what one paper does, runs it on the same scripted
candidates (real execution in the sandbox), and classifies the outcome. The bug labels in
cases.py are used only here, to score outcomes; the harness itself never reads them.

Outcome codes
  correct    emits an answer a correct program gives for the reading the question states
  guessed    emits a correct program's answer, but a convention or an input was assumed silently
  wrong      emits a buggy program's answer, or the wrong reading of a stated convention
  ask        asks the user, and every option offered is a legitimate reading or a missing input
  ask_noisy  asks, but offers options produced by bugs, or asks when nothing was ambiguous
  abstain    refuses
  noscope    has no scope gate, so it would still answer a question with no checkable answer
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

from . import harness, sandbox
from .cards import get_card, retrieve
from .cases import CASES, INSPECTORS
from .harness import _close, _clusters, _run_candidates
from .providers import ScriptedProvider

LENSES = [
    {
        "id": "verina", "papers": ["verina"], "title": "VERINA",
        "idea": "把可驗證程式生成拆成程式、規格、證明三件事分別評分。規格（前置與後置條件）以 soundness 與 completeness 對照標準規格來評：後置條件 sound 指它接受的輸出都是對的，complete 指所有對的輸出它都接受。證明失敗時退回用正例與反例測試，測試通過只代表「可能成立」。",
        "apply": "把每張卡片的不變量當成「待評分的規格」。正例：標註為正確的程式，在多組合法輸入上執行，卡片必須全部接受（completeness）。反例：標註的 bug 程式，加上把正確答案乘 1.001 或加一點點的變異程式，卡片必須至少在一組輸入上拒絕（soundness）。前置條件同理：合法輸入要放行，百分比寫成 3、年數 2.5 這類輸入要擋下。",
        "changes": [
            "上線前先給每張卡打分：卡片本身也要通過正例與反例，等於對檢查器做負控制。",
            "VERINA 發現把標準規格放進生成提示能提升程式正確率。可以把不變量也交給生成器，同時照樣在模型外執行，這和 VERAFI 只放進提示不同。",
            "VERINA 的證明成功率低於 5%。浮點數的財金公式更難在 Lean 裡證明，所以這裡和 VERINA 的退路一樣靠測試，結論是「在這些輸入上成立」，不是定理。",
        ],
    },
    {
        "id": "pal", "papers": ["pal", "pot"], "title": "PAL / PoT",
        "idea": "LLM 只寫程式，直譯器負責計算，模型不心算。",
        "apply": "抽到哪一份程式就執行哪一份，直接輸出結果。不看規格出處、不跑不變量、不問使用者。下面列出「如果抽到的是這一份」各會得到什麼。",
        "changes": ["保留：數字一律由程式算。", "補上：PAL 對題意誤讀與邊界錯誤沒有防護，抽到錯的程式就會安靜地輸出錯的數字。"],
    },
    {
        "id": "codet", "papers": ["codet", "alphacode"], "title": "CodeT / AlphaCode",
        "idea": "抽多份程式，在同一組輸入上執行，依輸出分群，從最大的群挑答案。",
        "apply": "在題目輸入上執行所有候選，取人數最多的那一群的答案。平手時依抽樣順序取第一群。",
        "changes": ["保留：多候選與依行為分群。", "補上：多數不等於正確。錯誤程式也可能和正確程式在題目輸入上給出相同數字（例如只在利率為 0 才會崩潰的程式）。"],
    },
    {
        "id": "clarifygpt", "papers": ["clarifygpt"], "title": "ClarifyGPT",
        "idea": "候選程式輸出不一致就判定需求有歧義，把分歧整理成澄清問題問使用者；一致就直接回答。",
        "apply": "在題目輸入上執行所有候選。只要出現兩種以上答案，就把每一種答案都當成一個選項拿去問使用者。",
        "changes": ["保留：用分歧當作歧義的證據，並把後果攤給使用者看。", "補上：先用不變量淘汰 bug，否則使用者會被要求在寫錯的答案之間做選擇；沒有歧義的題目也會被問。"],
    },
    {
        "id": "airepr", "papers": ["airepr"], "title": "AIRepr",
        "idea": "分析師交出工作流程與程式，檢查員只依工作流程重寫程式；結論能重現才接受。",
        "apply": "每份候選都交給檢查員：檢查員依這份候選宣告的約定，從規格重算一次。題目輸入上的答案一致就接受，不一致就退回。",
        "changes": ["保留：只看規格的獨立重算。", "補上：重現只在一組輸入上比對，只在邊界輸入才出錯的程式會被接受；檢查員沿用分析師的假設時，猜錯的輸入也會一起被重現。"],
    },
]

OUTCOME_LABEL = {
    "correct": "正確", "guessed": "自行假設", "wrong": "安靜地錯", "crash": "程式崩潰",
    "ask": "追問", "ask_noisy": "追問含雜訊", "abstain": "拒答", "noscope": "照樣作答",
}

# Input grids for the VERINA lens. The first row of each is the question's own input.
GRIDS = {
    "annuity_fv": [
        {"pmt": 100000, "annual_rate": 0.03, "years": 5, "freq": 1},
        {"pmt": 5000, "annual_rate": 0.02, "years": 10, "freq": 12},
        {"pmt": 2000, "annual_rate": 0.05, "years": 3, "freq": 4},
    ],
    "loan_pmt": [
        {"principal": 10000000, "annual_rate": 0.021, "years": 20},
        {"principal": 5000000, "annual_rate": 0.025, "years": 30},
        {"principal": 1000000, "annual_rate": 0.0, "years": 10},
    ],
    "cagr": [
        {"start_value": 100, "end_value": 240, "years": 5},
        {"start_value": 50, "end_value": 40, "years": 3},
        {"start_value": 1, "end_value": 2, "years": 1},
    ],
    "npv": [
        {"initial": 1000000, "cash_flow": 400000, "years": 3, "rate": 0.08},
        {"initial": 2000000, "cash_flow": 500000, "years": 5, "rate": 0.06},
        {"initial": 100000, "cash_flow": 10000, "years": 20, "rate": 0.0},
    ],
}

# Invalid inputs a precondition must reject (one bad field each, on top of the first grid row).
BAD_INPUTS = {
    "annuity_fv": [("利率寫成 3 而非 0.03", {"annual_rate": 3}), ("存款金額為負", {"pmt": -100000}),
                   ("年數 2.5", {"years": 2.5}), ("每年存 7 次", {"freq": 7})],
    "loan_pmt": [("利率寫成 2.1", {"annual_rate": 2.1}), ("本金為 0", {"principal": 0}), ("年數 0", {"years": 0})],
    "cagr": [("期初價值為 0", {"start_value": 0}), ("期末價值為負", {"end_value": -10}), ("年數 0", {"years": 0})],
    "npv": [("折現率寫成 8", {"rate": 8}), ("投入為負", {"initial": -1}), ("年數 2.5", {"years": 2.5})],
}

CASE_FOR_CARD = {"annuity_fv": "fv_ambiguous", "loan_pmt": "loan", "cagr": "cagr", "npv": "npv_missing"}


def _mutant(code: str, kind: str, rate_output: bool) -> str:
    body = {"scale": "_orig_solve(**kw) * 1.001",
            "offset": "_orig_solve(**kw) + (0.0001 if RATE_OUTPUT else 1.0)",
            "round": "round(_orig_solve(**kw), 4 if RATE_OUTPUT else 0)"}[kind]
    return (code + f"\nRATE_OUTPUT = {rate_output}\n_orig_solve = solve\n\n"
            f"def solve(**kw):\n    return {body}\n")


# ---------------------------------------------------------------- per-case setup

def _setup(case):
    card_id, _ = retrieve(case["question"])
    if not card_id:
        return None
    card = get_card(card_id)
    slots = case["intent"]["slots"]
    inputs = {k: v["value"] for k, v in slots.items()}           # LLM-claimed values, no grounding check
    ungrounded = [k for k, v in slots.items() if not v.get("quote") or v["quote"] not in case["question"]]
    stated = {k: v["value"] for k, v in (case["intent"].get("conventions") or {}).items() if v}
    rows = _run_candidates(card, case["candidates"], inputs)
    correct = [r for r in rows if r["bug"] is None]
    if stated:
        good = [r for r in correct if all(r["declared"].get(k) == v for k, v in stated.items())]
    else:
        good = correct
    guess_flag = bool(ungrounded) or bool([c for c in card["conventions"] if c["name"] not in stated])
    return {"card": card, "inputs": inputs, "ungrounded": ungrounded, "stated": stated, "rows": rows,
            "good_vals": [r["result"] for r in good], "valid_vals": [r["result"] for r in correct],
            "guess_flag": guess_flag}


def _classify(s, value):
    if value is None:
        return "crash"
    if any(_close(value, g) for g in s["good_vals"]):
        return "guessed" if s["guess_flag"] else "correct"
    return "wrong"


def _why(s):
    bits = []
    if s["ungrounded"]:
        bits.append("沿用 LLM 自行補上的 " + "、".join(s["ungrounded"]))
    open_c = [c["label"] for c in s["card"]["conventions"] if c["name"] not in s["stated"]]
    if open_c:
        bits.append("原文未決定的" + "、".join(open_c) + "被默默選定")
    return "；".join(bits)


# ---------------------------------------------------------------- lens runners

def _lens_pal(s):
    picks = []
    for r in s["rows"]:
        out = _classify(s, r["result"]) if r["ok"] else "crash"
        picks.append({"id": r["id"], "value": r["result"], "outcome": out, "bug": r["bug"]})
    counts = {}
    for p in picks:
        counts[p["outcome"]] = counts.get(p["outcome"], 0) + 1
    worst = "wrong" if counts.get("wrong") else ("guessed" if counts.get("guessed") else "correct")
    wrong = [p["id"] for p in picks if p["outcome"] in ("wrong", "crash")]
    lucky = [p["id"] for p in picks if p["bug"] and p["outcome"] not in ("wrong", "crash")]
    msg = f"{len(picks)} 份候選中，抽到 {'、'.join(wrong) or '任何一份都不'} 會安靜地輸出錯的數字"
    if lucky:
        msg += f"；{', '.join(lucky)} 是 bug，但在這組輸入上數字剛好對"
    return {"outcome": worst, "detail": picks, "counts": counts, "summary": msg + "。"}


def _lens_codet(s):
    cl = _clusters([r for r in s["rows"] if r["ok"]])
    top = max(len(c["members"]) for c in cl)
    leaders = [c for c in cl if len(c["members"]) == top]
    pick = leaders[0]
    out = _classify(s, pick["value"])
    bugs_in = [m for m in pick["members"] if next(r for r in s["rows"] if r["id"] == m)["bug"]]
    note = f"最大群有 {top} 份：{', '.join(pick['members'])}"
    if len(leaders) > 1:
        note += f"；和 {', '.join('/'.join(c['members']) for c in leaders[1:])} 平手，依抽樣順序取第一群"
    if bugs_in:
        note += f"；群內的 {', '.join(bugs_in)} 其實是 bug，只是在這組輸入上剛好答案相同"
    return {"outcome": out, "value": pick["value"], "clusters": cl, "summary": note + "。"}


def _lens_clarify(s):
    cl = _clusters([r for r in s["rows"] if r["ok"]])
    if len(cl) == 1:
        return {"outcome": _classify(s, cl[0]["value"]), "value": cl[0]["value"], "clusters": cl,
                "summary": "所有候選答案一致，直接回答。"}
    bogus = [c for c in cl if not any(_close(c["value"], v) for v in s["valid_vals"])]
    readings = len(cl) - len(bogus)
    noisy = bool(bogus) or readings < 2
    msg = f"出現 {len(cl)} 種答案，全部當成選項問使用者"
    if bogus:
        msg += f"；其中 {len(bogus)} 個選項來自寫錯的程式"
    if readings < 2:
        msg += "；這題其實沒有歧義"
    if s["ungrounded"]:
        msg += "；但沒有發現折現率根本是 LLM 自己補的"
    return {"outcome": "ask_noisy" if noisy else "ask", "clusters": cl,
            "options": [{"value": c["value"], "members": c["members"], "bogus": c in bogus} for c in cl],
            "summary": msg + "。"}


def _lens_airepr(s):
    card = s["card"]
    accepted, detail = [], []
    for r in s["rows"]:
        conv = {c["name"]: r["declared"].get(c["name"], "end") for c in card["conventions"]}
        code = INSPECTORS[card["id"]].replace("{timing}", conv.get("timing", "end"))
        insp = sandbox.run(code, s["inputs"])
        same = r["ok"] and insp["ok"] and _close(r["result"], insp["result"])
        detail.append({"id": r["id"], "value": r["result"], "inspector": insp["result"], "accepted": same,
                       "bug": r["bug"], "outcome": _classify(s, r["result"]) if same else "rejected"})
        if same:
            accepted.append(r)
    bad = [d["id"] for d in detail if d["accepted"] and d["bug"]]
    outs = {d["outcome"] for d in detail if d["accepted"]}
    worst = "wrong" if "wrong" in outs else ("guessed" if "guessed" in outs else "correct")
    msg = f"檢查員重現並接受了 {', '.join(d['id'] for d in detail if d['accepted']) or '無'}"
    if bad:
        msg += f"；其中 {', '.join(bad)} 是 bug，但在這組輸入上重現得出來"
    return {"outcome": worst, "detail": detail, "summary": msg + "。"}


RUNNERS = {"pal": _lens_pal, "codet": _lens_codet, "clarifygpt": _lens_clarify, "airepr": _lens_airepr}


def _harness_outcome(case, s):
    res = harness.run(ScriptedProvider(case), case["question"], {})
    d = res["decision"]
    if d["action"] == "ABSTAIN":
        return {"outcome": "abstain", "summary": d["headline"]}
    if d["action"] == "ASK":
        return {"outcome": "ask", "summary": d["headline"]}
    out = _classify(s, d["value"]) if s else "correct"
    return {"outcome": out, "value": d["value"], "summary": d["headline"]}


# ---------------------------------------------------------------- VERINA lens

def _grade_card(card_id):
    card = get_card(card_id)
    case = next(c for c in CASES if c["id"] == CASE_FOR_CARD[card_id])
    rate_out = card["output"]["unit"] == "小數"
    positives = [c for c in case["candidates"] if c["bug"] is None]
    negatives = [{"id": c["id"], "code": c["code"], "label": c["bug"]} for c in case["candidates"] if c["bug"]]
    for p in positives:
        negatives.append({"id": p["id"] + "×1.001", "code": _mutant(p["code"], "scale", rate_out),
                          "label": "正確答案放大 0.1%"})
        negatives.append({"id": p["id"] + ("+1bp" if rate_out else "+1元"), "code": _mutant(p["code"], "offset", rate_out),
                          "label": "正確答案加 0.0001" if rate_out else "正確答案加 1 元"})
        negatives.append({"id": p["id"] + "≈", "code": _mutant(p["code"], "round", rate_out),
                          "label": "四捨五入到 0.01%" if rate_out else "四捨五入到整數元"})
    grid = GRIDS[card_id]
    progs = [dict(p, bug=None) for p in positives] + [dict(n, bug=n["label"]) for n in negatives]

    def run_on(inp):
        return _run_candidates(card, progs, inp)

    with ThreadPoolExecutor(max_workers=3) as pool:
        per_input = list(pool.map(run_on, grid))

    inv_ids = [i["id"] for i in card["invariants"] if i["kind"] != "pre"]
    pos_rows, neg_rows = [], []
    for k, prog in enumerate(progs):
        rows = [pi[k] for pi in per_input]
        accepted_all = all(r["passed"] for r in rows)
        killers = sorted({f for r in rows for f in (r["failed"] or ([f"crash"] if not r["ok"] else []))})
        entry = {"id": prog["id"], "label": prog.get("bug") or "標註為正確的程式",
                 "per_input": [r["passed"] for r in rows], "killers": killers,
                 "errors": [r["error"] for r in rows if r["error"]]}
        (pos_rows if prog.get("bug") is None else neg_rows).append(entry | {"ok": accepted_all if prog.get("bug") is None
                                                                            else not accepted_all})
    pre = [i for i in card["invariants"] if i["kind"] == "pre"]
    pre_pos = [all(v["ok"] for v in sandbox.eval_pre(pre, g).values()) for g in grid]
    pre_neg = []
    for label, patch in BAD_INPUTS[card_id]:
        res = sandbox.eval_pre(pre, grid[0] | patch)
        pre_neg.append({"label": label, "rejected": not all(v["ok"] for v in res.values()),
                        "by": [k for k, v in res.items() if not v["ok"]]})
    kill = {i: [n["id"] for n in neg_rows if i in n["killers"]] for i in inv_ids}
    kill["claim"] = [n["id"] for n in neg_rows if any(k.startswith("claim:") for k in n["killers"])]
    return {
        "card": card_id, "title": card["title"], "grid": grid,
        "post_complete": {"ok": sum(p["ok"] for p in pos_rows), "n": len(pos_rows), "rows": pos_rows},
        "post_sound": {"ok": sum(n["ok"] for n in neg_rows), "n": len(neg_rows), "rows": neg_rows},
        "pre_complete": {"ok": sum(pre_pos), "n": len(pre_pos)},
        "pre_sound": {"ok": sum(p["rejected"] for p in pre_neg), "n": len(pre_neg), "rows": pre_neg},
        "kill": kill,
        "invariants": [{"id": i["id"], "kind": i["kind"], "text": i["text"]} for i in card["invariants"] if i["kind"] != "pre"],
    }


@lru_cache(maxsize=1)
def compute() -> dict:
    cases = [c for c in CASES]
    setups = {c["id"]: (_setup(c) if c["candidates"] else None) for c in cases}
    matrix = []
    for lens in LENSES:
        if lens["id"] == "verina":
            continue
        row = {"lens": lens["id"], "cells": {}}
        for c in cases:
            s = setups[c["id"]]
            if s is None:
                row["cells"][c["id"]] = {"outcome": "noscope", "summary": "這篇論文沒有範圍閘，會照樣產生一個答案。"}
                continue
            res = RUNNERS[lens["id"]](s)
            why = _why(s)
            if res["outcome"] in ("guessed",) and why:
                res["summary"] += f"（{why}）"
            row["cells"][c["id"]] = res
        matrix.append(row)
    full = {"lens": "harness", "cells": {c["id"]: _harness_outcome(c, setups[c["id"]]) for c in cases}}
    matrix.append(full)
    cards = [_grade_card(cid) for cid in CASE_FOR_CARD]
    return {
        "lenses": LENSES, "matrix": matrix, "verina": cards, "outcomes": OUTCOME_LABEL,
        "cases": [{"id": c["id"], "title": c["title"], "tag": c["tag"], "question": c["question"],
                   "inputs": setups[c["id"]]["inputs"] if setups[c["id"]] else None,
                   "unit": setups[c["id"]]["card"]["output"]["unit"] if setups[c["id"]] else None}
                  for c in cases],
    }
