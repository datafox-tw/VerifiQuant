"""The harness: ten steps from a question to an answer, a question back, or a refusal.

Only steps 2, 5 and 8 call an LLM (or read scripted stand-ins). Every other step
is deterministic, and every number in the final answer comes from executed code
that passed the card's invariants.
"""

from __future__ import annotations

import ast
import datetime as dt
import hashlib
import math
import re
from concurrent.futures import ThreadPoolExecutor

from . import sandbox, units
from .cards import get_card
from .papers import STEP_CONCEPTS

STEP_TITLES = {
    # the part before "：" matches the step names on the advisor slides
    "scope": "範圍閘：判斷問題類型，沒有卡就拒答",
    "intent": "意圖規格：把問題凍結成欄位，每個數值要有原文出處",
    "contract": "取出不變量：這類問題要檢查的條件",
    "precheck": "輸入檢查：前置條件",
    "generate": "生成並執行：LLM 寫多份候選程式，在沙箱中執行",
    "verify": "驗證：每條不變量跑在每份程式上",
    "ambiguity": "bug 或歧義：先淘汰 bug，必要時追問",
    "inspect": "檢查員：只看規格，用不同方法重算",
    "trace": "追溯連結：規格 ↔ 程式行",
    "decision": "決策與紀錄：輸出、追問或拒答，附稽核紀錄",
}
ORDER = list(STEP_TITLES)
# two programs "agree" within this relative tolerance: loose enough for numerical solvers
# (bisection, Newton), far tighter than any bug worth catching (the demo's mutants are 1e-3)
AGREE_REL = 1e-5


def _step(sid, status, summary, body=None):
    return {"id": sid, "n": ORDER.index(sid) + 1, "title": STEP_TITLES[sid], "status": status,
            "summary": summary, "concepts": STEP_CONCEPTS.get(sid, []), "body": body or {}}


def _close(a, b, rel=1e-6):
    return a is not None and b is not None and math.isclose(a, b, rel_tol=rel, abs_tol=1e-9)


def _assumptions(code: str) -> dict:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "ASSUMPTIONS" for t in node.targets):
            try:
                value = ast.literal_eval(node.value)
                return value if isinstance(value, dict) else {}
            except ValueError:
                return {}
    return {}


def _spec_tags(code: str) -> list[dict]:
    rows = []
    for i, line in enumerate(code.splitlines(), 1):
        m = re.search(r"#\s*@spec\s+(.+)$", line)
        if m:
            names = [n.strip() for n in m.group(1).split(",") if n.strip()]
            rows.append({"line": i, "code": line[: m.start()].rstrip(), "names": names})
    return rows


# ---------------------------------------------------------------- step 2

def _bind_slots(card, raw, question, user_slots, ground=None):
    """`ground(slot, quote, claimed, question) -> (status, value, note)` replaces the Chinese parser when given."""
    rows, inputs, missing = [], {}, []
    for slot in card["slots"]:
        name = slot["name"]
        claim = (raw.get("slots") or {}).get(name) or {}
        quote, claimed = claim.get("quote"), claim.get("value")
        row = {"name": name, "label": slot["label"], "unit": slot["unit"], "quote": quote,
               "claimed": claimed, "found": isinstance(quote, str) and quote in question, "parsed": None}
        if slot.get("optional") and name not in user_slots and not quote and claimed is None:
            row.update(status="default", note="原文沒提，沿用函式預設值")
        elif ground and name not in user_slots:
            status, value, note = ground(slot, quote, claimed, question)
            row.update(status=status, parsed=value, note=note)
            if status == "grounded":
                inputs[name] = value
            else:
                missing.append(name)
        elif name in user_slots and str(user_slots[name]).strip():
            parsed = units.parse(str(user_slots[name]), slot["kind"])
            if parsed is None:
                row.update(status="bad_user", note=f"無法解析「{user_slots[name]}」，請加上單位（例如 8%）")
                missing.append(name)
            else:
                row.update(status="user", parsed=parsed, quote=str(user_slots[name]),
                           note="使用者補充，記入稽核紀錄")
                inputs[name] = parsed
        elif not quote:
            row.update(status="ungrounded" if claimed is not None else "missing",
                       note="LLM 給了值，但原文找不到出處 → 不採用" if claimed is not None else "原文沒有提供")
            missing.append(name)
        elif not row["found"]:
            row.update(status="ungrounded", note="引用的文字不在原文中 → 不採用")
            missing.append(name)
        else:
            parsed = units.parse(quote, slot["kind"])
            row["parsed"] = parsed
            if parsed is None or not _close(parsed, float(claimed) if claimed is not None else None):
                row.update(status="mismatch", note="由引用文字重新解析的數值和 LLM 宣稱的不一致 → 不採用")
                missing.append(name)
            else:
                row.update(status="grounded", note="引用文字在原文中，重新解析的數值一致")
                inputs[name] = parsed
        rows.append(row)
    return rows, inputs, missing


def _bind_conventions(card, raw, question, user_conv):
    rows, resolved = [], {}
    for conv in card["conventions"]:
        name = conv["name"]
        claim = (raw.get("conventions") or {}).get(name)
        row = {"name": name, "label": conv["label"], "question": conv["question"], "options": conv["options"]}
        if name in user_conv:
            resolved[name] = user_conv[name]
            row.update(status="user", value=user_conv[name], note="使用者回答")
        elif claim and isinstance(claim.get("quote"), str) and claim["quote"] in question \
                and any(c in claim["quote"] for c in conv["cues"].get(claim.get("value"), [])):
            resolved[name] = claim["value"]
            row.update(status="grounded", value=claim["value"], quote=claim["quote"], note="原文明確寫出")
        else:
            row.update(status="open", value=None, note="原文沒有決定（Astrogator 的 p ∖ s）→ 留到步驟 7 處理")
        rows.append(row)
    return rows, resolved


# ---------------------------------------------------------------- steps 5-6

def _rounds(code: str) -> bool:
    """True if the program rounds its own output (round(...) or x.round(...))."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    return any(isinstance(n, ast.Call) and (getattr(n.func, "id", None) == "round" or getattr(n.func, "attr", None) == "round")
               for n in ast.walk(tree))


def _run_candidates(card, cands, inputs, path=None, no_round=False):
    """no_round: a program that rounds internally fails as a bug (rounding belongs to the presentation)."""
    checks = [i for i in card["invariants"] if i["kind"] != "pre"]
    with ThreadPoolExecutor(max_workers=8) as pool:
        reports = list(pool.map(lambda c: sandbox.run(c["code"], inputs, checks, path=path), cands))
    out = []
    for cand, rep in zip(cands, reports):
        declared = _assumptions(cand["code"])
        row = {"id": cand["id"], "code": cand["code"], "bug": cand.get("bug"), "origin": cand.get("origin", ""),
               "declared": declared, "ok": rep["ok"], "error": rep["error"], "result": rep["result"],
               "preview": rep.get("preview"),
               "checks": rep["checks"], "behaves_as": {}, "failed": [], "passed": False}
        if rep["ok"]:
            core = [i for i in checks if i["kind"] != "boundary"]
            row["failed"] = [i["id"] for i in core if not rep["checks"][i["id"]]["ok"]]
            for conv in card["conventions"]:
                fits = []
                for opt in conv["options"]:
                    b = [i for i in checks if i["kind"] == "boundary" and i["when"].get(conv["name"]) == opt["value"]]
                    if b and all(rep["checks"][i["id"]]["ok"] for i in b):
                        fits.append(opt["value"])
                row["behaves_as"][conv["name"]] = fits
                claim = declared.get(conv["name"])
                if claim is None:
                    row["failed"].append(f"declare:{conv['name']}")
                elif claim not in fits:
                    row["failed"].append(f"claim:{conv['name']}")
            if no_round and _rounds(cand["code"]):
                row["failed"].append("no_internal_round")
            row["passed"] = not row["failed"]
        out.append(row)
    return out


def _clusters(rows):
    groups = []
    for r in rows:
        if r["result"] is None:
            continue
        for g in groups:
            if _close(g["value"], r["result"], AGREE_REL):
                g["members"].append(r["id"])
                break
        else:
            groups.append({"value": r["result"], "members": [r["id"]]})
    return groups


# ---------------------------------------------------------------- main entry

def run(provider, question: str, answers: dict) -> dict:
    steps, answers = [], answers or {}
    audit = {"question": question, "time": dt.datetime.now().isoformat(timespec="seconds"),
             "mode": provider.mode, "llm_calls": []}

    def finish(decision, pending=None):
        done = {s["id"] for s in steps}
        for sid in ORDER:
            if sid not in done:
                steps.append(_step(sid, "waiting", "等待前一步完成"))
        audit["decision"] = decision["action"]
        audit["llm_calls"] = provider.calls
        return {"question": question, "steps": steps, "decision": decision, "pending": pending,
                "audit": audit, "baseline": provider.baseline(question), "mode": provider.mode}

    # 1 scope ------------------------------------------------------------
    card_id, table, how = provider.retrieve(question)
    if not card_id:
        steps.append(_step("scope", "abstain", "卡片庫裡沒有對應的問題類型：這題無法寫成可驗證的程式，拒答。",
                           {"table": table, "how": how}))
        return finish({"action": "ABSTAIN", "headline": "拒答：超出可驗證範圍",
                       "text": "這個問題沒有可以用程式確定計算、再用不變量驗證的答案。harness 選擇不回答，而不是給一個看似篤定的猜測。"})
    card = provider.get_card(card_id) if hasattr(provider, "get_card") else get_card(card_id)
    audit["card"] = {"id": card_id, "title": card["title"]}
    steps.append(_step("scope", "pass", f"歸類為「{card['title']}」（{card['topic']}）。",
                       {"table": table, "how": how, "card": card_id}))

    # 2 intent -----------------------------------------------------------
    raw = provider.intent(question, card)
    if raw.get("card_answers_question") is False:
        steps.append(_step("intent", "abstain", "題目問的量不是這張卡算的量：卡片不適用，拒答。",
                           {"asked": raw.get("asked_quantity"), "card_output": card["output"]["label"]}))
        return finish({"action": "ABSTAIN", "headline": "拒答：題目問的不是卡片的輸出",
                       "text": "卡片只能驗證它定義的輸出量；題目問的是別的量，硬算會變成沒被驗證的答案。"})
    structure = card["output"].get("structure")
    path, select_row = None, None
    if structure:
        ground_select = getattr(provider, "ground_select", None)
        status, path, note = ground_select(card, raw.get("select"), question) if ground_select else \
            ("unsupported", None, "這個 provider 無法驗證選取路徑")
        select_row = {"quote": (raw.get("select") or {}).get("quote"), "claimed": (raw.get("select") or {}).get("path"),
                      "path": path, "status": status, "note": note, "structure": structure["text"]}
        if status != "grounded":
            steps.append(_step("intent", "ask", f"卡片輸出是一個結構，題目要其中哪一個值無法由原文確認：{note}。",
                               {"select": select_row}))
            return finish({"action": "ASK", "headline": "追問：題目要的是結構中的哪一個值？",
                           "text": "卡片算出的是一整張表；要取哪一格必須有原文依據，否則就是沒被驗證的選擇。"},
                          {"type": "select", "structure": structure["text"], "claimed": select_row["claimed"]})
    slot_rows, inputs, missing = _bind_slots(card, raw, question, answers.get("slots", {}),
                                             getattr(provider, "ground", None))
    conv_rows, resolved = _bind_conventions(card, raw, question, answers.get("conventions", {}))
    open_convs = [r["name"] for r in conv_rows if r["status"] == "open"]
    intent_body = {"slots": slot_rows, "conventions": conv_rows, "fixed": card["fixed"],
                   "signature": card["signature"], "select": select_row}
    audit["inputs"] = {r["name"]: {"value": inputs.get(r["name"]), "source": r["status"], "quote": r["quote"]}
                       for r in slot_rows}
    if missing:
        labels = "、".join(r["label"] for r in slot_rows if r["name"] in missing)
        steps.append(_step("intent", "ask", f"缺少或無法驗證：{labels}。不猜，回頭問使用者。", intent_body))
        steps.append(_step("contract", "info", "這類問題要檢查的不變量（先列出，輸入補齊後執行）。", _contract_body(card)))
        fields = [{"name": r["name"], "label": r["label"], "unit": r["unit"], "note": r["note"]}
                  for r in slot_rows if r["name"] in missing]
        return finish({"action": "ASK", "headline": f"追問：請提供{labels}",
                       "text": "原文沒有給出這個數值。若讓模型自行假設，結論可能完全相反，所以 harness 停下來問。"},
                      {"type": "slots", "fields": fields})
    n_user = sum(r["status"] == "user" for r in slot_rows)
    n_default = sum(r["status"] == "default" for r in slot_rows)
    head = f"{len(slot_rows) - n_user - n_default} 個欄位有原文出處且重新解析一致" + (f"，{n_user} 個由使用者補充" if n_user else "")
    extra = "；約定「" + "、".join(r["label"] for r in conv_rows if r["status"] == "open") + "」原文未決定" if open_convs else ""
    if path:
        extra += f"；題目要的值是 result{''.join(f'[{k!r}]' for k in path)}"
    steps.append(_step("intent", "pass", f"{head}{extra}。", intent_body))

    # 3 contract ---------------------------------------------------------
    if not any(i["kind"] != "pre" for i in card["invariants"]):
        steps.append(_step("contract", "abstain", "卡片沒有任何可以檢查程式的不變量：無法驗證，拒答。",
                           _contract_body(card)))
        return finish({"action": "ABSTAIN", "headline": "拒答：卡片無法驗證程式",
                       "text": "沒有後置條件或變形關係，任何程式都會「通過」，所以不輸出答案。"})
    steps.append(_step("contract", "info",
                       f"卡片提供 {len(card['invariants'])} 條不變量；它們由專家按問題類型撰寫，不是 LLM 當場生成。",
                       _contract_body(card)))

    # 4 precheck ---------------------------------------------------------
    pre = [i for i in card["invariants"] if i["kind"] == "pre"]
    pre_res = sandbox.eval_pre(pre, inputs)
    pre_rows = [{"id": i["id"], "text": i["text"], "check": i["check"], "ok": pre_res[i["id"]]["ok"]} for i in pre]
    bad = [r for r in pre_rows if not r["ok"]]
    if bad:
        names = {s["name"] for s in card["slots"] for r in bad if re.search(rf"\b{s['name']}\b", r["check"])}
        steps.append(_step("precheck", "fail", "輸入違反合約：" + "；".join(r["text"] for r in bad), {"rows": pre_rows}))
        fields = [{"name": s["name"], "label": s["label"], "unit": s["unit"], "note": "違反輸入合約，請確認"}
                  for s in card["slots"] if s["name"] in names]
        return finish({"action": "ASK", "headline": "追問：輸入數值不合理", "text": "輸入沒通過前置條件，計算前就停下。"},
                      {"type": "slots", "fields": fields})
    steps.append(_step("precheck", "pass", f"{len(pre_rows)} 條前置條件全部通過。", {"rows": pre_rows, "inputs": inputs}))

    # 5 generate ---------------------------------------------------------
    stated = {r["name"]: r["value"] for r in conv_rows if r["status"] == "grounded"}
    spec = {"card": card_id, "signature": card["signature"], "inputs": inputs, "conventions": resolved,
            "stated_conventions": stated, "open_conventions": open_convs, "fixed": [f["text"] for f in card["fixed"]]}
    no_round = getattr(provider, "reject_rounding", True)  # rounding belongs to the presentation, not the program
    cands = provider.candidates(question, card, spec)
    rows = _run_candidates(card, cands, inputs, path, no_round)
    notes = []
    if not any(r["passed"] for r in rows) and provider.can_regenerate:
        feedback = [{"id": r["id"], "failed": r["failed"], "error": r["error"]} for r in rows]
        more = provider.candidates(question, card, spec, feedback=feedback)
        rows += _run_candidates(card, more, inputs, path, no_round)
        notes.append(f"第一輪全數未通過，帶著失敗的不變量回饋重新生成 {len(more)} 份。")
    inv_view = [i for i in card["invariants"] if i["kind"] != "pre"]

    def emit_5_6():
        # steps 5 and 6 are rebuilt if step 7 has to regenerate, so the page shows every candidate
        clusters = _clusters(rows)
        surv = [r for r in rows if r["passed"]]
        gen = _step("generate", "info", f"{len(rows)} 份候選程式在題目輸入上跑出 {len(clusters)} 種不同答案。" + "".join(notes),
                    {"candidates": [{k: r[k] for k in ("id", "code", "declared", "result", "error", "origin")}
                                    for r in rows],
                     "clusters": clusters, "unit": card["output"]["unit"], "inputs": inputs})
        ver = _step("verify", "pass" if surv else "fail",
                    f"{len(surv)} / {len(rows)} 份通過所有不變量。" if surv else "沒有任何候選通過不變量。",
                    {"invariants": [{k: i.get(k) for k in ("id", "kind", "text", "when")} for i in inv_view],
                     "rows": [{k: r[k] for k in ("id", "ok", "error", "result", "checks", "declared",
                                                  "behaves_as", "failed", "passed", "bug")} for r in rows],
                     "conventions": card["conventions"], "unit": card["output"]["unit"]})
        for new_step in (gen, ver):
            idx = next((i for i, x in enumerate(steps) if x["id"] == new_step["id"]), None)
            if idx is None:
                steps.append(new_step)
            else:
                steps[idx] = new_step

    # 6 verify -----------------------------------------------------------
    emit_5_6()
    survivors = [r for r in rows if r["passed"]]
    if not survivors:
        return finish({"action": "ABSTAIN", "headline": "拒答：沒有程式通過驗證",
                       "text": "所有候選程式都違反至少一條不變量。harness 不輸出任何數字。"})
    if path:
        # a card can pin a schedule's shape and still say nothing about the one cell the question asks for
        core = [i for i in inv_view if i["kind"] != "boundary"]
        probe = sandbox.run(survivors[0]["code"], inputs, core, path=path, mutate=True)
        caught = [k for k, c in probe["checks"].items() if not c["ok"]] if probe["ok"] else ["(程式當機)"]
        ver = next(x for x in steps if x["id"] == "verify")
        ver["body"]["element_probe"] = {"candidate": survivors[0]["id"], "caught_by": caught}
        if not caught:
            ver["status"] = "fail"
            ver["summary"] += " 但把題目要的那一格改動 0.1% 後，沒有任何不變量發現：卡片沒有約束這個值。"
            return finish({"action": "ABSTAIN", "headline": "拒答：卡片沒有約束題目要的值",
                           "text": "不變量檢查的是整張表的其他部分；題目要的那一格錯了也不會被發現，所以不輸出。"})
        ver["summary"] += f" 題目要的那一格改動 0.1% 會被 {len(caught)} 條不變量抓到。"

    # 7 ambiguity --------------------------------------------------------
    def build_amb(survivors):
        # rebuilt after regeneration too, so the table counts the new candidates
        body = {"conventions": [], "clusters": _clusters(survivors), "unit": card["output"]["unit"]}
        open_entries = []
        for conv in card["conventions"]:
            name = conv["name"]
            previews = []
            for opt in conv["options"]:
                vals = [r for r in survivors if r["declared"].get(name) == opt["value"]]
                previews.append({"value": opt["value"], "label": opt["label"],
                                 "result": vals[0]["result"] if vals else None, "members": [v["id"] for v in vals]})
            rel = conv.get("relation")
            rel_ok = None
            if rel:
                pv = {p["value"]: p["result"] for p in previews}
                if pv.get(rel["from"]) is not None and pv.get(rel["to"]) is not None:
                    rel_ok = bool(eval(rel["check"], {"__builtins__": {}},
                                       {"close": _close, f"v_{rel['from']}": pv[rel["from"]],
                                        f"v_{rel['to']}": pv[rel["to"]], **inputs}))
            entry = {"name": name, "label": conv["label"], "question": conv["question"], "previews": previews,
                     "relation": rel["text"] if rel else None, "relation_ok": rel_ok, "chosen": resolved.get(name)}
            body["conventions"].append(entry)
            if name not in resolved:
                open_entries.append(entry)
        body["rejected"] = [r["id"] for r in rows if not r["passed"]]
        return body, open_entries

    amb_body, asks = build_amb(survivors)
    rejected_bugs = amb_body["rejected"]
    if asks:
        e = asks[0]
        readings = [p for p in e["previews"] if p["members"]]
        if len(readings) > 1:
            msg = f"淘汰 {len(rejected_bugs)} 份寫錯的程式後，仍剩兩種合理讀法：{e['label']}未定。這是歧義，不是 bug，要問使用者。"
        else:
            msg = (f"通過驗證的候選全都選了「{readings[0]['label']}」，答案一致；但原文沒有決定{e['label']}，"
                   "大家一起猜不等於使用者的意思，仍要問。")
        steps.append(_step("ambiguity", "ask", msg, amb_body))
        return finish({"action": "ASK", "headline": f"追問：{e['question']}",
                       "text": "兩種讀法的程式都通過了驗證，差別只在約定。harness 把兩個答案都攤開，請使用者決定。"},
                      {"type": "convention", "conventions": asks})

    selected = [r for r in survivors if all(r["declared"].get(k) == v for k, v in resolved.items())]
    regenerated = False
    if not selected and provider.can_regenerate:
        more = _run_candidates(card, provider.candidates(question, card, spec), inputs, path, no_round)
        rows += more
        notes.append(f"沒有候選採用使用者選的約定，告知約定後重新生成 {len(more)} 份（{', '.join(r['id'] for r in more)}）。")
        emit_5_6()
        regenerated = True
        selected = [r for r in more if r["passed"] and all(r["declared"].get(k) == v for k, v in resolved.items())]
        amb_body, _ = build_amb(survivors + [r for r in more if r["passed"]])
    if not selected:
        steps.append(_step("ambiguity", "fail", "沒有任何通過驗證的程式符合已確定的約定。", amb_body))
        return finish({"action": "ABSTAIN", "headline": "拒答：沒有符合約定的程式",
                       "text": "通過驗證的程式都採用了另一種約定。"})
    sel_clusters = _clusters(selected)
    if len(sel_clusters) > 1:
        steps.append(_step("ambiguity", "fail", "通過驗證的程式仍給出不同答案：卡片的不變量不足以區分，拒答。", amb_body))
        return finish({"action": "ABSTAIN", "headline": "拒答：驗證條件不足", "text": "不變量無法唯一決定答案。"})
    dropped = [r["id"] for r in survivors if r not in selected]
    msg = "通過驗證的程式答案一致，沒有歧義，不需要問。" if not card["conventions"] else \
        "約定已確定（" + "、".join(
            f"{c['label']}：" + next((o["label"] for o in c["options"] if o["value"] == resolved[c["name"]]),
                                    str(resolved[c["name"]])) for c in card["conventions"]) + "）。"
    if dropped:
        msg += f" 採用其他約定的 {', '.join(dropped)} 不選。"
    if regenerated:
        msg += f" 改用重新生成且通過驗證的 {', '.join(r['id'] for r in selected)}。"
    amb_body["selected"] = [r["id"] for r in selected]
    steps.append(_step("ambiguity", "pass", msg, amb_body))
    chosen = selected[0]
    value = chosen["result"]

    # 8 inspect ----------------------------------------------------------
    # the inspector is one more program: when the provider asks for it, it must pass the same card first,
    # including the boundary checks of the decided conventions (it never sees the candidate or its answer)
    insp_checks = [i for i in inv_view if i["kind"] != "boundary" or all(resolved.get(k) == v for k, v in i["when"].items())] \
        if getattr(provider, "verify_inspector", False) else []

    def run_inspector(code):
        rep = sandbox.run(code, inputs, insp_checks, path=path)
        broken = [i for i in insp_checks if rep["ok"] and not rep["checks"][i["id"]]["ok"]]
        return rep, broken

    insp_code = provider.inspector(card, spec | {"conventions": resolved})
    insp, broken = run_inspector(insp_code)
    if (not insp["ok"] or broken) and provider.can_regenerate:
        # a crash or a card violation is not evidence against the answer; give the inspector one more attempt
        why = insp["error"] or "the program violates these properties of the calculation: " + \
            "; ".join(i["text"] for i in broken)
        insp_code = provider.inspector(card, spec | {"conventions": resolved}, retry=why)
        insp, broken = run_inspector(insp_code)
    if insp["ok"] and broken:
        steps.append(_step("inspect", "fail", "檢查員的程式本身沒通過卡片的不變量：無法當作獨立重算。",
                           {"code": insp_code, "result": insp["result"], "error": None,
                            "broken": [i["id"] for i in broken], "candidate": chosen["id"], "candidate_result": value,
                            "unit": card["output"]["unit"]}))
        return finish({"action": "ABSTAIN", "headline": "拒答：檢查員程式未通過卡片",
                       "text": "獨立重算的程式自己就違反卡片，它的答案不能拿來確認候選。"})
    agree = insp["ok"] and _close(insp["result"], value, AGREE_REL)
    steps.append(_step("inspect", "pass" if agree else "fail",
                       "檢查員用不同方法算出相同答案。" if agree else "檢查員的答案和候選不一致。",
                       {"code": insp_code, "result": insp["result"], "error": insp["error"],
                        "candidate": chosen["id"], "candidate_result": value, "unit": card["output"]["unit"],
                        "sees": ["凍結後的欄位與數值", "已確定的約定", "卡片的固定假設"],
                        "hidden": ["原始問題文字", "候選程式"]}))
    if not agree:
        return finish({"action": "ABSTAIN", "headline": "拒答：獨立重算不一致", "text": "兩條獨立路徑的答案不同。"})

    # 9 trace ------------------------------------------------------------
    tags = _spec_tags(chosen["code"])
    items = [{"name": s["name"], "label": s["label"], "kind": "欄位", "value": inputs[s["name"]]} for s in card["slots"]]
    items += [{"name": c["name"], "label": c["label"], "kind": "約定", "value": resolved.get(c["name"])}
              for c in card["conventions"]]
    items += [{"name": f["name"], "label": f["text"], "kind": "固定假設", "value": None, "by": f.get("by")}
              for f in card["fixed"]]
    for it in items:
        it["lines"] = [t["line"] for t in tags if it["name"] in t["names"]]
    uncovered = [it["label"] for it in items if not it["lines"] and it.get("by") != "harness"]
    steps.append(_step("trace", "pass" if not uncovered else "info",
                       "每個規格項目都連到實作它的程式行。" if not uncovered else f"未被標記的規格項目：{'、'.join(uncovered)}",
                       {"items": items, "code": chosen["code"], "candidate": chosen["id"]}))

    # 10 decision --------------------------------------------------------
    out = dict(card["output"])
    if path:
        field = next((f for f in structure["fields"] for k in path if f["key"] == k), None)
        out["label"] = f"{field['label'] if field else out['label']}（{select_row['quote']}）"
        out["unit"] = field["unit"] if field else structure["element_unit"] or out["unit"]
    text = f"{out['label']} = {units.fmt(value, out['unit'])}"
    if card["id"] == "npv":
        text += "，NPV > 0，值得投資。" if value > 0 else "，NPV ≤ 0，不值得投資。"
    audit.update({
        "conventions": {k: {"value": v, "source": next(r["status"] for r in conv_rows if r["name"] == k)}
                        for k, v in resolved.items()},
        "fixed_assumptions": [f["text"] for f in card["fixed"]],
        "program": {"candidate": chosen["id"], "sha256": hashlib.sha256(chosen["code"].encode()).hexdigest()[:16]},
        "invariants": {i["id"]: chosen["checks"][i["id"]]["ok"] for i in inv_view
                       if i["kind"] != "boundary" or all(chosen["declared"].get(k) == v for k, v in i["when"].items())},
        "rejected_candidates": {r["id"]: r["failed"] or [r["error"]] for r in rows if not r["passed"]},
        "inspector_agrees": agree,
        "answer": value,
    })
    decision = {"action": "EMIT", "headline": text, "value": value, "unit": out["unit"],
                "text": "數字來自通過全部不變量的程式，並由只看規格的檢查員獨立重算確認。稽核紀錄只記錄被執行過的東西，不記模型的自我解釋。"}
    steps.append(_step("decision", "pass", text, {"action": "EMIT"}))
    return finish(decision)


def _contract_body(card):
    return {"invariants": card["invariants"], "note": card["completeness_note"], "fixed": card["fixed"]}
