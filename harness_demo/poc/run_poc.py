"""Run the harness on the frozen sample, one jsonl row per question.

    python3 -m harness_demo.poc.run_poc --tag smoke --n 5
    python3 -m harness_demo.poc.run_poc --tag full

The harness sees context + question only. Gold fields are read after the run, for grading,
with the paper's scorer (1% relative tolerance, x100 percent allowance, nearest-integer rule).
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

from verifiquant.pipeline.run_error_classification_pipeline import _answer_match

from .. import harness
from .data import RUNS, fic_id, questions, sample_ids, text_of
from .provider import PocProvider


def _step(res, sid):
    return next((s for s in res["steps"] if s["id"] == sid), {})


def summarize(qid: str, res: dict, seconds: float, q: dict | None = None) -> dict:
    q = q or questions()[qid]
    dec = res["decision"]
    scope = _step(res, "scope")
    verify = _step(res, "verify")
    amb = _step(res, "ambiguity")
    gen = _step(res, "generate")
    rows = (verify.get("body") or {}).get("rows") or []
    selected = (scope.get("body") or {}).get("card")
    out = {
        "question_id": qid, "level": q.get("level"), "target_card": fic_id(q), "selected_card": selected,
        "retrieval_ok": selected == fic_id(q) if selected else None,
        "action": dec["action"], "headline": dec.get("headline"), "value": dec.get("value"),
        "stopped_at": next((s["id"] for s in res["steps"] if s["status"] in ("abstain", "ask", "fail")), None),
        "steps": {s["id"]: s["status"] for s in res["steps"]},
        "slots": [{k: r.get(k) for k in ("name", "status", "quote", "claimed", "parsed", "note")}
                  for r in (_step(res, "intent").get("body") or {}).get("slots") or []],
        "asked_quantity": None,
        "candidates": [{k: r.get(k) for k in ("id", "result", "error", "failed", "passed", "declared")} for r in rows],
        "clusters": (gen.get("body") or {}).get("clusters"),
        "conventions": [{"name": c["name"], "chosen": c.get("chosen"),
                         "previews": [{"value": p["value"], "result": p["result"], "members": p["members"]}
                                      for p in c["previews"]]}
                        for c in (amb.get("body") or {}).get("conventions") or []],
        "select": (_step(res, "intent").get("body") or {}).get("select"),
        "element_probe": (verify.get("body") or {}).get("element_probe"),
        "inspector": _step(res, "inspect").get("status"),
        "inspector_value": (_step(res, "inspect").get("body") or {}).get("result"),
        "inspector_error": (_step(res, "inspect").get("body") or {}).get("error"),
        "inspector_code": (_step(res, "inspect").get("body") or {}).get("code"),
        "pending": res.get("pending"),
        "llm_calls": len(res["audit"].get("llm_calls") or []),
        "seconds": round(seconds, 1),
    }
    for call in res["audit"].get("llm_calls") or []:
        if call.get("role") == "intent":
            try:
                out["asked_quantity"] = json.loads(call["reply"]).get("asked_quantity")
                out["card_answers_question"] = json.loads(call["reply"]).get("card_answers_question")
            except (json.JSONDecodeError, AttributeError):
                pass
    return out


def grade(row: dict) -> dict:
    """Gold is read here and only here."""
    q = questions()[row["question_id"]]
    gold = q["ground_truth"]
    gold_num = float(gold) if isinstance(gold, (int, float)) else None
    row["gold"] = gold
    row["correct"] = None
    if row["action"] == "EMIT" and row["value"] is not None and gold_num is not None:
        _, ok, note = _answer_match(q["question"], float(row["value"]), gold_num)
        row["correct"], row["scale_note"] = bool(ok), note
    # would any option of an open convention have been right? (no extra calls: previews already ran)
    row["ask_has_correct_option"] = None
    if row["action"] == "ASK" and row["conventions"] and gold_num is not None:
        vals = [p["result"] for c in row["conventions"] for p in c["previews"] if p["result"] is not None]
        row["ask_has_correct_option"] = any(_answer_match(q["question"], float(v), gold_num)[1] for v in vals)
    return row


def run_one(qid: str) -> dict:
    t = time.time()
    try:
        res = harness.run(PocProvider(), text_of(questions()[qid]), {})
        row = summarize(qid, res, time.time() - t)
    except Exception as exc:
        row = {"question_id": qid, "level": questions()[qid]["level"], "action": "ERROR",
               "error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()[-1500:],
               "seconds": round(time.time() - t, 1)}
    return grade(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--allow-rounding", action="store_true",
                    help="turn the rounding gate off (it is on by default since v5; v3 had it off)")
    ap.add_argument("--verify-inspector", action="store_true", help="v6: the inspector must pass the card first")
    ap.add_argument("--self-check", action="store_true", help="v6b: drop invariants the reference violates")
    ap.add_argument("--ids", default=None, help="comma-separated question ids (default: the frozen sample)")
    args = ap.parse_args()
    PocProvider.reject_rounding = not args.allow_rounding
    PocProvider.verify_inspector = args.verify_inspector
    PocProvider.self_check = args.self_check
    ids = args.ids.split(",") if args.ids else sample_ids()[: args.n] if args.n else sample_ids()
    out_dir = RUNS / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "results.jsonl"
    done = {json.loads(l)["question_id"] for l in path.read_text().splitlines()} if path.exists() else set()
    todo = [i for i in ids if i not in done]
    print(f"{len(todo)} to run ({len(done)} already in {path})")
    with ThreadPoolExecutor(max_workers=args.workers) as pool, path.open("a") as fh:
        for row in pool.map(run_one, todo):
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"{row['question_id']:10s} {row['action']:8s} correct={row.get('correct')} "
                  f"stop={row.get('stopped_at')} {str(row.get('headline') or row.get('error'))[:70]}")


if __name__ == "__main__":
    main()
