"""Why each question of a run got no answer: method, card quality, or card selection (deck slide 23).

    python3 -m harness_demo.poc.coverage_causes --run full250_v5_r1

A verification failure is split by running the library function (the reference) through the selected card:
if the reference itself fails the card, the card is wrong; if it passes, the candidates read the spec wrongly.
"""

from __future__ import annotations

import argparse
import collections
import json

from .data import RUNS
from .grade_cards import grade

KIND = {"檢查員和候選不一致": "方法", "LLM 沒抽到欄位而追問": "方法", "出處比對失敗而追問": "方法",
        "卡片正確，但候選讀錯欄位意義": "方法", "通過驗證的程式仍給出不同答案": "方法",
        "卡片的不變量有錯（參考函式也過不了）": "卡片品質", "卡片輸出不是題目要的量": "卡片品質",
        "驗證失敗，卡片無法評分": "卡片品質（未確定）", "選錯卡，被不變量擋下": "選卡"}


def cause(qid: str, r: dict, rows: dict) -> str:
    stop = r.get("stopped_at")
    if r.get("retrieval_ok") is False or not r.get("selected_card"):
        return "選錯卡，被不變量擋下"
    if stop == "intent" and r["action"] == "ASK":
        bad = [s for s in r["slots"] if s["status"] not in ("grounded", "default")]
        return "出處比對失敗而追問" if any(s["status"] == "mismatch" for s in bad) else "LLM 沒抽到欄位而追問"
    if stop == "intent":
        return "卡片輸出不是題目要的量"
    if stop in ("verify", "precheck", "contract"):
        try:
            g = grade(qid, rows)
        except Exception:  # an ungradeable card is reported as such, never guessed
            g = {"gradeable": False}
        if not g.get("gradeable"):
            return "驗證失敗，卡片無法評分"
        return "卡片的不變量有錯（參考函式也過不了）" if not g.get("complete") else "卡片正確，但候選讀錯欄位意義"
    if stop == "inspect":
        return "檢查員和候選不一致"
    return "通過驗證的程式仍給出不同答案"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="full250_v5_r1")
    args = ap.parse_args()
    rows = {r["question_id"]: r for r in map(json.loads, (RUNS / args.run / "results.jsonl").read_text().splitlines())}
    out = collections.defaultdict(list)
    for qid, r in rows.items():
        if r["action"] != "EMIT":
            out[cause(qid, r, rows)].append(qid)
    total = sum(map(len, out.values()))
    print(f"# {args.run}：{total} 題沒有輸出", "", "| 原因 | 題數 | 屬於 |", "|---|---|---|", sep="\n")
    for k, v in sorted(out.items(), key=lambda kv: -len(kv[1])):
        print(f"| {k} | {len(v)} | {KIND[k]} |")
    by_kind = collections.Counter()
    for k, v in out.items():
        by_kind[KIND[k]] += len(v)
    print("", "、".join(f"{k} {v}" for k, v in by_kind.most_common()), sep="\n")
    (RUNS / args.run / "causes.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
