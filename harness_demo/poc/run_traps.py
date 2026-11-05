"""The harness on the paper's 80 governance traps (F missing input, E bad input, I ambiguity, N out of scope).

    python3 -m harness_demo.poc.run_traps --tag traps_r1                     -> runs/traps_r1/results.jsonl
    python3 -m harness_demo.poc.run_traps --score traps_r1,...               -> runs/traps_r1/TRAPS.md
    python3 -m harness_demo.poc.run_traps --split test --tag test_traps_r1   (held-out set, build_test_traps.py)

Two splits: dev = the paper's 80 traps (the harness may be changed after looking at them), test = 80 new
traps on disjoint base questions with unseen out-of-scope asks, only ever scored.

Single turn, like the clean PoC: an ASK is the final answer (the scripted user replies are not used).
Scoring is the paper's own `score_traps` (FNAR = a number given on a trap, except an I_SOFT answer
that carries a warning). Harness decisions are mapped onto its R-PAL row format:
  scope abstain, card does not answer -> N     slots missing -> F     precheck fails -> E
  open convention / select path       -> I     verify / inspect / contract refusal -> ABSTAIN
The harness has no warn-and-proceed mode, so an I_SOFT answer always counts as a false acceptance.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

from verifiquant.pipeline.run_error_classification_pipeline import _answer_match

from .. import harness
from .data import ROOT, RUNS, text_of
from .provider import PocProvider
from .run_poc import summarize

SPLITS = {"dev": ROOT / "datasets/03_traps_80/trap_set.jsonl",
          "test": ROOT / "harness_demo/poc/traps_test/trap_set.jsonl"}
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
PAPER = ROOT / "verifiquant/data/runs/governance_confirmatory_1014/confirmatory_summary_top4.json"
# VQ runs scored from their own rows, so every system gets both FNAR columns on the same split
VQ_RUNS = {"dev": [ROOT / f"verifiquant/data/runs/governance_confirmatory_1014/vq_k1_top4_run{i}.jsonl" for i in (1, 2, 3)],
           "test": [RUNS / f"vq_test_traps/vq_k1_top4_test_run{i}.jsonl" for i in (1, 2, 3)]}
split = "dev"


def traps() -> list[dict]:
    return [json.loads(l) for l in SPLITS[split].read_text().splitlines() if l.strip()]


def diagnose(row: dict) -> str:
    stop, action = row.get("stopped_at"), row["action"]
    if action == "EMIT":
        return "READY"
    if stop == "scope" or (stop == "intent" and action == "ABSTAIN"):
        return "N"
    if stop == "intent":
        return "I_HARD" if (row.get("pending") or {}).get("type") == "select" else "F"
    if stop == "precheck":
        return "E"
    if stop == "ambiguity" and action == "ASK":
        return "I_HARD"
    return "ABSTAIN"


def run_one(t: dict) -> dict:
    start = time.time()
    try:
        res = harness.run(PocProvider(), text_of(t), {})
        row = summarize(t["trap_id"], res, time.time() - start, q=t)
    except Exception as exc:
        row = {"question_id": t["trap_id"], "action": "ERROR", "error": f"{type(exc).__name__}: {exc}",
               "trace": traceback.format_exc()[-1500:]}
    row.update(operator=t["operator"], family=t["family"], expected=t["expected_diagnostic"],
               grounding=t["grounding"], diagnostic=diagnose(row) if row["action"] != "ERROR" else "ERROR")
    row["correct"] = None
    if row["action"] == "EMIT" and row.get("value") is not None:
        try:
            row["correct"] = bool(_answer_match(t["question"], float(row["value"]), float(t["ground_truth"]))[1])
        except (TypeError, ValueError):
            pass
    return row


def as_rpal(row: dict) -> dict:
    """The paper scorer's R-PAL row shape, so FNAR and action accuracy use the paper's exact code."""
    pending = row.get("pending") or {}
    targets = [f["name"] for f in pending.get("fields", [])] + [c["name"] for c in pending.get("conventions", [])]
    decision = {"EMIT": "execute", "ASK": "intercept", "ABSTAIN": "refuse"}.get(row["action"], "refuse")
    return {"case_id": row["question_id"], "output": row.get("value") if row["action"] == "EMIT" else None,
            "assessment": {"diagnostic": row["diagnostic"], "decision": decision, "action_targets": targets,
                           "reason": row.get("headline") or "",
                           "clarification_question": row.get("headline") if row["action"] == "ASK" else "",
                           "warning": ""}}


def _fnar_pair(ts, rows, normalizer):
    """(FNAR on all traps, FNAR without I_SOFT, per-operator FNAR) from the paper's scorer."""
    from scripts.analyze_governance_eval import score_traps

    s = score_traps(ts, rows, normalizer)
    flags = {d["case_id"]: d["unsafe_numeric"] for d in s["details"]}
    hard = [t["trap_id"] for t in ts if t["expected_diagnostic"] != "I_SOFT"]
    return (s["overall"]["fnar"], sum(flags[i] for i in hard) / len(hard),
            {k: v["fnar"] for k, v in s["per_operator"].items()}, s["overall"]["action_accuracy"])


def score(tags: list[str]) -> str:
    from scripts.analyze_governance_eval import load_jsonl, normalize_rpal, normalize_vq

    ts = traps()
    ids = {t["trap_id"] for t in ts}
    # rows of traps dropped from the split (construction errors) are ignored; a run must cover every trap,
    # because the paper's scorer counts a missing row as safe
    runs = {t: [r for r in map(json.loads, (RUNS / t / "results.jsonl").read_text().splitlines()) if r["question_id"] in ids]
            for t in tags}
    short = [t for t, rows in runs.items() if len({r["question_id"] for r in rows}) < len(ids)]
    if short:
        raise SystemExit(f"incomplete runs (would score missing rows as safe): {short}")
    n_soft = sum(t["expected_diagnostic"] == "I_SOFT" for t in ts)
    L = [f"# harness 在 {split} 陷阱題上（{len(ts)} 題，{len(tags)} 次執行）", "",
         "FNAR = 在陷阱題上仍給出數字的比例（論文的主要指標，越低越好），評分直接呼叫論文的 `score_traps`。",
         f"兩欄都報：含 I_SOFT（論文規則）與不含 I_SOFT（{len(ts) - n_soft} 題；harness 沒有「附警告照做」模式，"
         "I_SOFT 一輸出就算錯，拿掉後比較不受計分方式影響）。", "",
         "| 系統 | FNAR 含 I_SOFT | FNAR 不含 I_SOFT | F | E | I | N | 動作正確率 |", "|---|---|---|---|---|---|---|---|"]

    def line(name, res):
        a = statistics.mean(r[0] for r in res)
        b = statistics.mean(r[1] for r in res)
        each = "（" + "、".join(f"{r[0]:.1%}" for r in res) + "）" if len(res) > 1 else ""
        op = {k: statistics.mean(r[2].get(k, 0) for r in res) for k in "FEIN"}
        acc = statistics.mean(r[3] for r in res)
        return (f"| {name} | **{a:.1%}**{each} | {b:.1%} | " + " | ".join(f"{op[k]:.0%}" for k in "FEIN")
                + f" | {acc:.1%} |")

    harness_res = [_fnar_pair(ts, [as_rpal(r) for r in rows], normalize_rpal) for rows in runs.values()]
    L.append(line(f"harness（{len(tags)} 次）", harness_res))
    vq = [p for p in VQ_RUNS[split]
          if p.exists() and ids <= {r.get("case_id") for r in load_jsonl(p)}]  # complete runs only
    if vq:
        L.append(line(f"VQ（論文設定 top-4、K=1，{len(vq)} 次）", [_fnar_pair(ts, load_jsonl(p), normalize_vq) for p in vq]))
    else:
        L.append("| VQ | 尚未在這個 split 上執行 | | | | | | |")
    if split == "dev":
        paper = json.loads(PAPER.read_text())
        for name, key in (("R-PAL prompted（論文，三次平均）", "rpal_prompted_mean"), ("R-PAL plain（論文）", "rpal_plain")):
            o, op = paper["overall"].get(key) or {}, paper["per_operator_fnar"].get(key) or {}
            L.append(f"| {name} | {o.get('fnar', 0):.1%} | — | " + " | ".join(f"{op.get(k, 0):.0%}" for k in "FEIN") + " | — |")
    first = runs[tags[0]]
    L += ["", f"## 逐類（{tags[0]}）", "", "| 運算子 | 家族 | 題數 | 輸出（對/錯） | 追問 | 拒答 | 診斷對 |", "|---|---|---|---|---|---|---|"]
    groups = collections.defaultdict(list)
    for r in first:
        groups[(r["operator"], r["expected"], r["family"])].append(r)
    for (op, exp, fam), rs in sorted(groups.items()):
        emit = [r for r in rs if r["action"] == "EMIT"]
        L.append(f"| {op} | {exp} / {fam} | {len(rs)} | {len(emit)}（{sum(bool(r['correct']) for r in emit)}/"
                 f"{sum(r['correct'] is False for r in emit)}） | {sum(r['action'] == 'ASK' for r in rs)} | "
                 f"{sum(r['action'] == 'ABSTAIN' for r in rs)} | {sum(r['diagnostic'] == exp for r in rs)} |")
    L += ["", "## 給出數字的陷阱題（逐題）", "", "| 題號 | 預期 | 停在 | 輸出 | 答案對照原題 | 說明 |", "|---|---|---|---|---|---|"]
    for r in first:
        if r["action"] == "EMIT":
            L.append(f"| {r['question_id']} | {r['expected']} | — | {r['value']:.6g} | {r['correct']} | "
                     f"{str(r['grounding'])[:90]} |")
    text = "\n".join(L) + "\n"
    (RUNS / tags[0] / "TRAPS.md").write_text(text)
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag")
    ap.add_argument("--split", choices=sorted(SPLITS), default="dev")
    ap.add_argument("--score", default=None, help="comma-separated run tags to score")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--ids", default=None, help="comma-separated trap ids (default: all 80)")
    args = ap.parse_args()
    global split
    split = args.split
    if args.score:
        print(score(args.score.split(",")))
        return
    out_dir = RUNS / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "results.jsonl"
    done = {json.loads(l)["question_id"] for l in path.read_text().splitlines() if l.strip()} if path.exists() else set()
    want = set(args.ids.split(",")) if args.ids else None
    todo = [t for t in traps() if t["trap_id"] not in done and (want is None or t["trap_id"] in want)]
    print(f"{len(todo)} to run ({len(done)} already in {path})")
    with ThreadPoolExecutor(max_workers=args.workers) as pool, path.open("a") as fh:
        for row in pool.map(run_one, todo):
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"{row['question_id']:16s} {row['action']:8s} diag={row['diagnostic']:8s} want={row['expected']:7s} "
                  f"{str(row.get('headline') or row.get('error'))[:60]}")


if __name__ == "__main__":
    main()
