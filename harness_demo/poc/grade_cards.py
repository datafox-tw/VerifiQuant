"""VERINA-style grading of the auto cards (G3), the same idea as the demo's VERINA lens.

Reference f = the library function. Inputs = the question's own values, taken from the harness run
(grounded slots) or, failing that, from literal assignments in the gold solution, plus two perturbed
grids. Gold is used here only, never by the harness.

  complete   every invariant holds for the reference on every grid (no false alarms)
  sound      share of non-equivalent mutants that some invariant rejects
  shape      which invariant forms the card has: base case, recurrence/relation, inequality only

    python3 -m harness_demo.poc.grade_cards --run full
"""

from __future__ import annotations

import argparse
import ast
import json
import math

from .. import sandbox
from . import cardgen
from .data import RUNS, fic_id, library_function, questions, sample_ids

ALLOW = ["math", "scipy", "numpy", "datetime"]

OUTPUT_MUTANTS = {
    "scale_1.001": "v * 1.001",
    "offset_small": "v + (1 if abs(v) > 10 else 0.01)",
    "round_2dp": "round(v, 2)",
    "percent_x100": "v * 100",
    "sign_flip": "-v",
}


_AT = """

def _at(v, path, fn):
    if not path:
        return fn(v)
    k = path[0]
    if isinstance(v, dict):
        return {**v, k: _at(v[k], path[1:], fn)}
    out = list(v)
    out[k] = _at(v[k], path[1:], fn)
    return tuple(out) if isinstance(v, tuple) else out
"""


def _wrapper(lib_name: str, args: list[str], pre: str = "", post: str = "v", path: list | None = None) -> str:
    if path and post != "v":  # structured output: the output mutant changes the asked cell only
        post = f"_at(v, {path!r}, lambda v: {post})"
    return ((_AT if path else "") + f"\n\ndef solve(*a, **k):\n"
            f"    ba = dict(zip({args!r}, a))\n    ba.update(k)\n{pre}"
            f"    v = {lib_name}(**ba)\n    return {post}\n")


def mutants(lib_name: str, args: list[str], inputs: dict, path: list | None = None) -> dict[str, str]:
    out = {name: _wrapper(lib_name, args, post=expr, path=path) for name, expr in OUTPUT_MUTANTS.items()}
    for a, v in inputs.items():
        if isinstance(v, bool):
            continue
        if isinstance(v, int) or (isinstance(v, float) and v == int(v) and 1 <= v <= 400):
            out[f"off_by_one_{a}"] = _wrapper(lib_name, args, pre=f"    ba[{a!r}] = ba[{a!r}] - 1\n")
        elif isinstance(v, float) and 0 < abs(v) < 1:
            out[f"percent_input_{a}"] = _wrapper(lib_name, args, pre=f"    ba[{a!r}] = ba[{a!r}] * 100\n")
        elif isinstance(v, list) and len(v) > 1:
            out[f"drop_last_{a}"] = _wrapper(lib_name, args, pre=f"    ba[{a!r}] = ba[{a!r}][:-1]\n")
    return out


def _literal_inputs(q: dict, args: list[str]) -> dict:
    tree = ast.parse(q["python_solution"])
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in args:
                try:
                    found[name] = ast.literal_eval(node.value)
                except ValueError:
                    pass
    return found


def _perturb(inputs: dict, factor: float) -> dict:
    out = {}
    for k, v in inputs.items():
        if isinstance(v, bool) or isinstance(v, str):
            out[k] = v
        elif isinstance(v, int):
            out[k] = v + 1 if factor > 1 else max(1, v - 1) if v > 1 else v
        elif isinstance(v, float):
            out[k] = v * factor
        elif isinstance(v, list):
            out[k] = [x * factor if isinstance(x, float) else x for x in v]
        else:
            out[k] = v
    return out


def _shape(card: dict) -> dict:
    kinds = {"base": 0, "relation": 0, "inequality": 0, "pre": 0, "boundary": 0}
    for inv in card["invariants"]:
        if inv["kind"] == "pre":
            kinds["pre"] += 1
            continue
        if inv["kind"] == "boundary":
            kinds["boundary"] += 1
            continue
        tree = ast.parse(inv["check"], mode="eval")
        closes = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "close"]
        if not closes:
            kinds["inequality"] += 1
            continue
        sides = closes[0].args[:2]
        uses_f = [any(isinstance(x, ast.Name) and x.id in ("f", "result") for x in ast.walk(s)) for s in sides]
        kinds["base" if uses_f.count(True) == 1 else "relation"] += 1
    kinds["pins"] = kinds["base"] >= 1 and kinds["relation"] >= 1
    kinds["weak"] = kinds["base"] == 0 and kinds["relation"] == 0
    return kinds


def grade(qid: str, run_rows: dict) -> dict:
    q = questions()[qid]
    fid = fic_id(q)
    card = cardgen.get(fid)["card"]
    code, fn = library_function(q)
    args = [a.arg for a in fn.args.args]
    row = {"question_id": qid, "card": fid, "title": card["title"], "n_invariants": len(card["invariants"]),
           "shape": _shape(card)}
    run = run_rows.get(qid) or {}
    path = None
    if card["output"].get("structure"):
        sel = run.get("select") or {}
        if run.get("selected_card") != fid or sel.get("status") != "grounded":
            row.update(gradeable=False, reason="structured output but the run has no grounded path to the asked cell")
            return row
        path = sel["path"]
        row["path"] = path
    inputs = {}
    if run.get("selected_card") == fid:
        inputs = {s["name"]: s["parsed"] for s in run.get("slots") or [] if s["status"] == "grounded"}
    source = "harness"
    if set(inputs) != {a for a in args if not any(s["name"] == a and s.get("optional") for s in card["slots"])}:
        inputs, source = _literal_inputs(q, args), "gold_literals"
    row["input_source"] = source
    gold = code + _wrapper(fn.name, args)
    checks = [i for i in card["invariants"] if i["kind"] != "pre"]
    grids = []
    for g in (inputs, _perturb(inputs, 1.1), _perturb(inputs, 0.9)):
        rep = sandbox.run(gold, g, checks, allow=ALLOW, timeout=10, path=path)
        if rep["ok"]:
            grids.append((g, rep))
    if not grids:
        row["gradeable"] = False
        row["reason"] = "reference does not run on the extracted inputs (multi-value output or inputs not found)"
        return row
    row["gradeable"], row["grids"] = True, len(grids)
    pre = [i for i in card["invariants"] if i["kind"] == "pre"]
    pre_res = sandbox.eval_pre(pre, grids[0][0]) if pre else {}
    false_alarms = sorted({i["id"] for i in pre if not pre_res[i["id"]]["ok"]} |
                          {iid for _, rep in grids for iid, c in rep["checks"].items()
                           if not c["ok"] and next(i for i in checks if i["id"] == iid)["kind"] != "boundary"})
    # boundary: the reference must fit at least one option of every convention
    unfit = []
    for conv in card["conventions"]:
        fits = False
        for opt in conv["options"]:
            b = [i for i in checks if i["kind"] == "boundary" and i["when"].get(conv["name"]) == opt["value"]]
            if b and all(rep["checks"][i["id"]]["ok"] for _, rep in grids for i in b):
                fits = True
        if not fits and any(i["kind"] == "boundary" and conv["name"] in i["when"] for i in checks):
            unfit.append(conv["name"])
    row["false_alarms"] = false_alarms
    row["unfit_conventions"] = unfit
    row["complete"] = not false_alarms and not unfit
    row["invariant_validity"] = round(1 - len(false_alarms) / max(1, len(card["invariants"])), 3)
    # soundness over mutants; invariants that false-alarm on the reference are excluded so they cannot "kill"
    good = [i for i in checks if i["id"] not in false_alarms and i["kind"] != "boundary"]
    killed, survived, equivalent, crashed = [], [], [], []
    for name, wrap in mutants(fn.name, args, grids[0][0], path).items():
        mcode = code + wrap
        reps = [sandbox.run(mcode, g, good, allow=ALLOW, timeout=10, path=path) for g, _ in grids]
        if not reps[0]["ok"]:
            crashed.append(name)
            continue
        same = all(r["ok"] and gr["result"] is not None and r["result"] is not None and
                   math.isclose(r["result"], gr["result"], rel_tol=1e-9, abs_tol=1e-12)
                   for r, (_, gr) in zip(reps, grids))
        if same:
            equivalent.append(name)
        elif any((not r["ok"]) or any(not c["ok"] for c in r["checks"].values()) for r in reps):
            killed.append(name)
        else:
            survived.append(name)
    row.update(killed=killed, survived=survived, equivalent=equivalent, crashed=crashed)
    n = len(killed) + len(survived)
    row["kill_rate"] = round(len(killed) / n, 3) if n else None
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="full")
    args = ap.parse_args()
    path = RUNS / args.run / "results.jsonl"
    run_rows = {r["question_id"]: r for r in map(json.loads, path.read_text().splitlines())} if path.exists() else {}
    out = RUNS / args.run / "card_grades.jsonl"
    with out.open("w") as fh:
        for qid in sample_ids():
            row = grade(qid, run_rows)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"{qid:10s} {row['title'][:18]:18s} complete={row.get('complete')} kill={row.get('kill_rate')} "
                  f"survived={row.get('survived')} fa={row.get('false_alarms')} src={row.get('input_source')}")


if __name__ == "__main__":
    main()
