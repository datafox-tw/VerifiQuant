"""Card self-check (v6b): drop the invariants that the reference implementation itself violates.

nl2postcond (FSE 2024) calls a postcondition correct when the reference implementation satisfies it; VERINA's
completeness asks the same of a specification. Building a card has no question, so the inputs are example input
sets that the LLM proposes from the library function alone (never a question, a context or a gold field). A set is
used only if it satisfies the card's own pre invariants and the reference runs on it. Each set is also run at
x1.1 and x0.9. Before checking, a guard written as a conjunction (`x > y and close(f(...), ...)`) is rewritten as
the implication it means (`not (x > y) or close(...)`): as a conjunction it is False, and rejects every program,
whenever the inputs do not meet the guard. An invariant (post / meta) that still fails or raises on any usable run
is dropped. Boundary invariants
pin conventions and are left alone. The cards in cards_auto.jsonl are never edited: the drop list lives in
self_check.jsonl and is applied when PocProvider.self_check is on.

    python3 -m harness_demo.poc.selfcheck fic_article_2311     # one card, prints what it drops
"""

from __future__ import annotations

import ast
import json
import threading

from .. import providers, sandbox
from . import cardgen
from .data import ROOT, fic_id, library_function, questions
from .grade_cards import ALLOW, _perturb, _wrapper

STORE = ROOT / "harness_demo/poc/self_check.jsonl"
_lock = threading.Lock()
_store: dict[str, dict] | None = None

PROMPT = """Give 4 realistic, varied example input sets for this finance function, as a JSON list of objects keyed
by the argument names. Use the magnitudes a real user would type (money in units, rates as decimals, counts as
integers, lists and dicts in the shape the function expects). Do not compute the output.

```python
{code}
```

Return only the JSON list."""


def _touches_f(node) -> bool:
    return any(isinstance(n, ast.Name) and n.id in ("f", "result") for n in ast.walk(node))


def as_implication(check: str) -> str:
    """`g1 and g2 and body` with input-only guards g1, g2 -> `not (g1 and g2) or (body)`; anything else unchanged."""
    try:
        tree = ast.parse(check, mode="eval").body
    except SyntaxError:
        return check
    if not (isinstance(tree, ast.BoolOp) and isinstance(tree.op, ast.And)):
        return check
    n = 0
    while n < len(tree.values) and not _touches_f(tree.values[n]):
        n += 1
    if n == 0 or n == len(tree.values):
        return check
    guard = " and ".join(f"({ast.unparse(v)})" for v in tree.values[:n])
    body = " and ".join(f"({ast.unparse(v)})" for v in tree.values[n:])
    return f"not ({guard}) or ({body})"


def repaired(card: dict) -> dict:
    invs = [i | {"check": as_implication(i["check"])} if i["kind"] in ("post", "meta") else i for i in card["invariants"]]
    return card | {"invariants": invs}


def _load() -> dict[str, dict]:
    if not STORE.exists():
        return {}
    return {r["fic_id"]: r for r in map(json.loads, STORE.read_text().splitlines()) if r}


def check(fid: str) -> dict:
    """The self-check record for a card, computed once and cached on disk."""
    global _store
    with _lock:
        if _store is None:
            _store = _load()
        if fid in _store:
            return _store[fid]
    card = repaired(cardgen.get(fid)["card"])
    q = next(q for q in questions().values() if fic_id(q) == fid)
    code, fn = library_function(q)
    args = [a.arg for a in fn.args.args]
    required = set(args[: len(args) - len(fn.args.defaults)])
    try:
        sets = json.loads(providers._ask(PROMPT.format(code=code), 0.0, as_json=True))
    except (json.JSONDecodeError, Exception):  # no examples -> nothing is dropped, and the record says why
        sets = []
    sets = [s for s in sets if isinstance(s, dict) and required <= set(s) <= set(args)] if isinstance(sets, list) else []
    pre = [i for i in card["invariants"] if i["kind"] == "pre"]
    checks = [i for i in card["invariants"] if i["kind"] in ("post", "meta")]
    ref = code + _wrapper(fn.name, args)
    failed, used = {}, 0
    for base in sets:
        if pre and not all(c["ok"] for c in sandbox.eval_pre(pre, base).values()):
            continue
        for inputs in (base, _perturb(base, 1.1), _perturb(base, 0.9)):
            rep = sandbox.run(ref, inputs, checks, allow=ALLOW, timeout=10, raw=True)
            if not rep["ok"]:
                continue  # the reference cannot run here: this run is no evidence about any invariant
            used += 1
            for iid, c in rep["checks"].items():
                if not c["ok"]:
                    failed.setdefault(iid, c["error"] or "False on the reference")
    rewritten = [i["id"] for i, j in zip(card["invariants"], cardgen.get(fid)["card"]["invariants"]) if i["check"] != j["check"]]
    rec = {"fic_id": fid, "n_sets": len(sets), "runs_used": used, "dropped": sorted(failed), "why": failed,
           "rewritten": rewritten, "examples": sets,
           "status": "checked" if used else "no_usable_inputs"}
    with _lock:
        _store[fid] = rec
        with STORE.open("a") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def apply(card: dict) -> dict:
    rec = check(card["id"])
    drop = set(rec["dropped"])
    card = repaired(card)
    return card | {"invariants": [i for i in card["invariants"] if i["id"] not in drop], "self_check": rec}


if __name__ == "__main__":
    import sys

    for fid in sys.argv[1:]:
        r = check(fid)
        print(fid, r["status"], f"runs {r['runs_used']}", "dropped", r["dropped"])
