"""Generate an invariant card from a library function, then audit it for oracle leaks.

The generator sees the library function and its docstring (the same source the paper's
FIC cards were built from), never a question, its context, or any gold field.

The audit is the hard gate (the prompt is aspirational):
  - names: only f, result, close, math, the slots, safe builtins and comprehension variables
  - no call to the library function itself
  - no closed-form anchor: `close(result, <expr without f or result>)` restates the formula,
    so verification would compare the answer to itself
  - pre invariants may not touch f or result; boundary invariants need a valid `when`
The signature and slot names are taken from the function, not from the LLM.
"""

from __future__ import annotations

import ast
import json
import threading

from .. import providers
from ..cards import CARDS
from .data import AUTO_CARDS, fic_id, library_function, questions

SAFE_NAMES = {"f", "result", "close", "math", "abs", "min", "max", "sum", "range", "len", "float", "int",
              "round", "pow", "all", "any", "zip", "enumerate", "list", "tuple", "sorted", "reversed",
              "isinstance", "bool", "True", "False", "None"}
KINDS = {"pre", "post", "meta", "boundary"}

_lock = threading.Lock()


def _example() -> str:
    c = CARDS["annuity_fv"]
    keep = {k: c[k] for k in ("title", "topic", "output", "slots", "conventions", "fixed", "invariants")}
    return json.dumps(keep, ensure_ascii=False, indent=1)


PROMPT = """You write an INVARIANT CARD for one kind of financial calculation.

A card does not contain the formula. It lists what must be known (slots, conventions) and what must be
TRUE of any correct program `f` (invariants). Another model will write `f`; the card only judges it.

Library function this card covers (the reference; you may read it, but must not copy its formula):
```python
{code}
```

The program under test has the signature `solve({args})`, with exactly these argument names.

Write invariants as Python expressions. In scope: f (the program), result = f(all original inputs),
close(a, b, rel=1e-6), math, every input as a bare name, and safe builtins (abs, min, max, sum, range,
len, round, all, any, zip, enumerate).

Kinds:
- pre: a check on inputs alone (ranges, types, consistency between inputs). No f, no result.
- post: a property of f at chosen points. Best are BASE CASES at special inputs where the answer is
  obvious (zero rate, one period, zero cash flow, equal prices) and RECURRENCES that relate f at n+1
  to f at n. A base case plus a recurrence pins the answer without a gold value.
- meta: a relation between two runs of f (scaling an input scales the output, monotonicity, symmetry).
- boundary: pins one option of a convention (e.g. payment timing) at a special point; give "when":
  {{"<convention name>": "<option value>"}}.

FORBIDDEN (these are deleted automatically):
- comparing `result` (or f at the original inputs) directly to a closed-form expression: that restates
  the formula, so it proves nothing. A base case at a DIFFERENT, special input is fine.
- guarding on the original inputs, e.g. `x == 0 and close(result, ...)`: it only fires when the given
  inputs happen to be special. Instead CALL f at the special input yourself:
  `close(f(price=strike + 10, strike=strike, ...), -premium * shares)`.
- calling the library function by name, importing, or using names not listed above.

Conventions are readings a question may leave open that change the answer (payment timing, compounding
frequency, 360 vs 365 days, percent vs decimal output). List only real ones for THIS calculation, with
"cues": phrases that, when quoted from a question, decide each option. Use [] if there are none.
Fixed assumptions are choices the card makes that a user should see.

Slots: one per argument, in order, with "name" equal to the argument name, a Traditional Chinese
"label", "kind" in [money, rate, count, number, list, text, bool], and "unit" ("小數" for rates given as
decimals). Output: {{"name", "label", "unit"}}.

Aim for 6-12 invariants. Prefer few strong ones (base case + recurrence, exact scaling) over many sign
checks. Write labels and "text" fields in Traditional Chinese.

Here is a finished card for a different calculation, showing the shape (its slots differ from yours):
{example}

Return JSON with keys: title, topic, output, slots, conventions, fixed, invariants, completeness_note.
"""


STRUCT_PROMPT = """This finance function returns a structure, not one number. Describe its LAYOUT so that
another program can return the same layout and a reader can pick one number out of it by a path such as
result[2], result["upper_band"][-1] or result[0]["interest_paid"]. Do not describe the formula.

```python
{code}
```

Return JSON:
- "text": one Traditional Chinese sentence on the layout (what each element is, in what order)
- "levels": the path steps from the outside in, each one of
    "index"  a position in a sequence of periods or items (years, months, CDs, prices)
    "field"  a fixed dict key or tuple/list position with a fixed meaning (listed in "fields")
    "key"    a dict key taken from the inputs (e.g. department names)
- "index_origin": when an "index" level counts periods, the period number of element 0. Trace the code:
  find the loop that appends, and the period the FIRST appended element belongs to (a loop over
  range(years) that appends the value at the end of each year starts at year 1, so 1; a list that starts
  with the value at time 0 gives 0). null when the index counts items, not periods
- "index_meaning" must agree with index_origin (origin 1 -> "第 i+1 年…", origin 0 -> "第 i 年…")
- "index_meaning": Traditional Chinese, what element i is
- "fields": for a "field" level, [{{"key": <dict key string or integer position>, "name": <English snake_case>,
  "label": <Traditional Chinese>, "unit": <unit or "">}}]; [] otherwise
- "element_unit": unit of the numbers when there is no "field" level ("小數" for decimal fractions)
"""

LEVELS = {"index", "field", "key"}


def _returns_structure(fn: ast.FunctionDef) -> bool:
    ann = ast.unparse(fn.returns).lower() if fn.returns else ""
    return ann.split("[")[0] in ("list", "dict", "tuple")


def describe_output(code: str) -> dict | None:
    """Layout of a structured return value, or None when the reply is unusable (the card stays scalar)."""
    reply = providers._ask(STRUCT_PROMPT.format(code=code), 0.0, as_json=True, think=True)
    try:
        raw = json.loads(reply)
    except json.JSONDecodeError:
        return None
    levels = raw.get("levels") if isinstance(raw, dict) else None
    if not isinstance(levels, list) or not 1 <= len(levels) <= 3 or any(x not in LEVELS for x in levels):
        return None
    fields = [x for x in raw.get("fields") or [] if isinstance(x, dict) and isinstance(x.get("key"), (str, int))
              and not isinstance(x.get("key"), bool) and x.get("name")]
    if "field" in levels and not fields:
        return None
    origin = raw.get("index_origin")
    return {"text": str(raw.get("text") or ""), "levels": levels,
            "index_origin": origin if isinstance(origin, int) and not isinstance(origin, bool) else None,
            "index_meaning": str(raw.get("index_meaning") or ""),
            "fields": [{"key": x["key"], "name": str(x["name"]), "label": str(x.get("label") or x["name"]),
                        "unit": str(x.get("unit") or "")} for x in fields],
            "element_unit": str(raw.get("element_unit") or "")}


def _signature(fn: ast.FunctionDef) -> tuple[list[str], str, set[str]]:
    args = [a.arg for a in fn.args.args]
    n_def = len(fn.args.defaults)
    optional = set(args[len(args) - n_def:]) if n_def else set()
    parts = []
    for a, d in zip(args, [None] * (len(args) - n_def) + list(fn.args.defaults)):
        parts.append(a if d is None else f"{a}={ast.unparse(d)}")
    return args, f"solve({', '.join(parts)})", optional


class _Names(ast.NodeVisitor):
    def __init__(self):
        self.used, self.bound, self.calls = set(), set(), []

    def visit_comprehension(self, node):
        for t in ast.walk(node.target):
            if isinstance(t, ast.Name):
                self.bound.add(t.id)
        self.generic_visit(node)

    def visit_Lambda(self, node):
        self.bound.update(a.arg for a in node.args.args)
        self.generic_visit(node)

    def visit_Name(self, node):
        self.used.add(node.id)

    def visit_Call(self, node):
        self.calls.append(node)
        self.generic_visit(node)


def _is_original_point(node, args) -> bool:
    """result, or f(...) called with every input unchanged."""
    if isinstance(node, ast.Name) and node.id == "result":
        return True
    if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "f":
        pos = dict(zip(args, node.args))
        kw = pos | {k.arg: k.value for k in node.keywords}
        return all(isinstance(v, ast.Name) and v.id == k for k, v in kw.items())
    return False


def _touches_f(node) -> bool:
    return any((isinstance(n, ast.Name) and n.id in ("f", "result")) for n in ast.walk(node))


def audit_invariant(inv: dict, args: list[str], fn_name: str, conv_options: dict) -> str | None:
    """Return a rejection reason, or None if the invariant is admissible."""
    if inv.get("kind") not in KINDS:
        return f"unknown kind {inv.get('kind')}"
    try:
        tree = ast.parse(inv.get("check", ""), mode="eval")
    except SyntaxError:
        return "syntax error"
    v = _Names()
    v.visit(tree)
    free = v.used - v.bound - SAFE_NAMES - set(args)
    if fn_name in v.used:
        return f"calls the library function {fn_name}"
    if free:
        return f"unknown names {sorted(free)}"
    if any(isinstance(n, (ast.Attribute)) and n.attr.startswith("__") for n in ast.walk(tree)):
        return "dunder attribute"
    if inv["kind"] == "pre" and _touches_f(tree):
        return "pre invariant uses f or result"
    if inv["kind"] == "boundary":
        when = inv.get("when") or {}
        if not when or any(o not in conv_options.get(c, ()) for c, o in when.items()):
            return "boundary without a valid convention option"
    # closed-form anchor: close(original point, expr) or original point == expr, with expr free of f
    pairs = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "close" and len(n.args) >= 2:
            pairs.append((n.args[0], n.args[1]))
        if isinstance(n, ast.Compare) and len(n.ops) == 1 and isinstance(n.ops[0], ast.Eq):
            pairs.append((n.left, n.comparators[0]))
    for a, b in pairs:
        for x, y in ((a, b), (b, a)):
            if _is_original_point(x, args) and not _touches_f(y):
                return "closed-form anchor (restates the formula at the original inputs)"
    return None


def build_card(raw: dict, fid: str, fn: ast.FunctionDef) -> tuple[dict, list[dict]]:
    args, signature, optional = _signature(fn)
    by_name = {s.get("name"): s for s in raw.get("slots") or [] if isinstance(s, dict)}
    ann = {a.arg: ast.unparse(a.annotation).lower() if a.annotation else "" for a in fn.args.args}
    slots = []
    for a in args:
        s = by_name.get(a) or {}
        # the type annotation decides the kind when it is not a plain number; the LLM only refines numbers
        kind = {"str": "text", "bool": "bool", "list": "list", "dict": "dict"}.get(ann[a].split("[")[0])
        if kind is None:
            kind = s.get("kind") if s.get("kind") in ("money", "rate", "count", "number") else "number"
        slots.append({"name": a, "label": s.get("label") or a, "kind": kind,
                      "unit": s.get("unit") or "", "optional": a in optional})
    convs = []
    for c in raw.get("conventions") or []:
        opts = [o for o in c.get("options") or [] if isinstance(o, dict) and o.get("value")]
        if c.get("name") and len(opts) >= 2:
            convs.append({"name": c["name"], "label": c.get("label") or c["name"],
                          "question": c.get("question") or c.get("label") or c["name"],
                          "options": [{"value": str(o["value"]), "label": o.get("label") or str(o["value"])}
                                      for o in opts],
                          "cues": {str(k): list(v) for k, v in (c.get("cues") or {}).items() if isinstance(v, list)}})
    conv_options = {c["name"]: {o["value"] for o in c["options"]} for c in convs}
    kept, rejected, seen = [], [], set()
    for i, inv in enumerate(raw.get("invariants") or []):
        if not isinstance(inv, dict):
            continue
        inv = dict(inv)
        inv["id"] = str(inv.get("id") or f"inv{i}")
        while inv["id"] in seen:
            inv["id"] += "_"
        seen.add(inv["id"])
        reason = audit_invariant(inv, args, fn.name, conv_options)
        if reason:
            rejected.append({"id": inv["id"], "check": inv.get("check"), "reason": reason})
            continue
        kept.append({"id": inv["id"], "kind": inv["kind"], "text": inv.get("text") or inv["check"],
                     "check": inv["check"], "when": inv.get("when") or {}, "source": inv.get("source", "auto")})
    out = raw.get("output") or {}
    card = {"id": fid, "title": raw.get("title") or fn.name, "topic": raw.get("topic") or "",
            "signature": signature, "slots": slots, "conventions": convs,
            "output": {"name": out.get("name") or "result", "label": out.get("label") or "答案",
                       "unit": out.get("unit") or ""},
            "fixed": [{"name": str(x.get("name") or f"fixed{i}"), "text": str(x.get("text") or "")}
                      for i, x in enumerate(raw.get("fixed") or []) if isinstance(x, dict)],
            "invariants": kept, "completeness_note": raw.get("completeness_note") or "",
            "keywords": []}
    return card, rejected


def _load() -> dict[str, dict]:
    if not AUTO_CARDS.exists():
        return {}
    return {r["fic_id"]: r for r in map(json.loads, AUTO_CARDS.read_text().splitlines())}


_store: dict[str, dict] | None = None


def get(fid: str) -> dict:
    """Card for a fic id, generated once and cached on disk."""
    global _store
    with _lock:
        if _store is None:
            _store = _load()
        rec = _store.get(fid)
    q = next(q for q in questions().values() if fic_id(q) == fid)
    code, fn = library_function(q)
    if rec is not None:
        if _returns_structure(fn) and "structure" not in rec:
            # added after the invariants were generated: the invariants stay as they were
            rec = rec | {"structure": describe_output(code)}
            with _lock:
                _store[fid] = rec
                AUTO_CARDS.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in _store.values()))
        return _with_structure(rec)
    args, _, _ = _signature(fn)
    prompt = PROMPT.format(code=code, args=", ".join(args), example=_example())
    for attempt in range(1, 4):  # malformed JSON or nothing executable left after the audit -> ask again
        reply = providers._ask(prompt, 0.2, as_json=True, think=True)
        try:
            raw = json.loads(reply)
        except json.JSONDecodeError:
            raw = {}
        card, rejected = build_card(raw if isinstance(raw, dict) else {}, fid, fn)
        if any(i["kind"] != "pre" for i in card["invariants"]):
            break
    rec = {"fic_id": fid, "card": card, "rejected": rejected, "raw": reply, "attempts": attempt}
    if _returns_structure(fn):
        rec["structure"] = describe_output(code)
    with _lock:
        _store[fid] = rec
        with AUTO_CARDS.open("a") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return _with_structure(rec)


def _with_structure(rec: dict) -> dict:
    if not rec.get("structure"):
        return rec
    card = rec["card"] | {"output": rec["card"]["output"] | {"structure": rec["structure"]}}
    return rec | {"card": card}


def rebuild() -> None:
    """Re-apply build_card and the audit to every cached LLM reply (no API calls)."""
    global _store
    recs = list(_load().values())
    lines = []
    for r in recs:
        q = next(q for q in questions().values() if fic_id(q) == r["fic_id"])
        _, fn = library_function(q)
        try:
            raw = json.loads(r["raw"])
        except json.JSONDecodeError:
            raw = {}
        card, rejected = build_card(raw if isinstance(raw, dict) else {}, r["fic_id"], fn)
        lines.append(json.dumps(r | {"card": card, "rejected": rejected}, ensure_ascii=False))
    AUTO_CARDS.write_text("\n".join(lines) + "\n")
    _store = None


if __name__ == "__main__":
    import sys
    from concurrent.futures import ThreadPoolExecutor

    from .data import sample_ids

    qs = questions()
    fids = [fic_id(qs[i]) for i in sample_ids()]
    if len(sys.argv) > 1:
        fids = fids[: int(sys.argv[1])]
    with ThreadPoolExecutor(max_workers=6) as pool:
        recs = list(pool.map(get, fids))
    for r in recs:
        c = r["card"]
        print(f"{r['fic_id']:22s} kept {len(c['invariants']):2d} rejected {len(r['rejected']):2d} "
              f"conv {len(c['conventions'])}  {c['title']}")
