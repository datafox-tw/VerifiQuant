"""Gemini provider for the 250-question PoC.

Differs from the demo's LiveProvider in four places only:
  retrieve  full-text search over the paper's 250 retrieval cards, then the LLM picks one or none
  get_card  auto-generated invariant cards (cardgen.py), never hand-written
  intent    English questions; also says whether the card's output is what the question asks
  ground    the paper's literal grounding (strict_binding.literal_candidates) replaces the Chinese
            parser; a value may also be a short arithmetic derivation over quoted numbers
  ground_select  when the card returns a structure, the number the question asks for is picked by a
            path (result[1], result["upper_band"][-1]); every step of the path must be backed by the quote
Candidates and the inspector are the demo's, unchanged.
"""

from __future__ import annotations

import ast
import json
import math
import re
import sqlite3

from verifiquant.pipeline.strict_binding import literal_candidates

from .. import providers
from ..providers import LiveProvider, _ask
from . import cardgen
from .data import CARDS_DB

TOP_K = 8
_STOP = {"what", "which", "will", "with", "that", "this", "from", "have", "your", "their", "would", "should",
         "answer", "decimal", "places", "place", "percentage", "nearest", "integer", "calculate", "value",
         "company", "there", "these", "they", "when", "into", "each", "over", "after", "before", "about"}


def _norm(s: str) -> str:
    s = (s or "").replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    s = s.replace("–", "-").replace("—", "-")
    s = s.replace("\\$", "$")  # the dataset escapes dollar signs for LaTeX
    return re.sub(r"\s+", " ", s).strip().lower()


def _in_text(quote, text) -> bool:
    return bool(quote) and _norm(str(quote)) in _norm(text)


def _match(value, cands) -> bool:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(v) and any(math.isclose(v, c, rel_tol=1e-4, abs_tol=1e-9) for c in cands)


def _arith(expr: str, cands: set[float]):
    """Evaluate +-*/** over numbers that are each literally quoted. Anything else -> None."""
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None
    ok_nodes = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div,
                ast.Pow, ast.USub, ast.UAdd)
    for n in ast.walk(tree):
        if not isinstance(n, ok_nodes):
            return None
        if isinstance(n, ast.Constant) and not (isinstance(n.value, (int, float)) and
                                                (_match(n.value, cands) or n.value in (1, 100, 12))):
            return None
    try:
        return float(eval(compile(tree, "<derive>", "eval"), {"__builtins__": {}}))
    except Exception:
        return None


def ground(slot, quote, claimed, text):
    """(status, value, note) for one slot claim. Mirrors the demo's statuses."""
    if claimed is None and not quote:
        return "missing", None, "原文沒有提供"
    if not quote:
        return "ungrounded", None, "LLM 給了值，但沒有原文出處 → 不採用"
    quotes = quote if isinstance(quote, list) else [quote]
    if not all(_in_text(q, text) for q in quotes):
        return "ungrounded", None, "引用的文字不在原文中 → 不採用"
    cands = set().union(*(literal_candidates(str(q)) for q in quotes))
    kind = slot.get("kind")
    numeric = isinstance(claimed, (int, float)) and not isinstance(claimed, bool)
    if kind in ("text", "bool") and not numeric:
        if kind == "bool" and isinstance(claimed, str):
            claimed = claimed.strip().lower() in ("true", "yes", "1")
        return "grounded", claimed, "引用文字在原文中（非數值欄位）"
    if kind == "dict":
        if not isinstance(claimed, dict) or not claimed:
            return "mismatch", None, "應為 {名稱: 數值} → 不採用"
        if not all(_in_text(k, text) for k in claimed) or not all(_match(v, cands) for v in claimed.values()):
            return "mismatch", None, "有名稱不在原文、或數值不在引用文字裡 → 不採用"
        return "grounded", {str(k): float(v) for k, v in claimed.items()}, "每個名稱在原文中，每個數值在引用文字中"
    if kind == "list" or isinstance(claimed, list):
        vals = claimed if isinstance(claimed, list) else None
        if not vals or not all(_match(v, cands) for v in vals):
            return "mismatch", None, "清單中有數值不在引用文字裡 → 不採用"
        return "grounded", [float(v) for v in vals], "清單每個數值都在引用文字中"
    if isinstance(claimed, dict) and claimed.get("derive"):
        v = _arith(str(claimed["derive"]), cands)
        if v is None or not _match(claimed.get("value"), {v}):
            return "mismatch", None, f"推導式「{claimed['derive']}」無法由引用數字重算 → 不採用"
        return "grounded", v, f"由引用數字推導：{claimed['derive']}"
    if _match(claimed, cands):
        v = float(claimed)
        return "grounded", int(v) if kind == "count" and v == int(v) else v, "數值在引用文字中（允許 %、百萬、×12 等換算）"
    return "mismatch", None, "由引用文字重新解析的數值和 LLM 宣稱的不一致 → 不採用"


_ORDINALS = {w: i for i, w in enumerate(
    "first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth thirteenth fourteenth "
    "fifteenth sixteenth seventeenth eighteenth nineteenth twentieth".split(), 1)}
_LAST = ("last", "final", "latest", "most recent", "longest", "ending")
_PERIOD = r"(?:year|month|day|week|quarter|period|semester)s?"


def _positions(quote: str, origin) -> set[int]:
    """Indices the quote can mean. An ordinal right before a period word ("sixth year") or a period word
    before a number ("year 6") is a period number and maps through the card's index_origin; any other
    ordinal ("the second value") counts elements."""
    words = re.findall(r"[a-z0-9]+", _norm(quote))
    out = {-1} if any(w in _LAST for w in words) or "most recent" in _norm(quote) else set()
    for i, w in enumerate(words):
        m = re.fullmatch(r"(\d+)(?:st|nd|rd|th)", w)
        k = _ORDINALS.get(w) or (int(m.group(1)) if m else None)
        nxt = words[i + 1] if i + 1 < len(words) else ""
        if k is not None:
            tied = re.fullmatch(_PERIOD, nxt) and origin is not None
            out.add(k - origin if tied else k - 1)
        elif re.fullmatch(_PERIOD, w) and nxt.isdigit() and origin is not None:
            out.add(int(nxt) - origin)
    return out


def _tokens(s: str) -> set[str]:
    return {w[:5] for w in re.findall(r"[a-z]{3,}", _norm(s).replace("_", " "))}


def ground_select(card, claim, text):
    """(status, path, note) for the path that picks the asked number out of a structured output."""
    st = card["output"].get("structure")
    if not st:
        return "grounded", None, "卡片輸出是單一數值"
    claim = claim if isinstance(claim, dict) else {}
    quote, path = claim.get("quote"), claim.get("path")
    if not isinstance(quote, str) or not _in_text(quote, text):
        return "ungrounded", None, "選取路徑沒有原文出處 → 不採用"
    if isinstance(path, list) and len(path) > len(st["levels"]):
        # the model sometimes writes the level names into the path: ["index", 1] -> [1]
        path = [x for x in path if x not in ("index", "field", "key")]
    if not isinstance(path, list) or len(path) != len(st["levels"]):
        return "mismatch", None, f"路徑應有 {len(st['levels'])} 層（{' / '.join(st['levels'])}）"
    out = []
    for level, step in zip(st["levels"], path):
        if level == "index":
            try:
                step = int(step)
            except (TypeError, ValueError):
                return "mismatch", None, f"位置「{step}」不是整數"
            if step not in _positions(quote, st["index_origin"]):
                return "mismatch", None, f"引用文字「{quote}」推不出位置 {step}"
        elif level == "field":
            f = next((x for x in st["fields"] if str(step) in (str(x["key"]), x["name"])), None)
            if f is None:
                return "mismatch", None, f"「{step}」不是卡片宣告的欄位"
            if not (_tokens(f["name"]) & _tokens(quote)):
                return "mismatch", None, f"引用文字沒有提到欄位「{f['name']}」"
            step = f["key"]
        elif not isinstance(step, str) or not _in_text(step, quote):
            return "mismatch", None, f"名稱「{step}」不在引用文字中"
        out.append(step)
    return "grounded", out, "路徑每一層都由引用文字決定"


def _fts_query(text: str) -> str:
    words = {w for w in re.findall(r"[a-z][a-z\-]{3,}", text.lower()) if w not in _STOP}
    return " OR ".join(f'"{w}"' for w in sorted(words)) or '"finance"'


def search(text: str, k: int = TOP_K) -> list[dict]:
    con = sqlite3.connect(CARDS_DB)
    rows = con.execute(
        "select f.fic_id, r.title, r.summary from retrieval_fts f join retrieval_cards r on r.fic_id = f.fic_id "
        "where retrieval_fts match ? order by bm25(retrieval_fts) limit ?", (_fts_query(text), k)).fetchall()
    return [{"fic_id": a, "title": b, "summary": c} for a, b, c in rows]


class PocProvider(LiveProvider):
    mode = "poc"
    verify_inspector = False  # on = v6a/v6b; measured harmful (test-1043), so off by default
    self_check = False  # v6b: drop the invariants the reference itself violates (selfcheck.py)

    def retrieve(self, question):
        hits = search(question)
        menu = "\n".join(f"- {h['fic_id']}: {h['title']}. {h['summary']}" for h in hits)
        prompt = (f"Pick the calculation card that computes what this finance question asks for, or answer none "
                  f"if no card below does.\n\n{menu}\n\nQuestion:\n{question}\n\nAnswer with only the card id or none.")
        reply = _ask(prompt, 0.0).strip().strip("`").strip()
        self._log("retrieve", prompt, reply)
        ids = [h["fic_id"] for h in hits]
        card_id = reply if reply in ids else None
        table = [{"card": h["title"], "id": h["fic_id"], "score": len(hits) - i} for i, h in enumerate(hits)]
        return card_id, table, f"全文檢索取前 {len(hits)} 張，由 {providers.MODEL} 選卡（回答：{reply}）"

    def get_card(self, card_id):
        card = cardgen.get(card_id)["card"]
        if self.self_check:
            from . import selfcheck
            card = selfcheck.apply(card)
        return card

    def ground(self, slot, quote, claimed, text):
        return ground(slot, quote, claimed, text)

    def ground_select(self, card, claim, text):
        return ground_select(card, claim, text)

    def intent(self, question, card):
        slots = "\n".join(f'- {s["name"]}: {s["label"]} (kind {s["kind"]}, unit {s["unit"] or "-"}'
                          f'{", optional" if s.get("optional") else ""})' for s in card["slots"])
        convs = "\n".join(f'- {c["name"]}: {c["question"]} options {[o["value"] for o in c["options"]]}'
                          for c in card["conventions"]) or "(none)"
        st = card["output"].get("structure")
        structure = select_rule = select_key = ""
        if st:
            fields = "; ".join(f'{x["key"]!r} = {x["name"]} ({x["label"]})' for x in st["fields"])
            structure = (f"\nThe output is a STRUCTURE: {st['text']}\nPath levels from the outside in: "
                         f"{st['levels']}" + (f"; index: {st['index_meaning']}" if "index" in st["levels"] else "")
                         + (f"; fields: {fields}" if fields else "") + "\n")
            select_rule = """
The card returns a structure, so also return "select": {"quote": <exact substring of the question that
says which number is asked>, "path": [<one step per level>]}. Index steps are 0-based integers; use -1
for the last / final / longest element. Field steps are the field key. Key steps are the key as written.
"card_answers_question" is then true when the asked number is one element of the structure."""
            example = [{"index": 0, "field": st["fields"][0]["key"] if st["fields"] else 0, "key": "<name>"}[lv]
                       for lv in st["levels"]]
            select_rule += f"\nThe path holds values only, one per level, e.g. {json.dumps(example)} for levels {st['levels']}."
            select_key = ', "select": {"quote": "...", "path": [...]}'
        prompt = f"""You extract an intent spec from a finance question for the calculation card below.
Do not compute the answer.

Card: {card['title']}. Output: {card['output']['label']} ({card['output']['unit'] or '-'}).
Signature: {card['signature']}
{structure}
Slots:
{slots}

Conventions:
{convs}

For each slot return {{"quote": <exact substring of the text that states it>, "value": <value>}}:
- numbers as plain numbers; rates as decimals (5% -> 0.05); scale words expanded (2.5 million -> 2500000)
- list slots: "value" is a JSON list and "quote" may be a list of exact substrings
- dict slots: "value" is a JSON object {{name: number}} and "quote" may be a list of exact substrings
- if the value needs simple arithmetic on quoted numbers (e.g. years between two dates), give
  "value": {{"derive": "<arithmetic on the quoted numbers>", "value": <result>}}
- an optional slot the text does not mention: {{"quote": null, "value": null}}
- a required slot the text does not give: {{"quote": null, "value": null}}. Never invent values.
For each convention return {{"quote": <exact substring that decides it, or null>, "value": <option or null>}}.

Also return "asked_quantity" (what the question asks for, in a few words) and "card_answers_question":
true if the card's output IS the asked quantity (the same number, possibly shown as a percentage or
rounded), false if the question needs a different or further-processed quantity.{select_rule}

Text:
{question}

Return JSON: {{"slots": {{...}}, "conventions": {{...}}, "asked_quantity": "...", "card_answers_question": true{select_key}}}"""
        reply = _ask(prompt, 0.0, as_json=True)
        self._log("intent", prompt, reply)
        try:
            raw = json.loads(reply)
        except json.JSONDecodeError:
            return {"slots": {}, "conventions": {}}
        return raw if isinstance(raw, dict) else {"slots": {}, "conventions": {}}

    def baseline(self, question):
        return {"kind": "none", "text": ""}
