"""Where the LLM-shaped pieces come from.

ScriptedProvider reads hand-written stand-ins from cases.py (no network).
LiveProvider calls Gemini. Both expose the same five methods to harness.run().
"""

from __future__ import annotations

import json
import os
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .cards import CARDS, retrieve
from .cases import INSPECTORS

MODEL = os.environ.get("HARNESS_DEMO_MODEL", "gemini-2.5-flash")


class ScriptedProvider:
    mode = "scripted"
    can_regenerate = False

    def __init__(self, case: dict):
        self.case = case
        self.calls = []

    def retrieve(self, question):
        card_id, table = retrieve(question)
        return card_id, table, "關鍵字比對（確定性）"

    def intent(self, question, card):
        self.calls.append({"role": "intent", "source": "示範用抽取結果"})
        return self.case["intent"] or {"slots": {}, "conventions": {}}

    def candidates(self, question, card, spec, feedback=None):
        self.calls.append({"role": "candidates", "source": "示範用候選程式（人工撰寫，重現常見 LLM 錯誤）"})
        return [dict(c, origin="示範") for c in self.case["candidates"]]

    def inspector(self, card, spec, retry=None):
        self.calls.append({"role": "inspector", "source": "示範用檢查員程式"})
        return INSPECTORS[card["id"]].replace("{timing}", spec["conventions"].get("timing", "end"))

    def baseline(self, question):
        return self.case["baseline"]


# ---------------------------------------------------------------- live

def _load_key():
    if os.environ.get("GEMINI_API_KEY"):
        return os.environ["GEMINI_API_KEY"]
    env = Path(__file__).resolve().parent.parent / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("GEMINI_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"')
    return None


def live_available() -> bool:
    try:
        import google.genai  # noqa: F401
    except ImportError:
        return False
    return bool(_load_key())


_client = None
_client_lock = threading.Lock()
_cache: dict = {}


def _gemini():
    global _client
    with _client_lock:
        if _client is None:
            from google import genai
            from google.genai import types
            # a hung request must fail (and be retried) instead of stalling a whole run; thinking calls fit in 180 s
            _client = genai.Client(api_key=_load_key(), http_options=types.HttpOptions(timeout=180_000))
    return _client


def _ask(prompt: str, temperature: float = 0.2, as_json: bool = False, think: bool = False) -> str:
    from google.genai import types
    cfg = types.GenerateContentConfig(temperature=temperature,
                                      response_mime_type="application/json" if as_json else "text/plain",
                                      thinking_config=None if think else types.ThinkingConfig(thinking_budget=0))
    for attempt in range(5):  # rate limits and 5xx are transient; anything else is a real error
        try:
            resp = _gemini().models.generate_content(model=MODEL, contents=prompt, config=cfg)
            return resp.text or ""
        except Exception as exc:
            if attempt == 4 or not any(t in str(exc) for t in ("429", "500", "502", "503", "504", "RESOURCE_EXHAUSTED",
                                                                "UNAVAILABLE", "DEADLINE", "imeout", "ConnectError",
                                                                "nodename", "Connection")):
                raise
            time.sleep(2 ** attempt * 3)


def _code_block(text: str) -> str:
    m = re.search(r"```(?:python)?\n(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip() + "\n"


def _spec_names(card):
    return [s["name"] for s in card["slots"]] + [c["name"] for c in card["conventions"]] + \
        [f["name"] for f in card["fixed"]]


class LiveProvider:
    mode = "live"
    can_regenerate = True
    verify_inspector = True  # the PoC turns this off until it is measured (v6)

    def __init__(self, k: int = 4):
        self.k = k
        self.calls = []
        self._round = 0
        self._pool = ThreadPoolExecutor(max_workers=1)
        self._baseline = None

    def _log(self, role, prompt, reply):
        self.calls.append({"role": role, "source": MODEL, "prompt": prompt, "reply": reply})

    def retrieve(self, question):
        # the no-harness baseline is independent of everything else; start it now
        self._baseline = self._pool.submit(self._baseline_text, question)
        card_id, table = retrieve(question)
        if card_id:
            return card_id, table, "關鍵字比對（確定性）"
        menu = "\n".join(f"- {cid}: {c['title']}" for cid, c in CARDS.items())
        prompt = (f"Classify the user's finance question into exactly one card id below, or answer none if the "
                  f"question cannot be answered by a deterministic calculation of that card's type.\n{menu}\n\n"
                  f"Question: {question}\n\nAnswer with only the id or none.")
        key = ("retrieve", question)
        reply = _cache.get(key) or _ask(prompt, 0.0).strip().lower()
        _cache[key] = reply
        self._log("retrieve", prompt, reply)
        card_id = reply if reply in CARDS else None
        return card_id, table, f"關鍵字未命中，由 {MODEL} 分類（回答：{reply}）"

    def intent(self, question, card):
        slots = "\n".join(f'- {s["name"]}: {s["label"]} ({s["kind"]}, unit {s["unit"]})' for s in card["slots"])
        convs = "\n".join(f'- {c["name"]}: {c["question"]} options {[o["value"] for o in c["options"]]}'
                          for c in card["conventions"]) or "(none)"
        prompt = f"""You extract an intent spec from a finance question. Do not compute anything.

Slots:
{slots}

Conventions:
{convs}

For each slot return {{"quote": <the exact substring of the question that states it, or null>, "value": <number>}}.
Rates are decimals (3% -> 0.03). Money in yuan (10 萬 -> 100000).
A frequency is the count per year, stated by words such as 每月 / 按月 (12), 每季 (4), 每半年 (2), 每年 (1); quote that word.
For each convention return {{"quote": <exact substring that decides it, or null>, "value": <option or null>}}.

Question: {question}

Return JSON: {{"slots": {{...}}, "conventions": {{...}}}}"""
        key = ("intent", question, card["id"])
        if key not in _cache:
            _cache[key] = _ask(prompt, 0.0, as_json=True)
        reply = _cache[key]
        self._log("intent", prompt, reply)
        try:
            return json.loads(reply)
        except json.JSONDecodeError:
            return {"slots": {}, "conventions": {}}

    def _candidate_prompt(self, question, card, spec, feedback):
        conv_lines = "\n".join(
            f'- {c["name"]}: options {[o["value"] for o in c["options"]]}' +
            (f' (already decided: {spec["conventions"][c["name"]]})' if c["name"] in spec["conventions"] else
             " (not stated in the question: pick the reading you think is intended)")
            for c in card["conventions"]) or "(none)"
        fixed = "\n".join(f'- {f["name"]}: {f["text"]}' for f in card["fixed"]) or "(none)"
        # the input contract is part of the spec; the checks on the program's behaviour stay hidden
        pre = "\n".join(f'- {i["text"]}  `{i["check"]}`' for i in card["invariants"] if i["kind"] == "pre") or "(none)"
        slots = "\n".join(f'- {s["name"]}: {s["label"]} ({s["unit"] or s["kind"]})' for s in card["slots"])
        out = card["output"]
        st = out.get("structure")
        returns = "the structure described under Output (not one number)" if st else "one float"
        layout = ""
        if st:
            layout = f"\nLayout: {st['text']}" + (f" Index: {st['index_meaning']}." if st["index_meaning"] else "") + \
                "".join(f"\n  {x['key']!r}: {x['name']} ({x['label']}, {x['unit'] or '-'})" for x in st["fields"])
        fb = ""
        if feedback:
            fb = "\nA previous attempt failed these executed checks; avoid the same mistakes:\n" + \
                 json.dumps(feedback, ensure_ascii=False)
        return f"""Write a Python function for this finance question.

Question: {question}

Rules:
- Define `def {card['signature']}:` returning {returns}. Inputs arrive already normalised (rates are decimals).
- Return the output in the unit stated under Output, unrounded, whatever rounding the question asks for.
- Only `import math` is allowed.
- At module level define ASSUMPTIONS = {{...}} with one entry per convention below, and nothing else
  (use ASSUMPTIONS = {{}} when there are no conventions; fixed assumptions are not conventions).
- Tag each line that implements a spec item with a trailing comment `# @spec name1, name2`.
  Spec item names: {', '.join(_spec_names(card))}

Inputs (what each argument means, as defined by the card):
{slots}

Output: {out['label']} ({out['unit'] or '-'}){layout}

Input contract (already checked on these inputs; it also tells you what each input is allowed to be):
{pre}

Conventions:
{conv_lines}

Fixed assumptions:
{fixed}
{fb}
Return only one ```python``` code block."""

    def candidates(self, question, card, spec, feedback=None):
        # Round 1 samples from the question as stated, so a clarification answer reuses the same
        # candidates the user was shown. Later rounds are told every decided convention.
        self._round += 1
        if self._round == 1:
            spec = spec | {"conventions": spec["stated_conventions"]}
            key = ("cands", question, card["id"], json.dumps(spec["inputs"], sort_keys=True))
        else:
            key = ("cands", question, card["id"], json.dumps(spec, sort_keys=True), json.dumps(feedback))
        prompt = self._candidate_prompt(question, card, spec, feedback)
        if key not in _cache:
            with ThreadPoolExecutor(max_workers=self.k) as pool:
                replies = list(pool.map(lambda _: _ask(prompt, 1.0), range(self.k)))
            _cache[key] = replies
        replies = _cache[key]
        self._log("candidates", prompt, f"{len(replies)} samples at temperature 1.0")
        letters = "ABCDEFGH" if self._round == 1 else "PQRSTUVW"
        return [{"id": letters[i], "code": _code_block(r), "bug": None, "origin": MODEL}
                for i, r in enumerate(replies)]

    def inspector(self, card, spec, retry=None):
        frozen = {"quantity": card["title"], "signature": card["signature"], "inputs": spec["inputs"],
                  "conventions": spec["conventions"], "fixed_assumptions": spec["fixed"],
                  "inputs_meaning": {s["name"]: f'{s["label"]} ({s["unit"] or s["kind"]})' for s in card["slots"]},
                  # what each decided convention value means; the bare value ("end") is easy to misread
                  "conventions_meaning": {c["name"]: next((f'{o["label"]}（{c["question"]}）' for o in c["options"]
                                                           if o["value"] == spec["conventions"].get(c["name"])), None)
                                          for c in card["conventions"] if c["name"] in spec["conventions"]},
                  "output": f'{card["output"]["label"]} ({card["output"]["unit"] or "-"})'}
        st = card["output"].get("structure")
        if st:  # the inspector rebuilds the whole structure; the harness picks the asked element itself
            frozen["output_layout"] = {"text": st["text"], "index": st["index_meaning"],
                                       "fields": {str(x["key"]): x["name"] for x in st["fields"]}}
        again = f"\nYour previous attempt failed: {retry}. Avoid unbounded loops.\n" if retry else ""
        prompt = f"""You are an inspector. You see only this frozen spec, not the original question.
Write `def {card['signature']}:` that computes the output by a DIFFERENT method than the textbook closed form
(for example: simulate period by period, or solve numerically by bisection). Only `import math` is allowed.
Return the unrounded value{" inside the structure given under output_layout" if st else ""}.{again}

Spec:
{json.dumps(frozen, ensure_ascii=False, indent=2)}

Return only one ```python``` code block."""
        key = ("inspect", json.dumps(frozen, sort_keys=True), retry)
        if key not in _cache:
            _cache[key] = _ask(prompt, 0.2)
        self._log("inspector", prompt, _cache[key])
        return _code_block(_cache[key])

    def _baseline_text(self, question):
        key = ("baseline", question)
        if key not in _cache:
            try:  # default thinking on: the baseline gets the model at full strength
                _cache[key] = _ask(f"{question}\n請直接給出答案與簡短說明。", 0.7, think=True)
            except Exception as exc:  # the baseline is decoration; never break the run for it
                return f"(baseline call failed: {exc})"
        return _cache[key]

    def baseline(self, question):
        text = self._baseline.result() if self._baseline else self._baseline_text(question)
        return {"kind": "live", "text": text}
