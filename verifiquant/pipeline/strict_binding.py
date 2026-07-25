"""Literal-grounding audit for bound inputs (anti-over-normalization guard).

Motivation (paper §binding): with a stronger model (Pro), the extractor sometimes
*recomputes* a literally-given input before binding (e.g., $2.5M silently became
$2.875M by adding profit the contract code would add again). E-checks validate the
bound value's range, not its provenance.

This module checks that every numeric bound value is derivable from a literal
number in the question/context via an ALLOWED normalization only:
  - comma / currency-sigil stripping        ($2,500,000 -> 2500000)
  - scale-word expansion                    (2.5 million / 2.5M -> 2500000)
  - percent <-> decimal                     (15% -> 0.15; 0.055 -> 5.5)
  - small word-numbers                      (three years -> 3)
Anything else (derived, adjusted, invented values) fails the audit. Intended use:
ungrounded REQUIRED fields are downgraded to missing -> the F gate requests them,
converting a potential silent binding slip into a structured ask.

Pure functions, no LLM. Deployed behind VERIFIQUANT_STRICT_BINDING=1 (opt-in).
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Set, Tuple

_SCALE_WORDS = {
    "k": 1e3, "thousand": 1e3,
    "m": 1e6, "mm": 1e6, "million": 1e6, "millions": 1e6,
    "bn": 1e9, "b": 1e9, "billion": 1e9, "billions": 1e9,
    "trillion": 1e12,
}
_WORD_NUMS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "hundred": 100,
}

_NUM_RE = re.compile(
    r"(?<![\w.])(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d+|\d+)\s*(%|percent\b|k\b|m\b|mm\b|bn\b|b\b|thousand\b|million[s]?\b|billion[s]?\b|trillion\b)?",
    re.IGNORECASE,
)


# Frequency words bind schema-declared period counts (compounding/payment frequency).
_FREQ_WORDS = {
    "annually": (1,), "annual": (1,), "yearly": (1,), "per year": (1,),
    "semi-annual": (2,), "semiannual": (2,), "semi-annually": (2,),
    "quarterly": (4,), "per quarter": (4,),
    "monthly": (12,), "per month": (12,),
    "weekly": (52,), "daily": (365, 360),
}


def literal_candidates(text: str) -> Set[float]:
    """All numeric values literally present in text, expanded by allowed normalizations."""
    out: Set[float] = set()

    def add(v: float) -> None:
        # Allowed schema-declared conversions only: percent<->decimal, sign convention
        # (outflow/loss phrasing), and months<->years time basis.
        for base in (v, -v):
            for x in (base, base / 100.0, base * 100.0, base * 12.0, base / 12.0):
                if math.isfinite(x):
                    out.add(round(x, 12))

    for m in _NUM_RE.finditer(text or ""):
        raw, suffix = m.group(1), (m.group(2) or "").lower().rstrip(".")
        try:
            v = float(raw.replace(",", ""))
        except ValueError:
            continue
        if suffix in ("%", "percent"):
            add(v)          # 15% -> {15, 0.15, 1500}
        elif suffix in _SCALE_WORDS:
            add(v * _SCALE_WORDS[suffix])
            add(v)          # keep unscaled too ("2.5M" question may bind 2.5)
        else:
            add(v)

    for w, v in _WORD_NUMS.items():
        if re.search(rf"\b{w}\b", text or "", re.IGNORECASE):
            add(float(v))
    low = (text or "").lower()
    for w, vals in _FREQ_WORDS.items():
        if w in low:
            for v in vals:
                add(float(v))
    return out


def _grounded_value(v: Any, cands: Set[float], rel_tol: float = 1e-4) -> bool:
    if isinstance(v, bool):
        return True  # booleans are schema flags, out of scope
    try:
        f = float(v)
    except (TypeError, ValueError):
        return True  # non-numeric (str) is out of scope
    if not math.isfinite(f):
        return False
    for c in cands:
        if math.isclose(f, c, rel_tol=rel_tol, abs_tol=1e-9):
            return True
    return False


def audit_binding(
    provided_inputs: Dict[str, Any], question: str, context: str,
) -> Tuple[List[str], List[str]]:
    """Return (grounded_names, ungrounded_names) for numeric bound values.

    Array inputs are grounded iff every element is grounded.
    """
    cands = literal_candidates(f"{question}\n{context}")
    grounded, ungrounded = [], []
    for name, v in (provided_inputs or {}).items():
        vals = v if isinstance(v, (list, tuple)) else [v]
        ok = all(_grounded_value(x, cands) for x in vals) if vals else True
        (grounded if ok else ungrounded).append(name)
    return grounded, ungrounded
