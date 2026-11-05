"""Deterministic re-parsing of quoted spans. The LLM's number is only accepted if the quote yields it."""

from __future__ import annotations

import re

_NUM = r"([0-9][0-9,]*(?:\.[0-9]+)?)"
_MULT = {"千": 1e3, "萬": 1e4, "億": 1e8}


def parse(quote: str, kind: str) -> float | None:
    if not quote:
        return None
    q = quote.replace("，", ",").replace(" ", "")
    if kind == "rate":
        m = re.search(_NUM + r"%", q) or re.search(r"百分之" + _NUM, q)
        if m:
            return float(m.group(1).replace(",", "")) / 100
        m = re.search(r"^0?\.[0-9]+$", q)
        return float(q) if m else None
    if kind == "freq":
        for word, per_year in (("每月", 12), ("每個月", 12), ("按月", 12), ("月繳", 12), ("每季", 4), ("按季", 4),
                               ("季繳", 4), ("每半年", 2), ("半年", 2), ("每年", 1), ("按年", 1), ("年繳", 1)):
            if word in q:
                return per_year
        m = re.search(_NUM, q)
        return int(m.group(1)) if m else None
    m = re.search(_NUM + r"([千萬億]?)", q)
    if not m:
        return None
    value = float(m.group(1).replace(",", "")) * _MULT.get(m.group(2), 1)
    if kind == "count":
        return int(value) if value == int(value) else value
    return value


def fmt(value: float, unit: str) -> str:
    if unit == "小數":
        return f"{value * 100:.2f}%"
    if unit == "元":
        return f"{value:,.2f} 元"
    return f"{value:g} {unit}"
