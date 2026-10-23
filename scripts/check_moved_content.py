#!/usr/bin/env python3
"""
check_moved_content.py
----------------------
Rule for the camera-ready (2026-10-16): whatever leaves the ACM body must still be in the
body in other words or in the arXiv appendix. This script narrows the check to what a person
has to read.

Every body sentence of a BASELINE version is classified against the CURRENT body and the
current appendix:

  kept     verbatim in the current body
  moved    verbatim in the appendix
  changed  neither: rewritten, merged, or deleted

For each changed sentence, every number (including ``.74``-style decimals) and every
zero-argument macro (``\\govVQfnar{}``) is looked up in the current body + appendix. A token
found nowhere is information that is gone and is printed as MISSING. Changed sentences
without a missing token still need reading, because a deleted claim may carry no number.
Table cells are not compared (tables move whole), and Figure 1's in-figure text is not in
the LaTeX source.

Usage:
  python3 scripts/check_moved_content.py paper/archive/2026-10-16_main_9p_before_cuts.tex
  python3 scripts/check_moved_content.py BASELINE.tex [CURRENT.tex] [APPENDIX.tex]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NUM = re.compile(r"(?<![\d.])(?:\d+(?:\.\d+)?|\.\d+)")


def strip_comments(t: str) -> str:
    return re.sub(r"(?<!\\)%.*", "", t)


def body(path: Path) -> str:
    t = path.read_text(encoding="utf-8")
    t = t[t.index(r"\begin{document}"):]
    if r"\bibliography{" in t:
        t = t[: t.index(r"\bibliography{")]
    t = strip_comments(t)
    return re.sub(r"\\begin\{tabular\}.*?\\end\{tabular\}", " ", t, flags=re.S)


def norm(x: str) -> str:
    return re.sub(r"\s+", " ", x).strip()


def sentences(t: str) -> list[str]:
    parts = re.split(r"(?<=[.?!])\s+(?=[A-Z\\$(])", norm(t))
    return [p.strip() for p in parts if len(p.strip()) > 25]


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    base = Path(sys.argv[1])
    cur = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "paper/main_camera_ready_9p.tex"
    app = Path(sys.argv[3]) if len(sys.argv) > 3 else cur.parent / "appendix_camera_ready.tex"

    cur_body = norm(body(cur))
    cur_app = norm(strip_comments(app.read_text(encoding="utf-8")))
    everywhere = cur_body + " " + cur_app

    kept = moved = 0
    changed: list[tuple[str, list[str]]] = []
    for s in sentences(body(base)):
        if s in cur_body:
            kept += 1
        elif s in cur_app:
            moved += 1
        else:
            missing = [n for n in sorted(set(NUM.findall(s)))
                       if not re.search(r"(?<![\d.])" + re.escape(n) + r"(?!\d)", everywhere)]
            missing += [m for m in sorted(set(re.findall(r"\\[a-zA-Z]{4,}(?=\{\})", s))) if m not in everywhere]
            changed.append((s, missing))

    print(f"baseline {base.name}: {kept + moved + len(changed)} sentences | kept {kept} | "
          f"moved verbatim {moved} | changed {len(changed)}")
    for i, (s, missing) in enumerate(changed, 1):
        print(f"\n[{i}] {'MISSING ' + ', '.join(missing) if missing else ''}\n    {s[:600]}")
    n_missing = sum(bool(m) for _, m in changed)
    print(f"\nchanged sentences carrying a number or macro found nowhere now: {n_missing}")
    print("Read every changed sentence: a deleted claim may carry no number.")


if __name__ == "__main__":
    main()
