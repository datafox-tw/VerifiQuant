"""Repeated runs side by side: totals per run, which questions flip, and what changed against a base run.

    python3 -m harness_demo.poc.compare_runs --runs full_v3_r1,full_v3_r2,full_v3_r3 --base full_v2
        -> runs/<first run>/COMPARE.md
"""

from __future__ import annotations

import argparse
import collections
import json

from .data import RUNS, sample_ids
from .report import outcome

LABEL = {"correct": "正確", "silent_wrong": "輸出但錯", "ask": "追問", "abstain": "拒答", "error": "錯誤"}


def load(tag):
    path = RUNS / tag / "results.jsonl"
    return {r["question_id"]: r for r in map(json.loads, path.read_text().splitlines()) if r}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--base", default=None)
    args = ap.parse_args()
    tags = args.runs.split(",")
    runs = {t: load(t) for t in tags}
    ids = [i for i in sample_ids() if all(i in r for r in runs.values())]
    L = [f"# 重複執行比較（{len(ids)} 題，{len(tags)} 次）", "",
         "| 執行 | 正確 | 輸出但錯 | 追問 | 拒答 | 錯誤 |", "|---|---|---|---|---|---|"]
    per = {}
    for t in tags:
        c = collections.Counter(outcome(runs[t][i]) for i in ids)
        per[t] = c
        L.append(f"| {t} | " + " | ".join(str(c[k]) for k in LABEL) + " |")
    if args.base:
        base = load(args.base)
        c = collections.Counter(outcome(base[i]) for i in ids if i in base)
        L.append(f"| {args.base}（對照） | " + " | ".join(str(c[k]) for k in LABEL) + " |")
    outs = {i: [outcome(runs[t][i]) for t in tags] for i in ids}
    stable = [i for i in ids if len(set(outs[i])) == 1]
    always = sum(outs[i][0] == "correct" for i in stable)
    ever = sum("correct" in outs[i] for i in ids)
    L += ["", f"{len(tags)} 次結果完全相同：{len(stable)}/{len(ids)} 題；每次都正確 {always} 題，至少一次正確 {ever} 題；"
          f"至少一次輸出但錯 {sum('silent_wrong' in outs[i] for i in ids)} 題。", ""]
    flips = [i for i in ids if i not in stable]
    if flips:
        L += ["### 會翻轉的題目", "", "| 題號 | " + " | ".join(tags) + " | 停在 |", "|---|" + "---|" * len(tags) + "---|"]
        for i in flips:
            L.append(f"| {i} | " + " | ".join(LABEL[o] for o in outs[i]) + " | "
                     + ", ".join(sorted({str(runs[t][i].get('stopped_at')) for t in tags})) + " |")
        L.append("")
    if args.base:
        moved = [(i, outcome(base[i]), collections.Counter(outs[i]).most_common(1)[0][0])
                 for i in ids if i in base and outcome(base[i]) != collections.Counter(outs[i]).most_common(1)[0][0]]
        L += [f"### 和 {args.base} 不同的題目（取多數結果）", "", "| 題號 | 之前 | 現在 | 現在停在 |", "|---|---|---|---|"]
        for i, a, b in moved:
            L.append(f"| {i} | {LABEL[a]} | {LABEL[b]} | {runs[tags[0]][i].get('stopped_at')} |")
        L.append("")
    sel = [(i, runs[t][i]) for t in tags for i in ids if runs[t][i].get("select")]
    if sel:
        L += ["### 結構輸出題：選取路徑與元素探針", "", "| 執行 | 題號 | 路徑 | 狀態 | 探針抓到 | 結果 |", "|---|---|---|---|---|---|"]
        for t in tags:
            for i in ids:
                r = runs[t][i]
                s = r.get("select")
                if s:
                    probe = (r.get("element_probe") or {}).get("caught_by")
                    L.append(f"| {t} | {i} | `{s.get('path') or s.get('claimed')}` | {s['status']} | "
                             f"{'—' if probe is None else len(probe)} | {LABEL[outcome(r)]}（{r.get('stopped_at') or '輸出'}） |")
    out = RUNS / tags[0] / "COMPARE.md"
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
