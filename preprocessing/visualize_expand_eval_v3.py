from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _norm_diag(v: Any) -> str:
    if v is None:
        return "None"
    s = str(v).strip()
    return s if s else "None"


def _pct(num: int, den: int) -> float:
    if den == 0:
        return 0.0
    return 100.0 * num / den


def _merge_rows(
    expanded_rows: List[Dict[str, Any]],
    result_rows: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    in_map = {str(r.get("case_id", "")): r for r in expanded_rows if str(r.get("case_id", "")).strip()}
    out_map = {str(r.get("case_id", "")): r for r in result_rows if str(r.get("case_id", "")).strip()}

    warnings: List[str] = []
    all_case_ids = sorted(set(in_map.keys()) | set(out_map.keys()))
    merged: List[Dict[str, Any]] = []
    for cid in all_case_ids:
        i = in_map.get(cid)
        o = out_map.get(cid)
        if i is None:
            warnings.append(f"Missing in expanded input: {cid}")
            continue
        if o is None:
            warnings.append(f"Missing in run output: {cid}")
            continue
        merged.append(
            {
                "case_id": cid,
                "source_sample_id": i.get("source_sample_id"),
                "variant_type": i.get("variant_type"),
                "update_method": i.get("update_method", ""),
                "expected_status": i.get("expected_status"),
                "expected_diagnostic_type": _norm_diag(i.get("expected_diagnostic_type")),
                "actual_status": o.get("status"),
                "actual_diagnostic_type": _norm_diag(o.get("diagnostic_type")),
                "result_reason": o.get("reason", ""),
                "question": i.get("question", ""),
            }
        )
    return merged, warnings


def _summarize(merged: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(merged)
    status_counts = Counter(r["actual_status"] for r in merged)
    diag_counts = Counter(r["actual_diagnostic_type"] for r in merged)
    variant_counts = Counter(r["variant_type"] for r in merged)
    update_method_counts = Counter(r["update_method"] for r in merged)

    status_match = 0
    diag_match = 0
    full_match = 0

    confusion: Dict[str, Counter] = defaultdict(Counter)
    by_variant: Dict[str, Dict[str, Any]] = {}

    for r in merged:
        expected_status = r["expected_status"]
        expected_diag = r["expected_diagnostic_type"]
        actual_status = r["actual_status"]
        actual_diag = r["actual_diagnostic_type"]
        vtype = r["variant_type"]

        confusion[expected_diag][actual_diag] += 1
        if expected_status == actual_status:
            status_match += 1
        if expected_diag == actual_diag:
            diag_match += 1
        if expected_status == actual_status and expected_diag == actual_diag:
            full_match += 1

        if vtype not in by_variant:
            by_variant[vtype] = {
                "count": 0,
                "expected_status": Counter(),
                "actual_status": Counter(),
                "expected_diag": Counter(),
                "actual_diag": Counter(),
                "full_match": 0,
            }
        row = by_variant[vtype]
        row["count"] += 1
        row["expected_status"][expected_status] += 1
        row["actual_status"][actual_status] += 1
        row["expected_diag"][expected_diag] += 1
        row["actual_diag"][actual_diag] += 1
        if expected_status == actual_status and expected_diag == actual_diag:
            row["full_match"] += 1

    e_rows = [r for r in merged if r["variant_type"] == "E_trap"]
    e_total = len(e_rows)
    e_alert = sum(1 for r in e_rows if r["actual_status"] == "alert")
    e_diag_e = sum(1 for r in e_rows if r["actual_diagnostic_type"] == "E")

    mismatch_samples = [
        {
            "case_id": r["case_id"],
            "variant_type": r["variant_type"],
            "expected_status": r["expected_status"],
            "actual_status": r["actual_status"],
            "expected_diag": r["expected_diagnostic_type"],
            "actual_diag": r["actual_diagnostic_type"],
            "update_method": r["update_method"],
            "result_reason": r["result_reason"],
        }
        for r in merged
        if not (
            r["expected_status"] == r["actual_status"]
            and r["expected_diagnostic_type"] == r["actual_diagnostic_type"]
        )
    ]

    return {
        "total_cases": total,
        "status_counts": dict(status_counts),
        "diagnostic_counts": dict(diag_counts),
        "variant_counts": dict(variant_counts),
        "update_method_counts": dict(update_method_counts),
        "status_match_count": status_match,
        "status_match_rate_pct": _pct(status_match, total),
        "diag_match_count": diag_match,
        "diag_match_rate_pct": _pct(diag_match, total),
        "full_match_count": full_match,
        "full_match_rate_pct": _pct(full_match, total),
        "e_trap_total": e_total,
        "e_trap_alert_count": e_alert,
        "e_trap_alert_rate_pct": _pct(e_alert, e_total),
        "e_trap_diag_e_count": e_diag_e,
        "e_trap_diag_e_rate_pct": _pct(e_diag_e, e_total),
        "by_variant": {
            k: {
                "count": v["count"],
                "expected_status": dict(v["expected_status"]),
                "actual_status": dict(v["actual_status"]),
                "expected_diag": dict(v["expected_diag"]),
                "actual_diag": dict(v["actual_diag"]),
                "full_match": v["full_match"],
                "full_match_rate_pct": _pct(v["full_match"], v["count"]),
            }
            for k, v in by_variant.items()
        },
        "diag_confusion_expected_vs_actual": {
            exp: dict(cnt) for exp, cnt in confusion.items()
        },
        "mismatch_samples": mismatch_samples,
    }


def _bar(label: str, value: int, total: int) -> str:
    pct = _pct(value, total)
    return (
        f'<div class="bar-row"><div class="bar-label">{label}</div>'
        f'<div class="bar-wrap"><div class="bar" style="width:{pct:.2f}%"></div></div>'
        f'<div class="bar-val">{value} ({pct:.1f}%)</div></div>'
    )


def _render_html(summary: Dict[str, Any], warnings: List[str]) -> str:
    total = int(summary["total_cases"])
    status_counts: Dict[str, int] = summary["status_counts"]
    diag_counts: Dict[str, int] = summary["diagnostic_counts"]
    by_variant: Dict[str, Dict[str, Any]] = summary["by_variant"]
    confusion: Dict[str, Dict[str, int]] = summary["diag_confusion_expected_vs_actual"]

    status_order = ["success", "refusal", "error", "alert"]
    diag_order = ["None", "M", "F", "E"]

    status_bars = "".join(_bar(s, int(status_counts.get(s, 0)), total) for s in status_order)
    diag_bars = "".join(_bar(d, int(diag_counts.get(d, 0)), total) for d in diag_order)

    variant_rows = []
    for v in ["clean", "M_trap", "F_trap", "E_trap"]:
        row = by_variant.get(v, {})
        variant_rows.append(
            "<tr>"
            f"<td>{v}</td>"
            f"<td>{row.get('count', 0)}</td>"
            f"<td>{row.get('full_match', 0)}</td>"
            f"<td>{row.get('full_match_rate_pct', 0.0):.1f}%</td>"
            f"<td>{json.dumps(row.get('actual_status', {}), ensure_ascii=False)}</td>"
            f"<td>{json.dumps(row.get('actual_diag', {}), ensure_ascii=False)}</td>"
            "</tr>"
        )

    conf_rows = []
    for exp in diag_order:
        cells = "".join(
            f"<td>{int(confusion.get(exp, {}).get(act, 0))}</td>" for act in diag_order
        )
        conf_rows.append(f"<tr><td>{exp}</td>{cells}</tr>")

    mismatches = summary["mismatch_samples"][:20]
    mismatch_rows = "".join(
        "<tr>"
        f"<td>{m['case_id']}</td>"
        f"<td>{m['variant_type']}</td>"
        f"<td>{m['expected_status']}/{m['expected_diag']}</td>"
        f"<td>{m['actual_status']}/{m['actual_diag']}</td>"
        f"<td>{m['update_method']}</td>"
        f"<td>{m['result_reason']}</td>"
        "</tr>"
        for m in mismatches
    )

    warn_html = "".join(f"<li>{w}</li>" for w in warnings) or "<li>(none)</li>"

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Expand40 MFE Visualization</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #17212b; }}
    .kpi-grid {{ display:grid; grid-template-columns: repeat(auto-fit,minmax(220px,1fr)); gap: 12px; margin-bottom: 16px; }}
    .kpi {{ border:1px solid #d8e1ea; border-radius:10px; padding:12px; background:#f8fbff; }}
    .kpi .title {{ font-size:13px; color:#4a5a6a; }}
    .kpi .val {{ font-size:28px; font-weight:700; margin-top:2px; }}
    .panel {{ border:1px solid #d8e1ea; border-radius:10px; padding:12px; margin-bottom:16px; }}
    .bar-row {{ display:grid; grid-template-columns: 120px 1fr 120px; gap:10px; align-items:center; margin:6px 0; }}
    .bar-wrap {{ background:#eef3f8; border-radius:8px; height:14px; overflow:hidden; }}
    .bar {{ background:#2b7fff; height:14px; }}
    table {{ width:100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ border: 1px solid #d8e1ea; padding: 6px; text-align: left; vertical-align: top; }}
    th {{ background:#f3f7fb; }}
    .warn {{ color:#9a5a00; }}
  </style>
</head>
<body>
  <h1>Expand40 M/F/E Evaluation Dashboard</h1>
  <div class="kpi-grid">
    <div class="kpi"><div class="title">Total Cases</div><div class="val">{summary["total_cases"]}</div></div>
    <div class="kpi"><div class="title">Full Match Rate</div><div class="val">{summary["full_match_rate_pct"]:.1f}%</div></div>
    <div class="kpi"><div class="title">Status Match Rate</div><div class="val">{summary["status_match_rate_pct"]:.1f}%</div></div>
    <div class="kpi"><div class="title">E_trap Alert Rate</div><div class="val">{summary["e_trap_alert_rate_pct"]:.1f}%</div></div>
  </div>

  <div class="panel">
    <h2>Status Distribution</h2>
    {status_bars}
  </div>

  <div class="panel">
    <h2>Diagnostic Distribution</h2>
    {diag_bars}
  </div>

  <div class="panel">
    <h2>Per-Variant Performance</h2>
    <table>
      <thead><tr><th>Variant</th><th>Count</th><th>Full Match</th><th>Full Match %</th><th>Actual Status</th><th>Actual Diag</th></tr></thead>
      <tbody>
        {''.join(variant_rows)}
      </tbody>
    </table>
  </div>

  <div class="panel">
    <h2>Diagnostic Confusion (Expected -> Actual)</h2>
    <table>
      <thead><tr><th>Expected \\ Actual</th><th>None</th><th>M</th><th>F</th><th>E</th></tr></thead>
      <tbody>
        {''.join(conf_rows)}
      </tbody>
    </table>
  </div>

  <div class="panel">
    <h2>Top Mismatches (first 20)</h2>
    <table>
      <thead><tr><th>Case ID</th><th>Variant</th><th>Expected</th><th>Actual</th><th>Update Method</th><th>Reason</th></tr></thead>
      <tbody>
        {mismatch_rows}
      </tbody>
    </table>
  </div>

  <div class="panel warn">
    <h2>Merge Warnings</h2>
    <ul>{warn_html}</ul>
  </div>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize expand40 input/output alignment for v3 M/F/E pipeline."
    )
    parser.add_argument("--expanded-input", required=True, type=Path, help="expanded_40 input JSONL")
    parser.add_argument("--run-output", required=True, type=Path, help="run_mfe_pipeline output JSONL")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("verifiquant_v3/data/viz_expand40"),
        help="output directory for summary and dashboard",
    )
    args = parser.parse_args()

    expanded_rows = _load_jsonl(args.expanded_input)
    result_rows = _load_jsonl(args.run_output)
    merged, warnings = _merge_rows(expanded_rows, result_rows)
    summary = _summarize(merged)

    args.outdir.mkdir(parents=True, exist_ok=True)
    merged_path = args.outdir / "merged_eval.jsonl"
    with merged_path.open("w", encoding="utf-8") as fh:
        for row in merged:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_path = args.outdir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    html = _render_html(summary, warnings)
    dashboard_path = args.outdir / "dashboard.html"
    dashboard_path.write_text(html, encoding="utf-8")

    print(f"Wrote merged rows: {len(merged)} -> {merged_path}")
    print(f"Wrote summary: {summary_path}")
    print(f"Wrote dashboard: {dashboard_path}")
    print(
        "E-trap alert rate: "
        f"{summary['e_trap_alert_count']}/{summary['e_trap_total']} "
        f"({summary['e_trap_alert_rate_pct']:.1f}%)"
    )


if __name__ == "__main__":
    main()
