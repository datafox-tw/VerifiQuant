from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_group_ids(config_path: Path) -> Tuple[Set[str], Set[str]]:
    if yaml is None:
        raise RuntimeError("PyYAML is required for this script. Please install pyyaml.")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    target = payload.get("target_ids", {}) or {}
    correct = {str(x).strip() for x in target.get("correct_samples", []) if str(x).strip()}
    error = {str(x).strip() for x in target.get("error_samples", []) if str(x).strip()}
    return correct, error


def _norm_diag(v: Any) -> str:
    if v is None:
        return "None"
    s = str(v).strip()
    return s if s else "None"


def _pct(num: int, den: int) -> float:
    if den == 0:
        return 0.0
    return 100.0 * num / den


def _summarize_zone(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    status_counts = Counter(str(r.get("status", "")) for r in rows)
    diag_counts = Counter(_norm_diag(r.get("diagnostic_type")) for r in rows)
    success_count = int(status_counts.get("success", 0))
    clarification_count = int(status_counts.get("needs_clarification", 0))
    refusal_count = int(status_counts.get("refusal", 0))
    error_count = int(status_counts.get("error", 0))
    has_i_soft_count = sum(1 for r in rows if bool(r.get("has_i_soft", False)))

    judged = [r for r in rows if isinstance(r.get("is_correct"), bool)]
    judged_total = len(judged)
    judged_correct = sum(1 for r in judged if bool(r.get("is_correct")))

    return {
        "count": total,
        "status_counts": dict(status_counts),
        "diagnostic_counts": dict(diag_counts),
        "success_count": success_count,
        "success_rate_pct": _pct(success_count, total),
        "needs_clarification_count": clarification_count,
        "needs_clarification_rate_pct": _pct(clarification_count, total),
        "refusal_count": refusal_count,
        "refusal_rate_pct": _pct(refusal_count, total),
        "error_count": error_count,
        "error_rate_pct": _pct(error_count, total),
        "has_i_soft_count": has_i_soft_count,
        "has_i_soft_rate_pct": _pct(has_i_soft_count, total),
        "judged_total": judged_total,
        "judged_correct_count": judged_correct,
        "judged_correct_rate_pct": _pct(judged_correct, judged_total),
    }


def _bar(label: str, value: int, total: int, color: str) -> str:
    pct = _pct(value, total)
    return (
        f'<div class="bar-row"><div class="bar-label">{label}</div>'
        f'<div class="bar-wrap"><div class="bar" style="width:{pct:.2f}%; background:{color};"></div></div>'
        f'<div class="bar-val">{value} ({pct:.1f}%)</div></div>'
    )


def _bars(counts: Dict[str, int], total: int, order: List[str], color: str) -> str:
    return "".join(_bar(k, int(counts.get(k, 0)), total, color) for k in order)


def _render_html(summary: Dict[str, Any]) -> str:
    overall = summary["overall"]
    z1 = summary["zones"]["group_a"]
    z2 = summary["zones"]["group_b"]
    missing = summary["missing_case_ids_in_output"]

    status_order = ["success", "needs_clarification", "error", "refusal", "alert"]
    diag_order = ["None", "M", "N", "F", "E", "I", "C"]

    overall_status_bars = _bars(overall["status_counts"], overall["count"], status_order, "#1463ff")
    overall_diag_bars = _bars(overall["diagnostic_counts"], overall["count"], diag_order, "#ff6a3d")
    z1_diag_bars = _bars(z1["diagnostic_counts"], z1["count"], diag_order, "#2e9f5d")
    z2_diag_bars = _bars(z2["diagnostic_counts"], z2["count"], diag_order, "#8b5cf6")

    missing_html = "".join(f"<li>{x}</li>" for x in missing) or "<li>(none)</li>"

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>50Q Evaluation Dashboard</title>
  <style>
    :root {{
      --bg: #f5f7fb;
      --card: #ffffff;
      --ink: #16202a;
      --line: #d7e0ea;
      --muted: #5e6a76;
    }}
    body {{ margin: 0; background: var(--bg); color: var(--ink); font-family: "Avenir Next", "Segoe UI", sans-serif; }}
    .wrap {{ max-width: 1160px; margin: 24px auto; padding: 0 16px; }}
    .hero {{ background: linear-gradient(120deg, #e9f1ff, #fdf1e9); border: 1px solid var(--line); border-radius: 12px; padding: 16px; }}
    .hero h1 {{ margin: 0 0 6px 0; font-size: 26px; }}
    .hero p {{ margin: 0; color: var(--muted); }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; margin-top: 14px; }}
    .kpi {{ background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 10px 12px; }}
    .kpi .t {{ color: var(--muted); font-size: 12px; }}
    .kpi .v {{ font-size: 28px; font-weight: 700; margin-top: 3px; }}
    .panel {{ background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 14px; margin-top: 14px; }}
    .two {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }}
    .bar-row {{ display: grid; grid-template-columns: 140px 1fr 140px; align-items: center; gap: 8px; margin: 6px 0; }}
    .bar-wrap {{ background: #edf2f8; border-radius: 8px; height: 14px; overflow: hidden; }}
    .bar {{ height: 14px; }}
    .bar-label, .bar-val {{ font-size: 12px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ border: 1px solid var(--line); padding: 6px; text-align: left; }}
    th {{ background: #f3f7fc; }}
    ul {{ margin: 6px 0 0 16px; }}
    @media (max-width: 900px) {{ .two {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>50Q Run Visualization</h1>
      <p>Input from config zones + run result JSONL. Focus on status and M/N/F/E/I/C distribution.</p>
      <div class="grid">
        <div class="kpi"><div class="t">Total Cases</div><div class="v">{overall["count"]}</div></div>
        <div class="kpi"><div class="t">Overall Success Rate</div><div class="v">{overall["success_rate_pct"]:.1f}%</div></div>
        <div class="kpi"><div class="t">Needs Clarification</div><div class="v">{overall["needs_clarification_rate_pct"]:.1f}%</div></div>
        <div class="kpi"><div class="t">Execution/Other Errors</div><div class="v">{overall["error_rate_pct"]:.1f}%</div></div>
      </div>
    </section>

    <section class="panel">
      <h2>Overall Status Distribution</h2>
      {overall_status_bars}
    </section>

    <section class="panel">
      <h2>Overall Diagnostic Distribution (M/N/F/E/I/C)</h2>
      {overall_diag_bars}
    </section>

    <section class="two">
      <div class="panel">
        <h2>Group A ({z1["count"]})</h2>
        <p>Judged-correct (where available): {z1["judged_correct_count"]}/{z1["judged_total"]} ({z1["judged_correct_rate_pct"]:.1f}%)</p>
        {z1_diag_bars}
      </div>
      <div class="panel">
        <h2>Group B ({z2["count"]})</h2>
        <p>Judged-correct (where available): {z2["judged_correct_count"]}/{z2["judged_total"]} ({z2["judged_correct_rate_pct"]:.1f}%)</p>
        {z2_diag_bars}
      </div>
    </section>

    <section class="panel">
      <h2>Zone Summary Table</h2>
      <table>
        <thead>
          <tr>
            <th>Zone</th><th>Count</th><th>Success %</th><th>Need Clarification %</th><th>Error %</th><th>Refusal %</th><th>Has I_soft %</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Group A</td><td>{z1["count"]}</td><td>{z1["success_rate_pct"]:.1f}%</td><td>{z1["needs_clarification_rate_pct"]:.1f}%</td><td>{z1["error_rate_pct"]:.1f}%</td><td>{z1["refusal_rate_pct"]:.1f}%</td><td>{z1["has_i_soft_rate_pct"]:.1f}%</td>
          </tr>
          <tr>
            <td>Group B</td><td>{z2["count"]}</td><td>{z2["success_rate_pct"]:.1f}%</td><td>{z2["needs_clarification_rate_pct"]:.1f}%</td><td>{z2["error_rate_pct"]:.1f}%</td><td>{z2["refusal_rate_pct"]:.1f}%</td><td>{z2["has_i_soft_rate_pct"]:.1f}%</td>
          </tr>
        </tbody>
      </table>
    </section>

    <section class="panel">
      <h2>Missing Case IDs in Output</h2>
      <ul>{missing_html}</ul>
    </section>
  </div>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize config-based 50Q evaluation result.")
    parser.add_argument("--config", type=Path, required=True, help="config-with-npv.yaml")
    parser.add_argument("--result", type=Path, required=True, help="testing_50Q_result.jsonl")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("verifiquant/data/runs/demo_50q_0408/viz"),
        help="output directory for summary and dashboard",
    )
    args = parser.parse_args()

    correct_ids, error_ids = _load_group_ids(args.config)
    result_rows = _load_jsonl(args.result)
    by_id = {str(r.get("case_id", "")).strip(): r for r in result_rows if str(r.get("case_id", "")).strip()}

    group_a_rows = [by_id[cid] for cid in sorted(correct_ids) if cid in by_id]
    group_b_rows = [by_id[cid] for cid in sorted(error_ids) if cid in by_id]
    all_rows = group_a_rows + group_b_rows

    expected_ids = sorted(correct_ids | error_ids)
    missing_ids = [cid for cid in expected_ids if cid not in by_id]

    summary = {
        "config_path": str(args.config),
        "result_path": str(args.result),
        "expected_total_ids": len(expected_ids),
        "result_total_rows": len(result_rows),
        "matched_rows": len(all_rows),
        "missing_case_ids_in_output": missing_ids,
        "zones": {
            "group_a": _summarize_zone(group_a_rows),
            "group_b": _summarize_zone(group_b_rows),
        },
        "overall": _summarize_zone(all_rows),
    }

    args.outdir.mkdir(parents=True, exist_ok=True)
    summary_path = args.outdir / "summary.json"
    dashboard_path = args.outdir / "dashboard.html"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    dashboard_path.write_text(_render_html(summary), encoding="utf-8")

    print(f"Wrote summary: {summary_path}")
    print(f"Wrote dashboard: {dashboard_path}")
    print(f"Matched rows: {summary['matched_rows']} / expected {summary['expected_total_ids']}")


if __name__ == "__main__":
    main()

