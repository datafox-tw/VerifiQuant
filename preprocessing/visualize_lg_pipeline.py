"""Generate an interactive HTML visualization of the LangGraph pipeline.

Usage:
    python preprocessing/visualize_lg_pipeline.py [--out PATH]

Output: a self-contained HTML file that renders the DAG in the browser.
No server needed — just open the file with any browser.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from verifiquant.pipeline.run_error_classification_pipeline_lg import PipelineDeps, build_pipeline

# ── Color palette (Mermaid classDef) ──────────────────────────────────────────
# Processing nodes: blue-ish
# Exit nodes: per funnel class
# Finalize: dark slate
_STYLE = """
    classDef processing fill:#dbeafe,stroke:#3b82f6,color:#1e3a8a,font-weight:bold
    classDef exit_m     fill:#fee2e2,stroke:#ef4444,color:#7f1d1d
    classDef exit_n     fill:#ffedd5,stroke:#f97316,color:#7c2d12
    classDef exit_f     fill:#fef9c3,stroke:#eab308,color:#713f12
    classDef exit_e     fill:#fde68a,stroke:#d97706,color:#78350f
    classDef exit_i     fill:#ede9fe,stroke:#8b5cf6,color:#4c1d95
    classDef exit_c     fill:#f3f4f6,stroke:#6b7280,color:#111827
    classDef exit_ok    fill:#dcfce7,stroke:#22c55e,color:#14532d
    classDef finalize   fill:#1e293b,stroke:#0f172a,color:#f8fafc,font-weight:bold
"""

_NODE_CLASSES = {
    "retrieve":     "processing",
    "mn_select":    "processing",
    "extract":      "processing",
    "fe_checks":    "processing",
    "i_gate":       "processing",
    "execute":      "processing",
    "exit_m":       "exit_m",
    "exit_n":       "exit_n",
    "exit_f":       "exit_f",
    "exit_e":       "exit_e",
    "exit_i":       "exit_i",
    "exit_c":       "exit_c",
    "exit_success": "exit_ok",
    "finalize":     "finalize",
}

# Human-readable labels for nodes
_LABELS = {
    "retrieve":     "① Retrieve\\n(RAG top-k)",
    "mn_select":    "② M/N Select\\n(LLM selector)",
    "extract":      "③ Extract\\n(LLM extractor)",
    "fe_checks":    "④ F/E Checks\\n(deterministic)",
    "i_gate":       "⑤ I-gate\\n(Critic agent)",
    "execute":      "⑥ Execute\\n(Python compute)",
    "exit_m":       "EXIT: M\\nIntent ambiguous",
    "exit_n":       "EXIT: N\\nOut of scope",
    "exit_f":       "EXIT: F\\nMissing inputs",
    "exit_e":       "EXIT: E\\nBoundary violation",
    "exit_i":       "EXIT: I\\nSemantic ambiguity",
    "exit_c":       "EXIT: C\\nExec error",
    "exit_success": "✓ SUCCESS",
    "finalize":     "Finalize\\n_build_result()",
}


def _enhance_mermaid(raw: str) -> str:
    """Post-process LangGraph's raw Mermaid output to add colors and labels."""
    lines = raw.splitlines()
    out: list[str] = []

    for line in lines:
        # Replace plain node declarations with labeled ones
        stripped = line.strip()
        added = False
        for node_id, label in _LABELS.items():
            if stripped == f"{node_id}({node_id})":
                cls = _NODE_CLASSES.get(node_id, "processing")
                out.append(f'\t{node_id}("{label}"):::{cls}')
                added = True
                break
        if not added:
            out.append(line)

    # Append custom classDefs (before the default ones)
    result = "\n".join(out)
    result = result.replace(
        "\tclassDef default",
        _STYLE + "\n\tclassDef default",
    )
    return result


_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>VerifiQuant LangGraph Pipeline</title>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: #0f172a; color: #e2e8f0;
    margin: 0; padding: 24px;
  }}
  h1 {{ font-size: 1.4rem; color: #93c5fd; margin-bottom: 4px; }}
  .subtitle {{ font-size: 0.85rem; color: #64748b; margin-bottom: 24px; }}
  .legend {{
    display: flex; flex-wrap: wrap; gap: 10px;
    margin-bottom: 20px; font-size: 0.78rem;
  }}
  .legend-item {{
    padding: 4px 10px; border-radius: 99px; border: 1px solid;
  }}
  .diagram-wrap {{
    background: #f8fafc; border-radius: 12px;
    padding: 32px; overflow-x: auto;
  }}
  .mermaid {{ display: flex; justify-content: center; }}
  details {{ margin-top: 24px; }}
  summary {{ cursor: pointer; color: #93c5fd; font-size: 0.85rem; }}
  pre {{
    background: #1e293b; border-radius: 8px; padding: 16px;
    font-size: 0.75rem; color: #94a3b8; overflow-x: auto;
    max-height: 400px;
  }}
</style>
</head>
<body>
<h1>VerifiQuant — LangGraph Pipeline DAG</h1>
<p class="subtitle">
  M/N/F/E/I/C funnel · {node_count} nodes · {edge_count} edges
</p>

<div class="legend">
  <span class="legend-item" style="background:#dbeafe;border-color:#3b82f6;color:#1e3a8a">Processing</span>
  <span class="legend-item" style="background:#fee2e2;border-color:#ef4444;color:#7f1d1d">M exit</span>
  <span class="legend-item" style="background:#ffedd5;border-color:#f97316;color:#7c2d12">N exit</span>
  <span class="legend-item" style="background:#fef9c3;border-color:#eab308;color:#713f12">F exit</span>
  <span class="legend-item" style="background:#fde68a;border-color:#d97706;color:#78350f">E exit</span>
  <span class="legend-item" style="background:#ede9fe;border-color:#8b5cf6;color:#4c1d95">I exit</span>
  <span class="legend-item" style="background:#f3f4f6;border-color:#6b7280;color:#111827">C exit</span>
  <span class="legend-item" style="background:#dcfce7;border-color:#22c55e;color:#14532d">Success</span>
  <span class="legend-item" style="background:#1e293b;border-color:#0f172a;color:#f8fafc">Finalize</span>
</div>

<div class="diagram-wrap">
  <div class="mermaid">
{mermaid_code}
  </div>
</div>

<details>
  <summary>▸ Raw Mermaid source</summary>
  <pre>{mermaid_escaped}</pre>
</details>

<details>
  <summary>▸ LangSmith tracing setup</summary>
  <pre>
# Add to your .env file:
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=&lt;your-key-from-smith.langchain.com&gt;
LANGCHAIN_PROJECT=verifiquant-pipeline

# LangGraph automatically sends every .invoke() call to LangSmith.
# Traces show per-node state diffs, execution time, and branching path.
# No code changes needed — just set the env vars before running.
  </pre>
</details>

<script>
  mermaid.initialize({{ startOnLoad: true, theme: 'default' }});
</script>
</body>
</html>
"""


def generate_html(out_path: Path) -> None:
    deps = PipelineDeps(
        client=MagicMock(), store=None,
        core_by_id={}, retrieval_cards=[], repair_index={},
    )
    app = build_pipeline(deps)
    graph = app.get_graph()

    raw_mermaid = graph.draw_mermaid()
    enhanced = _enhance_mermaid(raw_mermaid)

    node_count = len([n for n in graph.nodes if n not in ("__start__", "__end__")])
    edge_count = len(graph.edges)

    html = _HTML_TEMPLATE.format(
        node_count=node_count,
        edge_count=edge_count,
        mermaid_code=enhanced,
        mermaid_escaped=enhanced.replace("<", "&lt;").replace(">", "&gt;"),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"Saved → {out_path}")
    print(f"  nodes: {node_count}  edges: {edge_count}")


def generate_mermaid_md(out_path: Path) -> None:
    """Also save a raw .md file for Markdown-aware viewers (VSCode, GitHub)."""
    deps = PipelineDeps(
        client=MagicMock(), store=None,
        core_by_id={}, retrieval_cards=[], repair_index={},
    )
    app = build_pipeline(deps)
    raw = app.get_graph().draw_mermaid()
    enhanced = _enhance_mermaid(raw)
    md = f"# VerifiQuant LangGraph Pipeline\n\n```mermaid\n{enhanced}\n```\n"
    out_path.write_text(md, encoding="utf-8")
    print(f"Saved → {out_path}")


def print_ascii() -> None:
    deps = PipelineDeps(
        client=MagicMock(), store=None,
        core_by_id={}, retrieval_cards=[], repair_index={},
    )
    app = build_pipeline(deps)
    try:
        print(app.get_graph().draw_ascii())
    except Exception as e:
        print(f"ASCII draw failed: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize VerifiQuant LangGraph pipeline")
    parser.add_argument("--out-html", type=Path,
                        default=Path("docs/pipeline_dag.html"),
                        help="Output HTML path")
    parser.add_argument("--out-md", type=Path,
                        default=Path("docs/pipeline_dag.md"),
                        help="Output Mermaid .md path")
    parser.add_argument("--ascii", action="store_true", help="Print ASCII to stdout")
    args = parser.parse_args()

    if args.ascii:
        print_ascii()
        return

    generate_html(args.out_html)
    generate_mermaid_md(args.out_md)
    print()
    print("Open in browser:")
    print(f"  open {args.out_html}")


if __name__ == "__main__":
    main()
