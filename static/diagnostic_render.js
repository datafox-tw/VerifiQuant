/* ── Shared diagnostic renderers ──────────────────────────────────────────
 * Loaded before demo.js and conversation.js. Both classic scripts share global
 * scope on /demo, so all shared helpers live on the single VQRender namespace to
 * avoid redeclaration collisions. Pure HTML-string builders over the
 * /api/diagnose response shape — no DOM side effects.
 */
const VQRender = (() => {
  function escapeHtml(text) {
    return String(text || "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;");
  }

  function statusBadge(status, diagnosticType) {
    return `<span class="badge badge-${status}">${escapeHtml(status)} / ${escapeHtml(diagnosticType || "Unknown")}</span>`;
  }

  /* Compact M→N→F→E→I→C funnel stepper, highlighting the gate that fired. */
  const FUNNEL_LAYERS = [
    { key: "M", title: "Match — is there a contract for this?" },
    { key: "N", title: "Null — is the question in scope?" },
    { key: "F", title: "Fields — are required inputs present?" },
    { key: "E", title: "Evaluate — bounds & scale checks" },
    { key: "I", title: "Infer — ambiguity / clarification" },
    { key: "C", title: "Compute — verified calculation" },
  ];

  // funnel_layer holds descriptive words, not the M/N/F/E/I/C letter. The canonical
  // class is diagnostic_type; map it (and the descriptive word, as a fallback) to a letter.
  const FUNNEL_WORD_TO_LETTER = {
    intent: "M", scope: "N", schema: "F", boundary: "E", critic: "I", logic: "C",
  };

  function funnelActiveLetter(data) {
    if (String(data.status || "") === "success") return "C";
    const dt = String(data.diagnostic_type || "").trim().toUpperCase();
    if (dt && dt !== "NONE" && dt !== "UNKNOWN") return dt.charAt(0); // M,N,F,E,I(_HARD),C
    const word = String(data.funnel_layer || "").trim().toLowerCase();
    return FUNNEL_WORD_TO_LETTER[word] || "";
  }

  function renderFunnelStepper(data) {
    const active = funnelActiveLetter(data);
    const steps = FUNNEL_LAYERS.map((layer) => {
      const isActive = layer.key === active;
      return `<span class="funnel-step${isActive ? " active" : ""}" title="${escapeHtml(layer.title)}">${layer.key}</span>`;
    }).join('<span class="funnel-arrow">›</span>');
    const gate = data.gate_action ? `<span class="funnel-gate">gate: <code>${escapeHtml(data.gate_action)}</code></span>` : "";
    return `<div class="funnel-stepper">${steps}${gate}</div>`;
  }

  function renderSelectionSummary(selectionTrace) {
    if (!selectionTrace || Object.keys(selectionTrace).length === 0) return "";
    const selector = selectionTrace.selector || {};
    const candidates = selectionTrace.retrieval_candidates || [];
    const candidateRows = candidates.map((c) => `
      <tr>
        <td><code>${escapeHtml(c.fic_id || "-")}</code></td>
        <td>${escapeHtml(c.title || "-")}</td>
        <td>${escapeHtml(c.domain || "-")} / ${escapeHtml(c.topic || "-")}</td>
        <td>${escapeHtml(c.score)}</td>
      </tr>`).join("");
    return `
      <section class="trace-section">
        <h3>Card Selection &amp; M/N Decision</h3>
        <p>Decision: <code>${escapeHtml(selector.decision || "-")}</code></p>
        <p>Chosen FIC: <code>${escapeHtml(selector.chosen_fic_id || "-")}</code></p>
        <p>Reason: ${escapeHtml(selector.reason || "-")}</p>
        ${candidateRows ? `
          <table class="history-table compact-table">
            <thead><tr><th>fic_id</th><th>title</th><th>domain/topic</th><th>score</th></tr></thead>
            <tbody>${candidateRows}</tbody>
          </table>` : ""}
      </section>`;
  }

  function renderPipelineTimeline(timeline) {
    if (!Array.isArray(timeline) || !timeline.length) return "";
    const rows = timeline.map((step) => `
      <article class="timeline-step">
        <div class="timeline-head">
          <span class="step-status step-${escapeHtml(step.status || "pending")}">${escapeHtml(step.status || "pending")}</span>
          <strong>${escapeHtml(step.label || step.key || "")}</strong>
        </div>
        <p>${escapeHtml(step.detail || "")}</p>
      </article>`).join("");
    return `<section class="trace-section"><h3>Pipeline Timeline</h3><div class="timeline-grid">${rows}</div></section>`;
  }

  function renderPipelineLogs(logs) {
    if (!Array.isArray(logs) || !logs.length) return "";
    const items = logs.map((line) => `<li>${escapeHtml(line)}</li>`).join("");
    return `<section class="trace-section"><h3>Pipeline Logs</h3><ol class="pipeline-log-list">${items}</ol></section>`;
  }

  function renderTraceBlock(title, value) {
    if (!value || (typeof value === "object" && Object.keys(value).length === 0)) return "";
    return `<section class="trace-section"><h3>${escapeHtml(title)}</h3><pre>${escapeHtml(JSON.stringify(value, null, 2))}</pre></section>`;
  }

  function renderPythonExecution(executionTrace) {
    if (!executionTrace || Object.keys(executionTrace).length === 0) return "";
    return `
      <section class="trace-section">
        <h3>Python Execution</h3>
        <p>Engine: <code>${escapeHtml(executionTrace.engine || "-")}</code> | FIC: <code>${escapeHtml(executionTrace.fic_id || "-")}</code></p>
        <p>Code:</p><pre>${escapeHtml(executionTrace.code || "")}</pre>
        <p>Input:</p><pre>${escapeHtml(JSON.stringify(executionTrace.inputs || {}, null, 2))}</pre>
        <p>Output (raw):</p><pre>${escapeHtml(JSON.stringify(executionTrace.raw_output, null, 2))}</pre>
        <p>Output (parsed): <code>${escapeHtml(executionTrace.parsed_output_value)}</code></p>
        ${executionTrace.error ? `<p>Error: <code>${escapeHtml(executionTrace.error)}</code></p>` : ""}
      </section>`;
  }

  /* The full "how this was derived & verified" trace, ordered for the reverse-trace
   * narrative: funnel gate → chosen contract → inputs/execution → supporting traces. */
  function renderDerivationTrace(data) {
    return [
      renderFunnelStepper(data),
      renderSelectionSummary(data.selection_trace),
      renderPythonExecution(data.execution_trace),
      renderPipelineTimeline(data.pipeline_timeline),
      renderTraceBlock("Extraction Trace", data.extraction_trace),
      renderTraceBlock("F/E Checks", data.echeck_trace),
      renderTraceBlock("I Gate", data.critic_trace),
      renderPipelineLogs(data.pipeline_logs),
    ].join("");
  }

  return {
    escapeHtml,
    statusBadge,
    renderFunnelStepper,
    renderSelectionSummary,
    renderPipelineTimeline,
    renderPipelineLogs,
    renderTraceBlock,
    renderPythonExecution,
    renderDerivationTrace,
  };
})();
