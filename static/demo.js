/* ── VerifiQuant Demo Console JS ── */

const form = document.getElementById("question-form");
const loader = document.getElementById("loader");
const resultEl = document.getElementById("result");
const cardsLoader = document.getElementById("cards-loader");
const cardsResult = document.getElementById("cards-result");
const cardsMeta = document.getElementById("cards-meta");
const contextEl = document.getElementById("context");
const questionEl = document.getElementById("question");
const sampleQuestionEl = document.getElementById("sample-question");
const sampleMetaEl = document.getElementById("sample-meta");
const loadSampleBtn = document.getElementById("load-sample");

const tabChat = document.getElementById("tab-chat");
const tabQa = document.getElementById("tab-qa");
const tabCards = document.getElementById("tab-cards");
const panelChat = document.getElementById("panel-chat");
const panelQa = document.getElementById("panel-qa");
const panelCards = document.getElementById("panel-cards");

// Shared renderers (defined in diagnostic_render.js, loaded first).
const {
  escapeHtml,
  statusBadge,
  renderSelectionSummary,
  renderPipelineTimeline,
  renderPipelineLogs,
  renderTraceBlock,
  renderPythonExecution,
} = VQRender;

let demoQuestions = [];

/* ── Rate limit display ── */
async function fetchRateLimit() {
  try {
    const res = await fetch("/api/rate-limit");
    const data = await res.json();
    const bar = document.getElementById("rate-limit-bar");
    if (bar && data.status === "ok") {
      bar.textContent = `Remaining diagnoses: ${data.remaining} / ${data.limit} per hour`;
    }
  } catch (_) {}
}
fetchRateLimit();

/* ── Helpers ── */
function setQaLoading(isLoading) {
  loader.classList.toggle("hidden", !isLoading);
  form.querySelector("button[type=submit]").disabled = isLoading;
}

function setCardsLoading(isLoading) {
  cardsLoader.classList.toggle("hidden", !isLoading);
}

function activateTab(target) {
  if (tabChat) tabChat.classList.toggle("active", target === "chat");
  tabQa.classList.toggle("active", target === "qa");
  tabCards.classList.toggle("active", target === "cards");
  if (panelChat) panelChat.classList.toggle("active", target === "chat");
  panelQa.classList.toggle("active", target === "qa");
  panelCards.classList.toggle("active", target === "cards");
}

function toSlotToken(text) {
  return String(text || "").toLowerCase().replace(/[^a-z0-9_]+/g, "_").replace(/^_+|_+$/g, "");
}

function truncateText(text, maxLength = 90) {
  const value = String(text || "").replace(/\s+/g, " ").trim();
  return value.length > maxLength ? `${value.slice(0, maxLength - 1)}...` : value;
}

function formatSampleLabel(item) {
  const title = item.article_title || item.function_id || item.case_id || `Sample ${item.index}`;
  return `${String(item.index).padStart(2, "0")} · ${title} · ${truncateText(item.question, 72)}`;
}

function applySampleQuestion(index) {
  const item = demoQuestions[index];
  if (!item) return;
  questionEl.value = item.question || "";
  contextEl.value = item.context || "";
  sampleMetaEl.textContent = [
    item.case_id ? `case: ${item.case_id}` : "",
    item.level ? `level: ${item.level}` : "",
    item.ground_truth !== undefined && item.ground_truth !== null ? `gold: ${item.ground_truth}` : "",
  ].filter(Boolean).join(" · ");
}

async function loadDemoQuestionBank() {
  if (!sampleQuestionEl) return;
  sampleQuestionEl.innerHTML = `<option value="">Loading latest 50...</option>`;
  try {
    const response = await fetch("/api/demo/questions");
    const data = await response.json();
    if (data.status !== "ok") throw new Error(data.message || "load sample questions failed");
    demoQuestions = data.questions || [];
    sampleQuestionEl.innerHTML = demoQuestions.map((item, idx) =>
      `<option value="${idx}">${escapeHtml(formatSampleLabel(item))}</option>`
    ).join("");
    if (demoQuestions.length) {
      applySampleQuestion(0);
      const source = data.source || "questions";
      sampleMetaEl.textContent = `${sampleMetaEl.textContent} · source: ${source}`;
    } else {
      sampleQuestionEl.innerHTML = `<option value="">No samples found</option>`;
      sampleMetaEl.textContent = "No demo samples found.";
    }
  } catch (error) {
    sampleQuestionEl.innerHTML = `<option value="">Failed to load samples</option>`;
    sampleMetaEl.textContent = error.message;
  }
}

/* ── Repair helpers ── */
function collectRepairUpdates() {
  const updates = {};
  document.querySelectorAll(".repair-input").forEach((el) => {
    const slot = el.dataset.slot;
    if (!slot) return;
    const value = el.value.trim();
    if (value !== "") updates[slot] = value;
  });
  return updates;
}

function collectInteractiveUpdates() {
  const updates = {};
  document.querySelectorAll(".interactive-answer-card").forEach((card) => {
    const slot = card.dataset.slot;
    if (!slot) return;
    const optionEl = card.querySelector(".interactive-option");
    const freeEl = card.querySelector(".interactive-free");
    const optionValue = optionEl ? String(optionEl.value || "").trim() : "";
    const freeValue = freeEl ? String(freeEl.value || "").trim() : "";
    const chosen = freeValue || optionValue;
    if (chosen) updates[slot] = chosen;
  });
  const contextNote = document.getElementById("interactive-context-note");
  if (contextNote && contextNote.value.trim()) {
    updates.user_context_note = contextNote.value.trim();
  }
  return updates;
}

async function composeAndApplyUpdates(updates) {
  if (!updates || Object.keys(updates).length === 0) return;
  const payload = { question: questionEl.value, context: contextEl.value, updates };
  const response = await fetch("/api/repair/compose", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await response.json();
  if (data.status !== "ok") throw new Error(data.message || "repair compose failed");
  questionEl.value = data.repaired_question || questionEl.value;
  contextEl.value = data.repaired_context || contextEl.value;
}

async function composeAndApplyRepair() {
  await composeAndApplyUpdates(collectRepairUpdates());
}

async function applyInteractiveAdjustments() {
  const qRewriteEl = document.getElementById("interactive-question-rewrite");
  const rewrittenQuestion = qRewriteEl ? qRewriteEl.value.trim() : "";
  if (rewrittenQuestion) questionEl.value = rewrittenQuestion;
  await composeAndApplyUpdates(collectInteractiveUpdates());
}

/* ── Render functions ── */
function renderRepairHints(repairHints) {
  if (!repairHints?.length) return "";
  const blocks = repairHints.map((hint) => {
    const asks = (hint.ask_user_for || []).map((ask) => {
      const slot = escapeHtml(ask.slot || "");
      const label = escapeHtml(ask.label || ask.slot || "");
      return `<label class="repair-field">${label}<input class="repair-input" data-slot="${slot}" type="text" placeholder="${slot}" /></label>`;
    }).join("");
    return `
      <article class="repair-card">
        <h4>${escapeHtml(hint.title || hint.rule_id || "Repair Hint")}</h4>
        <p>${escapeHtml(hint.user_message || "")}</p>
        ${asks ? `<div class="repair-grid">${asks}</div>` : ""}
      </article>`;
  }).join("");
  return `
    <section class="repair-section">
      <h3>Repair Guide</h3>
      ${blocks}
      <div class="repair-actions">
        <button id="apply-repair-btn" type="button" class="btn btn-outline btn-sm">Apply repair to context</button>
        <button id="rerun-btn" type="button" class="btn btn-primary btn-sm">Re-diagnose with fix</button>
      </div>
    </section>`;
}

function renderInteractiveAssistant(data) {
  const clarification = data.clarification_request || null;
  const softWarnings = Array.isArray(data.soft_warnings) ? data.soft_warnings : [];
  if (!clarification && !softWarnings.length) return "";

  // Options may be plain strings or structured {label, value} objects.
  const optionHtml = (opt) => {
    const label = (opt && typeof opt === "object") ? (opt.label || opt.value || "") : opt;
    const value = (opt && typeof opt === "object") ? (opt.value || opt.label || "") : opt;
    return `<option value="${escapeHtml(value)}">${escapeHtml(label)}</option>`;
  };

  const clarificationCards = clarification
    ? (clarification.questions || []).map((q, idx) => {
        const slot = `clarification_${idx + 1}`;
        const options = (clarification.options || []).map(optionHtml).join("");
        return `
          <article class="interactive-answer-card" data-slot="${slot}">
            <h4>Clarification ${idx + 1}</h4>
            <p class="interactive-question">${escapeHtml(q)}</p>
            ${options ? `<label>Select answer<select class="interactive-option"><option value="">Choose...</option>${options}</select></label>` : ""}
            <label>Or enter manually<input class="interactive-free" type="text" placeholder="Enter a more precise answer" /></label>
          </article>`;
      }).join("") : "";

  const softCards = softWarnings.map((w, idx) => {
    const token = toSlotToken(w.hint_id || `i_soft_${idx + 1}`) || `i_soft_${idx + 1}`;
    const slot = `i_soft_${token}`;
    const options = (w.options || []).map(optionHtml).join("");
    return `
      <article class="interactive-answer-card" data-slot="${slot}">
        <h4>Soft Warning ${idx + 1}</h4>
        <p><strong>Default assumption</strong>: ${escapeHtml(w.assumption_if_not_clarified || "-")}</p>
        <p class="interactive-question"><strong>Question</strong>: ${escapeHtml(w.clarification_question || "-")}</p>
        ${options ? `<label>Options<select class="interactive-option"><option value="">Choose...</option>${options}</select></label>` : ""}
        <label>Or enter manually<input class="interactive-free" type="text" placeholder="Enter your answer" /></label>
      </article>`;
  }).join("");

  return `
    <section class="trace-section interaction-priority">
      <h3>Interactive Clarification (Priority)</h3>
      <label>Rewrite question (optional)<input id="interactive-question-rewrite" type="text" placeholder="Rewrite the question for more precision" /></label>
      <label>Add context (optional)<input id="interactive-context-note" type="text" placeholder="e.g., use end-of-year cash flows, 10% discount rate" /></label>
      <div class="interactive-grid">${clarificationCards}${softCards}</div>
      <div class="repair-actions">
        <button id="apply-interactive-btn" type="button" class="btn btn-outline btn-sm">Apply to context</button>
        <button id="rerun-interactive-btn" type="button" class="btn btn-primary btn-sm">Apply & re-diagnose</button>
      </div>
    </section>`;
}

function renderResult(data) {
  resultEl.classList.remove("hidden");
  const summary = `
    <div class="meta-row">
      ${statusBadge(data.status, data.diagnostic_type)}
      <span>Gate: <code>${escapeHtml(data.funnel_layer || "-")}</code> / <code>${escapeHtml(data.gate_action || "-")}</code></span>
    </div>
    <p>Reason: ${escapeHtml(data.reason || "-")}</p>
    <p>Card: <code>${escapeHtml(data.fic_id || "(none)")}</code></p>
    <p>Candidates: <code>${escapeHtml((data.candidate_ids || []).join(", "))}</code></p>
    <p>Inputs:</p><pre>${escapeHtml(JSON.stringify(data.provided_inputs || {}, null, 2))}</pre>`;

  const successPart = data.status === "success"
    ? `<p class="result-output">${escapeHtml(data.output_var)} = <strong>${escapeHtml(data.output_value)}</strong></p>` : "";
  const rulePart = data.triggered_rule_ids?.length
    ? `<p>Triggered rules: <code>${escapeHtml(data.triggered_rule_ids.join(", "))}</code></p>` : "";

  resultEl.innerHTML = [
    summary, rulePart, successPart,
    renderPythonExecution(data.execution_trace),
    renderInteractiveAssistant(data),
    renderRepairHints(data.repair_hints),
    renderSelectionSummary(data.selection_trace),
    renderPipelineTimeline(data.pipeline_timeline),
    renderTraceBlock("Extraction Trace", data.extraction_trace),
    renderTraceBlock("F/E Checks", data.echeck_trace),
    renderTraceBlock("I Gate", data.critic_trace),
    renderPipelineLogs(data.pipeline_logs),
  ].join("");

  // Bind interactive buttons
  const applyInteractiveBtn = document.getElementById("apply-interactive-btn");
  const rerunInteractiveBtn = document.getElementById("rerun-interactive-btn");
  if (applyInteractiveBtn) applyInteractiveBtn.addEventListener("click", async () => { try { await applyInteractiveAdjustments(); } catch (e) { alert(e.message); } });
  if (rerunInteractiveBtn) rerunInteractiveBtn.addEventListener("click", async () => { try { await applyInteractiveAdjustments(); form.requestSubmit(); } catch (e) { alert(e.message); } });

  const applyBtn = document.getElementById("apply-repair-btn");
  const rerunBtn = document.getElementById("rerun-btn");
  if (applyBtn) applyBtn.addEventListener("click", async () => { try { await composeAndApplyRepair(); } catch (e) { alert(e.message); } });
  if (rerunBtn) rerunBtn.addEventListener("click", async () => { try { await composeAndApplyRepair(); form.requestSubmit(); } catch (e) { alert(e.message); } });

  fetchRateLimit();
}

/* ── Cards overview ── */
async function loadCardsOverview() {
  setCardsLoading(true);
  try {
    const response = await fetch("/api/cards/overview");
    const data = await response.json();
    if (data.status !== "ok") throw new Error(data.message || "load cards failed");
    cardsMeta.textContent = `${data.domain_count} domains, ${data.card_count} cards`;
    const domains = Object.keys(data.grouped || {}).sort();
    cardsResult.innerHTML = domains.map((domain) => {
      const domainData = data.grouped[domain];
      const topicsHtml = Object.keys(domainData.topics || {}).sort().map((topic) => {
        const t = domainData.topics[topic];
        const cards = (t.cards || []).slice(0, 20).map((c) =>
          `<li><code>${escapeHtml(c.fic_id)}</code> ${escapeHtml(c.title || "")}</li>`).join("");
        return `<details><summary>${escapeHtml(topic)} (${t.count})</summary><ul class="card-list">${cards}</ul></details>`;
      }).join("");
      return `<section class="domain-block"><h3>${escapeHtml(domain)} <span style="color:var(--text-light)">(${domainData.total})</span></h3>${topicsHtml}</section>`;
    }).join("") || "<p>No cards found.</p>";
  } catch (error) {
    cardsResult.innerHTML = `<p>Failed to load: ${escapeHtml(error.message)}</p>`;
  } finally {
    setCardsLoading(false);
  }
}

/* ── Event listeners ── */
form.addEventListener("submit", async (event) => {
  event.preventDefault();
  resultEl.classList.add("hidden");
  resultEl.textContent = "";
  setQaLoading(true);
  try {
    const payload = {
      question: questionEl.value,
      context: contextEl.value,
      domain: document.getElementById("domain").value || null,
      topic: document.getElementById("topic").value || null,
      top_k: Number(document.getElementById("top-k").value || 3),
      m_min_top_score: Number(document.getElementById("m-min-top-score").value || 0.05),
      debug_sanity: true,
    };
    const response = await fetch("/api/diagnose", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    renderResult(data);
  } catch (error) {
    renderResult({ status: "error", diagnostic_type: "Unknown", message: error.message, reason: error.message });
  } finally {
    setQaLoading(false);
  }
});

document.getElementById("refresh-cards").addEventListener("click", loadCardsOverview);
if (loadSampleBtn) loadSampleBtn.addEventListener("click", () => applySampleQuestion(Number(sampleQuestionEl.value || 0)));
if (sampleQuestionEl) sampleQuestionEl.addEventListener("change", () => applySampleQuestion(Number(sampleQuestionEl.value || 0)));
if (tabChat) tabChat.addEventListener("click", () => activateTab("chat"));
tabQa.addEventListener("click", () => activateTab("qa"));
tabCards.addEventListener("click", () => {
  activateTab("cards");
  if (!cardsResult.innerHTML.trim()) loadCardsOverview();
});
loadDemoQuestionBank();
