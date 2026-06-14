/* ── VerifiQuant Conversation Console ─────────────────────────────────────
 * Chat-style, multi-turn diagnosis driven entirely client-side over the
 * single-shot /api/diagnose endpoint. The original question/context are captured
 * once and never overwritten: each repair/clarification turn re-diagnoses with
 * the immutable original context plus accumulated answers (composed server-side
 * via /api/repair/compose). This fixes the destructive-repair F-error and frames
 * the funnel as a screen → estimate → solve agent loop with an on-demand,
 * reverse-trace "how this was derived & verified" disclosure per answer.
 *
 * Shares global scope with demo.js — every identifier here is prefixed `cv` /
 * `CHAT_` to avoid collisions. Shared renderers come from VQRender.
 */
(() => {
  const { escapeHtml, statusBadge, renderDerivationTrace } = VQRender;

  const threadEl = document.getElementById("chat-thread");
  const inputEl = document.getElementById("chat-input");
  const contextEl = document.getElementById("chat-context");
  const sendBtn = document.getElementById("chat-send");
  const resetBtn = document.getElementById("chat-reset");
  const sampleEl = document.getElementById("chat-sample");
  const sampleMetaEl = document.getElementById("chat-sample-meta");
  const showcaseEl = document.getElementById("chat-showcase");
  const showcaseNoteEl = document.getElementById("chat-showcase-note");
  const compareEl = document.getElementById("chat-compare");
  const baselineModelEl = document.getElementById("chat-baseline-model");

  if (!threadEl) return; // Conversation panel not present.

  /* ── Conversation state (browser-held, original is immutable) ── */
  let cvOriginalQuestion = "";
  let cvOriginalContext = "";
  let cvAccumulatedUpdates = {}; // structured slot fields (e.g. annual_interest_rate)
  let cvNotes = []; // running transcript of clarification answers (Q→A), never overwritten
  let cvAnsweredHints = new Set(); // hint_ids the user already answered → don't re-prompt
  let cvShowcase = [];
  let cvActiveGroundTruth = null; // numeric gold for the active showcase/sample, if any
  let cvActiveExpectedClass = null; // expected funnel class (M/N/F/E/I) for trap/showcase cases
  let cvSeededNew = false; // a dropdown item was staged → next Send starts a fresh question
  let cvStarted = false;
  let cvBusy = false;
  let cvSamples = [];

  /* ── Thread helpers ── */
  function scrollThread() {
    threadEl.scrollTop = threadEl.scrollHeight;
  }

  function appendBubble(role, innerHtml) {
    const bubble = document.createElement("div");
    bubble.className = `chat-bubble ${role}`;
    bubble.innerHTML = innerHtml;
    threadEl.appendChild(bubble);
    scrollThread();
    return bubble;
  }

  function appendUser(text) {
    return appendBubble("user", `<p class="chat-text">${escapeHtml(text)}</p>`);
  }

  function appendStatus(text) {
    const bubble = appendBubble("assistant pending", `<p class="chat-text"><span class="spinner spinner-sm"></span> ${escapeHtml(text)}</p>`);
    return bubble;
  }

  function detailsDisclosure(data) {
    const trace = renderDerivationTrace(data);
    if (!trace.trim()) return "";
    return `
      <details class="bubble-details">
        <summary>How this was derived &amp; verified</summary>
        <div class="bubble-details-body">${trace}</div>
      </details>`;
  }

  /* ── Ground-truth verdict (who's right) ── */
  function parseNumber(text) {
    const matches = String(text == null ? "" : text).match(/-?\d[\d,]*\.?\d*/g);
    if (!matches) return null;
    const n = Number(matches[matches.length - 1].replace(/,/g, ""));
    return Number.isFinite(n) ? n : null;
  }

  // Classes where the *correct* behaviour is to NOT just answer (refuse / ask / flag).
  const FLAG_CLASSES = new Set(["M", "N", "F", "E", "I"]);
  function expectedFlagClass() {
    const c = String(cvActiveExpectedClass || "").trim().toUpperCase().charAt(0);
    return FLAG_CLASSES.has(c) ? c : null;
  }

  // Numeric ✓/✗ vs gold, or "" when there is no numeric gold.
  function verdictChip(value) {
    if (cvActiveGroundTruth === null || cvActiveGroundTruth === undefined || cvActiveGroundTruth === "") return "";
    const gold = Number(cvActiveGroundTruth);
    const v = (typeof value === "number") ? value : parseNumber(value);
    if (!Number.isFinite(gold) || v === null) return "";
    const ok = Math.abs(v - gold) <= Math.max(1e-6, Math.abs(gold) * 0.01);
    return ok
      ? `<span class="verdict verdict-ok">✓ matches gold (${escapeHtml(gold)})</span>`
      : `<span class="verdict verdict-bad">✗ differs from gold (${escapeHtml(gold)})</span>`;
  }

  // VerifiQuant did the right thing if it raised the EXACT gate the case expects.
  function vqVerdict(data) {
    const exp = expectedFlagClass();
    if (exp) {
      const dt = String(data.diagnostic_type || "").trim().toUpperCase().charAt(0);
      if (dt === exp) return `<span class="verdict verdict-ok">✓ correctly raised ${escapeHtml(exp)}-gate</span>`;
      if (data.status === "success") return `<span class="verdict verdict-bad">✗ answered (expected ${escapeHtml(exp)}-gate)</span>`;
      // Flagged, but at a different gate than expected.
      return `<span class="verdict verdict-bad">✗ raised ${escapeHtml(dt || "?")}-gate (expected ${escapeHtml(exp)})</span>`;
    }
    return data.status === "success" ? verdictChip(data.output_value) : "";
  }

  // The raw model "fails" a flag-class case the moment it returns a confident number.
  function baselineVerdict(answerNum) {
    const exp = expectedFlagClass();
    if (exp) {
      return answerNum !== null
        ? `<span class="verdict verdict-bad">✗ answered anyway (expected ${escapeHtml(exp)}-gate)</span>`
        : `<span class="verdict verdict-ok">✓ declined to answer</span>`;
    }
    return answerNum !== null ? verdictChip(answerNum) : "";
  }

  /* ── API calls ── */
  async function apiDiagnose(question, context) {
    const payload = { question, context, debug_sanity: true };
    const res = await fetch("/api/diagnose", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    return res.json();
  }

  async function apiCompose(context, updates) {
    const res = await fetch("/api/repair/compose", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: cvOriginalQuestion, context, updates }),
    });
    const data = await res.json();
    if (data.status !== "ok") throw new Error(data.message || "compose failed");
    return data.repaired_context;
  }

  /* ── Assistant bubble: render by status ── */
  function renderAssistant(data) {
    const status = data.status || "error";
    const header = `<div class="chat-verdict">${statusBadge(status, data.diagnostic_type)} ${vqVerdict(data)}</div>`;

    let body = "";
    if (status === "success") {
      body = `
        <p class="chat-answer"><span class="verified-check">✓ verified</span>
          <code>${escapeHtml(data.output_var || "result")}</code> =
          <strong>${escapeHtml(data.output_value)}</strong></p>
        <p class="chat-derived">Derived from contract <code>${escapeHtml(data.fic_id || "(none)")}</code> and verified numerically.</p>`;
      // A successful answer may still carry soft (I-class) refinements / transforms.
      body += softWarningWidget(data);
    } else if (status === "needs_clarification" || (Array.isArray(data.soft_warnings) && data.soft_warnings.length)) {
      body = `<p class="chat-text">${escapeHtml(data.reason || "I need a clarification before I can give a verified answer.")}</p>`;
      body += hardClarificationWidget(data);
      body += softWarningWidget(data);
    } else if (Array.isArray(data.repair_hints) && data.repair_hints.some((h) => (h.ask_user_for || []).length)) {
      body = `<p class="chat-text">${escapeHtml(data.reason || "Some required fields are missing.")}</p>`;
      body += repairWidget(data);
    } else {
      // refusal / alert / error
      const msg = data.reason || data.message || "Unable to produce a verified answer.";
      body = `<p class="chat-text">${escapeHtml(msg)}</p>`;
    }

    const bubble = appendBubble("assistant", header + body + detailsDisclosure(data));
    wireBubble(bubble, data);
    return bubble;
  }

  function optLabel(o) { return typeof o === "object" && o ? (o.label || o.value || "") : String(o || ""); }
  function optValue(o) { return typeof o === "object" && o ? (o.value || o.label || "") : String(o || ""); }

  /* ── Inline widgets ── */

  // Render an option as a button. Options that declare a verifiable transform trigger
  // /api/transform/apply; others re-diagnose with the choice as a resolving note.
  function optionButton(o, idx, oi, hintId) {
    const label = optLabel(o);
    const value = optValue(o);
    const hasTransform = typeof o === "object" && o && o.has_transform;
    const cls = hasTransform ? "btn btn-primary btn-sm cv-opt" : "btn btn-outline btn-sm cv-opt";
    const tag = hasTransform ? ' <span class="cv-tag">verified transform</span>' : "";
    return `<button type="button" class="${cls}" data-w="${idx}" data-o="${oi}"
              data-hint="${escapeHtml(hintId || "")}"
              data-value="${escapeHtml(value)}" data-label="${escapeHtml(label)}"
              data-transform="${hasTransform ? "1" : "0"}">${escapeHtml(label)}${tag}</button>`;
  }

  // I_hard: intercept with NO number. Renders ONE selector per ambiguity (group); a single
  // group keeps the transform-aware option buttons (the loan hero); multiple groups become
  // dropdowns answered together in one submit. Already-answered hints are not re-shown.
  function hardClarificationWidget(data) {
    const clar = data.clarification_request || null;
    if (!clar) return "";
    const hadGroups = (clar.groups || []).length > 0;
    const groups = (clar.groups || []).filter((g) => g.hint_id && !cvAnsweredHints.has(g.hint_id));

    if (groups.length === 1) {
      const g = groups[0];
      const buttons = (g.options || []).map((o, oi) => optionButton(o, 0, oi, g.hint_id)).join("");
      return `
        <div class="cv-widget" data-kind="clarification" data-hint="${escapeHtml(g.hint_id)}">
          <div class="cv-soft" data-hint="${escapeHtml(g.hint_id)}">
            <p class="cv-q">${escapeHtml(g.question)}</p>
            <div class="cv-opts">${buttons}</div>
          </div>
          <div class="cv-field"><input class="cv-free" type="text" placeholder="or type a precise answer" /></div>
          <button type="button" class="btn btn-outline btn-sm cv-submit">Submit typed answer</button>
        </div>`;
    }
    if (groups.length > 1) {
      const blocks = groups.map((g) => {
        const opts = (g.options || []).map((o) =>
          `<option value="${escapeHtml(optValue(o))}">${escapeHtml(optLabel(o))}</option>`).join("");
        return `
          <div class="cv-group" data-hint="${escapeHtml(g.hint_id)}">
            <p class="cv-q">${escapeHtml(g.question)}</p>
            <select class="cv-gselect"><option value="">Choose…</option>${opts}</select>
          </div>`;
      }).join("");
      return `
        <div class="cv-widget" data-kind="clarification-multi">
          <p class="cv-hint-note">Answer each question, then submit together.</p>
          ${blocks}
          <button type="button" class="btn btn-primary btn-sm cv-multi-submit">Submit answers</button>
        </div>`;
    }

    // All structured groups already answered → don't re-prompt via the flat fallback.
    if (hadGroups) return "";
    // Open mode (no structured hints) → flat fallback.
    if (!(clar.questions || []).length) return "";
    const questions = (clar.questions || []).map((q) => `<p class="cv-q">${escapeHtml(q)}</p>`).join("");
    const buttons = (clar.options || []).map((o, oi) => optionButton(o, 0, oi)).join("");
    return `
      <div class="cv-widget" data-kind="clarification">
        <div class="cv-soft">${questions}${buttons ? `<div class="cv-opts">${buttons}</div>` : ""}</div>
        <div class="cv-field"><input class="cv-free" type="text" placeholder="or type a precise answer" /></div>
        <button type="button" class="btn btn-outline btn-sm cv-submit">Submit typed answer</button>
      </div>`;
  }

  // I_soft: each option is a button. Options that declare a verifiable transform
  // trigger /api/transform/apply; others re-diagnose with the choice as a note.
  function softWarningWidget(data) {
    const soft = (Array.isArray(data.soft_warnings) ? data.soft_warnings : [])
      .filter((w) => !cvAnsweredHints.has(w.hint_id));
    if (!soft.length) return "";
    const blocks = soft.map((w, idx) => {
      const buttons = (w.options || []).map((o, oi) => optionButton(o, idx, oi, w.hint_id)).join("");
      return `
        <div class="cv-soft" data-w="${idx}" data-hint="${escapeHtml(w.hint_id || "")}">
          <p class="cv-assumption">Default if unanswered: <em>${escapeHtml(w.assumption_if_not_clarified || "-")}</em></p>
          <p class="cv-q">${escapeHtml(w.clarification_question || "-")}</p>
          <div class="cv-opts">${buttons || "<em>(no options)</em>"}</div>
        </div>`;
    }).join("");
    return `<div class="cv-widget" data-kind="soft">${blocks}</div>`;
  }

  function repairWidget(data) {
    const fields = [];
    (data.repair_hints || []).forEach((hint) => {
      (hint.ask_user_for || []).forEach((ask) => {
        const slot = ask.slot || "";
        if (!slot) return;
        const label = ask.label || slot;
        fields.push(`
          <div class="cv-field" data-slot="${escapeHtml(slot)}">
            <label class="cv-q">${escapeHtml(label)}</label>
            <input class="cv-free" type="text" placeholder="${escapeHtml(slot)}" />
          </div>`);
      });
    });
    if (!fields.length) return "";
    return `
      <div class="cv-widget" data-kind="repair">
        ${fields.join("")}
        <button type="button" class="btn btn-primary btn-sm cv-submit">Provide &amp; continue</button>
      </div>`;
  }

  function lockWidgets(bubble) {
    bubble.querySelectorAll(".cv-widget input, .cv-widget select, .cv-widget button")
      .forEach((el) => (el.disabled = true));
  }

  function wireBubble(bubble, data) {
    // Submit-style widgets (single-group hard clarification / repair, typed answer).
    bubble.querySelectorAll(".cv-widget .cv-submit").forEach((submit) => {
      submit.addEventListener("click", async () => {
        if (cvBusy) return;
        const widget = submit.closest(".cv-widget");
        const kind = widget.dataset.kind;
        const { updates, summary } = collectWidget(widget, kind);
        if (!Object.keys(updates).length) {
          alert("Please answer at least one field.");
          return;
        }
        if (widget.dataset.hint) cvAnsweredHints.add(widget.dataset.hint);
        lockWidgets(bubble);
        await runRepairTurn(updates, summary);
      });
    });

    // Multi-question hard clarification: collect every group's selection, submit once.
    bubble.querySelectorAll(".cv-widget .cv-multi-submit").forEach((submit) => {
      submit.addEventListener("click", async () => {
        if (cvBusy) return;
        const widget = submit.closest(".cv-widget");
        const parts = [];
        widget.querySelectorAll(".cv-group").forEach((g) => {
          const sel = g.querySelector(".cv-gselect");
          const val = sel ? sel.value.trim() : "";
          if (!val) return;
          const q = (g.querySelector(".cv-q") || {}).textContent || "";
          parts.push(`Regarding "${q.trim()}", the answer is: ${val}`);
          if (g.dataset.hint) cvAnsweredHints.add(g.dataset.hint);
        });
        if (!parts.length) { alert("Please answer at least one question."); return; }
        lockWidgets(bubble);
        await runRepairTurn({ user_context_note: parts.join(" ; ") }, parts.join(" ; "));
      });
    });

    // Option buttons (single-group hard clarification, or soft warnings): transform or note.
    bubble.querySelectorAll(".cv-widget .cv-opt").forEach((btn) => {
      btn.addEventListener("click", async () => {
        if (cvBusy) return;
        const value = btn.dataset.value;
        const label = btn.dataset.label;
        // Pair the answer with its question so the transcript reads as clear Q→A.
        const qEl = btn.closest(".cv-soft, .cv-widget").querySelector(".cv-q");
        const q = qEl ? qEl.textContent.trim() : "";
        const note = q ? `Regarding "${q}", the answer is: ${label}` : label;
        if (btn.dataset.hint) cvAnsweredHints.add(btn.dataset.hint);
        lockWidgets(bubble);
        if (btn.dataset.transform === "1") {
          await applyTransform(data, value, label);
        } else {
          await runRepairTurn({ user_context_note: note }, label);
        }
      });
    });
  }

  function collectWidget(widget, kind) {
    const updates = {};
    const parts = [];
    widget.querySelectorAll(".cv-field").forEach((field) => {
      const slot = field.dataset.slot;
      const sel = field.querySelector(".cv-select");
      const free = field.querySelector(".cv-free");
      const value = (free && free.value.trim()) || (sel && sel.value.trim()) || "";
      if (!value) return;
      if (kind === "repair" && slot) {
        // F-class: slot is a real missing input name → maps straight to context.
        updates[slot] = value;
        parts.push(`${slot} = ${value}`);
      } else {
        // I-class clarification: feed the choice back as a guiding context note.
        const q = field.querySelector(".cv-q");
        parts.push(q ? `${q.textContent.trim()} → ${value}` : value);
      }
    });
    if (kind !== "repair" && parts.length) {
      updates.user_context_note = parts.join("; ");
    }
    return { updates, summary: parts.join("; ") };
  }

  /* ── Verifiable atomic transform ── */
  function fmtNum(v) {
    const n = Number(v);
    return Number.isFinite(n) ? (Math.round(n * 1e6) / 1e6).toString() : String(v);
  }

  async function applyTransform(data, chosenValue, chosenLabel) {
    const et = data.execution_trace || {};
    const ficId = et.fic_id || data.fic_id || "";
    // I_HARD blocks execution, so inputs come from provided_inputs; I_soft has execution_trace.
    const inputs = (et.inputs && Object.keys(et.inputs).length) ? et.inputs : (data.provided_inputs || {});
    cvBusy = true;
    appendUser(`Use: ${chosenLabel}`);
    const pending = appendStatus("Applying & verifying atomic transform…");
    try {
      const res = await fetch("/api/transform/apply", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ fic_id: ficId, chosen_value: chosenValue, inputs }),
      });
      const td = await res.json();
      pending.remove();
      renderTransformResult(td, chosenLabel);
    } catch (err) {
      pending.remove();
      appendBubble("assistant", `<p class="chat-text">Transform request failed: ${escapeHtml(err.message)}</p>`);
    } finally {
      cvBusy = false;
    }
  }

  // The storyboard payoff: a UI-beautified Diagnostic / Audit Report.
  function renderTransformResult(td, label) {
    if (td.status === "ok") {
      const ts = new Date().toLocaleString();
      const codeBlock = td.python_code
        ? `<p>Python execution (<code>${escapeHtml(td.entrypoint || "")}</code>):</p><pre>${escapeHtml(td.python_code)}</pre>`
        : "";
      appendBubble("assistant audit", `
        <div class="chat-verdict"><span class="badge badge-success">Diagnostic Report · verified transform</span></div>
        <p class="chat-answer"><span class="verified-check">✓ verified</span> ${escapeHtml(label)}:
          <strong>${escapeHtml(fmtNum(td.result_old))} → ${escapeHtml(fmtNum(td.result_new))}</strong></p>
        <div class="audit-report">
          <div class="audit-row"><span class="audit-k">Adopted contract</span>
            <span class="audit-v">${escapeHtml(td.fic_name || "-")} · <code>${escapeHtml(td.fic_id || "-")}</code></span></div>
          <div class="audit-row"><span class="audit-k">Human-in-the-loop</span>
            <span class="audit-v">User selected "${escapeHtml(label)}" at ${escapeHtml(ts)}</span></div>
          <div class="audit-row"><span class="audit-k">Default (auto)</span>
            <span class="audit-v">${escapeHtml(fmtNum(td.result_old))} — what a raw model would have returned</span></div>
          <div class="audit-row"><span class="audit-k">Verified result</span>
            <span class="audit-v"><strong>${escapeHtml(fmtNum(td.result_new))}</strong></span></div>
        </div>
        <details class="bubble-details" open>
          <summary>Computation &amp; verification</summary>
          <div class="bubble-details-body">
            <section class="trace-section">
              <p>Transform: <code>result → ${escapeHtml(td.result_expr)}</code></p>
              <p>Invariant: <code>${escapeHtml(td.invariant)}</code> → <strong>${escapeHtml(td.invariant_check)}</strong></p>
              <p>Checked on ${escapeHtml(td.numerical_samples)} numerical sample(s) · rule <code>${escapeHtml(td.rule_id || "-")}</code></p>
              ${codeBlock}
            </section>
          </div>
        </details>`);
    } else if (td.status === "rejected") {
      appendBubble("assistant", `
        <div class="chat-verdict"><span class="badge badge-alert">transform rejected</span></div>
        <p class="chat-text">${escapeHtml(td.message || "Transform failed verification; result unchanged.")}</p>
        <p class="chat-derived">Invariant <code>${escapeHtml(td.invariant || "-")}</code> → ${escapeHtml(td.invariant_check || "-")}</p>`);
    } else {
      appendBubble("assistant", `<p class="chat-text">${escapeHtml(td.message || "Transform error.")}</p>`);
    }
  }

  /* ── Control group: raw LLM, no funnel ── */
  async function runBaseline(question, context) {
    if (!compareEl || !compareEl.checked) return;
    const provider = baselineModelEl ? baselineModelEl.value : "gemini";
    const pending = appendBubble("assistant baseline pending",
      `<p class="chat-text"><span class="spinner spinner-sm"></span> Raw ${escapeHtml(provider)} (no guardrails) is answering…</p>`);
    try {
      const res = await fetch("/api/baseline", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, context, provider }),
      });
      const bd = await res.json();
      pending.remove();
      if (bd.status !== "ok") {
        appendBubble("assistant baseline", `<p class="chat-text">Raw ${escapeHtml(provider)} failed: ${escapeHtml(bd.message || "error")}</p>`);
        return;
      }
      const parsed = parseNumber(bd.answer);
      appendBubble("assistant baseline", `
        <div class="chat-verdict"><span class="badge badge-refusal">Raw ${escapeHtml(bd.model)} · no funnel</span>
          ${baselineVerdict(parsed)}</div>
        <p class="chat-text">${escapeHtml(bd.answer)}</p>
        <p class="chat-derived">No FIC, no gate — it commits to an assumption without flagging it.</p>`);
    } catch (err) {
      pending.remove();
      appendBubble("assistant baseline", `<p class="chat-text">Raw model request failed: ${escapeHtml(err.message)}</p>`);
    }
  }

  /* ── Turn drivers ── */
  async function runFirstTurn() {
    const question = (inputEl.value || "").trim();
    if (!question) {
      alert("Enter a question to start.");
      return;
    }
    if (cvBusy) return;
    cvBusy = true;
    sendBtn.disabled = true;

    cvOriginalQuestion = question;
    cvOriginalContext = (contextEl.value || "").trim();
    cvAccumulatedUpdates = {};
    cvNotes = [];
    cvAnsweredHints = new Set();
    cvStarted = true;

    appendUser(question + (cvOriginalContext ? `\n\ncontext: ${cvOriginalContext}` : ""));
    inputEl.value = "";
    const pending = appendStatus("Running the M/N/F/E/I/C funnel…");

    try {
      const data = await apiDiagnose(cvOriginalQuestion, cvOriginalContext);
      pending.remove();
      renderAssistant(data);
    } catch (err) {
      pending.remove();
      appendBubble("assistant", `<p class="chat-text">Request failed: ${escapeHtml(err.message)}</p>`);
    } finally {
      cvBusy = false;
      sendBtn.disabled = false;
      updateComposerMode();
    }

    // Control group runs independently (raw LLM, no funnel) for side-by-side comparison.
    await runBaseline(cvOriginalQuestion, cvOriginalContext);
  }

  async function runRepairTurn(updates, summary) {
    cvBusy = true;
    // Free-text clarification answers reuse the same key, so APPEND them to a transcript
    // instead of letting each turn overwrite the last (e.g. "use NPV" then "end of period"
    // must BOTH stay in context). Structured slot fields keep accumulating by their own key.
    const incoming = { ...updates };
    if (incoming.user_context_note) {
      cvNotes.push(String(incoming.user_context_note));
      delete incoming.user_context_note;
    }
    cvAccumulatedUpdates = { ...cvAccumulatedUpdates, ...incoming };
    appendUser(summary || "(answer provided)");
    const pending = appendStatus("Re-diagnosing with your answer…");

    // Compose original + structured fields + the FULL clarification transcript.
    const composeUpdates = { ...cvAccumulatedUpdates };
    if (cvNotes.length) composeUpdates.user_context_note = cvNotes.join(" | ");

    let composedContext = cvOriginalContext;
    try {
      // Always recompose from the IMMUTABLE original + ALL accumulated answers.
      composedContext = await apiCompose(cvOriginalContext, composeUpdates);
      const data = await apiDiagnose(cvOriginalQuestion, composedContext);
      pending.remove();
      renderAssistant(data);
    } catch (err) {
      pending.remove();
      appendBubble("assistant", `<p class="chat-text">Request failed: ${escapeHtml(err.message)}</p>`);
    } finally {
      cvBusy = false;
    }

    // Control group fires on every turn the box is checked, with the same effective context.
    await runBaseline(cvOriginalQuestion, composedContext);
  }

  /* A free-text message after the conversation has started is treated as an added
   * note layered on the immutable original (not a brand-new question). */
  async function runFollowUp() {
    const text = (inputEl.value || "").trim();
    if (!text) return;
    if (cvBusy) return;
    inputEl.value = "";
    await runRepairTurn({ user_context_note: text }, text);
  }

  function onSend() {
    if (cvSeededNew) sendAsNewQuestion();
    else if (!cvStarted) runFirstTurn();
    else runFollowUp();
  }

  // A staged dropdown item is a NEW question: clear the prior thread/state at send-time
  // (not at click-time), keep the composer values + gold/expected class, then run turn 1.
  function sendAsNewQuestion() {
    cvSeededNew = false;
    const q = inputEl.value;
    const c = contextEl ? contextEl.value : "";
    const gt = cvActiveGroundTruth;
    const ec = cvActiveExpectedClass;
    threadEl.innerHTML = "";
    cvOriginalQuestion = "";
    cvOriginalContext = "";
    cvAccumulatedUpdates = {};
    cvNotes = [];
    cvAnsweredHints = new Set();
    cvStarted = false;
    inputEl.value = q;
    if (contextEl) contextEl.value = c;
    cvActiveGroundTruth = gt;
    cvActiveExpectedClass = ec;
    runFirstTurn();
  }

  function resetConversation() {
    cvOriginalQuestion = "";
    cvOriginalContext = "";
    cvAccumulatedUpdates = {};
    cvNotes = [];
    cvAnsweredHints = new Set();
    cvActiveGroundTruth = null;
    cvActiveExpectedClass = null;
    cvStarted = false;
    threadEl.innerHTML = "";
    if (contextEl) contextEl.value = "";
    inputEl.value = "";
    updateComposerMode();
  }

  function updateComposerMode() {
    if (!contextEl) return;
    const wrap = document.getElementById("chat-context-wrap");
    if (wrap) wrap.style.display = cvStarted ? "none" : "";
    inputEl.placeholder = cvStarted
      ? "Add a follow-up or correction…"
      : "Ask a financial question to start a verified diagnosis…";
  }

  function seedFrom(item, note) {
    if (!item) return;
    // Fill the composer fields only — do NOT clear the thread now (no disruption on click).
    // The NEXT Send will start this as a fresh question (see onSend / sendAsNewQuestion).
    inputEl.value = item.question || "";
    if (contextEl) contextEl.value = item.context || "";
    cvActiveGroundTruth = (item.ground_truth === undefined ? null : item.ground_truth);
    cvActiveExpectedClass = item.expected_class || item.expected_diagnostic || null;
    cvSeededNew = true;
    if (showcaseNoteEl) showcaseNoteEl.textContent = note || "";
    // Make sure the (possibly hidden/collapsed) context field is visible so the seed shows.
    const wrap = document.getElementById("chat-context-wrap");
    if (wrap) wrap.style.display = "";
    const det = wrap && wrap.querySelector("details");
    if (det && (item.context || "")) det.open = true;
  }

  function applySample() {
    if (sampleEl.value === "") return;
    const item = cvSamples[Number(sampleEl.value)];
    if (!item) return;
    if (showcaseEl) showcaseEl.value = "";
    seedFrom(item, "");
  }

  /* ── Curated showcase + trap set (grouped) ── */
  let cvTrap = [];
  const TRAP_PATH = "verifiquant/data/runs/paper_v1/trap/trap_set.jsonl";

  async function loadShowcase() {
    if (!showcaseEl) return;
    try {
      const [scRes, trRes] = await Promise.all([
        fetch("/api/demo/questions?showcase=1").then((r) => r.json()).catch(() => ({})),
        fetch(`/api/demo/questions?path=${encodeURIComponent(TRAP_PATH)}`).then((r) => r.json()).catch(() => ({})),
      ]);
      cvShowcase = (scRes && scRes.status === "ok") ? (scRes.questions || []) : [];
      // Surface a couple of E/I trap cases (the gates worth demoing) + a few others.
      const allTrap = (trRes && trRes.status === "ok") ? (trRes.questions || []) : [];
      const pref = allTrap.filter((q) => /^(E|I)/.test(String(q.expected_diagnostic || "")));
      const rest = allTrap.filter((q) => !/^(E|I)/.test(String(q.expected_diagnostic || "")));
      cvTrap = pref.slice(0, 4).concat(rest.slice(0, 2));

      let html = `<option value="">Choose a showcase…</option>`;
      if (cvShowcase.length) {
        html += `<optgroup label="Curated (one per class)">` +
          cvShowcase.map((item, idx) => `<option value="s:${idx}">${escapeHtml(item.demo_label || item.question)}</option>`).join("") +
          `</optgroup>`;
      }
      if (cvTrap.length) {
        html += `<optgroup label="Trap set">` +
          cvTrap.map((item, idx) => {
            const d = item.expected_diagnostic || "?";
            return `<option value="t:${idx}">trap · ${escapeHtml(d)} · ${escapeHtml(truncate(item.question, 48))}</option>`;
          }).join("") +
          `</optgroup>`;
      }
      showcaseEl.innerHTML = html;
    } catch (err) {
      showcaseEl.innerHTML = `<option value="">Showcase unavailable</option>`;
    }
  }

  function truncate(s, n) { s = String(s || ""); return s.length > n ? s.slice(0, n - 1) + "…" : s; }

  function applyShowcase() {
    const raw = showcaseEl.value;
    if (!raw) return;
    if (sampleEl) sampleEl.value = "";
    if (raw.startsWith("t:")) {
      const item = cvTrap[Number(raw.slice(2))];
      const naive = (item.ground_truth !== undefined && item.ground_truth !== null)
        ? ` A blind calculation yields ${item.ground_truth} — which is the trap.` : "";
      const note = `Trap · expected ${item.expected_diagnostic || "?"} (${item.expected_behavior || "recover"}).${naive}`;
      // The trap's stored ground_truth is the NAIVE (wrong) number; correct behaviour is to
      // flag/ask, so judge by behaviour (expected class), not by matching that number.
      seedFrom({ ...item, ground_truth: null }, note);
    } else {
      const item = cvShowcase[Number(raw.slice(2))];
      const cls = item.expected_class ? `Expected: ${item.expected_class}-class. ` : "";
      seedFrom(item, cls + (item.demo_note || ""));
    }
  }

  /* ── Sample seeding (independent fetch; chat tab may open before QA tab) ── */
  async function loadSamples() {
    if (!sampleEl) return;
    try {
      const res = await fetch("/api/demo/questions");
      const data = await res.json();
      if (data.status !== "ok") throw new Error(data.message || "failed");
      cvSamples = data.questions || [];
      sampleEl.innerHTML = `<option value="">Choose a sample…</option>` + cvSamples.map((item, idx) => {
        const title = item.article_title || item.function_id || `Sample ${idx + 1}`;
        return `<option value="${idx}">${String(idx + 1).padStart(2, "0")} · ${escapeHtml(title)}</option>`;
      }).join("");
      if (sampleMetaEl) sampleMetaEl.textContent = `${cvSamples.length} samples · source: ${data.source || "questions"}`;
    } catch (err) {
      if (sampleMetaEl) sampleMetaEl.textContent = `Failed to load samples: ${err.message}`;
    }
  }

  /* ── Wiring ── */
  sendBtn.addEventListener("click", onSend);
  inputEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      onSend();
    }
  });
  if (resetBtn) resetBtn.addEventListener("click", resetConversation);
  if (sampleEl) sampleEl.addEventListener("change", applySample);
  if (showcaseEl) showcaseEl.addEventListener("change", applyShowcase);
  // Manual editing of the seed invalidates the loaded gold / expected class.
  inputEl.addEventListener("input", () => { cvActiveGroundTruth = null; cvActiveExpectedClass = null; });
  if (contextEl) contextEl.addEventListener("input", () => { cvActiveGroundTruth = null; cvActiveExpectedClass = null; });

  loadSamples();
  loadShowcase();
  updateComposerMode();

  // Expose a hook so demo.js's tab switch can lazy-load samples if needed.
  window.VQChat = { loadSamples };
})();
