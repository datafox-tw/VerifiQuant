/* ── VerifiQuant Paper Replay ─────────────────────────────────────────────
 * Renders static/data/paper_replay.json (built by scripts/build_paper_demo.py):
 * recorded outputs of R-PAL, R-PAL+ and VerifiQuant on the paper's 80 traps and
 * three clean questions, scored with the paper's own scorer. No model calls.
 *
 * Also drives the "Verified repair" tab, which calls /api/transform/apply (the
 * deterministic verifier, no LLM) for the loan contract's annuity-due transform.
 *
 * Identifiers are prefixed `rp` to stay clear of demo.js / conversation.js globals.
 */
(() => {
  const { escapeHtml } = VQRender;
  const DATA_URL = `/static/data/paper_replay.json?v=${document.body.dataset.assetVersion || ""}`;

  const SYSTEMS = {
    rpal: { name: "R-PAL", desc: "Writes a program from the four retrieved contracts and runs it." },
    rpal_plus: { name: "R-PAL+", desc: "Same, with the full M/N/F/E/I taxonomy and its checks described in the prompt." },
    vq: { name: "VerifiQuant", desc: "Selects a contract. Executable gates decide whether a number may leave." },
  };
  const EXIT_LABEL = {
    N: "Out of scope · N",
    M: "Which quantity? · M",
    F: "Missing input · F",
    E: "Boundary alert · E",
    I_HARD: "Clarify convention · I (hard)",
    I_SOFT: "Number with warning · I (soft)",
    I_SOFT_MISMATCH: "Number with an unrelated warning",
    READY: "Number",
    C: "Stopped at compute",
  };
  const OPERATOR_LABEL = { E: "boundary", I: "ambiguity", F: "missing input", N: "scope" };
  const STAGES = [
    { key: "retrieve", label: "Retrieve", kind: "code" },
    { key: "select", label: "Select", kind: "llm" },
    { key: "bind", label: "Bind", kind: "llm" },
    { key: "inputs", label: "Inputs · F", kind: "code" },
    { key: "boundary", label: "Boundary · E", kind: "code" },
    { key: "ambiguity", label: "Ambiguity · I", kind: "llm" },
    { key: "compute", label: "Compute", kind: "code" },
  ];
  const STOP_AT = { N: 1, M: 1, F: 3, E: 4, I_HARD: 5, C: 6 };

  const storyEl = document.getElementById("story-list");
  const viewEl = document.getElementById("case-view");
  const filterEl = document.getElementById("trap-filter");
  const selectEl = document.getElementById("trap-select");
  const countEl = document.getElementById("trap-count");
  if (!viewEl) return;

  let rpData = null;
  let rpById = {};
  let rpFeatured = {};
  let rpCurrent = null;
  let rpRun = {}; // system -> selected run index for the current case
  let rpFilter = "all";

  /* ── Formatting ── */
  // Recorded outputs are shown as emitted: a string stays verbatim ("6500.00%"), a number
  // only gains thousands separators.
  function fmtNum(value) {
    if (value === null || value === undefined) return "";
    if (typeof value === "number") return value.toLocaleString("en-US", { maximumFractionDigits: 6 });
    return String(value).trim();
  }

  function plainExpr(expr) {
    return String(expr || "").replace(/inputs\['([^']+)'\]/g, "$1");
  }

  function falseAccepts(records) {
    return records.filter((r) => r.false_accept).length;
  }

  /* ── Case selection ── */
  function caseIds() {
    return rpData.cases.map((c) => c.id);
  }

  function trapMatches(c) {
    if (c.set !== "trap") return false;
    if (rpFilter === "all") return true;
    if (["E", "I", "F", "N"].includes(rpFilter)) return c.operator === rpFilter;
    if (rpFilter === "vq_miss") return falseAccepts(c.systems.vq) > 0;
    if (rpFilter === "plus_miss") return falseAccepts(c.systems.rpal_plus) > 0;
    return true;
  }

  function renderStory() {
    storyEl.innerHTML = rpData.featured.map((f, i) => `
      <li><button class="story-btn" type="button" data-case="${escapeHtml(f.id)}">
        <span class="n">${i + 1}</span>
        <span class="ch">${escapeHtml(f.chapter)}</span>
        <span class="tt">${escapeHtml(f.title)}</span>
      </button></li>`).join("");
    storyEl.querySelectorAll("[data-case]").forEach((btn) => {
      btn.addEventListener("click", () => selectCase(btn.dataset.case, true));
    });
  }

  function renderFilter() {
    const opts = [
      ["all", "All"], ["E", "E"], ["I", "I"], ["F", "F"], ["N", "N"],
      ["vq_miss", "VerifiQuant let one through"], ["plus_miss", "R-PAL+ let one through"],
    ];
    filterEl.innerHTML = opts.map(([v, label]) =>
      `<button type="button" class="filter-btn ${v === rpFilter ? "active" : ""}" data-filter="${v}">${escapeHtml(label)}</button>`
    ).join("");
    filterEl.querySelectorAll("[data-filter]").forEach((btn) => {
      btn.addEventListener("click", () => { rpFilter = btn.dataset.filter; renderFilter(); renderTrapSelect(); });
    });
  }

  function renderTrapSelect() {
    const traps = rpData.cases.filter(trapMatches);
    selectEl.innerHTML = `<option value="">Choose a trap (${traps.length})…</option>` + traps.map((c) => {
      const vq = falseAccepts(c.systems.vq);
      const plus = falseAccepts(c.systems.rpal_plus);
      const q = c.question.length > 54 ? `${c.question.slice(0, 53)}…` : c.question;
      return `<option value="${escapeHtml(c.id)}" ${c.id === rpCurrent ? "selected" : ""}>${escapeHtml(c.id)} · VQ ${vq}/3 · R-PAL+ ${plus}/3 · ${escapeHtml(q)}</option>`;
    }).join("");
    countEl.textContent = "Counts are false numeric acceptances out of three runs.";
  }

  function selectCase(id, fromUser) {
    const c = rpById[id];
    if (!c) return;
    rpCurrent = id;
    rpRun = {};
    storyEl.querySelectorAll("[data-case]").forEach((btn) => btn.classList.toggle("active", btn.dataset.case === id));
    if (selectEl.value !== id) selectEl.value = rpFeatured[id] ? "" : id;
    renderCase(c);
    if (!fromUser) return;
    if (history.replaceState) history.replaceState(null, "", `#${id}`);
    if (window.innerWidth < 900) viewEl.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  /* ── Question card ── */
  function highlightContext(c) {
    let html = escapeHtml(c.context || "");
    const edit = c.edit || {};
    if (edit.kind === "removed_field") {
      html = html.replace(/\[REDACTED\]/g, `<mark class="edit">[REDACTED]</mark>`);
    } else if (edit.kind === "value" && edit.to !== null && edit.to !== undefined) {
      const v = Number(edit.to);
      const forms = new Set([String(edit.to), String(v), v.toLocaleString("en-US"), v.toLocaleString("en-US", { maximumFractionDigits: 6 })]);
      for (const form of forms) {
        const esc = form.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
        const re = new RegExp(`(?<![\\d.,])${esc}(?![\\d])`);
        if (re.test(html)) {
          const from = edit.from !== null && edit.from !== undefined ? `<s>${escapeHtml(fmtNum(Number(edit.from)))}</s>` : "";
          html = html.replace(re, `<mark class="edit">${from}${escapeHtml(form)}</mark>`);
          break;
        }
      }
    }
    return html;
  }

  function editLine(c) {
    const e = c.edit || {};
    if (c.set !== "trap") return "";
    if (e.kind === "value") {
      return `Trap edit: <code>${escapeHtml(e.field)}</code> changed from ${escapeHtml(fmtNum(Number(e.from)))} to
        ${escapeHtml(fmtNum(Number(e.to)))}, which violates the contract check <code>${escapeHtml(e.rule_id)}</code>
        (<code>${escapeHtml(plainExpr(e.expression))}</code>).`;
    }
    if (e.kind === "removed_field") {
      return `Trap edit: the required input <code>${escapeHtml(e.field)}</code> (value ${escapeHtml(fmtNum(Number(e.value)))}) was redacted.`;
    }
    if (e.kind === "removed_phrase") {
      return `Trap edit: the phrase &ldquo;${escapeHtml(e.phrase)}&rdquo; was removed, leaving the convention the contract declares in
        <code>${escapeHtml(e.rule_id)}</code> unstated.`;
    }
    return `Trap edit: the question was replaced with an out-of-scope ask (template <code>${escapeHtml(e.template)}</code>); no contract should match.`;
  }

  function expectedChip(c) {
    if (c.set === "clean") {
      return `<span class="chip chip-muted">clean question · ${escapeHtml(c.level || "")}</span>
        <span class="chip chip-num">gold answer ${escapeHtml(fmtNum(c.gold))}</span>`;
    }
    return `<span class="chip chip-muted">trap · ${escapeHtml(c.operator)} (${escapeHtml(OPERATOR_LABEL[c.operator] || "")}) · ${escapeHtml(c.family || "")}</span>
      <span class="chip chip-exit">expected exit: ${escapeHtml(EXIT_LABEL[c.expected] || c.expected)}</span>
      <span class="chip chip-risk">no valid number exists</span>`;
  }

  /* ── Funnel strip (VerifiQuant's path for the selected run) ── */
  function renderStrip(rec, risky) {
    let stop = STOP_AT[rec.exit];
    if (rec.exit === "N" && !(rec.candidates || []).length) stop = 0;
    const emitted = rec.numeric;
    if (emitted) stop = undefined;
    const nodes = STAGES.map((s, i) => {
      let cls = `strip-node ${s.kind}`;
      if (stop === undefined || i < stop) cls += " pass";
      if (stop === i) cls += " stop";
      if (emitted && s.key === "ambiguity" && /I_SOFT/.test(rec.exit)) cls += " warn";
      const label = stop === i ? `${s.label} · stopped` : s.label;
      return `<span class="${cls}">${escapeHtml(label)}</span>`;
    });
    if (emitted) nodes.push(`<span class="strip-node emit ${risky ? "risk" : ""}">${risky ? "Number (silent wrong)" : "Number"}</span>`);
    return `<div class="strip" aria-label="VerifiQuant's path"><span class="strip-label">VerifiQuant's path</span>${nodes.join('<span class="strip-arrow">→</span>')}</div>`;
  }

  /* ── One system column ── */
  function recordRisky(c, rec) {
    if (c.set === "trap") return Boolean(rec.false_accept);
    return rec.verdict === "silent_wrong";
  }

  function verdictBox(c, sysKey, rec) {
    const risky = recordRisky(c, rec);
    if (rec.numeric) {
      let label = "Number";
      if (c.set === "trap") {
        label = risky ? "False numeric acceptance" : "Number with the expected warning";
      } else {
        label = rec.verdict === "correct" ? "Correct · matches gold" : (risky ? "Silent wrong" : "Number");
      }
      return `<div class="verdict-box ${risky ? "risk" : "num"}"><div class="vlabel">${escapeHtml(label)}</div>
        <div class="vvalue">${escapeHtml(fmtNum(rec.output))}</div></div>`;
    }
    let label;
    if (c.set === "trap") {
      label = rec.exit_correct ? "No number · expected exit" : "No number · different exit";
    } else {
      label = "No number · counts against coverage";
    }
    let exit = EXIT_LABEL[rec.exit] || rec.exit;
    if (sysKey !== "vq" && rec.exit === "READY") exit = "Program failed to return a number";
    return `<div class="verdict-box"><div class="vlabel">${escapeHtml(label)}</div><div class="vexit">${escapeHtml(exit)}</div></div>`;
  }

  function runTabs(c, sysKey, records) {
    if (records.length < 2) return `<div class="run-tabs"><span class="chip chip-muted">1 recorded run</span></div>`;
    const sel = rpRun[sysKey] || 0;
    return `<div class="run-tabs">${records.map((r, i) => {
      const pip = recordRisky(c, r) ? "risk" : (r.numeric ? "num" : "");
      return `<button type="button" class="run-tab ${i === sel ? "active" : ""}" data-sys="${sysKey}" data-run="${i}">
        <span class="pip ${pip}"></span>Run ${i + 1}</button>`;
    }).join("")}</div>`;
  }

  function vqDetails(c, rec) {
    const first = c.systems.vq[0];
    const cands = rec.candidates || first.candidates || [];
    const inputs = rec.inputs || first.inputs || {};
    const selector = rec.selector_reason || first.selector_reason || "";
    const parts = [];
    if (cands.length) {
      parts.push(`<div><p class="sub-label">Retrieved contracts (BM25, top 4)</p><ul class="cand-list">${cands.map((k) =>
        `<li class="${k.fic_id === rec.fic_id ? "chosen" : ""}"><span>${escapeHtml(k.title || k.fic_id)}${k.fic_id === rec.fic_id ? " ← selected" : ""}</span>
        <span class="s">${k.score !== null && k.score !== undefined ? escapeHtml(String(k.score)) : ""}</span></li>`).join("")}</ul></div>`);
    }
    if (selector) parts.push(`<div><p class="sub-label">Selector (LLM 1)</p><p class="sys-msg">${escapeHtml(selector)}</p></div>`);
    const keys = Object.keys(inputs);
    if (keys.length) {
      parts.push(`<div><p class="sub-label">Bound inputs (LLM 2)</p><dl class="kv">${keys.map((k) =>
        `<dt>${escapeHtml(k)}</dt><dd>${escapeHtml(typeof inputs[k] === "object" ? JSON.stringify(inputs[k]) : String(inputs[k]))}</dd>`).join("")}</dl></div>`);
    }
    if ((rec.rule_ids || []).length) {
      parts.push(`<div><p class="sub-label">Checks that fired</p><p class="sys-msg">${rec.rule_ids.map((r) => `<code>${escapeHtml(r)}</code>`).join(", ")}</p></div>`);
    }
    if (!parts.length) return "";
    return `<details><summary>DiagnosticReport</summary><div class="inner">${parts.join("")}</div></details>`;
  }

  function systemColumn(c, sysKey) {
    const records = c.systems[sysKey] || [];
    if (!records.length) return "";
    const rec = records[rpRun[sysKey] || 0];
    const meta = SYSTEMS[sysKey];
    let body = verdictBox(c, sysKey, rec);
    if (sysKey === "vq") {
      if (rec.fic_id) body += `<p class="sys-msg"><span class="who">Contract</span>${escapeHtml(rec.fic_title || rec.fic_id)} <code>${escapeHtml(rec.fic_id)}</code></p>`;
      if ((rec.requested_fields || []).length) {
        body += `<p class="sys-msg"><span class="who">Asks the user for</span>${rec.requested_fields.map((r) => `<code>${escapeHtml(r)}</code>`).join(", ")}</p>`;
      }
      if (rec.message) body += `<p class="sys-msg"><span class="who">What the user sees</span>${escapeHtml(rec.message)}</p>`;
      if ((rec.options || []).length) {
        body += `<div><p class="sub-label">Contract's declared options</p><ul class="opt-list">${rec.options.map((o) =>
          `<li>${escapeHtml(o.label)}</li>`).join("")}</ul></div>`;
      }
      if (rec.numeric && (rec.soft_warnings || []).length && !rec.message) {
        body += `<p class="sys-msg"><span class="who">Warning</span>${escapeHtml(rec.soft_warnings[0].question || "")}</p>`;
      }
      body += vqDetails(c, rec);
    } else {
      if (sysKey === "rpal_plus" && rec.decision) {
        body += `<p class="sys-msg"><span class="who">Its decision</span>${escapeHtml(rec.decision)} · ${escapeHtml(rec.exit)}</p>`;
      }
      if (rec.message) {
        body += `<details><summary>${sysKey === "rpal_plus" ? "Its stated reason" : "Note"}</summary><div class="inner"><p class="sys-msg">${escapeHtml(rec.message)}</p></div></details>`;
      }
      const code = records[0].code;
      if (code) body += `<details><summary>Program it wrote${records.length > 1 ? " (run 1)" : ""}</summary><div class="inner"><pre class="code">${escapeHtml(code)}</pre></div></details>`;
    }
    return `<div class="sys-col ${sysKey === "vq" ? "vq" : ""}">
      <div><div class="sys-name">${escapeHtml(meta.name)}</div><div class="sys-desc">${escapeHtml(meta.desc)}</div></div>
      ${runTabs(c, sysKey, records)}
      ${body}
    </div>`;
  }

  function summaryLine(c) {
    if (c.set !== "trap") return "";
    const parts = ["rpal", "rpal_plus", "vq"].map((k) => {
      const recs = c.systems[k];
      return `${SYSTEMS[k].name} ${falseAccepts(recs)} of ${recs.length}`;
    });
    return `False numeric acceptances across recorded runs: ${parts.join(" · ")}.`;
  }

  function renderCase(c) {
    const f = rpFeatured[c.id];
    const vqRec = c.systems.vq[rpRun.vq || 0];
    const head = f
      ? `<p class="eyebrow">${escapeHtml(f.chapter)}</p><h2>${escapeHtml(f.title)}</h2><p class="case-caption">${escapeHtml(f.caption)}</p>`
      : `<p class="eyebrow">Trap ${escapeHtml(c.id)}</p><h2>${escapeHtml(c.question)}</h2><p class="case-caption">${escapeHtml(summaryLine(c))}</p>`;
    const order = c.set === "trap" ? ["rpal", "rpal_plus", "vq"] : ["rpal", "vq"];
    const prov = rpData.meta.sources_display[c.set];
    viewEl.innerHTML = `
      <div class="case-head">${head}</div>
      <div class="case-meta">${expectedChip(c)}</div>
      <div class="quote-card case-question">
        <p class="q">&ldquo;${escapeHtml(c.question)}&rdquo;</p>
        <p class="ctx">${highlightContext(c)}</p>
        ${c.set === "trap" ? `<p class="edit-line">${editLine(c)}</p>` : ""}
      </div>
      <div class="case-actions">
        <button type="button" class="btn btn-outline btn-sm" id="rp-live">Send this question to the live system →</button>
        <span class="hint">The live run makes fresh LLM calls and can differ from the recorded ones.</span>
      </div>
      ${renderStrip(vqRec, recordRisky(c, vqRec))}
      <div class="sys-grid ${order.length === 2 ? "two" : ""}">${order.map((k) => systemColumn(c, k)).join("")}</div>
      ${f && c.set === "trap" ? `<p class="table-note" style="margin-top:0.8rem">${escapeHtml(summaryLine(c))}</p>` : ""}
      <p class="provenance">Recorded in <code>verifiquant/data/runs/</code>: ${escapeHtml(prov)}. Scored with <code>${escapeHtml(rpData.meta.scorer)}</code>.</p>`;

    viewEl.querySelectorAll(".run-tab").forEach((btn) => {
      btn.addEventListener("click", () => {
        rpRun[btn.dataset.sys] = Number(btn.dataset.run);
        renderCase(c);
      });
    });
    const liveBtn = document.getElementById("rp-live");
    if (liveBtn) liveBtn.addEventListener("click", () => sendLive(c));
  }

  function sendLive(c) {
    if (!window.VQChat || !window.VQChat.seed) return;
    const note = c.set === "trap"
      ? `Paper trap ${c.id} · expected exit ${c.expected}. ${c.edit_note || ""}`
      : `Paper clean question ${c.id} · gold ${c.gold}.`;
    window.VQChat.seed({
      question: c.question,
      context: c.context,
      ground_truth: c.set === "clean" ? c.gold : null,
      expected_class: c.set === "trap" ? c.expected : null,
    }, note);
    if (window.VQTabs) window.VQTabs.activate("chat");
    if (history.replaceState) history.replaceState(null, "", "#live");
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  /* ── Load + routing ── */
  async function load() {
    try {
      const res = await fetch(DATA_URL);
      rpData = await res.json();
    } catch (err) {
      viewEl.innerHTML = `<p>Could not load the recorded runs: ${escapeHtml(err.message)}</p>`;
      return;
    }
    rpById = Object.fromEntries(rpData.cases.map((c) => [c.id, c]));
    rpFeatured = Object.fromEntries(rpData.featured.map((f) => [f.id, f]));
    window.VQReplay = { data: rpData };
    renderStory();
    renderFilter();
    renderTrapSelect();
    selectEl.addEventListener("change", () => { if (selectEl.value) selectCase(selectEl.value, true); });
    const hash = decodeURIComponent(location.hash.replace(/^#/, ""));
    selectCase(rpById[hash] ? hash : rpData.featured[0].id, false);
    document.dispatchEvent(new CustomEvent("vq-replay-ready", { detail: rpData }));
  }

  function routeHash() {
    const hash = decodeURIComponent(location.hash.replace(/^#/, ""));
    if (!hash || !window.VQTabs) return;
    const tab = { live: "chat", chat: "chat", repair: "repair", qa: "qa", cards: "cards", replay: "replay" }[hash];
    if (tab) window.VQTabs.activate(tab);
    else window.VQTabs.activate("replay");
  }

  /* ── Verified repair tab ── */
  const rpRunBtn = document.getElementById("rp-run");
  const rpResultEl = document.getElementById("rp-result");

  function repairInputs() {
    return {
      principal: Number(document.getElementById("rp-principal").value),
      annual_interest_rate: Number(document.getElementById("rp-rate").value),
      loan_term_years: Number(document.getElementById("rp-term").value),
    };
  }

  async function runRepair() {
    const choice = (document.querySelector('input[name="rp-timing"]:checked') || {}).value || "end_of_period";
    const inputs = repairInputs();
    rpResultEl.innerHTML = `<div class="loader"><div class="spinner"></div><span>Verifying…</span></div>`;
    try {
      // The default option has no transform (the contract's code is the default), so the
      // verifier is always asked about the declared alternative; result_old is the default.
      const res = await fetch("/api/transform/apply", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ fic_id: "fic_article_2164", chosen_value: "beginning_of_period", inputs }),
      });
      const data = await res.json();
      if (data.status === "rejected") {
        rpResultEl.innerHTML = `<div class="verdict-box risk"><div class="vlabel">Rejected by the verifier</div>
          <div class="vexit">Result not changed</div></div><p class="sys-msg">${escapeHtml(data.error || data.message || "")}</p>`;
        return;
      }
      if (data.status !== "ok") throw new Error(data.message || "verifier error");
      const fmt = (v) => Number(v).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
      if (choice === "end_of_period") {
        rpResultEl.innerHTML = `<div class="verdict-box num"><div class="vlabel">Contract code, default convention</div>
          <div class="vvalue">${fmt(data.result_old)}</div></div>
          <p class="sys-msg">No edit applies: the contract's own code computes the ordinary annuity. Switch the convention to see the declared transform.</p>`;
        return;
      }
      const samples = data.numerical_samples;
      const nSamples = Array.isArray(samples) ? samples.length : (typeof samples === "number" ? samples : null);
      rpResultEl.innerHTML = `
        <div class="flow-nums"><span class="old">${fmt(data.result_old)}</span><span class="arrow">→</span><span class="new">${fmt(data.result_new)}</span></div>
        <ul class="check-list">
          <li><span class="ok">&check;</span><span>Declared transform found on rule <code>${escapeHtml(data.rule_id || "")}</code>.</span></li>
          <li><span class="ok">&check;</span><span>AST bound, safe names and determinism passed.</span></li>
          <li><span class="ok">&check;</span><span>Invariant <code>${escapeHtml(data.invariant)}</code> holds${nSamples ? ` on ${nSamples} sampled binding${nSamples === 1 ? "" : "s"}` : ""}.</span></li>
        </ul>
        <details><summary class="sub-label" style="cursor:pointer">The contract's locked code</summary>
          <pre class="code" style="margin-top:0.5rem">${escapeHtml(data.python_code || "")}</pre></details>`;
    } catch (err) {
      rpResultEl.innerHTML = `<p class="sys-msg">Verifier unavailable: ${escapeHtml(err.message)}</p>`;
    }
  }

  if (rpRunBtn) {
    rpRunBtn.addEventListener("click", runRepair);
    document.querySelectorAll('input[name="rp-timing"]').forEach((input) => {
      input.addEventListener("change", () => {
        document.querySelectorAll("#rp-choices .choice").forEach((el) => {
          el.classList.toggle("active", Boolean(el.querySelector("input:checked")));
        });
        runRepair();
      });
    });
  }

  document.addEventListener("DOMContentLoaded", () => {
    routeHash();
    load();
  });
})();
