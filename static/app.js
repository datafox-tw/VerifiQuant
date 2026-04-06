const form = document.getElementById("question-form");
const loader = document.getElementById("loader");
const resultEl = document.getElementById("result");
const cardsLoader = document.getElementById("cards-loader");
const cardsResult = document.getElementById("cards-result");
const cardsMeta = document.getElementById("cards-meta");
const contextEl = document.getElementById("context");
const questionEl = document.getElementById("question");

const tabQa = document.getElementById("tab-qa");
const tabCards = document.getElementById("tab-cards");
const panelQa = document.getElementById("panel-qa");
const panelCards = document.getElementById("panel-cards");

function setQaLoading(isLoading) {
  loader.classList.toggle("hidden", !isLoading);
  form.querySelector("button").disabled = isLoading;
}

function setCardsLoading(isLoading) {
  cardsLoader.classList.toggle("hidden", !isLoading);
}

function activateTab(target) {
  const isQa = target === "qa";
  tabQa.classList.toggle("active", isQa);
  tabCards.classList.toggle("active", !isQa);
  panelQa.classList.toggle("active", isQa);
  panelCards.classList.toggle("active", !isQa);
}

function escapeHtml(text) {
  return String(text || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function statusBadge(status, diagnosticType) {
  return `<span class="badge badge-${status}">${escapeHtml(status)} / ${escapeHtml(
    diagnosticType || "Unknown"
  )}</span>`;
}

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

async function composeAndApplyRepair() {
  const updates = collectRepairUpdates();
  const payload = {
    question: questionEl.value,
    context: contextEl.value,
    updates,
  };
  const response = await fetch("/api/repair/compose", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await response.json();
  if (data.status !== "ok") {
    throw new Error(data.message || "repair compose failed");
  }
  questionEl.value = data.repaired_question || questionEl.value;
  contextEl.value = data.repaired_context || contextEl.value;
}

function renderRepairHints(repairHints) {
  if (!repairHints?.length) return "";
  const blocks = repairHints
    .map((hint) => {
      const asks = (hint.ask_user_for || [])
        .map((ask) => {
          const slot = escapeHtml(ask.slot || "");
          const label = escapeHtml(ask.label || ask.slot || "");
          return `
            <label class="repair-field">
              ${label}
              <input class="repair-input" data-slot="${slot}" type="text" placeholder="${slot}" />
            </label>
          `;
        })
        .join("");
      return `
        <article class="repair-card">
          <h4>${escapeHtml(hint.title || hint.rule_id || "Repair Hint")}</h4>
          <p>${escapeHtml(hint.user_message || "")}</p>
          ${asks ? `<div class="repair-grid">${asks}</div>` : ""}
        </article>
      `;
    })
    .join("");
  return `
    <section class="repair-section">
      <h3>Repair Guide</h3>
      ${blocks}
      <div class="repair-actions">
        <button id="apply-repair-btn" type="button">套用 repair 到 context</button>
        <button id="rerun-btn" type="button">用修正版重新診斷</button>
      </div>
    </section>
  `;
}

function renderResult(data) {
  resultEl.classList.remove("hidden");
  const summary = `
    <div class="meta-row">
      ${statusBadge(data.status, data.diagnostic_type)}
      <span>Gate: <code>${escapeHtml(data.funnel_layer || "-")}</code> / <code>${escapeHtml(
        data.gate_action || "-"
      )}</code></span>
    </div>
    <p>原因：${escapeHtml(data.reason || "-")}</p>
    <p>卡片：<code>${escapeHtml(data.fic_id || "(none)")}</code></p>
    <p>候選：<code>${escapeHtml((data.candidate_ids || []).join(", "))}</code></p>
    <p>inputs：</p>
    <pre>${escapeHtml(JSON.stringify(data.provided_inputs || {}, null, 2))}</pre>
  `;

  const successPart =
    data.status === "success"
      ? `<p class="result-output">${escapeHtml(data.output_var)} = <strong>${escapeHtml(
          data.output_value
        )}</strong></p>`
      : "";
  const rulePart = data.triggered_rule_ids?.length
    ? `<p>觸發規則：<code>${escapeHtml(data.triggered_rule_ids.join(", "))}</code></p>`
    : "";
  resultEl.innerHTML = `${summary}${rulePart}${successPart}${renderRepairHints(data.repair_hints)}`;

  const applyBtn = document.getElementById("apply-repair-btn");
  const rerunBtn = document.getElementById("rerun-btn");
  if (applyBtn) {
    applyBtn.addEventListener("click", async () => {
      try {
        await composeAndApplyRepair();
      } catch (error) {
        alert(error.message);
      }
    });
  }
  if (rerunBtn) {
    rerunBtn.addEventListener("click", async () => {
      try {
        await composeAndApplyRepair();
        form.requestSubmit();
      } catch (error) {
        alert(error.message);
      }
    });
  }
}

async function loadCardsOverview() {
  setCardsLoading(true);
  try {
    const response = await fetch("/api/cards/overview");
    const data = await response.json();
    if (data.status !== "ok") throw new Error(data.message || "load cards failed");
    cardsMeta.textContent = `domain: ${data.domain_count}, cards: ${data.card_count}`;
    const domains = Object.keys(data.grouped || {}).sort();
    const html = domains
      .map((domain) => {
        const domainData = data.grouped[domain];
        const topicKeys = Object.keys(domainData.topics || {}).sort();
        const topicsHtml = topicKeys
          .map((topic) => {
            const t = domainData.topics[topic];
            const cards = (t.cards || [])
              .slice(0, 20)
              .map(
                (c) => `
                <li>
                  <code>${escapeHtml(c.fic_id)}</code>
                  <span>${escapeHtml(c.title || "")}</span>
                </li>
              `
              )
              .join("");
            return `
              <details>
                <summary>${escapeHtml(topic)} (${t.count})</summary>
                <ul class="card-list">${cards}</ul>
              </details>
            `;
          })
          .join("");
        return `
          <section class="domain-block">
            <h3>${escapeHtml(domain)} <span>(${domainData.total})</span></h3>
            ${topicsHtml}
          </section>
        `;
      })
      .join("");
    cardsResult.innerHTML = html || "<p>目前沒有卡片資料。</p>";
  } catch (error) {
    cardsResult.innerHTML = `<p>載入失敗：${escapeHtml(error.message)}</p>`;
  } finally {
    setCardsLoading(false);
  }
}

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
tabQa.addEventListener("click", () => activateTab("qa"));
tabCards.addEventListener("click", () => {
  activateTab("cards");
  if (!cardsResult.innerHTML.trim()) loadCardsOverview();
});
