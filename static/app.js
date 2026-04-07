const form = document.getElementById("question-form");
const loader = document.getElementById("loader");
const resultEl = document.getElementById("result");
const cardsLoader = document.getElementById("cards-loader");
const cardsResult = document.getElementById("cards-result");
const cardsMeta = document.getElementById("cards-meta");
const contextEl = document.getElementById("context");
const questionEl = document.getElementById("question");

const tabQa = document.getElementById("tab-qa");
const tabBatch = document.getElementById("tab-batch");
const tabCards = document.getElementById("tab-cards");
const panelQa = document.getElementById("panel-qa");
const panelBatch = document.getElementById("panel-batch");
const panelCards = document.getElementById("panel-cards");
const batchLoader = document.getElementById("batch-loader");
const batchResult = document.getElementById("batch-result");
const historyResult = document.getElementById("history-result");
const historySummary = document.getElementById("history-summary");
const historyMeta = document.getElementById("history-meta");
const uploadForm = document.getElementById("upload-form");
const batchForm = document.getElementById("batch-form");

function setQaLoading(isLoading) {
  loader.classList.toggle("hidden", !isLoading);
  form.querySelector("button").disabled = isLoading;
}

function setCardsLoading(isLoading) {
  cardsLoader.classList.toggle("hidden", !isLoading);
}

function activateTab(target) {
  const isQa = target === "qa";
  const isBatch = target === "batch";
  const isCards = target === "cards";
  tabQa.classList.toggle("active", isQa);
  tabBatch.classList.toggle("active", isBatch);
  tabCards.classList.toggle("active", isCards);
  panelQa.classList.toggle("active", isQa);
  panelBatch.classList.toggle("active", isBatch);
  panelCards.classList.toggle("active", isCards);
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

function setBatchLoading(isLoading) {
  batchLoader.classList.toggle("hidden", !isLoading);
  document.getElementById("batch-run-btn").disabled = isLoading;
  document.getElementById("upload-btn").disabled = isLoading;
}

function renderBatchSummary(data) {
  const sc = data.status_counts || {};
  const dc = data.diagnostic_counts || {};
  batchResult.innerHTML = `
    <div class="meta-row">
      <span class="badge badge-success">batch / ${escapeHtml(data.status || "ok")}</span>
      <span>processed: <code>${escapeHtml(data.processed)}</code></span>
    </div>
    <p>input: <code>${escapeHtml(data.input_path || "")}</code></p>
    <p>output: <code>${escapeHtml(data.output_path || "")}</code></p>
    <p>top_k: <code>${escapeHtml(data.top_k)}</code>, m_min_top_score: <code>${escapeHtml(
      data.m_min_top_score
    )}</code></p>
    <p>status_counts:</p>
    <pre>${escapeHtml(JSON.stringify(sc, null, 2))}</pre>
    <p>diagnostic_counts:</p>
    <pre>${escapeHtml(JSON.stringify(dc, null, 2))}</pre>
  `;
}

function renderHistoryList(data) {
  const files = data.files || [];
  historyMeta.textContent = `files: ${files.length}`;
  if (!files.length) {
    historyResult.innerHTML = "<p>目前沒有歷史輸出檔。</p>";
    return;
  }
  const rows = files
    .map(
      (f) => `
      <tr>
        <td><code>${escapeHtml(f.name)}</code></td>
        <td>${escapeHtml(f.modified_at)}</td>
        <td>${escapeHtml(f.size_bytes)}</td>
        <td>
          <button class="summary-btn" data-path="${escapeHtml(f.relative_path)}" type="button">檢視摘要</button>
          <a class="download-link" href="/api/files/download?path=${encodeURIComponent(f.relative_path)}">下載</a>
        </td>
      </tr>
    `
    )
    .join("");
  historyResult.innerHTML = `
    <table class="history-table">
      <thead>
        <tr>
          <th>檔名</th>
          <th>修改時間</th>
          <th>大小(bytes)</th>
          <th>操作</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
  `;

  document.querySelectorAll(".summary-btn").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const path = btn.dataset.path;
      if (!path) return;
      await loadHistorySummary(path);
    });
  });
}

async function loadHistoryList() {
  try {
    const response = await fetch("/api/files/history");
    const data = await response.json();
    if (data.status !== "ok") throw new Error(data.message || "load history failed");
    renderHistoryList(data);
  } catch (error) {
    historyResult.innerHTML = `<p>讀取歷史清單失敗：${escapeHtml(error.message)}</p>`;
  }
}

async function loadHistorySummary(path) {
  try {
    const response = await fetch(`/api/files/summary?path=${encodeURIComponent(path)}`);
    const data = await response.json();
    if (data.status !== "ok") throw new Error(data.message || "summary failed");
    historySummary.innerHTML = `
      <h3>摘要：<code>${escapeHtml(path)}</code></h3>
      <p>records: <code>${escapeHtml(data.records)}</code></p>
      <p>status_counts:</p>
      <pre>${escapeHtml(JSON.stringify(data.status_counts || {}, null, 2))}</pre>
      <p>diagnostic_counts:</p>
      <pre>${escapeHtml(JSON.stringify(data.diagnostic_counts || {}, null, 2))}</pre>
      <p>sample:</p>
      <pre>${escapeHtml(JSON.stringify(data.sample || [], null, 2))}</pre>
    `;
  } catch (error) {
    historySummary.innerHTML = `<p>讀取摘要失敗：${escapeHtml(error.message)}</p>`;
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
      top_k: Number(document.getElementById("top-k").value || 3),
      m_min_top_score: Number(document.getElementById("m-min-top-score").value || 0.05),
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
tabBatch.addEventListener("click", () => {
  activateTab("batch");
  loadHistoryList();
});
tabCards.addEventListener("click", () => {
  activateTab("cards");
  if (!cardsResult.innerHTML.trim()) loadCardsOverview();
});
document.getElementById("refresh-history").addEventListener("click", loadHistoryList);

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const fileInput = document.getElementById("batch-file");
  const file = fileInput.files?.[0];
  if (!file) {
    alert("請先選擇 JSONL/JSON 檔案");
    return;
  }
  setBatchLoading(true);
  try {
    const formData = new FormData();
    formData.append("file", file);
    const response = await fetch("/api/files/upload", { method: "POST", body: formData });
    const data = await response.json();
    if (data.status !== "ok") throw new Error(data.message || "upload failed");
    document.getElementById("batch-input-path").value = data.relative_path;
    if (!document.getElementById("batch-output-path").value.trim()) {
      const base = data.relative_path.replace(/\.jsonl?$/i, "");
      document.getElementById("batch-output-path").value = `${base}_output.jsonl`;
    }
    loadHistoryList();
  } catch (error) {
    alert(error.message);
  } finally {
    setBatchLoading(false);
  }
});

batchForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  setBatchLoading(true);
  try {
    const payload = {
      input_path: document.getElementById("batch-input-path").value.trim(),
      output_path: document.getElementById("batch-output-path").value.trim(),
      max_records: Number(document.getElementById("batch-max-records").value || 0),
      top_k: Number(document.getElementById("batch-top-k").value || 3),
      m_min_top_score: Number(document.getElementById("batch-m-min-top-score").value || 0.05),
      domain: document.getElementById("batch-domain").value.trim() || null,
      topic: document.getElementById("batch-topic").value.trim() || null,
      include_results: false,
    };
    const response = await fetch("/api/diagnose/batch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (data.status !== "ok") throw new Error(data.message || "batch failed");
    renderBatchSummary(data);
    loadHistoryList();
    if (data.output_path) {
      loadHistorySummary(data.output_path);
    }
  } catch (error) {
    batchResult.innerHTML = `<p>批次執行失敗：${escapeHtml(error.message)}</p>`;
  } finally {
    setBatchLoading(false);
  }
});
