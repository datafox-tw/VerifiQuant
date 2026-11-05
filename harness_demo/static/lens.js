"use strict";

// Single-paper lens view. Uses helpers from harness.js: S, $, esc, fmtVal, paperBadges, paperLinks.
const L = { data: null, lens: "verina", card: "annuity_fv" };
const OUT_CLASS = { correct: "o-correct", guessed: "o-guessed", wrong: "o-wrong", crash: "o-wrong", ask: "o-ask",
  ask_noisy: "o-noisy", abstain: "o-abstain", noscope: "o-noscope", rejected: "o-muted" };

async function openLensView() {
  const view = $("#lens-view");
  if (!L.data) {
    view.innerHTML = `<div class="panel loading">在沙箱裡執行每一篇論文的機制，約需幾秒…</div>`;
    try {
      const res = await fetch("/api/lenses");
      L.data = await res.json();
    } catch (e) {
      view.innerHTML = `<div class="panel err">無法載入：${esc(e.message)}</div>`;
      return;
    }
  }
  renderLensButtons();
  renderLens();
}

function outChip(code) {
  return `<span class="chip ${OUT_CLASS[code] || ""}">${esc(L.data.outcomes[code] || (code === "rejected" ? "退回" : code))}</span>`;
}

function renderLensButtons() {
  $("#lens-buttons").innerHTML = L.data.lenses.map((l) =>
    `<button class="case ${l.id === L.lens ? "active" : ""}" data-lens="${l.id}"><div class="t">${esc(l.title)}</div>
      <div class="tag">${l.id === "verina" ? "替卡片本身打分" : "只用這個機制回答"}</div></button>`).join("");
  document.querySelectorAll("[data-lens]").forEach((b) => b.addEventListener("click", () => {
    L.lens = b.dataset.lens; renderLensButtons(); renderLens();
  }));
}

function renderLens() {
  const lens = L.data.lenses.find((l) => l.id === L.lens);
  const papers = lens.papers.map((k) => {
    const p = S.meta.papers[k];
    return `<div class="lens-paper"><b>${esc(p.short)}</b> <span class="sub">${esc(p.cite)}</span><div class="sub">${paperBadges(p)} ${paperLinks(p)}</div></div>`;
  }).join("");
  const body = lens.id === "verina" ? verinaBody() : answerBody(lens);
  $("#lens-view").innerHTML = `
    <div class="panel">
      <div class="eyebrow">單篇論文套用</div>
      <h2 class="lens-title">${esc(lens.title)}：只用這篇論文的想法處理財金例子</h2>
      ${papers}
      <div class="two lens-two">
        <div><div class="block-title">論文的想法</div><p>${esc(lens.idea)}</p></div>
        <div><div class="block-title">套到財金例子上</div><p>${esc(lens.apply)}</p></div>
      </div>
    </div>
    ${body}
    <div class="panel"><div class="block-title">這篇論文會讓 harness 保留或改什麼</div><ul>${lens.changes.map((c) => `<li>${esc(c)}</li>`).join("")}</ul></div>
    ${matrixPanel()}`;
  if (lens.id === "verina") {
    document.querySelectorAll("[data-card]").forEach((b) => b.addEventListener("click", () => { L.card = b.dataset.card; renderLens(); }));
  }
}

// ------------------------------------------------------------ answer-producing lenses

function answerBody(lens) {
  const row = L.data.matrix.find((m) => m.lens === lens.id);
  const cards = L.data.cases.map((c) => {
    const cell = row.cells[c.id];
    return `<div class="lens-case"><div class="lens-case-h"><b>${esc(c.title)}</b>${outChip(cell.outcome)}</div>
      <div class="sub">${esc(c.question)}</div><p>${esc(cell.summary)}</p>${detail(lens.id, cell, c)}</div>`;
  }).join("");
  return `<div class="panel"><div class="block-title">同一批候選程式，真的執行後的結果</div><div class="lens-cases">${cards}</div></div>`;
}

function detail(lensId, cell, c) {
  const u = c.unit;
  if (lensId === "pal" && cell.detail) {
    return `<table class="mini"><thead><tr><th>抽到</th><th>輸出</th><th>結果</th></tr></thead><tbody>${cell.detail.map((d) =>
      `<tr><td><b>${d.id}</b></td><td class="mono">${fmtVal(d.value, u)}</td><td>${outChip(d.outcome)}</td></tr>`).join("")}</tbody></table>`;
  }
  if ((lensId === "codet" || lensId === "clarifygpt") && cell.clusters) {
    const opts = cell.options || cell.clusters.map((x) => ({ ...x, bogus: false }));
    return `<div class="clusters">${opts.map((o) => `<div class="cluster ${o.bogus ? "bogus" : ""}"><b>${fmtVal(o.value, u)}</b>
      <span class="sub">← ${o.members.join(", ")}${o.bogus ? "（bug 造成）" : ""}</span></div>`).join("")}</div>`;
  }
  if (lensId === "airepr" && cell.detail) {
    return `<table class="mini"><thead><tr><th>候選</th><th>輸出</th><th>檢查員</th><th>結果</th></tr></thead><tbody>${cell.detail.map((d) =>
      `<tr><td><b>${d.id}</b></td><td class="mono">${fmtVal(d.value, u)}</td><td class="mono">${fmtVal(d.inspector, u)}</td><td>${outChip(d.outcome)}${d.bug && d.accepted ? ' <span class="bad">bug</span>' : ""}</td></tr>`).join("")}</tbody></table>`;
  }
  return "";
}

// ------------------------------------------------------------ VERINA lens

function score(x, label, hint) {
  const full = x.ok === x.n;
  return `<div class="score ${full ? "full" : "miss"}"><div class="sv">${x.ok} / ${x.n}</div><div class="sl">${label}</div><div class="sub">${hint}</div></div>`;
}

function verinaBody() {
  const g = L.data.verina.find((v) => v.card === L.card);
  const tabs = L.data.verina.map((v) => `<button class="tab ${v.card === L.card ? "active" : ""}" data-card="${v.card}">${esc(v.title)}</button>`).join("");
  const gridHead = g.grid.map((_, i) => `<th class="c" title="${esc(JSON.stringify(g.grid[i]))}">輸入 ${i + 1}</th>`).join("");
  const invLabel = (id) => {
    if (id.startsWith("claim:")) return "約定宣稱不符";
    if (id === "crash") return "崩潰";
    const inv = g.invariants.find((i) => i.id === id);
    return inv ? inv.text : id;
  };
  const negRows = g.post_sound.rows.map((r) => `<tr class="${r.ok ? "" : "survivor"}"><td><b>${esc(r.id)}</b><div class="sub">${esc(r.label)}</div></td>
      ${r.per_input.map((passed) => `<td class="c ${passed ? "bad" : "ok"}">${passed ? "放行" : "拒絕"}</td>`).join("")}
      <td>${r.ok ? '<span class="chip o-correct">抓到</span>' : '<span class="chip o-wrong">漏掉</span>'}<div class="sub">${r.killers.map(invLabel).map(esc).join("；") || "沒有任何不變量拒絕它"}</div></td></tr>`).join("");
  const posRows = g.post_complete.rows.map((r) => `<tr><td><b>${esc(r.id)}</b><div class="sub">${esc(r.label)}</div></td>
      ${r.per_input.map((passed) => `<td class="c ${passed ? "ok" : "bad"}">${passed ? "接受" : "拒絕"}</td>`).join("")}
      <td>${r.ok ? '<span class="chip o-correct">全部接受</span>' : '<span class="chip o-wrong">誤殺</span>'}</td></tr>`).join("");
  const preRows = g.pre_sound.rows.map((r) => `<tr><td>${esc(r.label)}</td><td>${r.rejected ? '<span class="chip o-correct">擋下</span>' : '<span class="chip o-wrong">放行</span>'}</td><td class="sub mono">${esc(r.by.join(", "))}</td></tr>`).join("");
  const killRows = g.invariants.filter((i) => i.kind !== "boundary").map((i) => {
    const k = g.kill[i.id] || [];
    return `<tr><td>${esc(i.text)}</td><td>${k.length ? k.map(esc).join(", ") : '<span class="na">這組反例中沒有殺掉任何一個</span>'}</td></tr>`;
  }).join("") + (g.kill.claim.length ? `<tr><td>約定釘子（宣稱與行為比對）</td><td>${g.kill.claim.map(esc).join(", ")}</td></tr>` : "");
  const survivors = g.post_sound.rows.filter((r) => !r.ok);
  const firstOnly = g.post_sound.rows.filter((r) => r.ok && r.per_input[0]);
  let finding = "";
  if (survivors.length) {
    finding += `<div class="note callout warn"><b>沒抓到：</b>${survivors.map((r) => `${esc(r.id)}（${esc(r.label)}）`).join("、")} 在所有輸入上都通過了卡片。依 VERINA 的定義，這張卡的後置條件對這個反例不 sound。要不要算 bug，取決於規格允許多少誤差，這是卡片作者要做的決定。</div>`;
  }
  if (firstOnly.length) {
    finding += `<div class="note callout"><b>只看題目那一組輸入不夠：</b>${firstOnly.map((r) => esc(r.id)).join("、")} 在題目的輸入上被放行，換到其他輸入才被拒絕。這就是 VERINA 要用一整組正反例測試、而不是單一範例的原因。</div>`;
  }
  return `<div class="panel">
    <div class="block-title">用 VERINA 的規格評分方式，替每張卡片的不變量打分</div>
    <div class="tabs">${tabs}</div>
    <div class="scores">
      ${score(g.post_complete, "後置條件 completeness", "正確程式在每組輸入上都被接受")}
      ${score(g.post_sound, "後置條件 soundness", "每個反例至少在一組輸入上被拒絕")}
      ${score(g.pre_complete, "前置條件 completeness", "合法輸入都被放行")}
      ${score(g.pre_sound, "前置條件 soundness", "不合法輸入都被擋下")}
    </div>
    ${finding}
    <div class="block-title">反例（應該被拒絕）</div>
    <div class="table-wrap"><table class="matrix"><thead><tr><th>程式</th>${gridHead}<th>結果與拒絕它的不變量</th></tr></thead><tbody>${negRows}</tbody></table></div>
    <div class="block-title">正例（應該被接受）</div>
    <div class="table-wrap"><table class="matrix"><thead><tr><th>程式</th>${gridHead}<th>結果</th></tr></thead><tbody>${posRows}</tbody></table></div>
    <div class="two">
      <div><div class="block-title">前置條件反例</div><table class="mini"><tbody>${preRows}</tbody></table></div>
      <div><div class="block-title">每條不變量殺掉哪些反例</div><table class="mini"><tbody>${killRows}</tbody></table></div>
    </div>
    <div class="sub">輸入組：${g.grid.map((x, i) => `輸入 ${i + 1} = <code>${esc(JSON.stringify(x))}</code>`).join("　")}</div>
    <div class="note">VERINA 先嘗試證明，證明不出來才退回測試；它的證明成功率低於 5%。這裡只做測試，所以「全部抓到」的意思是「在這些輸入與這些反例上成立」，不是定理。</div>
  </div>`;
}

// ------------------------------------------------------------ comparison matrix

function matrixPanel() {
  const names = { pal: "PAL / PoT", codet: "CodeT / AlphaCode", clarifygpt: "ClarifyGPT", airepr: "AIRepr", harness: "完整 harness" };
  const head = L.data.cases.map((c) => `<th>${esc(c.title)}</th>`).join("");
  const rows = L.data.matrix.map((m) => `<tr class="${m.lens === "harness" ? "hl-row" : ""} ${m.lens === L.lens ? "cur-row" : ""}"><td><b>${names[m.lens]}</b></td>${
    L.data.cases.map((c) => `<td title="${esc(m.cells[c.id].summary)}">${outChip(m.cells[c.id].outcome)}</td>`).join("")}</tr>`).join("");
  const legend = ["correct", "guessed", "wrong", "ask", "ask_noisy", "abstain", "noscope"].map(outChip).join(" ");
  return `<div class="panel"><div class="block-title">每篇論文單獨使用 vs. 完整 harness（同一批例子）</div>
    <div class="table-wrap"><table class="matrix lens-matrix"><thead><tr><th>只用</th>${head}</tr></thead><tbody>${rows}</tbody></table></div>
    <div class="legend-row">${legend}</div>
    <div class="note">「追問含雜訊」指選項裡有 bug 造成的答案，或題目根本沒有歧義卻被問。PAL 一列取「抽到最壞的那一份」；滑過格子可看細節。VERINA 不產生答案，它評的是卡片，所以不在這張表裡。評分用 cases.py 裡人工標註的 bug，harness 本身從不讀這些標註。</div></div>`;
}
