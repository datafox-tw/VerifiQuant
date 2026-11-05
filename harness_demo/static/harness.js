"use strict";

const S = { meta: null, mode: "scripted", caseId: null, question: "", answers: {}, data: null };
const $ = (sel, root = document) => root.querySelector(sel);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const STATUS = { pass: "通過", fail: "失敗", ask: "追問", abstain: "拒答", info: "資訊", waiting: "等待" };
const ACTION = { EMIT: ["輸出答案", "pass"], ASK: ["追問使用者", "ask"], ABSTAIN: ["拒答", "abstain"] };
const KIND = { pre: "前置條件", post: "後置條件", meta: "變形關係", boundary: "約定釘子" };
const SOURCE = { dbc: "Design by Contract", metamorphic: "變形測試", induction: "歸納法", simulation: "模擬", round_trip: "反推" };

function fmtVal(v, unit) {
  if (v === null || v === undefined) return "—";
  if (typeof v === "object") return esc(JSON.stringify(v));
  if (unit === "小數") return (v * 100).toFixed(2) + "%";
  if (unit === "元") return Number(v).toLocaleString("en-US", { maximumFractionDigits: 2 }) + " 元";
  return String(v);
}

async function init() {
  S.meta = await (await fetch("/api/meta")).json();
  renderCases();
  renderLit();
  renderNav();
  if (!S.meta.live) {
    $("#live-btn").disabled = true;
    $("#live-btn").title = "需要 GEMINI_API_KEY";
  }
  document.querySelectorAll(".mode-btn").forEach((b) => b.addEventListener("click", () => setMode(b.dataset.mode)));
  $("#live-run").addEventListener("click", () => {
    S.question = $("#live-q").value.trim();
    S.answers = {};
    if (S.question) run(0);
  });
  const ex = $("#live-examples");
  ["我每月存 5,000 元，年利率 2%，存 10 年，最後帳戶裡有多少錢？", ...S.meta.cases.slice(0, 5).map((c) => c.question)]
    .forEach((q) => {
      const b = document.createElement("button");
      b.textContent = q;
      b.onclick = () => { $("#live-q").value = q; };
      ex.appendChild(b);
    });
  document.addEventListener("click", (e) => { if (!e.target.closest(".pop, .paper")) $("#pop").classList.add("hidden"); });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") $("#pop").classList.add("hidden"); });
  selectCase(S.meta.cases[0].id);
}

function setMode(mode) {
  document.querySelectorAll(".mode-btn").forEach((b) => b.classList.toggle("active", b.dataset.mode === mode));
  document.body.classList.toggle("lens-mode", mode === "lens");
  if (mode === "lens") {
    openLensView();
    return;
  }
  S.mode = mode;
  $("#case-list").classList.toggle("hidden", mode === "live");
  $("#live-box").classList.toggle("hidden", mode !== "live");
  if (mode === "live") {
    S.data = null; S.answers = {};
    $("#question").innerHTML = `<div class="eyebrow">即時模式</div><div class="q">在左側輸入任何財金計算題</div>
      <div class="story">規格抽取、候選程式、檢查員都由 Gemini 現場生成；不變量與驗證仍由卡片和沙箱執行。問題類型目前限於四張卡：年金終值、房貸月付、CAGR、NPV。</div>`;
    $("#pipeline").innerHTML = ""; $("#baseline").innerHTML = ""; $("#decision").innerHTML = "";
  } else {
    selectCase(S.caseId || S.meta.cases[0].id);
  }
}

function renderCases() {
  const firstPoc = S.meta.cases.findIndex((c) => c.id.startsWith("poc_"));
  $("#cases").innerHTML = S.meta.cases.map((c, i) => (i === firstPoc
    ? `<div class="hint"><b>PoC 實測紀錄</b>：論文的英文題與陷阱題。卡片、候選程式、檢查員全部由 Gemini 自動產生，錄下完整執行後在這裡重播，失敗案例也一併列出。</div>`
    : "") +
    `<button class="case" data-id="${c.id}"><div class="t">${esc(c.title)}</div><div class="tag">${esc(c.tag)}</div></button>`).join("");
  document.querySelectorAll(".case").forEach((b) => b.addEventListener("click", () => selectCase(b.dataset.id)));
}

function selectCase(id) {
  S.caseId = id; S.answers = {};
  const c = S.meta.cases.find((x) => x.id === id);
  S.question = c.question;
  document.querySelectorAll(".case").forEach((b) => b.classList.toggle("active", b.dataset.id === id));
  $("#question").innerHTML = `<div class="eyebrow">使用者的問題</div><div class="q">${esc(c.question)}</div><div class="story">${esc(c.story)}</div>`;
  run(0);
}

function renderNav() {
  $("#nav").innerHTML = Object.entries(S.meta.steps).map(([id, t], i) =>
    `<li data-step="${id}"><span class="dot" id="dot-${id}"></span>${i + 1}. ${esc(t)}</li>`).join("");
  document.querySelectorAll("#nav li").forEach((li) => li.addEventListener("click", () =>
    document.getElementById("step-" + li.dataset.step)?.scrollIntoView({ behavior: "smooth", block: "start" })));
}

async function run(revealFrom) {
  const body = S.mode === "live" ? { mode: "live", question: S.question, answers: S.answers }
    : { mode: "scripted", case_id: S.caseId, answers: S.answers };
  $("#pipeline").innerHTML = `<div class="loading">${S.mode === "live" ? "Gemini 生成中，約 15–40 秒…" : "執行中…"}</div>`;
  $("#live-run").disabled = true;
  let data;
  try {
    const res = await fetch("/api/run", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    data = await res.json();
    if (!res.ok) throw new Error(data.error || res.statusText);
  } catch (e) {
    $("#pipeline").innerHTML = `<div class="panel err">${esc(e.message)}</div>`;
    $("#live-run").disabled = false;
    return;
  }
  $("#live-run").disabled = false;
  S.data = data;
  if (S.mode === "live") {
    $("#question").innerHTML = `<div class="eyebrow">使用者的問題（即時）</div><div class="q">${esc(data.question)}</div>`;
  }
  renderBaseline(data.baseline);
  renderDecision(data.decision);
  renderPipeline(data, revealFrom);
}

function renderBaseline(b) {
  const src = { illustration: "示意：常見錯誤類型，非實測輸出", measured: "實測紀錄", live: "Gemini 2.5 Flash 直接作答（無 harness）" }[b.kind];
  const txt = esc(b.text).replace(/\*\*(.+?)\*\*/g, "<b>$1</b>");
  $("#baseline").innerHTML = `<div class="txt">${txt}</div><div class="src">${esc(src)}</div>`;
}

function renderDecision(d) {
  const [label, cls] = ACTION[d.action];
  $("#decision").innerHTML = `<span class="chip s-${cls}">${label}</span><div class="head">${esc(d.headline)}</div><div class="txt">${esc(d.text)}</div>`;
}

function renderPipeline(data, revealFrom) {
  const root = $("#pipeline");
  root.innerHTML = "";
  data.steps.forEach((s, i) => {
    const el = document.createElement("article");
    el.className = `step ${s.status === "waiting" ? "waiting collapsed" : ""}`;
    el.id = "step-" + s.id;
    el.innerHTML = `
      <div class="step-head">
        <div class="num s-${s.status}">${s.n}</div>
        <div>
          <div class="step-title">${esc(s.title)}</div>
          <div class="step-sum">${esc(s.summary)}</div>
          <div class="papers">${s.concepts.map(conceptBtn).join("")}</div>
        </div>
        <span class="chip s-${s.status}">${STATUS[s.status]}</span>
      </div>
      <div class="step-body"></div>`;
    if (s.status !== "waiting") $(".step-body", el).appendChild(renderBody(s, data));
    $(".step-head", el).addEventListener("click", (e) => { if (!e.target.closest(".paper")) el.classList.toggle("collapsed"); });
    root.appendChild(el);
    const dot = $("#dot-" + s.id);
    dot.className = "dot s-" + s.status;
    const delay = i < revealFrom ? 0 : (i - revealFrom) * 180;
    setTimeout(() => el.classList.add("shown"), delay);
  });
  root.querySelectorAll(".concept").forEach((b) => b.addEventListener("click", (e) => showConcept(b.dataset.cid, e)));
  if (data.pending && revealFrom > 0) {
    const target = data.steps.find((s) => s.status === "ask" || s.status === "fail");
    if (target) setTimeout(() => document.getElementById("step-" + target.id).scrollIntoView({ behavior: "smooth", block: "center" }),
      (data.steps.indexOf(target) - revealFrom) * 180 + 200);
  }
}

const CHECKED = { "1029": ["10/29 查證", "pass"], bib: ["ICAIF bib 已查證", "pass"], memo: ["先前讀過原文", "info"], none: ["未查證", "fail"] };
const stepNo = (sid) => Object.keys(S.meta.steps).indexOf(sid) + 1;
const concept = (id) => S.meta.concepts.find((c) => c.id === id);

function conceptBtn(id) {
  const c = concept(id);
  return `<button class="paper concept ${c.own ? "own" : ""}" data-cid="${id}">${esc(c.name)}</button>`;
}

function paperLinks(p) {
  return p.links.map(([label, url]) => `<a href="${esc(url)}" target="_blank" rel="noopener">${esc(label)}</a>`).join(" · ");
}

function paperBadges(p) {
  const [t, cls] = CHECKED[p.checked];
  return `${p.read ? '<span class="chip s-pass">讀過</span>' : '<span class="chip s-ask">建議補讀</span>'} <span class="chip s-${cls}">${t}</span>`;
}

function linkBlock(l) {
  const p = S.meta.papers[l.paper];
  return `<div class="plink"><div class="ph"><b>${esc(p.short)}</b> <span class="sub">${esc(p.cite)}</span></div>
    <div><span class="lbl">論文怎麼說</span>${esc(l.says)}</div>
    <div><span class="lbl">差別</span>${esc(l.gap)}</div>
    <div class="sub">${paperBadges(p)} ${paperLinks(p)}</div></div>`;
}

function showConcept(id, ev) {
  ev.stopPropagation();
  const c = concept(id);
  const pop = $("#pop");
  pop.innerHTML = `<h3>${esc(c.name)} ${c.own ? '<span class="chip s-info">本系統的設計</span>' : ""}</h3>
    <div class="cite">${esc(c.where)}</div><div class="here">${esc(c.here)}</div>${c.links.map(linkBlock).join("")}
    <div class="sub"><a href="#concept-${c.id}" data-jump="${c.id}">在對照表中查看</a></div>`;
  pop.classList.remove("hidden");
  const jump = pop.querySelector("[data-jump]");
  jump.onclick = (e) => { e.preventDefault(); pop.classList.add("hidden"); flashRow(c.id); };
  const r = ev.target.getBoundingClientRect();
  pop.style.left = Math.max(8, Math.min(r.left, window.innerWidth - pop.offsetWidth - 8)) + "px";
  const below = r.bottom + 8;
  pop.style.top = (below + pop.offsetHeight > window.innerHeight ? Math.max(8, r.top - pop.offsetHeight - 8) : below) + "px";
}

function flashRow(id) {
  const row = document.getElementById("concept-" + id);
  row.scrollIntoView({ behavior: "smooth", block: "center" });
  row.classList.add("flash");
  setTimeout(() => row.classList.remove("flash"), 1600);
}

function renderLit() {
  const rows = S.meta.concepts.map((c) => `<tr id="concept-${c.id}">
      <td><b>${esc(c.name)}</b>${c.own ? '<div><span class="chip s-info">本系統的設計</span></div>' : ""}<div class="sub">步驟 ${c.steps.map(stepNo).join("、")}</div></td>
      <td>${esc(c.here)}<div class="sub mono">${esc(c.where)}</div></td>
      <td>${c.links.map(linkBlock).join("")}</td></tr>`).join("");
  $("#lit-table").innerHTML = `<thead><tr><th>概念</th><th>harness 怎麼做（程式位置）</th><th>真實討論這個概念的論文</th></tr></thead><tbody>${rows}</tbody>`;
  const used = (k) => S.meta.concepts.filter((c) => c.links.some((l) => l.paper === k)).map((c) => c.name);
  $("#paper-table").innerHTML = `<thead><tr><th>論文</th><th>出處</th><th>論文本身的主張</th><th>用在哪些概念</th><th>連結</th></tr></thead><tbody>${
    Object.entries(S.meta.papers).map(([k, p]) => `<tr><td><b>${esc(p.short)}</b><div>${paperBadges(p)}</div></td><td class="sub">${esc(p.cite)}</td>
      <td>${esc(p.said)}</td><td class="sub">${used(k).map(esc).join("、")}</td><td class="links">${paperLinks(p) || '<span class="na">—</span>'}</td></tr>`).join("")}</tbody>`;
}

// ------------------------------------------------------------ step bodies

function renderBody(s, data) {
  const div = document.createElement("div");
  const fn = BODY[s.id];
  div.innerHTML = fn ? fn(s.body, s, data) : "";
  if (s.id === "ambiguity" || s.id === "intent" || s.id === "precheck") wireForms(div, data);
  if (s.id === "trace") wireTrace(div);
  return div;
}

function codeBlock(code, hl = []) {
  return `<pre class="code">${code.replace(/\n$/, "").split("\n").map((line, i) => {
    const html = esc(line).replace(/(#\s*@spec.*)$/, '<span class="tag-spec">$1</span>');
    return `<span class="${hl.includes(i + 1) ? "hl" : ""}" data-ln="${i + 1}"><span class="ln">${i + 1}</span>${html || " "}</span>`;
  }).join("")}</pre>`;
}

const BODY = {
  scope(b) {
    // PoC replays come from full-text search over the 250-card library: rows carry card / id, no keyword hits
    const poc = b.table.length && b.table[0].card !== undefined;
    const rows = b.table.map((r) => poc
      ? `<tr class="${r.id === b.card ? "" : "lost"}"><td>${esc(r.card)}</td><td class="mono">${esc(r.id)}${r.id === b.card ? " ← 選中" : ""}</td><td class="mono">${r.score}</td></tr>`
      : `<tr><td>${esc(r.title)}</td><td>${r.hits.map(esc).join("、") || '<span class="na">—</span>'}</td><td class="mono">${r.score}</td></tr>`).join("");
    const head = poc ? "<th>卡片</th><th>卡片編號</th><th>檢索名次分</th>" : "<th>卡片</th><th>命中關鍵字</th><th>分數</th>";
    return `<table><thead><tr>${head}</tr></thead><tbody>${rows}</tbody></table>
      <div class="note">方法：${esc(b.how)}。${poc ? "卡片庫是論文的 250 張卡，卡片內容由 LLM 自動產生。" : "分數不足或兩張卡同分就不硬選，直接拒答。這是 VerifiQuant N 閘的簡化版。"}</div>`;
  },

  intent(b, s, data) {
    const label = { grounded: ["有出處", "pass"], user: ["使用者補充", "info"], missing: ["原文沒有", "ask"],
      ungrounded: ["無出處，不採用", "fail"], mismatch: ["數值不符，不採用", "fail"], bad_user: ["無法解析", "fail"],
      default: ["沿用預設值", "info"] };
    const show = (v) => (v !== null && typeof v === "object" ? JSON.stringify(v) : v ?? "—");
    const quote = (q) => (Array.isArray(q) ? q.map((x) => "「" + esc(x) + "」").join(" ") : q ? "「" + esc(q) + "」" : '<span class="na">—</span>');
    const sel = b.select ? `<div class="block-title">選取路徑（卡片輸出是一個結構，題目要其中一格）</div>
      <div class="note">結構：${esc(b.select.structure)}<br>LLM 宣稱的路徑 <code>${esc(JSON.stringify(b.select.claimed))}</code>，原文引用 ${quote(b.select.quote)}
      → <span class="chip s-${b.select.status === "grounded" ? "pass" : "fail"}">${b.select.status === "grounded" ? "每一層都由引用文字決定" : "不採用"}</span> ${esc(b.select.note)}</div>` : "";
    if (!b.slots) {
      if (b.asked !== undefined) return `<div class="note">題目問的量：「${esc(b.asked)}」。卡片算的是「${esc(b.card_output && b.card_output.label)}」。兩者不同，卡片不適用。</div>`;
      return sel;
    }
    const rows = b.slots.map((r) => {
      const [t, c] = label[r.status];
      return `<tr><td>${esc(r.label)}<div class="sub mono">${r.name}</div></td>
        <td class="mono">${esc(show(r.claimed))}</td><td>${quote(r.quote)}</td>
        <td class="mono">${esc(show(r.parsed))}</td><td><span class="chip s-${c}">${t}</span><div class="sub">${esc(r.note)}</div></td></tr>`;
    }).join("");
    const conv = b.conventions.map((c) => `<li><b>${esc(c.label)}</b>：${c.value ? `<span class="chip s-${c.status === "user" ? "info" : "pass"}">${esc(c.value)}</span>` : '<span class="chip s-ask">未決定</span>'} <span class="sub">${esc(c.note)}${c.quote ? "（原文：「" + esc(c.quote) + "」）" : ""}</span></li>`).join("");
    const fixed = b.fixed.map((f) => `<li>${esc(f.text)}</li>`).join("");
    let html = `<div class="table-wrap"><table><thead><tr><th>欄位</th><th>LLM 宣稱</th><th>原文引用</th><th>重新解析</th><th>判定</th></tr></thead><tbody>${rows}</tbody></table></div>
      <div class="note callout">LLM 只負責「指出原文哪幾個字是這個欄位」。數值由程式從引用文字重新解析；引用不在原文裡、或解析結果對不上，一律不採用。</div>`;
    html += sel;
    if (conv) html += `<div class="block-title">約定（convention）</div><ul>${conv}</ul>`;
    if (fixed) html += `<div class="block-title">卡片的固定假設（程式會比問題多做的部分，先攤給使用者看）</div><ul>${fixed}</ul>`;
    html += `<div class="block-title">凍結後的函式簽名</div><code>${esc(b.signature)}</code>`;
    if (data.pending && data.pending.type === "slots" && s.status === "ask") html += slotForm(data.pending);
    return html;
  },

  contract(b) {
    const groups = ["pre", "post", "meta", "boundary"].map((k) => {
      const items = b.invariants.filter((i) => i.kind === k);
      if (!items.length) return "";
      return `<div class="block-title">${KIND[k]}</div><div class="inv-group">${items.map((i) =>
        `<div class="inv"><span class="k k-${k}">${KIND[k]}${i.when ? "（" + Object.values(i.when).join("") + "）" : ""} · ${SOURCE[i.source] || i.source}</span><div>${esc(i.text)}</div><code>${esc(i.check)}</code></div>`).join("")}</div>`;
    }).join("");
    return groups + `<div class="note callout"><b>為什麼不需要標準答案：</b>${esc(b.note)}</div>
      <div class="note">這些是可執行的數值檢查，不是 SMT 規格。避開 Logic-LM / SATLM 的翻譯瓶頸，也不像 VERAFI 只把規則塞進 prompt。</div>`;
  },

  precheck(b, s, data) {
    let html = `<table><thead><tr><th></th><th>前置條件</th><th>檢查式</th></tr></thead><tbody>${b.rows.map((r) =>
      `<tr><td class="${r.ok ? "ok" : "bad"}">${r.ok ? "✓" : "✗"}</td><td>${esc(r.text)}</td><td><code>${esc(r.check)}</code></td></tr>`).join("")}</tbody></table>`;
    if (b.inputs) html += `<div class="note">綁定後的輸入：<code>${esc(JSON.stringify(b.inputs))}</code></div>`;
    if (data.pending && data.pending.type === "slots" && s.status === "fail") html += slotForm(data.pending);
    return html;
  },

  generate(b) {
    const clusters = b.clusters.map((c, i) => `<div class="cluster">答案 ${i + 1}：<b>${fmtVal(c.value, b.unit)}</b> <span class="sub">← ${c.members.join(", ")}</span></div>`).join("");
    const cands = b.candidates.map((c) => `<div class="cand"><div class="cand-h"><span class="id">候選 ${c.id}</span>
      <span class="sub">宣稱 ${esc(JSON.stringify(c.declared))}</span><span class="mono">${c.error ? '<span class="bad">' + esc(c.error) + "</span>" : fmtVal(c.result, b.unit)}</span></div>${codeBlock(c.code)}</div>`).join("");
    return `<div class="block-title">在題目輸入上執行的結果，依答案分群（CodeT / ClarifyGPT 的做法）</div><div class="clusters">${clusters}</div>
      <div class="note">分群只告訴我們「有分歧」，不告訴我們誰對。分歧可能來自 bug，也可能來自讀法不同。下一步用不變量把兩者分開。</div>
      <div class="block-title">候選程式（每份都必須宣告 ASSUMPTIONS，並用 @spec 標記實作了哪個規格項目）</div><div class="cands">${cands}</div>`;
  },

  verify(b) {
    const invs = b.invariants;
    const head = invs.map((inv, i) => `<th class="c" title="${esc(inv.text)}"><span class="k-${inv.kind}">I${i + 1}</span></th>`).join("");
    const rows = b.rows.map((r) => {
      const cells = invs.map((inv) => {
        if (!r.ok) return `<td class="c na">·</td>`;
        const c = r.checks[inv.id];
        if (inv.kind === "boundary") {
          const claimed = Object.entries(inv.when).every(([k, v]) => r.declared[k] === v);
          if (!claimed) return `<td class="c na" title="不適用：程式沒有宣稱這個約定（但仍拿來推斷實際行為）">${c.ok ? "◦" : "·"}</td>`;
        }
        return `<td class="c ${c.ok ? "ok" : "bad"}" title="${esc(inv.text)}${c.error ? "\n" + esc(c.error) : ""}">${c.ok ? "✓" : "✗"}</td>`;
      }).join("");
      const conv = b.conventions.map((cv) => {
        const d = r.declared[cv.name] ?? "—";
        const act = (r.behaves_as[cv.name] || []).join("/") || "都不是";
        return `${esc(cv.label)}：宣稱 <b>${esc(d)}</b>，行為 <b>${esc(act)}</b>`;
      }).join("<br>");
      const verdict = !r.ok ? `<span class="chip s-fail">執行失敗</span><div class="sub">${esc(r.error)}</div>`
        : r.passed ? `<span class="chip s-pass">通過</span>` : `<span class="chip s-fail">淘汰</span><div class="sub">${r.failed.map(failName(invs)).join("、")}</div>`;
      const bug = r.bug ? `<div class="sub">成因（人工標註）：${esc(r.bug)}</div>` : "";
      return `<tr class="${r.passed ? "" : "lost"}"><td><b>${r.id}</b></td><td class="mono">${fmtVal(r.result, b.unit)}</td>${cells}
        ${b.conventions.length ? `<td class="sub conv">${conv}</td>` : ""}<td class="verdict">${verdict}${bug}</td></tr>`;
    }).join("");
    const legend = invs.map((inv, i) => `<div><b class="k-${inv.kind}">I${i + 1}</b> ${esc(inv.text)}</div>`).join("");
    const probe = b.element_probe ? `<div class="note callout"><b>元素探針：</b>把題目要的那一格改動 0.1%，${b.element_probe.caught_by.length
      ? `有 ${b.element_probe.caught_by.length} 條不變量發現（${b.element_probe.caught_by.map(failName(invs)).join("、")}），卡片管得到這一格。`
      : "沒有任何不變量發現：卡片沒有約束題目要的值，拒答。"}</div>` : "";
    return `<div class="table-wrap"><table class="matrix"><thead><tr><th>候選</th><th>答案</th>${head}${b.conventions.length ? "<th>約定：宣稱 vs 行為</th>" : ""}<th>判定</th></tr></thead><tbody>${rows}</tbody></table></div>
      <div class="legend">${legend}</div>${probe}
      <div class="note callout">規則不是寫在 prompt 裡請模型「自我檢查」（VERAFI 的做法），而是在模型外面真的執行。${b.conventions.length ? "約定釘子（橘色）只檢查程式宣稱的那個約定；◦ 表示程式沒宣稱、但行為符合。程式的自我說明不被信任，宣稱和行為對不上就淘汰。" : ""}</div>`;
  },

  ambiguity(b, s, data) {
    let html = "";
    if (b.rejected.length) html += `<div class="note">已排除的 bug：候選 ${b.rejected.join(", ")}。它們造成的分歧不算歧義，不拿去問使用者。</div>`;
    b.conventions.forEach((c) => {
      html += `<div class="block-title">${esc(c.label)}：${esc(c.question)}</div><div class="options">${c.previews.map((p) =>
        `<label class="opt"><input type="radio" name="conv-${c.name}" value="${p.value}" ${c.chosen === p.value ? "checked" : ""} ${c.chosen ? "disabled" : ""}>
          <b>${esc(p.label)}</b><div class="v">${fmtVal(p.result, b.unit)}</div><div class="sub">通過驗證的候選：${p.members.join(", ") || "無"}</div></label>`).join("")}</div>`;
      if (c.relation) html += `<div class="sub">兩種讀法的關係檢查：${esc(c.relation)} → ${c.relation_ok === null ? "—" : c.relation_ok ? '<span class="ok">成立</span>' : '<span class="bad">不成立</span>'}（確認差異只來自約定，不是其中一份另有 bug）</div>`;
    });
    if (!b.conventions.length) html += `<div class="clusters">${b.clusters.map((c) => `<div class="cluster">通過驗證的答案：<b>${fmtVal(c.value, b.unit)}</b> <span class="sub">← ${c.members.join(", ")}</span></div>`).join("")}</div>`;
    if (data.pending && data.pending.type === "convention" && s.status === "ask") {
      html += `<div class="ask-form"><b>請使用者決定。</b> 選好後按下方按鈕，harness 從頭重跑。
        <div class="note">就算所有候選剛好都猜同一種讀法，卡片列出的約定仍會被問（ClarifyGPT 只看分歧，抓不到「大家一起猜錯」）。</div>
        <button class="primary" data-submit="conv">確認讀法並繼續</button></div>`;
    }
    return html;
  },

  inspect(b) {
    return `<div class="two"><div><div class="block-title">檢查員看得到</div><ul>${b.sees.map((x) => `<li>${esc(x)}</li>`).join("")}</ul></div>
      <div><div class="block-title">檢查員看不到</div><ul>${b.hidden.map((x) => `<li>${esc(x)}</li>`).join("")}</ul></div></div>
      <div class="vs"><div>候選 ${b.candidate}：<b>${fmtVal(b.candidate_result, b.unit)}</b></div><div>檢查員：<b>${fmtVal(b.result, b.unit)}</b></div></div>
      <div class="block-title">檢查員的程式（不同方法）</div>${codeBlock(b.code)}`;
  },

  trace(b) {
    const items = b.items.map((it) => `<div class="trace-item ${it.lines.length || it.by ? "" : "miss"}" data-lines="${it.lines.join(",")}">
      <span class="sub">${esc(it.kind)}</span> ${esc(it.label)}${it.value !== null && it.value !== undefined ? ` = <code>${esc(it.value)}</code>` : ""}
      <div class="lines">${it.lines.length ? "第 " + it.lines.join(", ") + " 行" : it.by ? "由 harness 執行，不在程式內" : "沒有任何程式行實作"}</div></div>`).join("");
    return `<div class="trace"><div>${items}<div class="sub">滑過項目可在右側看到對應的程式行。</div></div><div><div class="block-title">採用的程式：候選 ${b.candidate}</div>${codeBlock(b.code)}</div></div>
      <div class="note">標記是程式自己宣稱的。數值相關的宣稱已在步驟 6 被行為驗證過，這一步只是讓人能沿著連結逐項審查。</div>`;
  },

  decision(b, s, data) {
    return `<div class="block-title">稽核紀錄（只記錄被執行過的東西，不記模型的自述）</div><pre class="audit">${esc(JSON.stringify(auditView(data.audit), null, 2))}</pre>`;
  },
};

function auditView(a) {
  const out = { ...a };
  out.llm_calls = (a.llm_calls || []).map((c) => ({ role: c.role, source: c.source }));
  return out;
}

function failName(invs) {
  return (id) => {
    if (id.startsWith("claim:")) return "約定宣稱與行為不符";
    if (id.startsWith("declare:")) return "沒有宣告約定";
    const i = invs.findIndex((x) => x.id === id);
    return i >= 0 ? "I" + (i + 1) : id;
  };
}

function slotForm(p) {
  return `<div class="ask-form"><b>請使用者補充（harness 不會替你猜）</b>${p.fields.map((f) =>
    `<label class="f">${esc(f.label)}（${esc(f.unit)}）：<input type="text" data-slot="${f.name}" placeholder="${f.unit === "小數" ? "例如 8%" : ""}"> <span class="sub">${esc(f.note)}</span></label>`).join("")}
    <button class="primary" data-submit="slots">補上並繼續</button></div>`;
}

function wireForms(div, data) {
  const btn = div.querySelector("[data-submit]");
  if (!btn) return;
  btn.addEventListener("click", () => {
    const from = data.steps.findIndex((s) => s.status === "ask" || s.status === "fail");
    if (btn.dataset.submit === "slots") {
      const slots = { ...(S.answers.slots || {}) };
      div.querySelectorAll("[data-slot]").forEach((inp) => { if (inp.value.trim()) slots[inp.dataset.slot] = inp.value.trim(); });
      S.answers = { ...S.answers, slots };
    } else {
      const conventions = { ...(S.answers.conventions || {}) };
      div.querySelectorAll("input[type=radio]:checked").forEach((r) => { conventions[r.name.replace("conv-", "")] = r.value; });
      if (!Object.keys(conventions).length) return;
      S.answers = { ...S.answers, conventions };
    }
    run(Math.max(0, from));
  });
}

function wireTrace(div) {
  const pre = div.querySelector(".trace pre.code");
  div.querySelectorAll(".trace-item").forEach((it) => {
    const lines = it.dataset.lines ? it.dataset.lines.split(",").map(Number) : [];
    it.addEventListener("mouseenter", () => pre.querySelectorAll("[data-ln]").forEach((l) => l.classList.toggle("hl", lines.includes(+l.dataset.ln))));
    it.addEventListener("mouseleave", () => pre.querySelectorAll(".hl").forEach((l) => l.classList.remove("hl")));
  });
}

init();
