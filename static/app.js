const form = document.getElementById("question-form");
const domainSelect = document.getElementById("domain");
const topicSelect = document.getElementById("topic");
const loader = document.getElementById("loader");
const resultEl = document.getElementById("result");

const topicOptions = window.TOPIC_OPTIONS || {};

function updateTopicOptions() {
  const domain = domainSelect.value;
  const topics = domain ? topicOptions[domain] || [] : [];
  topicSelect.innerHTML = '<option value="">全部主題</option>';
  topics.forEach((topic) => {
    const option = document.createElement("option");
    option.value = topic;
    option.textContent = topic;
    topicSelect.appendChild(option);
  });
}

domainSelect.addEventListener("change", () => {
  updateTopicOptions();
});

function setLoading(isLoading) {
  loader.classList.toggle("hidden", !isLoading);
  form.querySelector("button").disabled = isLoading;
}

function renderResult(data) {
  resultEl.classList.remove("hidden");
  if (data.status === "success") {
    const stepsList = data.steps
      .map(
        (step, idx) =>
          `<li><code>${idx + 1}. ${step.variable} = ${
            step.formula || "公式"
          } = ${Number(step.value).toFixed(4)}</code></li>`
      )
      .join("");
    const calcMode = data.is_fallback ? "LLM fallback" : "SymPy";
    const modeClass = data.is_fallback ? "badge-llm" : "badge-sympy";
    resultEl.innerHTML = `
      <div class="meta-row">
        <span>狀態：<strong>成功</strong></span>
        <span>計算模式：<span class="badge ${modeClass}">${calcMode}</span></span>
      </div>
      <p>使用卡片：<code>${data.card_id}</code></p>
      <p>LLM 選擇理由：${data.selection_reason}</p>
      <p>輸入值：</p>
      <pre>${JSON.stringify(data.inputs, null, 2)}</pre>
      <p>計算步驟：</p>
      <ol class="steps-list">${stepsList}</ol>
      <p class="result-output">${data.output_var} = <strong>${data.output_value}</strong></p>
    `;
  } else if (data.status === "refused") {
    const missing = data.missing_inputs?.length
      ? `<p>缺少欄位：${data.missing_inputs.join(", ")}</p>`
      : "";
    resultEl.innerHTML = `
      <p>狀態：<strong>拒絕</strong></p>
      <p>原因：${data.reason}</p>
      ${missing}
    `;
  } else {
    resultEl.textContent = data.message || "系統發生未知錯誤";
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  resultEl.classList.add("hidden");
  resultEl.textContent = "";
  setLoading(true);
  try {
    const payload = {
      question: document.getElementById("question").value,
      domain: domainSelect.value || null,
      topic: topicSelect.value || null,
    };
    const response = await fetch("/api/solve", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    renderResult(data);
  } catch (error) {
    renderResult({ status: "error", message: error.message });
  } finally {
    setLoading(false);
  }
});

updateTopicOptions();

