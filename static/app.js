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
    const stepsText = data.steps
      .map(
        (step, idx) =>
          `${idx + 1}. ${step.variable} = ${step.formula} = ${step.value.toFixed(
            4
          )}`
      )
      .join("\n");
    resultEl.textContent = [
      `狀態：成功`,
      `使用卡片：${data.card_id}`,
      `理由：${data.selection_reason}`,
      `輸入值：${JSON.stringify(data.inputs, null, 2)}`,
      `步驟：\n${stepsText}`,
      `輸出：${data.output_var} = ${data.output_value}`,
    ].join("\n\n");
  } else if (data.status === "refused") {
    resultEl.textContent = [
      `狀態：拒絕`,
      `原因：${data.reason}`,
      data.missing_inputs
        ? `缺少欄位：${data.missing_inputs.join(", ")}`
        : "",
    ]
      .filter(Boolean)
      .join("\n");
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

