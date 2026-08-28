const analyseBtn = document.getElementById("analyseBtn");
const urlInput = document.getElementById("urlInput");
const statusText = document.getElementById("statusText");
const resultCard = document.getElementById("resultCard");
const resultPanel = document.getElementById("resultPanel");
const breakdownEl = document.getElementById("breakdown");
const signalsEl = document.getElementById("signals");
const formError = document.getElementById("formError");

function clearErrors() {
  formError.textContent = "";
  formError.classList.add("hidden");
}

function setBusy(state) {
  analyseBtn.disabled = state;
  analyseBtn.textContent = state ? "ANALYSING..." : "ANALYSE URL";
}

function setResult(payload) {
  const phishingProbability = payload.selected_ensemble_probability;
  const confidence = payload.confidence;
  const verdict = payload.verdict;

  statusText.classList.add("hidden");
  resultCard.classList.remove("hidden");

  resultCard.className = verdict === "PHISHING" ? "card phishing" : "card legit";
  resultCard.innerHTML = `
    <div class="verdict">${verdict}</div>
    <p class="muted">Confidence: ${(confidence * 100).toFixed(1)}%</p>
    <p class="muted">Selected Ensemble: ${(phishingProbability * 100).toFixed(1)}% phishing probability</p>
    <p class="muted">Input URL: ${payload.url}</p>
  `;

  breakdownEl.innerHTML = "";
  const items = [
    ["Char-CNN", payload.cnn_probability],
    ["LightGBM", payload.lightgbm_probability],
    ["Reference Ensemble (60/40)", payload.reference_ensemble_probability],
    ["Selected Ensemble (95/5)", payload.selected_ensemble_probability],
  ];
  for (const [name, value] of items) {
    const li = document.createElement("li");
    li.textContent = `${name}: ${(value * 100).toFixed(1)}%`;
    breakdownEl.appendChild(li);
  }

  signalsEl.innerHTML = "";
  const orderedSignals = Object.entries(payload.important_features || {});
  for (const [name, value] of orderedSignals) {
    const li = document.createElement("li");
    if (typeof value === "boolean") {
      li.textContent = `${name}: ${value ? "Yes" : "No"}`;
    } else if (typeof value === "number") {
      if (Number.isInteger(value)) {
        li.textContent = `${name}: ${value}`;
      } else {
        li.textContent = `${name}: ${value.toFixed(4)}`;
      }
    } else {
      li.textContent = `${name}: ${value}`;
    }
    signalsEl.appendChild(li);
  }

  resultPanel.scrollIntoView({ behavior: "smooth", block: "start" });
}

async function analyse() {
  clearErrors();
  const url = urlInput.value.trim();
  if (!url) {
    formError.textContent = "Please enter a URL.";
    formError.classList.remove("hidden");
    return;
  }

  setBusy(true);
  statusText.textContent = "Running analysis...";
  resultCard.classList.add("hidden");

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
    });
    const payload = await response.json();
    if (!response.ok) {
      const detail = payload && payload.detail ? payload.detail : "Prediction failed.";
      throw new Error(detail);
    }
    setResult(payload);
  } catch (err) {
    formError.textContent = err instanceof Error ? err.message : "Prediction failed";
    formError.classList.remove("hidden");
  } finally {
    setBusy(false);
  }
}

analyseBtn.addEventListener("click", analyse);
urlInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    analyse();
  }
});
