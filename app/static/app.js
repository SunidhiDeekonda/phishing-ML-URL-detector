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
    <p class="muted">Canonical URL analysed: ${payload.url}</p>
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

(function mountResearchExtensions() {
  function mount() {
    const host = document.querySelector("main") || document.body;
    if (document.getElementById("research-extensions")) return;
    const section = document.createElement("section");
    section.id = "research-extensions";
    section.className = "panel research-extensions";
    section.innerHTML = `<details><summary>Research Extensions</summary>
      <p class="research-note">Optional context is analysed locally and is not mixed into the validated URL score.</p>
      <label for="context-html">Supplied HTML (optional)</label><textarea id="context-html" rows="3" placeholder="Paste HTML for local signal extraction"></textarea>
      <label for="context-email">Supplied email text (optional)</label><textarea id="context-email" rows="3" placeholder="Paste email text without sensitive information"></textarea>
      <div class="research-actions"><button type="button" id="analyse-context">Analyse Context</button><button type="button" id="submit-correction" class="secondary">Report Prediction as Incorrect</button></div>
      <pre id="context-output" aria-live="polite"></pre></details>`;
    host.appendChild(section);
    const output = section.querySelector("#context-output");
    section.querySelector("#analyse-context").addEventListener("click", async () => {
      const url = urlInput.value.trim(); if (!url) { output.textContent = "Enter a URL first."; return; }
      output.textContent = "Analysing supplied context...";
      const response = await fetch("/predict-context", {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({url,html:section.querySelector("#context-html").value||null,email_text:section.querySelector("#context-email").value||null})});
      const data = await response.json(); output.textContent = response.ok ? JSON.stringify(data.context_signals,null,2) : (data.detail||"Context analysis failed.");
    });
    section.querySelector("#submit-correction").addEventListener("click", async () => {
      const url=urlInput.value.trim(); if (!url) { output.textContent="Enter and analyse a URL first."; return; }
      const predicted=document.body.textContent.includes("PHISHING")?"PHISHING":"LEGITIMATE"; const correct=predicted==="PHISHING"?"LEGITIMATE":"PHISHING";
      const response=await fetch("/feedback",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({url,predicted_label:predicted,correct_label:correct,optional_notes:"Submitted from demo UI"})});
      const data=await response.json(); output.textContent=response.ok?"Feedback quarantined for human verification. No automatic retraining occurred.":(data.detail||"Feedback was not accepted.");
    });
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", mount); else mount();
})();
