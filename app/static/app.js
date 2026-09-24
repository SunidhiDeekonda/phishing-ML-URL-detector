const analyseBtn = document.getElementById("analyseBtn");
const urlInput = document.getElementById("urlInput");
const statusText = document.getElementById("statusText");
const resultCard = document.getElementById("resultCard");
const resultPanel = document.getElementById("resultPanel");
const breakdownEl = document.getElementById("breakdown");
const signalsEl = document.getElementById("signals");
const formError = document.getElementById("formError");
let latestPrediction = null;

function clearErrors() {
  formError.textContent = "";
  formError.classList.add("hidden");
}

function setBusy(state) {
  analyseBtn.disabled = state;
  analyseBtn.textContent = state ? "ANALYSING..." : "ANALYSE URL";
}

function setResult(payload) {
  latestPrediction = payload;
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

(function setupResearchExtensions() {
  const section = document.getElementById("research-extensions");
  if (!section) return;
  const htmlInput = document.getElementById("context-html");
  const emailInput = document.getElementById("context-email");
  const output = document.getElementById("context-output");
  const benignHtml = "<html>\n<body>\n<h1>Welcome</h1>\n<p>Documentation page.</p>\n</body>\n</html>";
  const suspiciousHtml = "<html>\n<body>\n<form action=\"/verify\">\n<input type=\"text\" name=\"username\">\n<input type=\"password\" name=\"password\">\n<button>Verify Account</button>\n</form>\n</body>\n</html>";
  const benignEmail = "Hi team,\nThe project meeting is tomorrow at 10 AM.\nPlease bring the final report.";
  const suspiciousEmail = "URGENT: Your account will be suspended.\nVerify your login immediately and confirm your password.";

  document.getElementById("load-safe-context").addEventListener("click", () => {
    htmlInput.value = benignHtml;
    emailInput.value = benignEmail;
    output.textContent = "Safe example loaded. Select Analyse Optional Context.";
  });
  document.getElementById("load-suspicious-context").addEventListener("click", () => {
    htmlInput.value = suspiciousHtml;
    emailInput.value = suspiciousEmail;
    output.textContent = "Synthetic suspicious-style example loaded. Select Analyse Optional Context.";
  });
  document.getElementById("analyse-context").addEventListener("click", async () => {
    const url = urlInput.value.trim();
    if (!url) { output.textContent = "Enter a URL in the main URL box first."; return; }
    const html = htmlInput.value.trim();
    const emailText = emailInput.value.trim();
    if (!html && !emailText) { output.textContent = "No optional context supplied."; return; }
    output.textContent = "Analysing supplied text locally...";
    const response = await fetch("/predict-context", {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({url,html:html||null,email_text:emailText||null})});
    const data = await response.json();
    if (!response.ok) { output.textContent = data.detail || "Context analysis failed."; return; }
    const signals = data.context_signals;
    const flags = signals.context_risk_flags.length ? signals.context_risk_flags.map(flag => `- ${flag}`).join("\n") : "- No context risk flags detected.";
    output.textContent = `${signals.message}\n\nHTML signals\n- Forms: ${signals.html.form_count}\n- Password inputs: ${signals.html.password_input_count}\n- Credential terms: ${signals.html.credential_term_count}\n- External targets: ${signals.html.external_target_count}\n\nEmail signals\n- URLs: ${signals.email.url_count}\n- Risk terms: ${signals.email.risk_term_count}\n- Credential requests: ${signals.email.credential_request_count}\n- Calls to action: ${signals.email.call_to_action_count}\n\nRisk flags\n${flags}\n\nThese signals do not change the validated URL probability.`;
  });
  document.getElementById("submit-correction").addEventListener("click", async () => {
    if (!latestPrediction) { output.textContent = "Run a URL prediction before submitting feedback."; return; }
    const predicted = latestPrediction.verdict;
    const correct = predicted === "PHISHING" ? "LEGITIMATE" : "PHISHING";
    const response = await fetch("/feedback",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({url:latestPrediction.url,predicted_label:predicted,correct_label:correct,optional_notes:"Submitted from demo UI"})});
    const data = await response.json();
    output.textContent = response.ok ? "Feedback quarantined for human verification. No automatic retraining occurred." : (data.detail || "Feedback was not accepted.");
  });
})();
