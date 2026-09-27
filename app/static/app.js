const analyseBtn = document.getElementById("analyseBtn");
const urlInput = document.getElementById("urlInput");
const statusText = document.getElementById("statusText");
const resultCard = document.getElementById("resultCard");
const resultPanel = document.getElementById("resultPanel");
const breakdownEl = document.getElementById("breakdown");
const signalsEl = document.getElementById("signals");
const formError = document.getElementById("formError");
const urlFeedbackStatus = document.getElementById("urlFeedbackStatus");
const htmlInput = document.getElementById("htmlInput");
const htmlError = document.getElementById("htmlError");
const htmlResult = document.getElementById("htmlResult");
const analyseHtmlBtn = document.getElementById("analyseHtmlBtn");
const emailInput = document.getElementById("emailInput");
const emailError = document.getElementById("emailError");
const emailResult = document.getElementById("emailResult");
const analyseEmailBtn = document.getElementById("analyseEmailBtn");
let latestPrediction = null;

const safeHtml = "<html>\n<body>\n<h1>Welcome</h1>\n<p>This is a documentation page.</p>\n</body>\n</html>";
const suspiciousHtml = "<html>\n<body>\n<form action=\"/verify\">\n<input type=\"text\" name=\"username\">\n<input type=\"password\" name=\"password\">\n<button>Verify Account</button>\n</form>\n</body>\n</html>";
const safeEmail = "Hi team,\n\nThe project meeting is tomorrow at 10 AM.\nPlease bring the final report.\n\nThanks.";
const suspiciousEmail = "URGENT: Your account will be suspended.\n\nVerify your login immediately and confirm your password to prevent account closure.";

function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, (character) => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"})[character]);
}

async function responseError(response, payload, fallback) {
  if (response.ok) return "";
  if (typeof payload.detail === "string") return payload.detail;
  if (Array.isArray(payload.detail)) return payload.detail.map((item) => item.msg).join("; ");
  return fallback;
}

function setButtonBusy(button, busy, normalText) {
  button.disabled = busy;
  button.textContent = busy ? "ANALYSING..." : normalText;
}

function setUrlResult(payload) {
  latestPrediction = payload;
  statusText.classList.add("hidden");
  resultCard.className = payload.verdict === "PHISHING" ? "card phishing" : "card legit";
  const explanation = payload.verdict === "PHISHING" ? `<div class="prediction-explanation"><h4>Why might the model be suspicious?</h4>${payload.notable_signals.length ? `<ul>${payload.notable_signals.map((value) => `<li>${escapeHtml(value)}</li>`).join("")}</ul>` : "<p>No single notable engineered value explains this result.</p>"}<p class="small">${escapeHtml(payload.explanation_caveat)}</p></div>` : "";
  resultCard.innerHTML = `<div class="verdict">${payload.verdict}</div><p class="muted">Confidence: ${(payload.confidence * 100).toFixed(1)}%</p><p class="muted">Selected Ensemble: ${(payload.selected_ensemble_probability * 100).toFixed(1)}% phishing probability</p><p class="muted">Canonical URL analysed: ${escapeHtml(payload.url)}</p>${explanation}`;
  breakdownEl.replaceChildren();
  [["Char-CNN",payload.cnn_probability],["LightGBM",payload.lightgbm_probability],["Reference Ensemble (60/40)",payload.reference_ensemble_probability],[`Production Ensemble (${Math.round(payload.selected_weights.cnn * 100)}/${Math.round(payload.selected_weights.lightgbm * 100)})`,payload.selected_ensemble_probability]].forEach(([name,value]) => {
    const item = document.createElement("li"); item.textContent = `${name}: ${(value * 100).toFixed(1)}%`; breakdownEl.appendChild(item);
  });
  signalsEl.replaceChildren();
  Object.entries(payload.important_features || {}).forEach(([name,value]) => {
    const item = document.createElement("li");
    item.textContent = `${name}: ${typeof value === "boolean" ? (value ? "Yes" : "No") : (typeof value === "number" && !Number.isInteger(value) ? value.toFixed(4) : value)}`;
    signalsEl.appendChild(item);
  });
  resultPanel.scrollIntoView({behavior:"smooth",block:"start"});
}

async function analyseUrl() {
  formError.classList.add("hidden");
  const url = urlInput.value.trim();
  if (!url) { formError.textContent = "Please enter a URL."; formError.classList.remove("hidden"); return; }
  setButtonBusy(analyseBtn, true, "ANALYSE URL");
  statusText.textContent = "Running URL analysis...";
  resultCard.classList.add("hidden");
  try {
    const response = await fetch("/predict", {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({url})});
    const payload = await response.json();
    const error = await responseError(response, payload, "URL prediction failed.");
    if (error) throw new Error(error);
    setUrlResult(payload);
  } catch (error) {
    formError.textContent = error instanceof Error ? error.message : "URL prediction failed.";
    formError.classList.remove("hidden");
  } finally { setButtonBusy(analyseBtn, false, "ANALYSE URL"); }
}

function riskMarkup(contextRisk) {
  const risk = contextRisk && typeof contextRisk === "object" ? contextRisk : {};
  const values = Array.isArray(risk.indicators) ? risk.indicators : [];
  const indicators = values.length ? `<ul>${values.map((value) => `<li>${escapeHtml(value)}</li>`).join("")}</ul>` : "<p>No elevated warning combination detected.</p>";
  return `<div class="risk-summary ${risk.level === "ELEVATED" ? "risk-elevated" : "risk-low"}"><strong>Context Risk: ${escapeHtml(risk.level || "UNKNOWN")}</strong><p>${escapeHtml(risk.summary || "No summary returned.")}</p>${indicators}</div>`;
}

function signalMarkup(signals, fields) {
  const source = signals && typeof signals === "object" ? signals : {};
  return `<dl class="signal-list">${fields.map(([key, label, format]) => `<div><dt>${escapeHtml(label)}</dt><dd>${escapeHtml(format ? format(source[key] || 0) : source[key] || 0)}</dd></div>`).join("")}</dl>`;
}

async function analyseHtmlContext() {
  htmlError.classList.add("hidden");
  const html = htmlInput.value.trim();
  if (!html) { htmlError.textContent = "Paste HTML source or an HTML snippet before analysing."; htmlError.classList.remove("hidden"); return; }
  setButtonBusy(analyseHtmlBtn, true, "ANALYSE HTML");
  htmlResult.innerHTML = "<h3>HTML Context Analysis</h3><p>Parsing the supplied HTML text...</p>";
  try {
    const response = await fetch("/analyze-html", {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({html})});
    const payload = await response.json();
    const error = await responseError(response, payload, "HTML analysis failed.");
    if (error) throw new Error(error);
    const yesNoCount = (value) => value ? `YES (${value})` : "NO";
    const fields = [["form_count","Forms detected"],["password_input_count","Password fields"],["iframe_count","Iframes"],["script_count","Scripts"],["external_target_count","External targets",yesNoCount],["meta_refresh_count","Meta refresh",yesNoCount],["credential_term_count","Credential-related terms"]];
    htmlResult.innerHTML = `<h3>HTML Context Analysis</h3>${riskMarkup(payload.context_risk)}${signalMarkup(payload.signals, fields)}<p class="separation-note">Context evidence only. HTML was not executed and no network request was made.</p>`;
  } catch (error) {
    htmlError.textContent = error instanceof Error ? error.message : "HTML analysis failed.";
    htmlError.classList.remove("hidden");
  } finally { setButtonBusy(analyseHtmlBtn, false, "ANALYSE HTML"); }
}

async function analyseEmailContext() {
  emailError.classList.add("hidden");
  const emailText = emailInput.value.trim();
  if (!emailText) { emailError.textContent = "Paste email text before analysing."; emailError.classList.remove("hidden"); return; }
  setButtonBusy(analyseEmailBtn, true, "ANALYSE EMAIL");
  emailResult.innerHTML = "<h3>Email Context Analysis</h3><p>Inspecting the supplied email text...</p>";
  try {
    const response = await fetch("/analyze-email", {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({email_text:emailText})});
    const payload = await response.json();
    const error = await responseError(response, payload, "Email analysis failed.");
    if (error) throw new Error(error);
    const fields = [["url_count","URLs found"],["risk_term_count","Implemented risk terms"],["credential_request_count","Credential-request phrases"],["call_to_action_count","Exact calls to action"]];
    emailResult.innerHTML = `<h3>Email Context Analysis</h3>${riskMarkup(payload.context_risk)}${signalMarkup(payload.signals, fields)}<p class="separation-note">Context evidence only. No Gmail or mailbox access occurred.</p>`;
  } catch (error) {
    emailError.textContent = error instanceof Error ? error.message : "Email analysis failed.";
    emailError.classList.remove("hidden");
  } finally { setButtonBusy(analyseEmailBtn, false, "ANALYSE EMAIL"); }
}

analyseBtn.addEventListener("click", analyseUrl);
urlInput.addEventListener("keydown", (event) => { if (event.key === "Enter") { event.preventDefault(); analyseUrl(); } });
document.getElementById("loadSafeHtml").addEventListener("click", () => { htmlInput.value = safeHtml; htmlError.classList.add("hidden"); });
document.getElementById("loadSuspiciousHtml").addEventListener("click", () => { htmlInput.value = suspiciousHtml; htmlError.classList.add("hidden"); });
document.getElementById("loadSafeEmail").addEventListener("click", () => { emailInput.value = safeEmail; emailError.classList.add("hidden"); });
document.getElementById("loadSuspiciousEmail").addEventListener("click", () => { emailInput.value = suspiciousEmail; emailError.classList.add("hidden"); });
analyseHtmlBtn.addEventListener("click", analyseHtmlContext);
analyseEmailBtn.addEventListener("click", analyseEmailContext);
document.getElementById("submit-correction").addEventListener("click", async () => {
  if (!latestPrediction) { urlFeedbackStatus.textContent = "Run a URL prediction before submitting feedback."; return; }
  const predicted = latestPrediction.verdict;
  const correct = predicted === "PHISHING" ? "LEGITIMATE" : "PHISHING";
  try {
    const response = await fetch("/feedback", {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({url:latestPrediction.url,predicted_label:predicted,correct_label:correct,optional_notes:"Submitted from demo UI"})});
    const data = await response.json();
    urlFeedbackStatus.textContent = response.ok ? "Feedback quarantined for human verification. No automatic retraining occurred." : (data.detail || "Feedback was not accepted.");
  } catch (error) { urlFeedbackStatus.textContent = "Feedback could not be submitted."; }
});
