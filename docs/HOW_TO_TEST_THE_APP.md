# How to Test the Phishing Detection App

The application contains three independent tools. HTML and email analysis do not require a URL.

## TEST URL

1. Open the deployed application.
2. In **URL Phishing Detector**, paste `https://www.google.com` and select **ANALYSE URL**.
3. Expected: `LEGITIMATE`; the canonical URL is `https://www.google.com`.
4. Repeat with `https://www.google.com/`. The canonical URL, verdict, and probability should match.
5. Paste `http://secure-account-login-example.xyz/verify` as text. Do not visit it.
6. Expected: `PHISHING`.

The URL tool uses the validated Character-Level CNN + LightGBM ensemble. It does not visit or download the URL.

## TEST HTML

HTML analysis works while the URL and email fields are empty.

### Safe HTML

```html
<html>
<body>
<h1>Welcome</h1>
<p>This is a documentation page.</p>
</body>
</html>
```

1. Paste it into **HTML Phishing Context Analyzer**.
2. Select **ANALYSE HTML**.
3. Expected: no password field, no credential terms, and low contextual risk.

### Suspicious-style HTML

```html
<html>
<body>
<form action="/verify">
<input type="text" name="username">
<input type="password" name="password">
<button>Verify Account</button>
</form>
</body>
</html>
```

1. Paste it into the HTML box.
2. Select **ANALYSE HTML**.
3. Expected: one form, one password field, credential-related terms, and elevated contextual risk.

HTML is parsed as inert text. The application does not open a website, execute HTML/JavaScript, or make a network request. The result is contextual evidence, not an ML probability.

## TEST EMAIL

Email analysis works while the URL and HTML fields are empty.

### Safe email

```text
Hi team,

The project meeting is tomorrow at 10 AM.
Please bring the final report.

Thanks.
```

1. Paste it into **Email Phishing Context Analyzer**.
2. Select **ANALYSE EMAIL**.
3. Expected: zero or fewer risk/credential indicators and low contextual risk.

### Suspicious synthetic email

```text
URGENT: Your account will be suspended.

Verify your login immediately and confirm your password to prevent account closure.
```

1. Paste it into the email box.
2. Select **ANALYSE EMAIL**.
3. Expected: multiple implemented risk terms, one credential-request phrase, and elevated contextual risk.

The analyzer does not connect to Gmail, access an inbox, or send email. It counts only implemented risk terms, credential-request patterns, exact call-to-action phrases, and URL strings in manually pasted text.

## WHAT TO EXPLAIN TO A PROFESSOR

- URL detection is the trained and evaluated CNN + LightGBM classifier.
- HTML and email are independent deterministic research extensions that provide understandable context evidence.
- HTML/email signals are not fused into URL probability because the project has no properly labelled content dataset for calibrated multimodal training.
- No submitted URL is fetched, no HTML is executed, and no mailbox is accessed.
