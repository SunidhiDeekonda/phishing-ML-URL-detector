# How to Test the Phishing Detection App

## Part 1 - Test normal URL prediction

Open the deployed application. Paste `https://www.google.com` into the main URL box and select **Analyse URL**. The result should be **LEGITIMATE**. The displayed canonical URL should be `https://www.google.com`.

## Part 2 - Test slash/no-slash behavior

First test `https://www.google.com`. Then test `https://www.google.com/`. Both should show the same canonical URL, verdict, and probability. Repeat with `https://github.com` and `https://github.com/`.

## Part 3 - Test known legitimate URLs

Try `https://www.wikipedia.org/`, `https://www.microsoft.com/`, and `https://openai.com/`. These are external regression examples, not an allowlist. A failure must be reported rather than hidden.

## Part 4 - Test a synthetic phishing string

Paste `http://secure-account-login-example.xyz/verify`. Do not open this string in a browser. Only submit the text to this application. The expected result is **PHISHING**.

## Part 5 - What the HTML box is

The HTML box analyses raw HTML text that you supply. It does not accept another URL and does not download a webpage.

## Part 6 - Exactly what to paste into HTML

Benign example:

```html
<html>
<body>
<h1>Welcome</h1>
<p>Documentation page.</p>
</body>
</html>
```

Synthetic suspicious-style example:

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

## Part 7 - Expected HTML output

The benign sample should have no password input and few or no credential terms. The suspicious-style sample should show one form, one password input, credential terms, and a credential-form risk flag.

## Part 8 - What the email box is

The email box analyses text pasted by you. It cannot open or read your mailbox. Never paste passwords or private information.

## Part 9 - Exactly what to paste into Email Text

Benign example: `Hi team, The project meeting is tomorrow at 10 AM. Please bring the final report.`

Synthetic suspicious-style example: `URGENT: Your account will be suspended. Verify your login immediately and confirm your password.`

## Part 10 - What context signals mean

Counts such as password inputs, credential terms, urgent language, and calls to action are supplementary warning signs. They are understandable indicators, not a second trained classifier.

## Part 11 - Why context does not change probability

The project has no properly labelled HTML/email training corpus. Mixing unvalidated rules into the calibrated URL score would make the reported accuracy misleading. The validated URL probability therefore remains separate.

## Part 12 - What feedback does

The **Report Prediction as Incorrect** button stores a pending correction containing the canonical URL and the opposite label. It does not store HTML/email content.

## Part 13 - Why feedback does not automatically retrain

Automatic retraining would allow malicious users to poison the model. Feedback requires human review, an explicit offline candidate build, and validation gates.

## Part 14 - What model-info means

`/model-info` lists registered model versions and validation status. It does not mean every registered candidate is deployed. The prediction response names the actual production artifact.

## Part 15 - Five-minute professor demonstration

1. Show Google without and with the root slash and compare identical results.
2. Show a scheme-less `www.google.com` canonicalized to HTTPS.
3. Submit the synthetic suspicious string as text and show the phishing verdict.
4. Load the safe context example and explain its low signal counts.
5. Load the suspicious-style example and explain the password/credential signals.
6. State that context is not fused into probability and no URL is fetched.
7. Open `/model-info` and explain verified feedback, drift monitoring, and model governance.
