# Pre-Reliability Audit

Audit date: 2026-09-24

- Branch: `main`
- Starting commit: `aaf8985`
- Remote: `https://github.com/SunidhiDeekonda/phishing-ML-URL-detector.git`
- Working tree before audit: clean.
- Baseline tests: 35 passed, 0 failed, with two dependency deprecation warnings.
- Production `/predict`: calibrated LightGBM ONNX plus character-CNN ONNX, combined with the validation-selected 95/5 CNN/LightGBM weights and threshold 0.5.
- Production `/predict-context`: the same `URLInference` prediction plus separate caller-supplied HTML/email signals. Context is not fused into probability.
- Production artifact identity: `url-ensemble-original-onnx-1.0`. The robust PyTorch/pickle candidate is recorded in the registry but is not loaded by the Vercel runtime.
- Current deployment: `https://phishing-ml-url-detector-seven.vercel.app`.
- Narrow Google slash/no-slash bug on current `main`: not reproduced because commit `5857815` already removes a root slash before inference.
- Remaining instability reproduced: scheme-less `www.google.com` was phishing at 99.611% while `https://www.google.com` was legitimate at 0.578% phishing probability.
- Architectural issue: canonicalization was embedded in the CNN tokenizer, lowercased the entire URL, and did not supply a consistent missing scheme before both production models.
- Model, threshold, training-data, split, and historical result files will not be rewritten for this reliability fix.
