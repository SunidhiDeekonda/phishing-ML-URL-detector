# Production Reliability Audit

Date: 2026-09-24

## Scope

This audit investigated equivalent root-URL instability, traced the live inference path, added a network-free external regression suite, clarified optional HTML/email analysis, and compared local predictions with the public Vercel deployment. No model was trained or modified.

## Production model path

Both `POST /predict` and `POST /predict-context` use `app.inference.URLInference`. The deployed model version is `url-ensemble-original-onnx-1.0`: the original ONNX character CNN and calibrated ONNX LightGBM model, combined with the validation-selected 95% CNN / 5% LightGBM weights at threshold 0.5. Context signals are returned separately and are not fused into this probability.

## Bug reproduction and root cause

On the audited baseline, the exact HTTPS Google root pair already produced the same result because an earlier change removed an empty root slash. Therefore the exact reported Google slash bug was not reproducible at commit `aaf8985`.

The audit did reproduce a broader equivalent-input defect: `www.google.com` was treated as a different model string from `https://www.google.com`. URL preparation was embedded in the CNN tokenizer, lowercased the entire URL, and did not consistently add a scheme. That made feature extraction and CNN sequence preparation depend on the user's textual representation. Character-level models are especially sensitive to even one added or removed character.

The frontend did not cause the defect. The API passed submitted strings to the shared inference class. Both prediction endpoints used the same model path.

## General fix

`src/url_normalization.py` now provides one network-free production canonicalizer. It:

- trims leading and trailing whitespace;
- adds `https` when the scheme is omitted;
- lowercases only the scheme and hostname;
- removes default ports 80 and 443;
- treats an empty path and `/` as the same root path;
- preserves non-root paths and their trailing slashes;
- preserves query strings, fragments, and percent encoding;
- performs no DNS lookup, redirect resolution, or URL fetch.

There is no brand/domain allowlist. The tokenizer keeps its released lowercase, rightmost-200-character behavior after the production boundary has canonicalized the URL.

## External URL regression

The deterministic suite contains 87 strings:

- 75 legitimate/benign cases across official roots, difficult account/login paths, format variants, project URLs, and reserved documentation domains;
- 12 synthetic suspicious cases;
- 5 root-format equivalence groups.

Results:

| Measure | Result |
|---|---:|
| Known-good passed | 62 / 75 |
| Known-good false positives | 13 |
| Synthetic phishing passed | 12 / 12 |
| Synthetic phishing false negatives | 0 |
| Format groups passed | 5 / 5 |
| Exact URLs absent from dataset | 87 |
| Registered domains absent from training | 66 |
| Unseen-domain benign accuracy | 88.89% |
| Unseen-domain benign false positives | 6 |
| Unseen-domain synthetic recall | 100.00% |

### Remaining false positives

| URL | Ensemble | CNN | LightGBM | Likely reason |
|---|---:|---:|---:|---|
| `https://www.paypal.com/` | 57.5323% | 60.4551% | 1.9986% | CNN sequence behavior |
| `https://github.com/login` | 99.9929% | 99.9953% | 99.9460% | login-path distribution behavior |
| `https://www.paypal.com/signin` | 99.9973% | 100.0000% | 99.9460% | signin-path distribution behavior |
| `https://www.amazon.com/ap/signin` | 99.9973% | 100.0000% | 99.9461% | signin-path distribution behavior |
| `https://www.dropbox.com/login` | 99.9973% | 100.0000% | 99.9460% | login-path distribution behavior |
| `https://www.linkedin.com/login` | 99.9972% | 99.9999% | 99.9460% | login-path distribution behavior |
| `https://dash.cloudflare.com/login` | 99.9970% | 99.9996% | 99.9460% | login-path distribution behavior |
| `https://phishing-ml-url-detector-seven.vercel.app/` | 99.6490% | 99.6488% | 99.6522% | long, hyphenated unseen host |
| `https://github.com/SunidhiDeekonda/phishing-ML-URL-detector` | 99.9970% | 100.0000% | 99.9395% | long, hyphenated path |
| `https://example.com/docs/getting-started?lang=en#intro` | 99.9973% | 100.0000% | 99.9460% | long external path/query |
| `https://example.org/account/preferences` | 99.9957% | 99.9983% | 99.9460% | account keyword/path behavior |
| `https://example.net/login/help` | 99.9967% | 99.9993% | 99.9460% | login keyword/path behavior |
| `https://example.org/docs/My%20Guide?Topic=Security#Part-1` | 99.9973% | 100.0000% | 99.9460% | long encoded external path |

These failures are reported rather than hidden. No threshold was tuned against this suite, and no domain was whitelisted.

## Research-metric integrity

Frozen historical selected-ensemble test metrics remain unchanged:

- accuracy 99.5248%;
- precision 100.0000%;
- recall 99.0476%;
- ROC-AUC 99.9312%;
- false negatives 19.

Running the same held-out rows through today's production canonicalization path gives:

- accuracy 93.9220%;
- precision 100.0000%;
- recall 87.8195%;
- ROC-AUC 99.6181%;
- false negatives 243.

The values are not substituted for one another. The frozen research metrics measure the original stored preprocessing artifacts; the production-normalized metrics measure a changed input representation. Retraining was deliberately not performed during this reliability fix because it would constitute a new experiment and require fresh validation without tuning on the held-out test.

## Robustness regression

The saved robustness results were preserved:

- baseline adversarial accuracy 99.0745%, recall 98.1454%, false negatives 37;
- robust candidate adversarial accuracy 99.5498%, recall 99.1479%, false negatives 17;
- robust candidate clean accuracy 99.6248%, recall 99.2982%, ROC-AUC 99.9450%.

The robust candidate remains registered as a validated candidate; production still uses the documented original ONNX ensemble.

## Context and adaptation checks

- Empty context returns `No optional context supplied.`
- Benign HTML produces no risk flag.
- Synthetic credential-form HTML reports one form, one password field, two credential terms, and a credential-form risk flag.
- Benign email produces no risk flag.
- Synthetic urgent credential-request email reports seven risk terms and a credential-request risk flag.
- Context does not alter the validated URL probability.
- Submitted URLs are never fetched.
- Feedback was accepted as `pending_verification`; automatic retraining is disabled.
- The current Vercel response reports `persistent_storage: false`, so durable production feedback storage remains a deployment limitation.

## Automated and deployed verification

- Python compile/import validation: PASS
- Full pytest: 48 passed, 0 failed, 2 dependency deprecation warnings
- `git diff --check`: PASS
- Google root slash/no-slash: PASS
- GitHub root slash/no-slash: PASS
- HTML context: PASS
- Email context: PASS
- Model info: PASS
- Feedback validation/acceptance: PASS
- Local/deployed mismatches across 87 cases: 0
- Public health: PASS; both ONNX models loaded

Deployment: `https://phishing-ml-url-detector-seven.vercel.app`

## Conclusion

Equivalent root representations are now stable locally and in production, and the UI accurately explains the experimental context feature. The system is suitable for demonstrating preprocessing reliability and honest evaluation methodology. It is not a universal URL oracle: the external suite exposes material false positives on legitimate login paths and unusual unseen URLs, and the production-normalized held-out result shows that a future model release should be retrained and validated on the canonicalized representation before claiming research-level production performance.
