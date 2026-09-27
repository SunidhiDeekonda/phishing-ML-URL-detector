# Viva / Professor Defense

## Part 1 — 60-second project explanation

We reproduced a CNN-LightGBM phishing URL detector with domain-separated splits, then added deterministic adversarial stress testing/training, safe context evidence, verified feedback, drift monitoring, and model versioning. No submitted URL is fetched.

## Part 2 — 3-minute technical explanation

The CNN consumes 200 characters and LightGBM consumes 36 features. Eight seed-42 mutation families augment training only; validation freezes selection before a 3998-row adversarial test. Original adversarial recall was 98.1454%; robust recall was 99.1479%. Context stays separate because no labeled content corpus supports calibrated fusion.

## Part 3 — What was in the original paper?

CNN, 36 URL features, LightGBM, weighted ensemble.

## Part 4 — What did we reproduce?

Independent preprocessing, domain splits, training, selection, test, deployment.

## Part 5 — What is actually new?

Robustness benchmark/training, context, verified adaptation, drift, versioning.

## Part 6 — Why not higher precision?

The reference already reports 100%; novelty answers different questions.

## Part 7 — Adversarial robustness

Label-preserving surface changes; clean and adversarial tests stay separate.

## Part 8 — Content/context

Only supplied text is parsed; it is not probability fusion.

## Part 9 — Continuous learning

Quarantine, human approval, deduplication, test protection, offline retraining, validation gate.

## Part 10 — Data leakage

Domains and mutation sources remain split-specific.

## Part 11 — Clean vs adversarial test

Ordinary generalization versus declared perturbation resilience.

## Part 12 — Why never fetch URLs

Avoid harm, SSRF, privacy loss, and nondeterminism.

## Part 13 — Limitations

Synthetic coverage, fixed subset, heuristic context, ephemeral serverless feedback.

## Part 14 — Files to open

`src/adversarial_urls.py`, `scripts/run_research_extension.py`, `src/context_features.py`, `src/continuous_learning.py`, `src/drift_monitor.py`, `results/robustness_final_metrics.json`, `tests/test_research_extensions.py`.

## Part 15 — 25 likely professor questions

### 1. Paper already has 100% precision. What is new?

**Short answer:** Robustness and adaptability, not higher precision.

**Detailed follow-up:** Robustness and adaptability, not higher precision. The design and measured evidence are recorded in the cited source rather than inferred from a headline metric.

**Proof to open:** `docs/NOVELTY_AND_EXTENSION.md`

### 2. Is this copied?

**Short answer:** No: independent reproduction plus measured extensions.

**Detailed follow-up:** No: independent reproduction plus measured extensions. The design and measured evidence are recorded in the cited source rather than inferred from a headline metric.

**Proof to open:** `RUN_LOG.md`

### 3. Which code did you implement?

**Short answer:** The reproducibility and extension pipeline.

**Detailed follow-up:** The reproducibility and extension pipeline. The design and measured evidence are recorded in the cited source rather than inferred from a headline metric.

**Proof to open:** `src/adversarial_urls.py`

### 4. Why 20,000 URLs?

**Short answer:** Feasible deterministic college-scale study.

**Detailed follow-up:** Feasible deterministic college-scale study. The design and measured evidence are recorded in the cited source rather than inferred from a headline metric.

**Proof to open:** `docs/FINAL_REPORT.md`

## Reliability-fix questions

### Why did `google.com/` previously produce a different prediction?

The CNN reads literal characters. Before reliable canonicalization, a root slash or missing scheme changed its sequence even when the user meant the same root website. The narrow slash symptom had already been hotfixed, but the audit found the scheme-less form remained unstable.

### What did you change?

We created one conservative string-only canonicalizer and apply it before both feature extraction and tokenization. We also added an external/OOD corpus, training-membership annotations, local/deployed parity checks, and explicit context UX.

### Did you whitelist Google?

No. There is no brand/domain allowlist in inference. Google is only one test record in a transparent regression dataset; the canonical rules apply equally to every host.

### What is URL canonicalization?

It maps safely equivalent text representations to one representation, such as trimming outer spaces, lowercasing scheme/host, supplying HTTPS when the scheme is absent, and treating an empty root path like `/`.

### Why only normalize equivalent root URLs?

The root paths `host` and `host/` identify the same resource. By contrast, `/app` and `/app/` can route differently, so both are preserved.

### Why not strip every slash?

Slashes inside paths carry structure and may change server routing. Removing them would be over-normalization and could hide phishing signals.

### How did you test URLs outside training data?

The external suite contains known official roots, difficult legitimate login/account paths, format variants, documented project URLs, reserved documentation domains, and inert synthetic phishing strings.

### How do you know whether an external URL was in training?

Each output records exact-string membership plus whether its registered domain occurs in train, validation, or test splits. Registered domains are extracted offline using the packaged public-suffix snapshot.

### Why not claim 100% external accuracy?

The suite is a transparent regression sample, not every possible URL. All failures are retained with component probabilities and relevant features.

### What is the HTML box?

It accepts raw HTML text pasted by the user and counts local structural/credential indicators. It does not accept or fetch a webpage URL.

### Is HTML analysis part of the trained ensemble?

No. It remains supplementary because the project lacks a labelled HTML/email corpus for calibrated training and evaluation.

### Does the app visit the URL?

No. URL prediction, canonicalization, context parsing, adversarial generation, and regression testing operate on strings only.

### How does context improve the research contribution?

It implements a safe, clearly separated version of the paper's context-analysis future work while preserving scientific honesty about what has and has not been validated.

### Original work versus today's fix

The paper supplied CNN, LightGBM, 36 URL features, ensembling, and reported 100% precision. We independently reproduced training/evaluation and a local UI. The research extension added adversarial robustness, context evidence, verified feedback, drift, and model governance. Today's work is a production reliability fix: shared canonicalization, external/OOD regression testing, production invariance checks, and clearer context UX.

### 5. How prevent domain leakage?

**Short answer:** Root domains are split-disjoint.

**Detailed follow-up:** Root domains are split-disjoint. The design and measured evidence are recorded in the cited source rather than inferred from a headline metric.

**Proof to open:** `tests/test_phase1.py`

### 6. What are the 36 features?

**Short answer:** Lexical and structural URL measurements.

**Detailed follow-up:** Lexical and structural URL measurements. The design and measured evidence are recorded in the cited source rather than inferred from a headline metric.

**Proof to open:** `src/features.py`

### 7. What does CNN add?

**Short answer:** Learned local character motifs.

**Detailed follow-up:** Learned local character motifs. The design and measured evidence are recorded in the cited source rather than inferred from a headline metric.

**Proof to open:** `src/train_charcnn.py`

### 8. Why 95/5?

**Short answer:** Validation-only selection chose it.

**Detailed follow-up:** Validation-only selection chose it. The design and measured evidence are recorded in the cited source rather than inferred from a headline metric.

**Proof to open:** `src/select_ensemble.py`

### 9. Why did LightGBM score higher cleanly?

**Short answer:** Engineered signals were strong here.

**Detailed follow-up:** Engineered signals were strong here. The design and measured evidence are recorded in the cited source rather than inferred from a headline metric.

**Proof to open:** `results/test_metrics.json`

### 10. Why keep validation selection?

**Short answer:** To avoid test-set tuning.

**Detailed follow-up:** To avoid test-set tuning. The design and measured evidence are recorded in the cited source rather than inferred from a headline metric.

**Proof to open:** `docs/FINAL_REPORT.md`

### 11. What is adversarial training?

**Short answer:** Training with label-preserving perturbations.

**Detailed follow-up:** Training with label-preserving perturbations. The design and measured evidence are recorded in the cited source rather than inferred from a headline metric.

**Proof to open:** `scripts/run_research_extension.py`

### 12. How generate mutations?

**Short answer:** Eight seeded string transformations.

**Detailed follow-up:** Eight seeded string transformations. The design and measured evidence are recorded in the cited source rather than inferred from a headline metric.

**Proof to open:** `src/adversarial_urls.py`

### 13. Did you visit phishing sites?

**Short answer:** No; URLs remained inert strings.

**Detailed follow-up:** No; URLs remained inert strings. The design and measured evidence are recorded in the cited source rather than inferred from a headline metric.

**Proof to open:** `src/adversarial_urls.py`

### 14. Could mutations leak?

**Short answer:** No; sources remain split-specific.

**Detailed follow-up:** No; sources remain split-specific. The design and measured evidence are recorded in the cited source rather than inferred from a headline metric.

**Proof to open:** `data/processed/adversarial_test.csv`

### 15. How did robustness change?

**Short answer:** Recall changed +1.0025%.

**Detailed follow-up:** Recall changed +1.0025%. The design and measured evidence are recorded in the cited source rather than inferred from a headline metric.

**Proof to open:** `results/robustness_final_metrics.json`

### 16. Did clean performance decrease?

**Short answer:** Robust clean accuracy was 99.6248%.

**Detailed follow-up:** Robust clean accuracy was 99.6248%. The design and measured evidence are recorded in the cited source rather than inferred from a headline metric.

**Proof to open:** `results/robustness_selection.json`

### 17. Why not retrain every report?

**Short answer:** Unverified feedback enables poisoning.

**Detailed follow-up:** Unverified feedback enables poisoning. The design and measured evidence are recorded in the cited source rather than inferred from a headline metric.

**Proof to open:** `src/continuous_learning.py`

### 18. What is model poisoning?

**Short answer:** Maliciously corrupting training feedback.

**Detailed follow-up:** Maliciously corrupting training feedback. The design and measured evidence are recorded in the cited source rather than inferred from a headline metric.

**Proof to open:** `src/continuous_learning.py`

### 19. How does validation gating work?

**Short answer:** Mean clean/adversarial AUC plus clean floor.

**Detailed follow-up:** Mean clean/adversarial AUC plus clean floor. The design and measured evidence are recorded in the cited source rather than inferred from a headline metric.

**Proof to open:** `results/robustness_selection.json`

### 20. How does drift monitoring work?

**Short answer:** PSI over all 36 features.

**Detailed follow-up:** PSI over all 36 features. The design and measured evidence are recorded in the cited source rather than inferred from a headline metric.

**Proof to open:** `src/drift_monitor.py`

### 21. Is context in validated accuracy?

**Short answer:** No; it is separate evidence.

**Detailed follow-up:** No; it is separate evidence. The design and measured evidence are recorded in the cited source rather than inferred from a headline metric.

**Proof to open:** `src/context_features.py`

### 22. Why not fetch HTML?

**Short answer:** Safety, privacy, SSRF, reproducibility.

**Detailed follow-up:** Safety, privacy, SSRF, reproducibility. The design and measured evidence are recorded in the cited source rather than inferred from a headline metric.

**Proof to open:** `app/research_api.py`

### 23. How is deployment different?

**Short answer:** Safe extension APIs and controls.

**Detailed follow-up:** Safe extension APIs and controls. The design and measured evidence are recorded in the cited source rather than inferred from a headline metric.

**Proof to open:** `app/main.py`

### 24. Biggest limitations?

**Short answer:** Synthetic attacks and no labeled content corpus.

**Detailed follow-up:** Synthetic attacks and no labeled content corpus. The design and measured evidence are recorded in the cited source rather than inferred from a headline metric.

**Proof to open:** `docs/NOVELTY_AND_EXTENSION.md`

### 25. What next?

**Short answer:** External temporal and labeled content evaluation.

**Detailed follow-up:** External temporal and labeled content evaluation. The design and measured evidence are recorded in the cited source rather than inferred from a headline metric.

**Proof to open:** `docs/FINAL_REPORT.md`

## Independent URL, HTML, and Email Product Questions

### What is the difference between URL, HTML, and Email analysis?

URL detection is the trained CNN + LightGBM classifier. HTML analysis parses structural and credential-related signals from pasted source. Email analysis counts implemented risk language, credential-request patterns, exact call-to-action phrases, and URL strings in pasted text.

### Is HTML analysis machine learning?

No. It is deterministic, explainable context extraction and is clearly labelled as a research extension.

### Is Email analysis machine learning?

No. It is deterministic language-signal extraction, not a calibrated email classifier.

### Why are HTML/email signals not included in the CNN-LightGBM probability?

The models were trained and evaluated on URLs. We do not have a properly labelled aligned HTML/email dataset for calibrated fusion. Adding hand-written weights would make the reported probability scientifically misleading.

### Why did you implement them?

The reference system focused on URL evidence and identified broader content/context analysis as future work. Phishing also uses credential forms and social-engineering language, so the extensions explore those evidence layers safely.

### What features does HTML analysis inspect?

Forms, password inputs, iframes, scripts, absolute external action/link/source targets, meta refresh, and occurrences of the implemented credential/risk terms in HTML text.

### What signals does Email analysis inspect?

URL count; occurrences of login, verify, urgent, account, password, credential, confirm, and suspend; credential-request phrase patterns; and the implemented exact call-to-action phrases.

### Does HTML analysis execute pasted code?

No. It parses inert text using Python's HTML parser. It does not execute HTML or JavaScript and makes no network request.

### Does Email analysis access Gmail?

No. It analyses only manually pasted text and has no mailbox connection.

### Can HTML and Email be used independently?

Yes. `POST /analyze-html` requires only HTML, and `POST /analyze-email` requires only email text. Neither requires a URL or the other context type.

### What came from the original paper and what did you add?

The original methodology provided URL classification using a character CNN, 36 engineered URL features, LightGBM, and ensembling. Our work reproduced that pipeline and added domain-separated evaluation, adversarial robustness, independent HTML/email context evidence, verified feedback, drift monitoring, model registry, reliability canonicalization, and production regression testing.

## GitHub false-positive defense

### Why did your own GitHub repository get classified as phishing?

**20-second answer:** The URL was legitimate, so the result was a false positive. The original data accidentally taught both models that almost any non-root path looks like phishing: all legitimate examples were roots, while 97% of training phishing examples had paths. GitHub was absent from every split, so ordinary repository paths were out of distribution.

**One-minute technical answer:** The original CNN returned `0.9999998700` and LightGBM returned `0.9993945513`. Counterfactual tests showed that neutral GitHub paths also saturated, proving the literal repository name was not the sole cause. For LightGBM, local contribution analysis identified path length as the strongest positive contribution. The selected ensemble was 95% CNN, so two correlated errors produced `0.9999696041`. We preserved this evidence, built a separately partitioned legitimate hard-negative corpus, trained versioned production artifacts with adversarial examples retained, and required a validation gate before inspecting the hard-negative test.

### Does that mean GitHub is phishing?

No. A normal repository page is not phishing merely because it is on GitHub. User-hosted `github.io` pages, raw content, links inside repositories, and abused platform content must be assessed separately.

### Why did you not whitelist GitHub?

A whitelist would hide the learned shortcut and would incorrectly treat all platform content as safe. Production inference contains no GitHub/domain allowlist. The repair uses labelled hard-negative training and transparent regression tests.

### What is `[object Object]`?

It is JavaScript's default text conversion for a nested object. It was a frontend rendering defect, not an ML result. The UI now renders each returned signal and nested context-risk field explicitly through reusable defensive renderers.

### Does HTML analysis determine whether a website is phishing?

No. It parses manually pasted, inert HTML text for supplementary indicators. It neither fetches nor executes the page, and it is not part of the calibrated URL probability.
