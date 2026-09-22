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
