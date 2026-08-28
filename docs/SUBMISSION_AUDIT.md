# Final College Submission and Demonstration Audit

**Project:** AI-Based Phishing Detection System Using Ensemble Learning<br>
**College:** KLH Bachupally<br>
**Audit scope:** Actual local repository evidence available for final submission and publication<br>
**Audit date:** 28 August 2026

## 1. Audit Scope and Method

This audit checks the files currently present in the local repository. It does not rely only on statements in `README.md`. Source implementations, processed data, saved models, metric records, predictions, plots, tests, application files, provenance, and the academic report were checked as distinct evidence categories.

No model was retrained, no metric was changed, no frontend file was redesigned, and no PDF was generated during this audit. The most recent recorded test result remains 24 passed and 0 failed. The audit did not rerun training or require live internet access.

## 2. Verified Repository Inventory by Phase

### Phase 1: Data Preparation, Features, Tokenization, and Splitting

The two frozen source snapshots exist at `data/raw/phishing_urls.csv` and `data/raw/legit_urls.csv`. Their provenance, source addresses, acquisition timestamp, row cleaning, hashes, sampling method, and seed are recorded in `DATA_PROVENANCE.md`.

The processed dataset exists at `data/processed/dataset.csv` with 20,001 CSV lines: one header and 20,000 data rows. It has two columns, representing the URL and class label. The engineered feature matrix exists at `data/processed/features.csv` with one header and 20,000 data rows, and its header contains exactly 36 feature columns.

Character artifacts exist at `data/processed/char_vocab.json` and `data/processed/char_sequences.npy`. The sequence array has verified shape `(20000, 200)`. The split arrays exist with verified shapes: train `(14002,)`, validation `(2000,)`, and test `(3998,)`. The leakage discovery and correction are recorded in `RUN_LOG.md`, and the relevant automated checks are in `tests/test_phase1.py`.

### Phase 2A: LightGBM

The training implementation exists at `src/train_lightgbm.py`. Both the raw and calibrated trained artifacts exist at `models/lightgbm_raw.pkl` and `models/lightgbm_calibrated.pkl`. Validation metrics, best iteration, training duration, confusion matrices, and top feature importances are preserved in `results/lightgbm_validation_metrics.json`. Row-level validation probabilities are stored in `results/lightgbm_validation_predictions.csv`, and the complete importance ranking is stored in `results/lightgbm_feature_importance.csv`.

### Phase 2B: Character-Level CNN

The CNN implementation and training procedure exist at `src/train_charcnn.py`. The trained state exists at `models/char_cnn.pt`. Epoch-by-epoch training and validation history, device, hyperparameters, best epoch, early stopping, duration, and final validation measurements are stored in `results/charcnn_training_history.json`. Validation probabilities are stored in `results/charcnn_validation_predictions.csv`.

### Phase 2C: Validation-Only Ensemble Selection

The weight-search implementation exists at `src/select_ensemble.py`. The complete 0.05-step search record exists at `results/ensemble_weight_search.csv`, with 22 lines consisting of a header and 21 candidate weight combinations. The frozen reference and selected weights, validation selection criterion, and validation metrics are stored in `results/ensemble_config.json`.

There is no generated `ensemble_weight_search.png`. The final report correctly does not reference one. The CSV is the authoritative weight-search evidence.

### Phase 2D: Held-Out Test Evaluation

The test evaluation implementation exists at `src/evaluate_test.py`. Final model and ensemble metrics are stored in `results/final_test_metrics.json` and summarized in `results/model_comparison.csv`. The latter contains one header and four evaluated configurations. Row-level test outcomes exist in `results/test_predictions.csv`, with one header and 3,998 prediction rows.

Six generated result figures exist under `results/plots/`: confusion matrix, ROC curve, precision-recall curve, model comparison, feature importance, and probability distribution. These are derived presentation artifacts; the JSON and CSV records remain the numerical source of truth.

### Phase 3A: FastAPI Demonstration Application

The FastAPI routes and page serving exist in `app/main.py`. Shared model loading, feature extraction, character encoding, component probabilities, ensemble calculation, threshold decision, and safety behavior exist in `app/inference.py`. The interface files exist at `app/templates/index.html`, `app/static/style.css`, and `app/static/app.js`. API and page tests exist in `tests/test_phase3a_api.py`.

The repository contains two successful frontend screenshots at `docs/assets/frontend_legitimate.png` and `docs/assets/frontend_phishing.png`. The earlier failed-connection screenshot is not part of the repository and is not submission evidence.

### Phase 3B and 3C: Academic Report

The academic report exists at `docs/FINAL_REPORT.md`. It contains 31 sequential sections, 13 titled tables, six verified figure references, completed team and guide metadata, the college name `KLH Bachupally`, source-supported references, honest limitations, and reproducibility instructions. Reference-study values and local experiment values are explicitly separated.

## 3. Evidence Map

| Major claim or artifact | Exact local evidence | What it proves |
|---|---|---|
| Frozen dataset origin and cleaning | `DATA_PROVENANCE.md` | Source URLs, timestamp, hashes, original counts, duplicate handling, and deterministic sampling. |
| Raw phishing snapshot | `data/raw/phishing_urls.csv` | The locally retained frozen phishing source data. |
| Raw legitimate snapshot | `data/raw/legit_urls.csv` | The locally retained frozen legitimate source data. |
| Balanced 20,000-row experiment | `data/processed/dataset.csv` | 20,000 labelled URL rows used by the local experiment. |
| Label convention and preparation history | `RUN_LOG.md` | Class definitions, counts, audit decisions, and phase chronology. |
| Exact 36-feature implementation | `src/features.py` | Names and calculations for every engineered feature. |
| Materialized 36-feature matrix | `data/processed/features.csv` | One aligned 36-column feature row per dataset URL. |
| Exact character tokenizer | `src/char_tokenizer.py` | PAD, explicit character indices, UNK, lowercase conversion, rightmost truncation, and padding. |
| Saved tokenizer metadata | `data/processed/char_vocab.json` | Released-compatible character-to-index mapping. |
| Saved character input | `data/processed/char_sequences.npy` | The `(20000, 200)` encoded sequence matrix. |
| Corrected train split | `data/processed/train_idx.npy` | 14,002 final training row indices. |
| Corrected validation split | `data/processed/val_idx.npy` | 2,000 final validation row indices. |
| Corrected test split | `data/processed/test_idx.npy` | 3,998 final held-out row indices. |
| Leakage discovery and correction | `RUN_LOG.md` | Initial overlap counts, deterministic grouped correction, final class counts, and zero overlap. |
| Automated split and tokenizer checks | `tests/test_phase1.py` | Executable checks for alignment, indices, sequence rules, split separation, and domain leakage. |
| LightGBM implementation | `src/train_lightgbm.py` | Model configuration, fitting, early stopping, calibration, and metric production. |
| Calibrated LightGBM artifact | `models/lightgbm_calibrated.pkl` | The fitted model used for final inference. |
| Raw LightGBM artifact | `models/lightgbm_raw.pkl` | The fitted uncalibrated model retained for audit. |
| LightGBM validation evidence | `results/lightgbm_validation_metrics.json` | Best iteration 88, validation results, timing, and leading feature importances. |
| CNN implementation | `src/train_charcnn.py` | Embedding, parallel convolutions, pooling, dense layers, training, and early stopping. |
| Trained CNN artifact | `models/char_cnn.pt` | The saved CNN state used by evaluation and the application. |
| CNN training evidence | `results/charcnn_training_history.json` | All seven completed epochs, best epoch 5, MPS use, timing, and validation metrics. |
| Ensemble selection implementation | `src/select_ensemble.py` | Validation-only probability combination and weight search. |
| Full ensemble search | `results/ensemble_weight_search.csv` | Every candidate from CNN weight 0.00 to 1.00 in 0.05 increments. |
| Frozen ensemble decision | `results/ensemble_config.json` | Selected CNN 0.95 and LightGBM 0.05 weights and validation ROC-AUC criterion. |
| Test evaluation implementation | `src/evaluate_test.py` | One held-out evaluation path for components and both ensembles. |
| Final test metrics | `results/final_test_metrics.json` | Exact test size, weights, metrics, and confusion-matrix counts. |
| Compact model comparison | `results/model_comparison.csv` | Side-by-side final results for four configurations. |
| Row-level test proof | `results/test_predictions.csv` | 3,998 held-out labels, probabilities, and predictions. |
| Generated quantitative figures | `results/plots/` | Six visual summaries derived from completed evaluation records. |
| FastAPI routes | `app/main.py` | Local page, health endpoint, and prediction endpoint. |
| End-to-end inference | `app/inference.py` | Loading saved models and applying the exact frozen preprocessing and ensemble. |
| Browser interface | `app/templates/index.html`, `app/static/style.css`, `app/static/app.js` | The completed local user interface and API call behavior. |
| API and frontend tests | `tests/test_phase3a_api.py` | Health, prediction, response, and page-loading checks. |
| Dependency list | `requirements.txt` | Python packages required by the project. |
| Complete audit history | `RUN_LOG.md` | Chronological evidence of every completed experimental phase. |
| Academic report | `docs/FINAL_REPORT.md` | Submission-ready narrative before final formatting and PDF generation. |

## 4. Recommended 8-12 Minute Professor/Viva Demonstration

The demonstration should show saved evidence and run inference, not retrain either model. Keep VS Code open at the project root and prepare the files as tabs in the following order.

### File 1: `DATA_PROVENANCE.md` - approximately 45 seconds

Point at the two reference snapshot URLs, source counts, SHA-256 hashes, and sampling statement. Say: “The frozen sources are traceable here. We sampled 10,000 phishing and 10,000 legitimate URLs deterministically with seed 42; we did not claim to train on the full reference corpus.”

### File 2: `RUN_LOG.md` - approximately 60 seconds

Point at the initial domain-overlap counts and final corrected split counts. Say: “A random stratified split leaked root domains, so we corrected it before training. The final 14,002/2,000/3,998 split has zero train-validation, train-test, or validation-test domain overlap.”

### File 3: `src/features.py` - approximately 60 seconds

Point at the feature-name list and extraction function. Say: “This is the actual implementation of all 36 numeric URL features, including lengths, punctuation, entropy, structure, suspicious TLDs, and keyword flags. It treats a URL as text and never opens it.”

### File 4: `src/char_tokenizer.py` - approximately 45 seconds

Point at the explicit alphabet, PAD and UNK indices, sequence length, and truncation logic. Say: “The CNN receives deterministic 200-character sequences. PAD is zero, explicit characters are 1 to 49, UNK is 50, and long URLs keep the rightmost 200 characters.”

### File 5: `src/train_lightgbm.py` - approximately 60 seconds

Point at the LightGBM configuration, early stopping, and calibration code. Say: “This branch learns from the 36 engineered features. It stopped at iteration 88, was calibrated with sigmoid calibration, and produced the saved model rather than being retrained for the demo.”

### File 6: `src/train_charcnn.py` - approximately 60 seconds

Point at the embedding, three convolution branches, pooling, dense layer, dropout, and training loop. Say: “This branch learns character patterns directly from URLs using kernels 3, 5, and 7. It trained on Apple MPS, selected epoch 5 through validation early stopping, and has 56,625 trainable parameters.”

### File 7: `src/select_ensemble.py` - approximately 45 seconds

Point at the validation-only weight loop and ROC-AUC selection. Say: “Weights were searched only on validation data from 0.00 to 1.00 in 0.05 steps. The test set was not loaded to choose the ensemble.”

### File 8: `results/ensemble_config.json` - approximately 30 seconds

Point at `best_weights`, `selection_criterion`, and `row_count`. Say: “This frozen record proves that validation ROC-AUC selected 0.95 CNN and 0.05 LightGBM before final testing.”

### File 9: `results/final_test_metrics.json` - approximately 60 seconds

Point at `test_size`, `selected_weights`, and `selected_ensemble`. Say: “On 3,998 held-out URLs, the selected ensemble achieved about 99.525% accuracy, 100% precision, 99.048% recall, and 99.931% ROC-AUC. These are our results, separate from the reference paper.”

### File 10: `results/plots/confusion_matrix.png` - approximately 30 seconds

Point at TN 2,003, FP 0, FN 19, and TP 1,976. Say: “At threshold 0.50, no legitimate test URL was flagged, while nineteen phishing URLs were missed. We did not alter the threshold after seeing this result.”

### File 11: `results/plots/model_comparison.png` - approximately 30 seconds

Point at differences among the component models and ensembles. Say: “LightGBM had the highest test accuracy, while the CNN had the highest test ROC-AUC. We retained the validation-selected ensemble instead of tuning to whichever test metric looked best.”

### File 12: `app/inference.py` - approximately 60 seconds

Point at model loading, `extract_features`, character encoding, component probabilities, and weighted combination. Say: “The frontend calls this same frozen inference pipeline. It loads both saved models, returns component and ensemble probabilities, and never fetches or visits the submitted URL.”

After the code evidence, run the tests and launch the existing app. Use one legitimate example and one synthetic suspicious-looking string. Do not retrain models during the viva.

## 5. Likely Professor Questions and Best Proof

| Professor may ask | Best file to open | What that file proves | Short answer to give |
|---|---|---|---|
| Where did your dataset come from? | `DATA_PROVENANCE.md` | Frozen PhishX snapshot URLs, hashes, cleaning, and acquisition record. | “The phishing snapshot is based on PhishTank and the legitimate snapshot on Tranco; exact frozen sources and hashes are recorded here.” |
| Why only 20,000 URLs? | `DATA_PROVENANCE.md` | Deterministic 10,000-per-class sampling. | “This is a balanced limited-compute reproduction. Seed 42 makes the 20,000-row subset reproducible, but we do not equate it with the full reference corpus.” |
| Can you prove there are 20,000 processed rows? | `data/processed/dataset.csv` | The actual processed URL and label records. | “The file contains one header and 20,000 aligned labelled rows.” |
| How do you know train and test data did not leak? | `RUN_LOG.md` and `tests/test_phase1.py` | Initial leakage discovery, corrected grouped split, and executable zero-overlap test. | “We found overlap in the first split, corrected it before training, and verified zero root-domain overlap across all final partitions.” |
| What are the 36 features? | `src/features.py` | Exact names and calculations. | “They are lexical, structural, statistical, keyword, and length-bucket features implemented here, not an undocumented external feature set.” |
| How does the CNN read a URL? | `src/char_tokenizer.py` and `src/train_charcnn.py` | Character indexing, 200-length encoding, embedding, and convolution architecture. | “The URL is lowercased and encoded to 200 indices, then processed by 3-, 5-, and 7-character convolution branches.” |
| Why combine CNN and LightGBM? | `src/train_charcnn.py`, `src/train_lightgbm.py`, and `src/select_ensemble.py` | Complementary learned character and engineered-feature branches. | “The CNN learns sequential text patterns, while LightGBM uses interpretable engineered indicators; their probabilities can provide complementary evidence.” |
| How did you choose 95/5 weights? | `results/ensemble_weight_search.csv` and `results/ensemble_config.json` | Full candidate search and frozen validation winner. | “We searched CNN weights in 0.05 steps and selected the highest validation ROC-AUC, which was 0.95 CNN and 0.05 LightGBM.” |
| Did you tune on the test set? | `src/select_ensemble.py`, `results/ensemble_config.json`, and `RUN_LOG.md` | Validation-only selection and phase order. | “No. Early stopping, calibration, and weights used validation data; the test set was evaluated only after decisions were frozen.” |
| Where are your actual results? | `results/final_test_metrics.json` and `results/model_comparison.csv` | Exact held-out metrics and confusion counts. | “The JSON is the exact source of truth, and the CSV provides a compact four-model comparison.” |
| Where are individual test predictions? | `results/test_predictions.csv` | All 3,998 held-out predictions and probabilities. | “Every final test row is preserved here for audit, not only aggregate percentages.” |
| Where is your trained LightGBM model? | `models/lightgbm_calibrated.pkl` | Saved calibrated fitted artifact. | “This is the trained calibrated model loaded by final evaluation and the app.” |
| Where is your trained CNN? | `models/char_cnn.pt` | Saved CNN state dictionary. | “This file contains the selected trained CNN state used without live retraining.” |
| Why does LightGBM have higher test accuracy than the selected ensemble? | `results/model_comparison.csv` | Metric-specific component and ensemble results. | “Weights were selected by validation ROC-AUC, not test accuracy. We retained the preselected weights to avoid test-set tuning.” |
| What does zero false positives mean? | `results/plots/confusion_matrix.png` | Selected ensemble error counts at threshold 0.50. | “None of the 2,003 legitimate test URLs was incorrectly blocked, although nineteen phishing URLs were missed.” |
| How does the frontend call the ML model? | `app/main.py` and `app/inference.py` | POST route and end-to-end saved-model inference. | “The JavaScript sends text to `/predict`; FastAPI applies the same features and tokenizer, runs both saved models, and returns probabilities and a verdict.” |
| Does the application visit phishing sites? | `app/inference.py` | Local string-only preprocessing with no fetch operation. | “No. The URL is inert text; the app performs no DNS request, webpage fetch, HTML download, or JavaScript execution.” |
| Can you prove your tests pass? | `tests/test_phase1.py`, `tests/test_phase3a_api.py`, and `RUN_LOG.md` | Test implementation and most recent recorded result of 24 passed, 0 failed. | “The test source is visible here, the last audited run records 24 passing tests, and I can run `pytest` live without retraining.” |
| Are your results the same as the paper's? | `docs/FINAL_REPORT.md` | Separate reference and local result tables. | “No. The paper reports about 99,361 URLs and 99.819% accuracy; our 20,000-URL selected ensemble achieved about 99.525%.” |

## 6. Final PDF Content Audit

### Must Include in the Final PDF

- Completed title page with `KLH Bachupally`, department, team, guide, and academic year.
- Abstract that clearly labels the 20,000-URL results as this reproduction's outcomes.
- Introduction, problem statement, and objectives.
- Concise related work and the reference-system architecture.
- Dataset provenance, source snapshot counts, and the exact 10,000/10,000 local sample.
- Data safety statement explaining that URLs were treated only as text.
- Preprocessing and exact character-tokenizer metadata.
- Domain-leakage discovery, why it mattered, and the corrected zero-overlap split.
- The 36-feature categories and exact feature count.
- LightGBM configuration, early stopping, calibration, and validation results.
- Char-CNN architecture, training procedure, early stopping, and validation results.
- Ensemble formula, validation-only weight selection, and frozen 0.95/0.05 weights.
- Experimental protocol separating train, validation, and held-out test roles.
- Final held-out comparison table with accuracy, precision, recall, F1, ROC-AUC, PR-AUC, FP, and FN.
- `results/plots/confusion_matrix.png`.
- `results/plots/roc_curve.png`.
- `results/plots/precision_recall_curve.png`.
- `results/plots/model_comparison.png`.
- Honest result interpretation, including LightGBM's higher test accuracy and CNN's higher test ROC-AUC.
- Limitations, conclusion, and source-supported references.

### Should Include in the Final PDF

- `results/plots/feature_importance.png` with a short non-causal interpretation.
- `results/plots/probability_distribution.png` to illustrate confidence and ambiguous regions.
- A concise FastAPI application section and one or two genuine working-frontend screenshots.
- Future work clearly separated from completed work.
- A short reproducibility appendix containing environment, seed, artifact locations, and launch commands.
- A compact comparison with the reference paper that states the 0.294-percentage-point accuracy difference without claiming superiority.

### Optional

- The four Phase 3A example prediction strings and probabilities.
- A short code excerpt for the ensemble formula or tokenizer if the college format permits implementation excerpts.
- An appendix listing the 36 feature names.
- A repository tree limited to important project directories.
- A test-summary excerpt, preferably rendered as text rather than a terminal screenshot.

### Do Not Include

- The failed `ERR_CONNECTION_REFUSED` browser screenshot.
- Raw terminal dumps or screenshots of routine installation commands.
- Full raw phishing or legitimate URL lists.
- Full `test_predictions.csv` contents.
- Binary model contents or serialized-file dumps.
- A reference to nonexistent `results/plots/ensemble_weight_search.png`.
- Claims of public deployment, live URL visiting, complete reference-dataset training, test-set tuning, or superiority over the reference study.
- Unverified institutional requirements or a fabricated PDF rubric.

No special college PDF rubric has been provided. Page layout, certificates, declarations, signatures, margins, and binding requirements must therefore be confirmed separately with the instructor or department before final PDF production.

## 7. Screenshot Audit

**Frontend screenshots exist in the repository: YES.**

The repository contains the six generated quantitative PNG figures and two successful frontend captures. `docs/assets/frontend_legitimate.png` shows the legitimate `https://www.google.com` result. `docs/assets/frontend_phishing.png` shows the synthetic suspicious-looking URL string classified as phishing; that string was analysed only as text and was not visited.

The two recommended screenshots were captured after starting the local FastAPI application:

1. A legitimate result using `https://www.google.com`, showing the input, legitimate verdict, selected probability, and component probabilities.
2. A synthetic suspicious-looking result using `http://secure-account-login-example.xyz/verify`, clearly described as an unvisited local test string, showing the phishing verdict and probabilities.

Terminal screenshots are not academically necessary. Generated plots, metric tables, source files, and reproducible commands provide stronger evidence. If a terminal image is included at all, one compact `pytest` success screenshot is sufficient; it remains optional.

## 8. Result Figure Audit

| Existing figure | What it demonstrates | Final PDF status | Suggested caption |
|---|---|---|---|
| `results/plots/confusion_matrix.png` | Selected ensemble TN, FP, FN, and TP counts at threshold 0.50. | MUST INCLUDE | “Confusion matrix of the validation-selected 95/5 ensemble on 3,998 held-out URLs.” |
| `results/plots/roc_curve.png` | Class-ranking behavior of the evaluated components and ensembles across thresholds. | MUST INCLUDE | “Receiver operating characteristic curves for LightGBM, Char-CNN, and ensemble configurations on the held-out test set.” |
| `results/plots/precision_recall_curve.png` | Precision-recall tradeoff across thresholds. | MUST INCLUDE | “Precision-recall curves for the evaluated phishing URL classifiers on the held-out test set.” |
| `results/plots/model_comparison.png` | Side-by-side final metrics and the fact that no model dominates every metric. | MUST INCLUDE | “Comparison of component and ensemble performance on the final held-out test set.” |
| `results/plots/feature_importance.png` | Ranked LightGBM feature contributions led by hostname entropy, vowel fraction, and host length. | SHOULD INCLUDE | “Top engineered URL features by LightGBM importance.” |
| `results/plots/probability_distribution.png` | Probability separation and regions of model uncertainty. | SHOULD INCLUDE | “Distribution of predicted phishing probabilities on held-out data.” |

No other result figure exists. In particular, `results/plots/ensemble_weight_search.png` is absent and must not be cited.

## 9. Recommended Submission Package

### A. Formal Academic Submission

- Final academic PDF after human proofreading, genuine screenshot capture, and confirmation of college formatting requirements.
- Any declaration, approval, certificate, plagiarism, or signature pages only if the instructor or department specifically requires them.
- No special PDF rubric has been supplied, so no unconfirmed institutional page should be invented.

### B. Source and Evidence Package

- A clean project archive or repository containing `src/`, `app/`, `tests/`, `docs/`, `results/`, `models/`, `DATA_PROVENANCE.md`, `RUN_LOG.md`, `README.md`, and `requirements.txt`.
- Processed dataset, feature, character, and split artifacts under `data/processed/` when file-size and dataset-distribution rules permit.
- Frozen raw snapshots under `data/raw/` only when redistribution is permitted and the instructor requests them; otherwise retain provenance and acquisition instructions.
- Saved model artifacts and all JSON/CSV result records so the demonstration does not depend on retraining.
- Exclude `.venv/`, caches, temporary files, operating-system metadata, and the failed browser screenshot.

### C. Live Demonstration Materials

- The project laptop with the existing virtual environment and saved models.
- VS Code with the twelve demonstration files pre-opened in the order listed above.
- A browser tab for `http://127.0.0.1:8000` opened only after the server is running.
- One legitimate URL string and one clearly labelled synthetic suspicious-looking URL string prepared for inference.
- An offline fallback consisting of the two genuine frontend screenshots, six plots, metric JSON/CSV files, and the report.
- No live model retraining and no navigation to any submitted URL.

## 10. Exact Demonstration Commands

Run from the project root:

```bash
source .venv/bin/activate
pytest
uvicorn app.main:app --reload
```

After Uvicorn reports that it is running, open:

```text
http://127.0.0.1:8000
```

The earlier `ERR_CONNECTION_REFUSED` screenshot indicates that no server process was listening at that moment. Starting Uvicorn before loading the browser resolves that operational condition; it does not require a code or model change.

## 11. Top 10 VS Code Proof Files

1. `DATA_PROVENANCE.md`
2. `RUN_LOG.md`
3. `src/features.py`
4. `src/char_tokenizer.py`
5. `src/train_lightgbm.py`
6. `src/train_charcnn.py`
7. `src/select_ensemble.py`
8. `results/ensemble_config.json`
9. `results/final_test_metrics.json`
10. `app/inference.py`

The two most useful visual files to keep open alongside this top-ten proof list are `results/plots/confusion_matrix.png` and `results/plots/model_comparison.png`.

## 12. Final Submission Status

- Human confirmation of any college-specific PDF layout, declaration, certificate, signature, or binding rules, because no special rubric has been supplied.
- Final instructor-specific submission checks, if any.

The final 17-page PDF exists at `docs/AI_Based_Phishing_Detection_System_KLH_Bachupally.pdf`. No ML, data, metric, test-source, application, screenshot, or academic-report artifact is missing from the completed core project.

---

## Final Audit Output

FINAL COLLEGE SUBMISSION AUDIT

Project completeness: COMPLETE<br>
Core ML pipeline: COMPLETE<br>
Held-out evaluation: COMPLETE<br>
Frontend: COMPLETE<br>
Tests: PASS<br>
Academic report draft: COMPLETE<br>
Evidence sufficient for professor demonstration: YES<br>
Frontend screenshots currently available: YES

Missing items before final PDF:

- Confirmation of any college-specific formatting requirements.
- Any instructor-specific submission forms or signatures, if required.

Recommended final PDF figures:

- `results/plots/confusion_matrix.png`
- `results/plots/roc_curve.png`
- `results/plots/precision_recall_curve.png`
- `results/plots/model_comparison.png`
- `results/plots/feature_importance.png`
- `results/plots/probability_distribution.png`

Frontend screenshots included:

- `docs/assets/frontend_legitimate.png`
- `docs/assets/frontend_phishing.png`

Most important VS Code proof files:

1. `DATA_PROVENANCE.md`
2. `RUN_LOG.md`
3. `src/features.py`
4. `src/char_tokenizer.py`
5. `src/train_lightgbm.py`
6. `src/train_charcnn.py`
7. `src/select_ensemble.py`
8. `results/ensemble_config.json`
9. `results/final_test_metrics.json`
10. `app/inference.py`

Exact demo commands:

```bash
source .venv/bin/activate
pytest
uvicorn app.main:app --reload
```

Recommended submission package:

- Formal academic PDF after the remaining presentation steps.
- Clean source/evidence archive with code, tests, docs, provenance, run log, saved models, metrics, predictions, plots, requirements, and permitted data artifacts.
- Live local demonstration using saved models, with an offline screenshot and plot fallback.

READY TO DECIDE FINAL PDF CONTENT: YES
