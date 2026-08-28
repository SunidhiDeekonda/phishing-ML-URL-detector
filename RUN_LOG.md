# Project Run Log

This public run log preserves the final audited outcomes of the completed reproduction. Repeated early setup stack traces and machine-specific absolute paths were removed during repository publication cleanup; no experimental result was changed.

## Phase 1: Data Preparation

- Frozen PhishX source snapshots were recorded in `DATA_PROVENANCE.md`.
- Cleaned source availability: 49,363 unique phishing URLs and 50,000 legitimate URLs.
- Deterministic balanced sample with seed 42: 10,000 phishing and 10,000 legitimate URLs.
- Processed dataset: 20,000 rows.
- Engineered feature matrix: 36 numeric features.
- Character sequences: `(20000, 200)`.
- Final tokenizer metadata: PAD 0, explicit characters 1-49, UNK 50.
- All submitted and dataset URLs were handled only as strings; no destination was visited.

## Reproducibility Corrections Before Training

- Replaced the initial dynamic tokenizer with the released 49-character PhishX alphabet.
- Audited the initial stratified split and detected root-domain overlap.
- Regenerated only the split indices using deterministic domain-grouped splitting with seed 42.
- Final split: 14,002 train, 2,000 validation, and 3,998 held-out test rows.
- Final train-validation, train-test, and validation-test root-domain overlaps: 0, 0, and 0.
- All model training occurred after these corrections.

FINAL PRE-TRAINING AUDIT
-------------------------
Timestamp: 2026-08-28 22:31:00

LightGBM version: 4.7.0
MPS available: True

Tokenizer:
PAD: 0
First explicit index: 1 (a)
Last explicit index: 49 (~)
UNK: 50
char_sequences shape: (20000, 200)
Explicit character mapping count: 49

Domain leakage before correction:
- train-val overlap: 153
- train-test overlap: 232
- val-test overlap: 108

Were splits regenerated: YES

Final split:
Train rows: 14002
Train class counts: {0: 6997, 1: 7005}

Validation rows: 2000
Validation class counts: {0: 1000, 1: 1000}

Test rows: 3998
Test class counts: {0: 2003, 1: 1995}

Final domain overlap:
train-val: 0
train-test: 0
val-test: 0

Tests passed: 18
Tests failed: 0

READY FOR PHASE 2: YES

PHASE 2A LIGHTGBM
------------------
Timestamp: 2026-08-28 22:29:56
LightGBM version: 4.7.0
Training configuration: objective=binary, n_estimators=1000, learning_rate=0.05, num_leaves=64, min_child_samples=20, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
Training duration (s): 0.8420972919993801
Best iteration: 88
Requested max estimators: 1000
Early stopping triggered: YES

RAW validation metrics:
accuracy: 0.991
precision: 1.0
recall: 0.982
f1: 0.9909182643794148
roc_auc: 0.9957699999999998
pr_auc: 0.9973232843288381
Raw confusion matrix [tn, fp, fn, tp]: [1000, 0, 18, 982]

Calibrated validation metrics:
accuracy: 0.99
precision: 0.9979674796747967
recall: 0.982
f1: 0.9899193548387096
roc_auc: 0.9957699999999998
pr_auc: 0.9973232843288381
Calibrated confusion matrix [tn, fp, fn, tp]: [998, 2, 18, 982]

Top 10 features:
- hostname_entropy: 1100
- vowel_fraction: 1069
- host_length: 1011
- url_length: 536
- letter_count: 412
- hyphen_count: 287
- num_dots: 242
- special_char_count: 202
- digit_count: 190
- digit_letter_ratio: 163

Artifacts:
- Raw model: models/lightgbm_raw.pkl
- Calibrated model: models/lightgbm_calibrated.pkl
- Metrics: results/lightgbm_validation_metrics.json
- Validation predictions: results/lightgbm_validation_predictions.csv
- Feature importance: results/lightgbm_feature_importance.csv
PHASE 2B CHAR-CNN
------------------
Timestamp: 2026-08-28 22:50:14
Torch version: 2.13.0
Device requested: MPS
Device actually used: MPS
Batch size: 64
Learning rate: 0.001
Epochs completed: 7
Best epoch: 5
Training duration (seconds): 16.898632499998712
Early stopping triggered: YES
Validation metrics for best epoch:
- accuracy: 0.988
- precision: 0.9979591836734694
- recall: 0.978
- f1: 0.9878787878787879
- roc_auc: 0.996759
- pr_auc: 0.997759016004921
Confusion matrix (TN FP FN TP):
- TN: 998
- FP: 2
- FN: 22
- TP: 978
Saved model: ./models/char_cnn.pt
Saved training history: ./results/charcnn_training_history.json
Saved validation predictions: ./results/charcnn_validation_predictions.csv
Tests (full pytest): 18 passed, 0 failed
Sequence shape check: PASS (20000, 200)
Max token index check: PASS (<= 50)
No negative token indices: PASS
Model vocab_size check: PASS (51)
Validation probability check: PASS (all finite, in [0,1])
Validation rows check: PASS (2000)
Confusion matrix sum check: PASS (2000)
Files existence check: PASS
Test split untouched during training/eval: PASS
Fallbacks/warnings:
- No fallback required; MPS training executed successfully.
Data splits:
Train rows: 14002
Train class counts: {0: 6997, 1: 7005}
Validation rows: 2000
Validation class counts: {0: 1000, 1: 1000}
Test rows: 3998
Test class counts: {0: 2003, 1: 1995}
Domain overlap checks:
- train-val: 0
- train-test: 0
- val-test: 0
Ready for Phase 3? YES
PHASE 2C ENSEMBLE SELECTION
---------------------------
Timestamp: 2026-08-28 23:06:30
Validation rows aligned by row_index: 2000

Requirements compliance:
- Test set was not loaded for model evaluation
- No retraining performed
- No threshold tuning performed (fixed at 0.5)

Alignment checks:
- Matched rows after join: 2000
- True label agreement between files: PASS
- Duplicate row_index values: None
- Missing probabilities: None
- Probability bounds [0,1]: PASS

Model comparison (validation only):
1) Calibrated LightGBM
  Accuracy: 0.990000
  Precision: 0.997967
  Recall: 0.982000
  F1: 0.989919
  ROC-AUC: 0.995770
  PR-AUC: 0.997323
  FP: 2, FN: 18

2) Char-CNN
  Accuracy: 0.988000
  Precision: 0.997959
  Recall: 0.978000
  F1: 0.987879
  ROC-AUC: 0.996759
  PR-AUC: 0.997759
  FP: 2, FN: 22

3) Reference Ensemble (cnn=0.60, lgbm=0.40)
  Accuracy: 0.989000
  Precision: 0.997963
  Recall: 0.980000
  F1: 0.988900
  ROC-AUC: 0.996741
  PR-AUC: 0.997760
  FP: 2, FN: 20

4) Best Ensemble (validation ROC-AUC search)
  Best weights: cnn=0.95, lightgbm=0.05
  Accuracy: 0.988000
  Precision: 0.997959
  Recall: 0.978000
  F1: 0.987879
  ROC-AUC: 0.997066
  PR-AUC: 0.997924
  FP: 2, FN: 22

Artifacts:
- Weight search CSV: ./results/ensemble_weight_search.csv
- Ensemble config JSON: ./results/ensemble_config.json

## PHASE 2D: FINAL HELD-OUT TEST EVALUATION
- Status: completed (evaluation only, no training)
- Timestamp: 2026-08-28
- Evaluated artifacts:
  - `models/lightgbm_calibrated.pkl`
  - `models/char_cnn.pt`
  - `results/lightgbm_feature_importance.csv`
  - `data/processed/{dataset.csv, features.csv, char_sequences.npy, train_idx.npy, val_idx.npy, test_idx.npy}`

Character-CNN architecture used for deterministic recomputation:
- vocab_size: 51
- embedding_dim: 16
- conv channels: 128 per kernel size [3,5,7]
- pooled width: adaptive max pool
- head MLP: 384 -> 64 -> 1
- dropout: 0.3
- output: sigmoid
- ensemble weights:
  - reference ensemble: 60% CNN + 40% LightGBM
  - selected ensemble: 95% CNN + 5% LightGBM
- threshold: 0.50 (fixed)

Data split sizes at evaluation time:
- Train: 14002
- Validation: 2000
- Test: 3998
- Character sequence shape: (20000, 200)

Class counts by split:
- Train: {0: 6997, 1: 7005}
- Validation: {0: 1000, 1: 1000}
- Test: {0: 2003, 1: 1995}

Test set metrics:
- dataset size: 20000
- test size: 3998
- tests threshold: 0.50

1) Calibrated LightGBM
- Accuracy: 0.995997999
- Precision: 0.997486174
- Recall: 0.994486216
- F1: 0.995983936
- ROC-AUC: 0.99851063
- PR-AUC: 0.999077959
- Confusion matrix: TN=1998, FP=5, FN=11, TP=1984
- FPR: 0.002496256
- FNR: 0.005513784

2) Char-CNN
- Accuracy: 0.995247624
- Precision: 1.0
- Recall: 0.990476190
- F1: 0.995215311
- ROC-AUC: 0.999448196
- PR-AUC: 0.999552710
- Confusion matrix: TN=2003, FP=0, FN=19, TP=1976
- FPR: 0.000000000
- FNR: 0.009523810

3) Reference Ensemble (0.60 CNN / 0.40 LightGBM)
- Accuracy: 0.995747874
- Precision: 0.999494949
- Recall: 0.991979950
- F1: 0.995723270
- ROC-AUC: 0.998916663
- PR-AUC: 0.999281893
- Confusion matrix: TN=2002, FP=1, FN=16, TP=1979
- FPR: 0.000499251
- FNR: 0.008020050

4) Selected Ensemble (0.95 CNN / 0.05 LightGBM)
- Accuracy: 0.995247624
- Precision: 1.0
- Recall: 0.990476190
- F1: 0.995215311
- ROC-AUC: 0.999311809
- PR-AUC: 0.999479097
- Confusion matrix: TN=2003, FP=0, FN=19, TP=1976
- FPR: 0.000000000
- FNR: 0.009523810

Reference study check:
- Reference study accuracy: 99.819%
- Selected ensemble accuracy: 99.524762%
- Absolute difference: 0.294238 percentage points (lower)

Artifacts generated:
- `results/final_test_metrics.json`
- `results/model_comparison.csv`
- `results/test_predictions.csv`
- `results/plots/confusion_matrix.png`
- `results/plots/roc_curve.png`
- `results/plots/precision_recall_curve.png`
- `results/plots/model_comparison.png`
- `results/plots/feature_importance.png`
- `results/plots/probability_distribution.png`

Tests executed in this audit phase:
- `pytest -q` (pytest result: 18 passed)

Sanity checks:
- No retraining performed.
- No threshold tuning performed during held-out test pass.
- Domain and data leakage checks were not modified in this phase.

## PHASE 3A — DEMO APPLICATION REPORT

Backend
-------
FastAPI status: running on startup
LightGBM loaded: YES
Char-CNN loaded: YES
Device: mps
Prediction endpoint: POST /predict
Health endpoint: GET /health

Frontend
--------
Page created: YES (app/templates/index.html)
URL input: input field + "ANALYSE URL" button
Prediction result: verdict card with confidence and probabilities
Model probability breakdown: displayed
URL feature display: displayed in "URL Signals" panel
Research metrics display: displayed in expandable "About the Model" section with local experiment and 99.525% accuracy

Safety
------
Dataset/submitted URLs fetched externally: NO
DNS performed: NO
Webpage content downloaded: NO

Tests
-----
Health endpoint: PASS
Prediction endpoint: PASS
Frontend: PASS (homepage returns HTML and prediction page loads)
pytest passed: 24
pytest failed: 0

Example predictions
-------------------
https://www.google.com
- verdict: LEGITIMATE
- selected_ensemble_probability: 0.005776692357105774
- reference_ensemble_probability: 0.009786069864076466
- cnn_probability: 0.0052039241418242455
- lightgbm_probability: 0.016659288447454797

https://example.com
- verdict: LEGITIMATE
- selected_ensemble_probability: 0.0007568632851563902
- reference_ensemble_probability: 0.0058009744328315245
- cnn_probability: 0.00003627597834565677
- lightgbm_probability: 0.014448022114560324

https://accounts.google.com
- verdict: LEGITIMATE
- selected_ensemble_probability: 0.031162129578839466
- reference_ensemble_probability: 0.24876019569780633
- cnn_probability: 0.0000766915618441999
- lightgbm_probability: 0.6217854519017495

http://secure-account-login-example.xyz/verify
- verdict: PHISHING
- selected_ensemble_probability: 0.9999637071743265
- reference_ensemble_probability: 0.999710491859639
- cnn_probability: 0.9999998807907104
- lightgbm_probability: 0.9992764084630318

Files created/modified
---------------------
- app/main.py
- app/inference.py
- app/templates/index.html
- app/static/style.css
- app/static/app.js
- tests/test_phase3a_api.py
- README.md

Run command
-----------
source .venv/bin/activate
uvicorn app.main:app --reload

Local URL
---------
http://127.0.0.1:8000

READY FOR COLLEGE DEMO: YES

## Phase 3B — Final Academic Report Draft

- Created `docs/FINAL_REPORT.md` with the required title and 31 report sections.
- Kept the reference PhishX study separate from the independent 20,000-URL local reproduction.
- Verified report claims against the stored provenance, validation, ensemble, final-test, feature-importance, and run-log records already audited for this phase.
- Documented the corrected domain-separated split: 14,002 train, 2,000 validation, 3,998 test, with zero final root-domain overlap.
- Recorded the frozen selected ensemble weights as CNN 0.95 and LightGBM 0.05; no test-set tuning was claimed.
- Referenced 7 generated plots and created 12 numbered evidence tables.
- Included honest limitations, reproducibility instructions, and a human-review placeholder list.
- Did not retrain models, change artifacts or results, modify the frontend, or generate a PDF.
- Unresolved institutional placeholders: college, department, team member names, university IDs, guide, and academic year.
- Bibliographic follow-up: confirm full PhishX publication metadata and the exact Yang et al. citation from the supplied paper bibliography.
- Data-count note: frozen source snapshots contain 99,363 unique cleaned URLs, while the reference study corpus is reported approximately as 99,361; these are presented as distinct quantities.

READY FOR HUMAN REVIEW BEFORE PDF: YES
