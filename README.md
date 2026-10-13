# AI-Based Phishing Detection System Using Ensemble Learning

An academic phishing-detection project that began with a Character-Level CNN and calibrated LightGBM ensemble. The historical reproduction used domain-separated splits. The current deployed model is a different experiment: CharCNN + BiLSTM + additive attention v3.3, whose later integrity audit found split contamination. Do not apply the historical ensemble's accuracy or leakage guarantees to the deployed model.

The deployed API now serves a newer character-level CNN + BiLSTM + Attention model (`charcnn-bilstm-attention-v3.3`); see [Production Model V3.1](#production-model-v31-charcnn--bilstm--attention) and the sections after it. The ensemble described below is the original academic reproduction, and its scores are still returned for reference.

## Architecture

Served model (v3.3):

```text
URL string --> canonicalize --> characters (256) --> Conv1d 3/5/7 --> 2-layer BiLSTM --> attention + max pooling --> probability --> verdict
```

Original academic ensemble (reference scores only):

```text
URL string
  +-- Character encoding (200) --> Char-CNN --------+
  +-- 36 engineered URL features --> LightGBM ------+--> weighted probability
```

The application analyses each URL only as text. It does not perform DNS lookups, fetch webpages, execute scripts, or visit submitted destinations.

For a consistent prediction, equivalent root URLs are canonicalized before both
models run: `https://example.com/` is analysed as `https://example.com`.
Only an empty root-path slash is removed; path-ending slashes such as
`https://example.com/account/` are retained because they can identify a
different resource.

## Local reproduction experiment

- Dataset: 20,000 URLs, balanced as 10,000 legitimate and 10,000 phishing.
- Sampling seed: 42.
- Final split: 14,002 train, 2,000 validation, and 3,998 held-out test rows.
- Root-domain overlap after correction: zero across every split pair.
- Engineered features: 36.
- Character sequence length: 200.
- Selected ensemble weights: 0.95 Char-CNN and 0.05 LightGBM.

### Selected-ensemble held-out results

| Metric | Result |
|---|---:|
| Accuracy | 99.525% |
| Precision | 100.000% |
| Recall | 99.048% |
| F1 | 99.522% |
| ROC-AUC | 99.931% |

These are results from this independent 20,000-URL reproduction. They are not the results of the reference PhishX study, which reports approximately 99,361 URLs and 99.819% accuracy.

## Repository contents

| Path | Purpose |
|---|---|
| `src/` | Feature extraction, tokenization, model training, ensemble selection, and final evaluation. |
| `app/` | FastAPI inference service and browser interface. |
| `tests/` | Preprocessing, leakage, tokenizer, API, and frontend tests. |
| `data/` | Frozen source snapshots and deterministic processed artifacts. |
| `models/` | Saved calibrated LightGBM and Char-CNN artifacts. |
| `results/` | Validation history, ensemble search, test predictions, metrics, and plots. |
| `docs/` | Academic Markdown report, evidence audit, screenshots, and final PDF. |

## Installation

### Run the demo/API

```bash
git clone https://github.com/SunidhiDeekonda/phishing-ML-URL-detector.git
cd phishing-ML-URL-detector
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The demo uses compact ONNX artifacts through ONNX Runtime: `models/charcnn_bilstm_v3_3.onnx` for the verdict, and `models/char_cnn_production_v2.onnx` / `models/lightgbm_production_v2.onnx` for the reference ensemble scores. It does not install PyTorch or the native LightGBM library at runtime.

### Train, export models, or run the full test suite

```bash
python -m pip install -r requirements-dev.txt
```

## Testing

```bash
pytest
```

The current suite contains 91 passing tests and 0 failures.

## Run the local demo

```bash
source .venv/bin/activate
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000`. The trained artifacts are included, so the demo does not require retraining.

## Three independent analysis tools

The browser application now separates three capabilities:

1. **URL Phishing Detector** - the character-level CNN + BiLSTM + Attention model (v3.3); the original CNN + LightGBM ensemble scores are shown for reference. It accepts only a URL string and never visits the URL.
2. **HTML Phishing Context Analyzer** - a network-free research extension that parses manually pasted HTML text for forms, password inputs, external targets, iframes, scripts, meta refresh, and credential-related language. `POST /analyze-html` does not require a URL and does not execute HTML or JavaScript.
3. **Email Phishing Context Analyzer** - a network-free research extension that inspects manually pasted email text for implemented risk terms, credential requests, exact call-to-action phrases, and URL strings. `POST /analyze-email` does not require a URL and never connects to a mailbox.

HTML and email results are qualitative context evidence, not CNN/LightGBM probabilities. They are not fused into the validated URL ensemble because this project does not have a properly labelled content dataset for calibrated multimodal training.

## Deploy to Vercel

Import `SunidhiDeekonda/phishing-ML-URL-detector` with the FastAPI preset, use the repository root (`./`), and leave build-command, output-directory, and environment-variable overrides empty. Vercel installs `requirements.txt` automatically. The repository pins Vercel to Python 3.12 and uses ONNX Runtime instead of PyTorch in production.

- Application entry point: `app.main:app`.
- Required environment variables: none.
- Build logs: Vercel project -> Deployments -> select a deployment -> Build Logs.
- Runtime logs: Vercel project -> Logs.
- CI logs: GitHub repository -> Actions -> Reproducibility CI.

See [Vercel deployment guide](docs/VERCEL_DEPLOYMENT.md) for the complete setup.

## Evidence and reports

- [Dataset provenance](DATA_PROVENANCE.md)
- [Experiment run log](RUN_LOG.md)
- [Final held-out metrics](results/final_test_metrics.json)
- [Academic report source](docs/FINAL_REPORT.md)
- [Final academic PDF](docs/AI_Based_Phishing_Detection_System_KLH_Bachupally.pdf)
- [Submission and demonstration audit](docs/SUBMISSION_AUDIT.md)

## Reference work

R. Dubey, A. M. Tripathi, A. Srivastava, and S. Singh, “Phishing Detection System: An Ensemble Approach Using Character-Level CNN and Feature Engineering,” arXiv:2512.16717, 2025. Reference implementation: [PhishX](https://github.com/dubeyrudra-1808/PhishX).

## Academic attribution

- College: KLH Bachupally
- Department: DEPARTMENT OF CSIT
- Guide: Dr. K Venkateshwara Rao
- Team: 2320090017 - D Sunidhi; 2320090060 - P Manvitha; 2320090069 - G Likith

## Research Extensions Beyond Reference Reproduction

Eight deterministic offline mutation families were evaluated. Only 1751 training-derived phishing mutations entered robust training; validation froze selection before the 3998-row adversarial test.

Original adversarial recall was 98.145% with 37 false negatives. The robust candidate achieved 99.148% with 17 false negatives and 99.625% clean accuracy. Validation selected `robust_candidate_95_5`. These are synthetic-mutation results, not universal protection.

Caller-supplied HTML/email produces separate context evidence and is never fetched or mixed into validated probability. Feedback is quarantined for human verification, deduplicated, protected from test contamination, and intended for explicit offline retraining. PSI and model versioning support controlled adaptation. Vercel local feedback storage is ephemeral unless external persistence is configured.

Human review and candidate-data preparation are explicit: `python scripts/prepare_verified_retraining.py --review-record RECORD_ID --decision approved`. This command never trains or promotes a model automatically; candidate evaluation must pass the recorded validation gate first.

## Production Reliability and URL Canonicalization

Production inference now uses one network-free canonicalization contract before both models: whitespace is trimmed, a missing scheme becomes HTTPS, scheme and hostname casing are normalized, default HTTP/HTTPS ports are removed, and only an equivalent root `/` is removed. Query strings, fragments, percent encoding, and non-root paths (including meaningful trailing slashes) are preserved. This is a reliability fix, not scientific novelty and not a domain whitelist. External/OOD results are generated by `python scripts/run_external_url_audit.py`; the student guide is [HOW_TO_TEST_THE_APP.md](docs/HOW_TO_TEST_THE_APP.md).

## Production Model V2

A post-experiment reliability audit found that the original dataset accidentally made path presence a near-perfect class cue: legitimate rows were roots, while almost all phishing rows contained paths. Production V2 adds partitioned legitimate hard paths and adversarial training, then applies a validation gate before held-out evaluation. It does not whitelist GitHub, alter the 0.5 threshold, overwrite historical models, or revise original academic metrics. See [GITHUB_FALSE_POSITIVE_ANALYSIS.md](docs/GITHUB_FALSE_POSITIVE_ANALYSIS.md).

## Production Model V3.1: CharCNN + BiLSTM + Attention

The API serves `charcnn-bilstm-attention-v3.3` (`models/charcnn_bilstm_v3_3.onnx`; earlier artifacts are `models/charcnn_bilstm_hostsplit.onnx` and `models/charcnn_bilstm.onnx`). It uses parallel character CNN kernels 3/5/7, a two-layer bidirectional LSTM, additive attention and temporal max pooling. Training used the V2 dataset plus v3.3 augmentation with hostname-based grouping; this did not establish registered-domain separation. The later reconstruction found overlap even between cleaned hostname groups. Cleaning uses `src/dataset_cleaning.py`; character encoding uses the same `canonicalize_url` contract as inference. No domain allowlist or post-hoc probability adjustment is applied. The previous ensemble scores remain reference outputs and do not decide the verdict. See the [claim audit](docs/LATEST_RESEARCH_CLAIM_AUDIT.md) for the version-specific evidence.

The label audit (`python scripts/audit_dataset_labels.py --evidence-dir <dir>`, output in `results/dataset_label_audit.json`) found no reversed or inconsistent label mappings in either dataset, but did find class-correlated formatting artefacts and that about 48% of V2 legitimate rows are generated URLs (`<PhishX root> + /login` etc.) whose existence was never verified.

The like-for-like comparison is in `results/production_model_comparison.json` (`python scripts/compare_production_models.py`). The new model has far fewer false positives on legitimate URLs with paths and on the hard-negative suite, but lower phishing recall than the previous ensemble on the previous held-out test, its adversarial variants, and the external OpenPhish/Phishing.Database feeds, and more false positives on popular root domains. It is not an across-the-board improvement.

A second, pre-registered run (`results/hostsplit_preregistration.json`) grouped the split by hostname instead (`python scripts/train_bilstm.py --group-by hostname --output-tag _hostsplit`) and was judged on 11,000 URLs that no model had scored before (`python scripts/evaluate_fresh_snapshot.py`, results in `results/fresh_snapshot_evaluation.json`). It caught more phishing than v3.1 (Phishing.Database recall 90.2% vs 81.1%) but flagged more legitimate URLs (12.6% vs 9.0%), so it failed the pre-registered false-positive limit. The maintainer chose to serve it anyway (`models/charcnn_bilstm_hostsplit.onnx`, best fresh-snapshot F1 0.904 and ROC-AUC 0.948 of the three models); v3.1 is kept as the previous version. Neither BiLSTM matches the previous ensemble's phishing recall (95.0%), although the ensemble flags 42.7% of the fresh legitimate URLs.

v3.3 adds 24,000 source-labelled URLs (`scripts/build_augmentation_data.py`, `data/processed/augmentation_v3_3.csv`) to address the V2 dataset's generated legitimate paths and observed errors. The second pre-registered snapshot contains 11,273 strings, of which 11,271 satisfy the API length limit. On those compatible rows, v3.3 has 384 false positives and 547 false negatives (accuracy 91.740%, recall 91.277%). It missed the pre-registered two-percentage-point recall gain; the maintainer chose to serve it anyway. The later audit found two cleaned training-input matches and 2,555 evaluation rows on training registered domains. The earlier description "never-used" is therefore withdrawn. This is a previously inspected observational benchmark, not proof of unseen-domain generalization. Known false positives include official login endpoints; a URL-text model cannot verify domain ownership.

## Verified output refresh: October 2026

| Experiment | Status | Interpretation |
|---|---|---|
| Initial 20,000-URL CNN-LightGBM reproduction | Historical | 99.525% accuracy belongs only to that frozen experiment. |
| CharCNN-BiLSTM-attention v3.3 | Current production | Integrity issues and maintainer promotion override disclosed. |
| Same architecture, domain-separated v3.4 | Local research candidate; not promoted | 92.876% accuracy / 90.696% recall on its own 31,795-row test; not a production score. |

On a common 11,271-row external snapshot, v3.4 reduces false positives from 384 to 324 but increases false negatives from 547 to 738. The different training histories and prior inspection limit causal conclusions. No application/model changes or deployment are part of this documentation-only release. The production-source test suite passed **91 tests**; 107 tests belongs to a separate local research-branch run, not this branch.

Final materials:
- [Five-page IEEE paper](paper/AI_Phishing_Research_Paper_Updated.pdf) and [LaTeX/build notes](paper/README.md).
- [Editable poster](output/poster/AI_Phishing_Detection_Project_Poster_UPDATED.pptx), [print PDF](output/poster/AI_Phishing_Detection_Project_Poster_UPDATED.pdf), [PNG](output/poster/AI_Phishing_Detection_Project_Poster_UPDATED.png).
- [Updated 30-page college report](docs/AI_Based_Phishing_Detection_System_KLH_Bachupally_UPDATED.pdf).
- [Claim audit](docs/LATEST_RESEARCH_CLAIM_AUDIT.md), [output decisions](docs/LATEST_OUTPUT_REFRESH_AUDIT.md), [local-change inventory](docs/LOCAL_CHANGE_CLASSIFICATION.csv), and [viva guide](docs/VIVA_DEFENSE.md).
