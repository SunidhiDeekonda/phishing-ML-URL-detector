# AI-Based Phishing Detection System Using Ensemble Learning

An academic phishing URL detector that combines a Character-Level CNN with a calibrated LightGBM model through a probability ensemble. The project includes the complete preprocessing pipeline, domain-separated splits, trained models, held-out evaluation, automated tests, a FastAPI demonstration application, and the final academic report.

## Architecture

```text
URL string
  +-- Character encoding (200) --> Char-CNN --------+
  +-- 36 engineered URL features --> LightGBM ------+--> weighted probability --> verdict
```

The application analyses each URL only as text. It does not perform DNS lookups, fetch webpages, execute scripts, or visit submitted destinations.

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

```bash
git clone https://github.com/dingdong-vamshi/phishing-ML-URL-detector.git
cd phishing-ML-URL-detector
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Testing

```bash
pytest
```

The final audited suite contains 24 passing tests and 0 failures.

## Run the local demo

```bash
source .venv/bin/activate
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000` and submit a URL string for local inference. The trained artifacts are included, so the demo does not require retraining.

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
