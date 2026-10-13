# Latest Research Claim Audit

Audit date: 2026-10-13. Production source: `origin/main` at `00c15d37fd62b18dc24018aa7fd4871e8ddf66df`. Output branch: `docs/verified-output-refresh`. No training, threshold adjustment, promotion or deployment occurs in this refresh.

All percentages below are rounded from saved evidence; counts are exact. A cohort label is mandatory. Historical reference/reproduction values are not current production claims.

| Claim | Experiment/version | Source | Verified value | Output locations |
|---|---|---|---|---|
| Current architecture | Production v3.3 | `src/charcnn_bilstm_model.py`, `app/inference.py` | Parallel CharCNN + two-layer BiLSTM + additive attention/max pool; 1,138,434 parameters | Paper Methodology; poster architecture; report Ch.4; viva latest-model section |
| Recurrent dimensions | v3.3 and candidate v3.4 | Same source; `evidence/v3_4_model_metadata.json` | 128 hidden per direction, 2 layers, bidirectional; embedding 64; head 512/128/32/1 | Paper/report architecture; poster boxes |
| Encoding/threshold | v3.3/v3.4 | `src/charcnn_bilstm_model.py`, `app/inference.py` | First 256 characters, 97 indices; sigmoid threshold 0.5 | Paper/report methodology; poster |
| Production artifact | v3.3 | `app/inference.py`; live `/health` | `models/charcnn_bilstm_v3_3.onnx`; `charcnn-bilstm-attention-v3.3` | Paper/report deployment; README/viva |
| Candidate status | Local v3.4 | `evidence/v3_4_evaluation.json` | Promotion gate FAILED; not deployed | All current academic outputs |
| Maintainer override | v3.3 | `results/fresh2_snapshot_evaluation.json`; live `/model-info` | Required 2-point recall gain not met; served by maintainer decision | Paper limitations; report; poster abstract; viva |
| Old private-domain overlap | Reconstructed v3.3 split | `evidence/latest_model_verification.json` | Train-val 385; train-test 400; val-test 187 | Paper/report integrity discussion; viva; poster discloses overlap without these counts |
| Cleaned-input contamination | v3.3 fresh2 | Same source | 2 cleaned input matches; 2,555 rows on training registered domains | Paper/report caveat; README/viva; poster mentions two matches |
| Candidate data | v3.4 | `evidence/v3_4_protocol.json` | 64,000 raw; 63,869 retained; train 25,961 / val 6,113 / test 31,795 | Paper/report data; poster candidate card |
| Candidate split overlap | v3.4 | Same source | 0/0/0 for cleaned key, hostname, public and private registered domains | Paper/report; poster candidate card; viva |
| Own test metrics | v3.4, 31,795 rows | `evidence/charcnn_bilstm_v3_4_metrics.json` | Accuracy 92.876%; precision 95.455%; recall 90.696%; F1 93.015%; ROC-AUC 98.289%; AP 98.419%; FP 718; FN 1,547 | Paper/report tables; poster shows accuracy/recall/AUC only |
| Common external metrics | v3.3, 11,271 rows | `evidence/v3_4_evaluation.json`: fresh2_api_compatible | Accuracy 91.740%; precision 93.713%; recall 91.277%; F1 92.479%; ROC-AUC 96.489%; AP 97.078%; FP 384; FN 547 | Paper/report tables; poster chart/status line; viva |
| Common external metrics | v3.4, same 11,271 rows | Same source | Accuracy 90.578%; precision 94.468%; recall 88.232%; F1 91.243%; ROC-AUC 96.276%; AP 97.207%; FP 324; FN 738 | Paper/report tables; poster chart; viva |
| External tradeoff | v3.4 minus v3.3 | Same source | 60 fewer FP; 191 more FN; no universal improvement | All current outputs |
| Adversarial replay | Historical 3,998-row suite, v3.3 | Same source | Recall 94.035%; FN 119; FP 30 | Paper/report robustness figure; viva |
| Adversarial replay | Same suite, v3.4 | Same source | Recall 91.328%; FN 173; FP 12 | Paper/report robustness figure; viva |
| Initial selected ensemble | Historical 20,000-URL reproduction | `results/final_test_metrics.json` | Accuracy 99.525%; precision 100%; recall 99.048%; ROC-AUC 99.931% | Historical poster chart/table; historical report sections; paper background |
| Earlier robustness gain | Historical 95/5 ensemble extension | `results/robustness_final_metrics.json` | Adversarial FN 37 to 17; NOT v3.3/v3.4 | Poster explicitly historical; historical report/viva sections |
| Context scope | Current app | `src/context_features.py`, `app/research_api.py` | Independent heuristic HTML/email evidence; not probability fusion or webpage/mailbox fetching | All current outputs |
| Adaptation scope | Current app | `src/continuous_learning.py`, `src/drift_monitor.py` | Human review, quarantine, registry and PSI; no automatic online training | All current outputs |
| Tests | Production-source docs branch | Current `pytest -q`; private JUnit receipt | 91 passed, 0 failed; 2 dependency warnings | Paper/report testing; poster badge; README/viva |
| Earlier research tests | Local candidate branch only | Earlier research JUnit receipt | 107 passed, 0 failed; not the production-source count | Paper/report caveat; audit documentation |

## Interpretation gate

No reported v3.4 value is called a production result. No common-snapshot comparison is called a matched architecture ablation. No historical 99.x% result is promised for current external URLs. v3.3 contamination, maintainer override, candidate gate failure and limited candidate reproducibility are disclosed. No new claim of calibrated HTML/email detection, online learning efficacy or universal attack resistance is made.
