# Versioned Evidence for Final Academic Outputs

These are copies of saved audit/experiment summaries, not newly trained models or replacement production results. Original files remain in their original checkouts. They are deliberately outside `models/` and the production registry.

| File | Provenance / interpretation |
|---|---|
| `latest_model_verification.json` | Prior replay and reconstructed v3.3 integrity audit against production source. |
| `v3_4_protocol.json` | Frozen local candidate protocol; local research commit `5543066`. |
| `charcnn_bilstm_v3_4_metrics.json` | Candidate's own validation and 31,795-row test, not v3.3's original test. |
| `v3_4_evaluation.json` | Common-input old/new replay and failed validation gate; saved with local research commit `287554c`. |
| `v3_4_model_metadata.json` | Candidate configuration metadata only; not a deployable model. |
| `latest_model_comparison.csv` | Versioned cohort rows; compare only matching cohort definitions, with training-exposure caveats. |
| `software_test_summary.json` | Current production-source branch's 91-test result; prior 107-test research run explicitly separate. |
| `api_smoke_summary.json` | Local/current-public endpoint status checks; no valid fabricated production feedback. |
| `output_validation.json` | Final artifact hashes, page counts, image resolution and format/visual acceptance. |

## Artifact identities

Production v3.3 ONNX SHA-256: `b7020bed0c89399e72b5e374c631c3553ec6fb2424c6b8de22cd70366da6b783`.

Local v3.4 ONNX SHA-256: `3ef86faede8d81a8f4aa1acca7fc91f199aa62b12398fe9897501688fbeb9df5`.

Hashes identify the evaluated files; they do not imply that the candidate binary is included here. The full candidate source, model checkpoints, split/per-row artifacts and temporary replay helpers remain local. The local commit identifiers above are provenance identifiers, not promised publicly reachable commits. This release supports inspection of saved claims but does not ship a complete independently executable v3.4 experiment.

## Comparability limits

The 11,271-row fresh2 comparison excludes two strings exceeding the API length limit. It is an inspected snapshot with two v3.3 cleaned training-input matches and registered-domain exposure. The v3.4 split repairs domain separation but reserves previously evaluated domains, so it is not a newly prospective blind test. Different training histories prevent interpreting the replay as a controlled architecture ablation. The candidate failed promotion; v3.3 remains deployed by the earlier maintainer decision.
