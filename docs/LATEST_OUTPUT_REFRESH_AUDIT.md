# Latest Output Refresh: Final Decision Audit

Date: 2026-10-13. Scope: necessary final academic outputs only, not model development.

## Remote and branch decision

- Remote: `https://github.com/SunidhiDeekonda/phishing-ML-URL-detector.git`.
- Fetched main: `00c15d37fd62b18dc24018aa7fd4871e8ddf66df`; no newer remote source changes were found.
- Approved output branch: `docs/verified-output-refresh`, created directly from that main commit.
- Original checkout's local main and the local `codex/domain-separated-v3-4` branch were not reset, rebased, merged, deleted or pushed.
- Research commits `5543066` and `287554c` include candidate implementation/model artifacts. They are deliberately not ancestors of this output branch.
- Current production remains `charcnn-bilstm-attention-v3.3`, artifact `models/charcnn_bilstm_v3_3.onnx`: CharCNN, two-layer bidirectional LSTM, additive attention and max pooling.
- v3.4 uses the same architecture, repaired whole-domain splits, and a failed promotion gate. It remains a local research candidate, not a replacement deployment.

## Complete local-change classification

`LOCAL_CHANGE_CLASSIFICATION.csv` inventories all **256 pre-existing local/unpushed paths**: 45 required (selective copying/editing), 153 optional (kept local), 58 temporary (not published), zero discarded. Every row records purpose, destination, push decision and action. Nothing was deleted. The inventory distinguishes a required output from permission to copy all prior edits in its file.

`OUTPUT_REFRESH_FILE_MANIFEST.csv` separately records every new/modified file in this documentation release and every new temporary/intermediate path excluded from it. `PUSHED_FILES.txt` gives the exact publication manifest. The inventories, rather than an abbreviated chat list, are the exhaustive file-level record.

| Change group | Classification | Push? | Action |
|---|---|---|---|
| Corrected paper source, bibliography, three already-generated figure pairs, final PDF | REQUIRED | Yes | Retain scientific story; fix column tables and scope wording. |
| Updated poster PPTX/PDF/PNG | REQUIRED | Yes | Retain layout/branding/team/historical chart; replace outdated architecture and current comparison panel. |
| Updated report PDF, template source and used images | REQUIRED | Yes | Preserve formatting; shorten repetition from the prior 35-page draft to 30 pages. |
| Targeted README/provenance/report/novelty/viva corrections | REQUIRED | Yes | Preserve historical bodies and label their scope. |
| Saved candidate protocol/metrics/comparison metadata and integrity summaries | REQUIRED | Yes, evidence only | Allow claim inspection; no candidate model binaries, activation or executable experiment package. |
| Candidate training code, tests, datasets, checkpoints, ONNX/PT binaries and per-row outputs | OPTIONAL | No | Leave on local research branch. |
| Temporary smoke/replay/export helpers, JUnit XML, intermediate PDFs/logs, chart staging directories and caches | TEMPORARY | No | Retain locally; never stage them. |
| Existing historical figures, frozen results and current manual app guide | UNCHANGED | No change | They still describe their stated experiment or current three-tool UI. |

## Artifact decision matrix

| Artifact | Old version still valid? | Update required? | Why / action | Final file |
|---|---|---|---|---|
| Research paper | Partially; historical paper remains valid for its original experiment | Yes | Current description needs v3.3/v3.4 distinction and integrity limitations. Existing updated draft already had this story; only table fitting and scope clarifications were needed. | `paper/AI_Phishing_Research_Paper_Updated.pdf` |
| Expo poster PPTX | Partially | Yes | Current architecture panel was historical CNN-LightGBM. Preserve layout/historical evidence; show actual recurrent architecture, candidate status and error tradeoff. | `output/poster/AI_Phishing_Detection_Project_Poster_UPDATED.pptx` |
| Expo poster PDF | Partially | Yes | Export corresponding edited source. | `output/poster/AI_Phishing_Detection_Project_Poster_UPDATED.pdf` |
| Expo poster PNG | Partially | Yes | Export corresponding edited source. | `output/poster/AI_Phishing_Detection_Project_Poster_UPDATED.png` |
| Project report PDF | Partially | Yes | Disclose current architecture/integrity/gate; shorten repeated explanations without redesigning the college template. | `docs/AI_Based_Phishing_Detection_System_KLH_Bachupally_UPDATED.pdf` |
| Viva guide | Partially | Yes | Add latest-model questions, numerical tradeoffs and evidence paths; keep historical answers under a scope warning. | `docs/VIVA_DEFENSE.md` |
| README | Partially | Yes | Correct registered-domain/never-used claims; link version-labelled outputs. | `README.md` |
| Figures | Historical plots and existing verified comparison figures remain valid | Only minimum publication use | Reuse three paper figures and six report figures; only the poster's obsolete current-results chart is replaced with saved FP/FN values. No diagnostic plot proliferation. | `paper/figures/`, `docs/report_updated/images/` |
| Manual app guide | Yes | No | Current URL/HTML/email workflow and endpoints did not change. | `docs/HOW_TO_TEST_THE_APP.md` unchanged |

## Paper section decisions

| Section | Classification | Decision |
|---|---|---|
| Title/authors, abstract, introduction, related work | UNCHANGED - KEEP in existing updated draft | Correctly frames a versioned reproducibility study. Historical predecessor is not used as current truth. |
| Methodology | UNCHANGED - KEEP scientific content | Actual recurrent architecture, corrected data and historical ensemble are already separated. |
| Experimental setup | OUTDATED - UPDATE scope wording only | Identify the earlier local candidate run, no training in this refresh, and unavailable candidate implementation in the docs-only package. |
| Results/tables | UNCHANGED values; FORMAT FIX | Narrow table cell spacing to fit IEEE column width. Do not retune or change values. |
| Operational/testing paragraph | WRONG scope if one count is used universally - FIX | 91 production-source tests versus 107 in separate local research branch. |
| Limitations/discussion/conclusion | UNCHANGED - KEEP | Gate failure, contamination, prior inspection, unequal exposure and no causal architecture claim are explicit. |
| Additional diagnostic figures or a new architecture experiment | OPTIONAL - LEAVE | Not required by current claims. |

## Code push decision

**No application/model code push was required.** No edits to `app/`, `src/`, model weights, registry, datasets, thresholds, tests, dependency declarations or deployment configuration are included. v3.4 binary/code commits were not pushed or deployed. Saved research evidence is included only to support documented claims. A documentation branch push is not a production promotion.

## Scientific outcomes retained

The candidate's own test is 31,795 rows: accuracy 92.876%, recall 90.696%, ROC-AUC 98.289%, FP 718, FN 1,547, with zero cross-split registered-domain overlap. On the same 11,271 compatible external inputs, v3.3/v3.4 FP are 384/324 and FN 547/738. This is a tradeoff, not general superiority. The snapshot was previously inspected, and old-model training exposure differs. v3.3 domain overlap, two cleaned training-input matches and the maintainer's promotion override are disclosed throughout relevant outputs.

Historical 99.x% ensemble results and FN 37-to-17 are retained only as historical results. The reference paper's cohort and values remain separately attributed. See `LATEST_RESEARCH_CLAIM_AUDIT.md` for the complete claim/evidence/output matrix.

## Testing and output validation

- Production-source `pytest -q`: **91 passed, zero failed/errors/skipped**, two dependency warnings. No model training.
- Earlier local research branch: 107 passed, recorded separately, not counted as this release's suite.
- Smoke checks were rerun against the documentation branch's unchanged application source and the public detector: health, model info, URL/root variants, synthetic input, independent HTML/email, combined context and feedback validation pass.
- Empty context/invalid feedback are correctly rejected with HTTP 400. Valid local feedback uses isolated temporary storage; no fabricated valid feedback was submitted to production.
- Public model version remains v3.3. No submitted URL destination was fetched; only detector API endpoints were contacted.
- IEEE paper: 5 pages, compiled successfully, all pages rendered; result tables fit their columns, with no overfull table warnings.
- Project report: 30 pages, compiled successfully, all pages rendered. College-template fonts/margins/chapter styles retained. The inherited title-page overfull-box warning remains, but visual inspection shows content within the page. Template hyperlink boxes and spacing are preserved, not redesigned.
- Poster acceptance passed: one editable slide, 200 editable shapes, two native charts with embedded workbooks, five visible architecture arrows, and an A0-size vector PDF. The 200-dpi PNG is 6,622 by 9,363 pixels. The historical chart/workbook is byte-preserved. Package import and LibreOffice rendering passed; native Microsoft PowerPoint was not tested. Final results and hashes are in `evidence/output_validation.json`.
- No secret credentials, local absolute paths, scratch scripts or candidate model binaries are included in the publication manifest.

## Publication

Publish only `docs/verified-output-refresh`, using a normal push. Do not merge it into main automatically. The exact commit is the tip of that branch; `git log origin/main..docs/verified-output-refresh` identifies the documentation-only change. Publication success is reported after the push, not inferred from a local commit. All excluded work remains available locally.
