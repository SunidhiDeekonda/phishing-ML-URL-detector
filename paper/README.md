# Verified IEEE Paper

**Title:** Domain-Separated Evaluation of Character CNN-BiLSTM Phishing Detection: A Reproducibility Study

Final PDF: `AI_Phishing_Research_Paper_Updated.pdf` (5 A4, two-column IEEE conference pages).

The existing updated draft's scientific content was retained; table spacing, version boundaries and software-test scope were corrected. The original CNN-LightGBM paper remains a historical artifact, not a current production description.

## Build

Run from this directory with Tectonic:

```sh
tectonic main.tex --keep-logs
cp main.pdf AI_Phishing_Research_Paper_Updated.pdf
```

`main.tex`, `references.bib` and the three PDF figures under `figures/` are the required project files. PNG counterparts are convenient previews. This multi-file project uses IEEEtran and BibTeX; it is not a standalone single-file LaTeX document.

## Scientific scope

v3.3 is deployed and has disclosed split-contamination/promotion-override limitations. v3.4 is an earlier local domain-separated candidate with the same architecture; it failed its promotion gate and is not deployed. No new training or application changes were made for this output refresh. The common external snapshot is previously inspected, not prospective blind evidence. Historical 20,000-URL ensemble metrics belong to their own experiment.

Evidence: `../docs/evidence/`; claim-to-output mapping: `../docs/LATEST_RESEARCH_CLAIM_AUDIT.md`. Candidate binaries and full experiment implementation remain on a local research branch and are not shipped in this documentation release. Consequently, saved evidence is auditable, but this release alone is not a complete executable v3.4 reproduction package.
