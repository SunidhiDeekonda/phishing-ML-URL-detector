# Data Provenance

Reference repository: https://github.com/dubeyrudra-1808/PhishX.git
Phishing source URL: https://raw.githubusercontent.com/dubeyrudra-1808/PhishX/main/data/raw/phishing_urls.csv
Legitimate source URL: https://raw.githubusercontent.com/dubeyrudra-1808/PhishX/main/data/raw/legit_urls.csv
Download timestamp (UTC): 2026-08-28 21:57:09

## Raw phishing source snapshot
Original rows: 49371
Null rows: 0
Duplicate rows (after null drop): 8
Unique cleaned rows: 49363
SHA256: 00c3646e48fd22c810c5408ca57456e0b24ec990be1d9b415f59ab6356c757d2

## Raw legitimate source snapshot
Original rows: 50000
Null rows: 0
Duplicate rows (after null drop): 0
Unique cleaned rows: 50000
SHA256: 032a9a04a5cf6a0f8938b7c04a370c89bf28dafff17e310c6a5c0dee97193db3

## Sampling strategy
Deterministic sampling: pandas.DataFrame.sample with random_state=42
Stratified by source class before merge.
Phishing sampled rows: 10000
Legitimate sampled rows: 10000
Final rows: 20000

The complete source dataset contains substantially more URLs, but a balanced deterministic subset of 20,000 URLs was selected for this local reproduction experiment to reduce computational requirements while preserving both classes.

## Production BiLSTM datasets (v3.1-v3.3)

### V2 balanced dataset
File: data/processed/balanced_dataset.csv (40,000 rows, 20,000 per class)
SHA256: 0231f1073f662e424797b8ed86753391cf6eeda3c00f4690403851e9576a7e67
Generator: scripts/prepare_balanced_dataset.py (alexkstern and pirocheto Hugging Face datasets, PhishX raw files, hard negatives)
Known issues (results/dataset_label_audit.json): 9,598 legitimate rows are generated as <PhishX root> + fixed paths such as /login; phishing rows carry formatting artefacts. Training repairs or rejects artefacts with src/dataset_cleaning.py; the file itself is unmodified.

### v3.3 augmentation
File: data/processed/augmentation_v3_3.csv (24,000 rows, 12,000 per class)
SHA256: 1b7cbb68242a362e78e8512d68b56e2f3337fe594ae93d9de6bdc1e1b2dab5d4
Generator: scripts/build_augmentation_data.py
Phishing: 12,000 Phishing.Database phishing-links-ACTIVE.txt links, minus bare roots of Tranco top-100k domains.
Legitimate: 9,000 unused alexkstern legitimate URLs with a path and 3,000 Tranco rank 100k-1M roots, both minus hosts in Phishing.Database phishing-domains-ACTIVE.txt.
The generator intended to exclude evaluation URLs and V2 rows. The later cleaned-input audit found two evaluation/training matches in fresh2, so complete exclusion must not be claimed. Registered-domain overlap also remained; see `docs/evidence/latest_model_verification.json`.

### Frozen evaluation sets
results/external_eval_sets.csv, results/fresh_eval_sets.csv (development set since its errors were inspected) and results/fresh2_eval_sets.csv (originally pre-registered, results/v3_3_preregistration.json; subsequently inspected and found contaminated). Sources: OpenPhish public feed, Phishing.Database, Tranco list 8PX5V, Mitake (provenance undocumented). Hashes are in the matching *_manifest.json files. Do not call these snapshots prospective blind tests.

## Local domain-separated research candidate v3.4

This is an earlier local experiment, not the deployed model and not training performed by the output-refresh task. From 64,000 raw rows, cleaning retains 63,869. Whole public registered domains are reserved with test precedence over validation over training. Final counts are 25,961 / 6,113 / 31,795; public/private registered-domain, hostname and cleaned-key split overlaps are all zero. Large reserved test groups explain the non-70/10/20 row proportions. Previously inspected evaluation domains are not moved into training, but this does not make the resulting study a newly blind temporal test.

The candidate failed its promotion gate. Saved protocol, metrics and comparison summaries are published under `docs/evidence/`; its model binaries, training code changes and per-row research outputs remain on the local research branch. Production remains v3.3.
