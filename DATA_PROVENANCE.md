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