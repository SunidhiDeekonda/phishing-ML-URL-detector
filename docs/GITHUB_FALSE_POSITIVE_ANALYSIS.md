# GitHub False-Positive Analysis

## Finding

`https://github.com/SunidhiDeekonda/phishing-ML-URL-detector` is a legitimate repository URL. The original production ensemble classified it as phishing with probability `0.9999696041` (`CNN=0.9999998700`, `LightGBM=0.9993945513`). This was a genuine false positive, not the earlier root-slash normalization bug.

## Why the CNN flagged it

The original legitimate training rows had no non-root paths, while 97% of training phishing rows had a path. The character CNN therefore learned path characters and path-like character motifs as an extremely strong phishing shortcut. Counterfactual checks showed that even neutral paths such as `/user/project` produced near-certain phishing output. The literal word `phishing` and hyphenated repository slug were additional lexical cues, but they were not the primary cause because ordinary repository paths failed too.

## Why LightGBM flagged it

The generated record contains all 36 feature values in `results/github_false_positive_explanation.json`. Per-record LightGBM contributions showed `path_length` as the dominant positive contribution, followed by special-character count, path segments, and URL length. The model was exploiting the same dataset shortcut in engineered-feature form.

## Why the ensemble flagged it

Both components were already near one, and the historical selected ensemble placed 95% weight on the CNN. The ensemble therefore could not correct either component's correlated mistake.

## Training membership

The exact URL and the `github.com` registered domain were absent from train, validation, and test. The original dataset also contains a structural label imbalance: all 10,000 legitimate examples are roots, while phishing examples overwhelmingly contain paths. Some shared-hosting domains such as `google.com` and `vercel.app` occur only with phishing labels, reflecting real abuse URLs but further encouraging platform/path shortcuts.

## General repair

Production V2 uses explicitly partitioned legitimate hard negatives, retains adversarial training, and is selected through a frozen validation gate before its held-out hard-negative test is opened. Historical models and metrics remain unchanged. GitHub is not checked or allowed by runtime code; repository URLs are training examples, not a whitelist.

The hard-negative corpus uses exact-URL separation. Major multi-tenant platforms appear across hard-negative partitions so unseen pages on a known platform can be measured; this exception is explicit and should not be confused with the original domain-separated research split.

## Does this mean GitHub is always safe?

No. A normal `github.com` user or repository page is not phishing merely because it is hosted by GitHub. `github.io` pages, raw/user-controlled content, links inside repositories, and abused platform content have different risk. No blanket safety rule is used.

## Evidence

- Pre-fix predictions and 36 features: `results/hard_negative_pre_fix.csv`
- Per-feature distribution and contribution analysis: `results/github_false_positive_explanation.json`
- Curated benchmark: `tests/data/hard_legitimate_urls.json`
- Frozen selection gate: `results/production_v2_validation_gate.json`
- Final comparison: `results/production_v2_final_metrics.json`
- Baseline and V2 hard-negative rows: `results/hard_negative_baseline.csv`, `results/hard_negative_v2.csv`
