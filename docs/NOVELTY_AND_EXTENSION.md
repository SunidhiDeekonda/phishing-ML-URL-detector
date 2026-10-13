# Novelty and Research Extension

## Current claim boundaries

The sections below describe the reference, initial reproduction and **historical** robustness experiment. In particular, adversarial FN 37 to 17 is not a result for v3.3/v3.4.

The newest implemented production architecture is CharCNN + two-layer BiLSTM + additive attention, not the historical probability ensemble. Recurrence and attention are established techniques; their presence alone does not establish scientific novelty. No controlled component ablation proves a causal gain here.

The current defensible contributions are versioned reproduction/integrity auditing, controlled robustness evaluation, safe independent context evidence, and human-reviewed adaptation infrastructure. A prior local v3.4 candidate repaired whole-domain separation but failed its promotion gate; its usefulness is an honest negative/tradeoff result, not a superior deployed detector. On the common external snapshot it reduced FP by 60 but increased FN by 191. Production v3.3's leakage and promotion override must be disclosed.

HTML/email signals remain heuristic context evidence, not calibrated ML probabilities. Feedback does not automatically train or deploy. See [the claim audit](LATEST_RESEARCH_CLAIM_AUDIT.md).

## Historical contributions and results

## Reference paper implemented
Character-level CNN, 36 engineered URL features, LightGBM, and a weighted ensemble. It reported approximately 99.819% accuracy, 100% precision, 99.635% recall, and 99.947% ROC-AUC.

## Original reproduction
Independent training on a deterministic 20,000-URL subset, strict root-domain-separated splits, validation-only ensemble selection, held-out evaluation, and a network-free FastAPI demo.

## New implemented extensions
- Eight deterministic offline mutation categories with seed 42.
- Conservative adversarial training with 1751 mutations derived only from training phishing rows.
- Separate clean and 3998-row adversarial held-out evaluation.
- Safe caller-supplied HTML/email context, never mixed into validated probability.
- Quarantined human-verified feedback with deduplication and test protection.
- PSI drift monitoring and a versioned validation-gated model registry.

Original adversarial recall was 98.1454% with 37 false negatives; robust-candidate recall was 99.1479% with 17 false negatives (+1.0025%). Robust clean accuracy was 99.6248%. The contribution is robustness and controlled adaptability, not a claim of higher saturated precision. Synthetic results do not prove protection against every attacker.
