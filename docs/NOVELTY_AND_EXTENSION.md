# Novelty and Research Extension

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
