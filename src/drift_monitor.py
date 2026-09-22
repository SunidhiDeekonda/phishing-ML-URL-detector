"""Population Stability Index (PSI) feature drift reporting."""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.features import build_features_dataframe


def population_stability_index(reference: np.ndarray, recent: np.ndarray, bins: int = 10) -> float:
    reference, recent = np.asarray(reference, float), np.asarray(recent, float)
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2: return 0.0 if np.allclose(recent, reference[0]) else 1.0
    edges[0], edges[-1] = -np.inf, np.inf
    expected = np.clip(np.histogram(reference, bins=edges)[0] / max(len(reference), 1), 1e-6, None)
    actual = np.clip(np.histogram(recent, bins=edges)[0] / max(len(recent), 1), 1e-6, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def create_drift_report(reference_features: pd.DataFrame, recent_urls: list[str]) -> dict[str, object]:
    if not recent_urls: return {"status": "insufficient_data", "recent_count": 0, "method": "PSI", "feature_psi": {}, "drifted_features": []}
    recent = build_features_dataframe(recent_urls)[list(reference_features.columns)]
    values = {column: population_stability_index(reference_features[column], recent[column]) for column in reference_features.columns}
    return {"status": "drift_detected" if any(value >= .25 for value in values.values()) else "stable", "recent_count": len(recent_urls),
            "method": "PSI", "thresholds": {"monitor": .1, "investigate": .25}, "feature_psi": values,
            "drifted_features": sorted(key for key, value in values.items() if value >= .25)}


def save_drift_report(report: dict[str, object], path: str | Path) -> None:
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
