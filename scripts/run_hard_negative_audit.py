"""Re-run the transparent production hard-negative benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from app.inference import URLInference

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    cases = json.loads((ROOT / "tests/data/hard_legitimate_urls.json").read_text())
    service = URLInference(); service.load_models()
    rows = []
    for case in cases:
        prediction = service.predict(case["url"])
        rows.append({**case, "canonical_url": prediction["url"], "prediction": prediction["verdict"],
                     "selected_probability": prediction["selected_ensemble_probability"],
                     "cnn_probability": prediction["cnn_probability"],
                     "lightgbm_probability": prediction["lightgbm_probability"],
                     "result": "PASS" if prediction["verdict"] == "LEGITIMATE" else "FAIL"})
    frame = pd.DataFrame(rows)
    frame.to_csv(ROOT / "results/hard_negative_production_audit.csv", index=False)
    print(frame.groupby(["hard_negative_split", "result"]).size().unstack(fill_value=0))
    failures = frame[frame.result == "FAIL"]
    if len(failures):
        print("\nRemaining false positives:\n" + failures[["url", "category", "selected_probability"]].to_string(index=False))


if __name__ == "__main__":
    main()
