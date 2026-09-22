"""Human review and candidate-dataset preparation for explicit offline retraining."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.continuous_learning import FeedbackStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=ROOT / "data/feedback/pending.jsonl")
    parser.add_argument("--base-dataset", type=Path, default=ROOT / "data/processed/dataset.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "data/feedback/candidate_dataset.csv")
    parser.add_argument("--review-record")
    parser.add_argument("--decision", choices=("approved", "rejected"))
    args = parser.parse_args()
    hashes = json.loads((ROOT / "models/test_url_hashes.json").read_text())
    store = FeedbackStore(args.store, test_hashes=hashes)
    if bool(args.review_record) != bool(args.decision):
        parser.error("--review-record and --decision must be supplied together")
    if args.review_record:
        reviewed = store.review(args.review_record, args.decision)
        print(f"reviewed {reviewed['record_id']}: {reviewed['status']}")
    added = store.build_candidate_dataset(args.base_dataset, args.output)
    print(f"candidate dataset: {args.output}")
    print(f"approved new rows: {added}")
    print("automatic training/promotion: disabled")


if __name__ == "__main__":
    main()
