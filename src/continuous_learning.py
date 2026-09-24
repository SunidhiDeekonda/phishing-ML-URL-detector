"""Human-verified feedback quarantine and model registry."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from src.url_normalization import canonicalize_url

VALID_LABELS = {"LEGITIMATE": 0, "PHISHING": 1}


def _label(value: str | int) -> int:
    if isinstance(value, bool): raise ValueError("Boolean labels are not accepted")
    if isinstance(value, int) and value in (0, 1): return value
    key = str(value).strip().upper()
    if key in VALID_LABELS: return VALID_LABELS[key]
    if key in {"0", "1"}: return int(key)
    raise ValueError("Label must be LEGITIMATE/PHISHING or 0/1")


class FeedbackStore:
    def __init__(self, path: str | Path, test_urls: Iterable[str] = (), test_hashes: Iterable[str] = ()) -> None:
        self.path = Path(path)
        self.test_urls = {canonicalize_url(url) for url in test_urls}
        self.test_hashes = set(test_hashes)

    def submit(self, url: str, predicted_label: str | int, correct_label: str | int, notes: str = "") -> dict[str, object]:
        normalized = canonicalize_url(str(url))
        if not normalized or len(normalized) > 4096: raise ValueError("URL must contain 1 to 4096 characters")
        digest = hashlib.sha256(normalized.encode()).hexdigest()
        if normalized in self.test_urls or digest in self.test_hashes: raise ValueError("Held-out test URLs cannot enter the feedback pipeline")
        predicted, correct = _label(predicted_label), _label(correct_label)
        if any(row["url_sha256"] == digest for row in self.read_all()):
            return {"accepted": True, "deduplicated": True, "record_id": digest[:16], "status": "pending_verification"}
        record = {"record_id": digest[:16], "url": normalized, "url_sha256": digest,
                  "predicted_label": predicted, "correct_label": correct, "notes": str(notes).strip()[:500],
                  "status": "pending_verification", "created_at": datetime.now(timezone.utc).isoformat()}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle: handle.write(json.dumps(record, sort_keys=True) + "\n")
        return {"accepted": True, "deduplicated": False, "record_id": record["record_id"], "status": record["status"]}

    def read_all(self) -> list[dict[str, object]]:
        if not self.path.exists(): return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def approved_rows(self) -> list[dict[str, object]]:
        return [row for row in self.read_all() if row.get("status") == "approved"]

    def review(self, record_id: str, decision: str) -> dict[str, object]:
        if decision not in {"approved", "rejected"}:
            raise ValueError("Decision must be approved or rejected")
        records = self.read_all()
        matched = False
        for record in records:
            if record.get("record_id") == record_id:
                record["status"] = decision
                record["reviewed_at"] = datetime.now(timezone.utc).isoformat()
                matched = True
        if not matched:
            raise ValueError("Feedback record was not found")
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in records), encoding="utf-8")
        temporary.replace(self.path)
        return next(row for row in records if row.get("record_id") == record_id)

    def build_candidate_dataset(self, base_dataset: str | Path, output: str | Path) -> int:
        import pandas as pd

        base = pd.read_csv(base_dataset)
        approved = []
        known = {canonicalize_url(url) for url in base["url"].astype(str)}
        for row in self.approved_rows():
            normalized = canonicalize_url(str(row["url"]))
            digest = hashlib.sha256(normalized.encode()).hexdigest()
            if normalized in self.test_urls or digest in self.test_hashes:
                raise ValueError("Approved feedback contains a held-out test URL")
            if normalized not in known:
                approved.append({"url": normalized, "label": int(row["correct_label"])})
                known.add(normalized)
        candidate = pd.concat([base[["url", "label"]], pd.DataFrame(approved, columns=["url", "label"])], ignore_index=True)
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        candidate.to_csv(target, index=False)
        return len(approved)


def validation_gate(
    incumbent_clean_auc: float,
    incumbent_robust_auc: float,
    candidate_clean_auc: float,
    candidate_robust_auc: float,
    clean_tolerance: float = 0.002,
) -> dict[str, object]:
    incumbent_score = (incumbent_clean_auc + incumbent_robust_auc) / 2
    candidate_score = (candidate_clean_auc + candidate_robust_auc) / 2
    accepted = candidate_score > incumbent_score and candidate_clean_auc >= incumbent_clean_auc - clean_tolerance
    return {"accepted": accepted, "incumbent_score": incumbent_score, "candidate_score": candidate_score,
            "clean_floor": incumbent_clean_auc - clean_tolerance,
            "reason": "candidate passed both gates" if accepted else "candidate failed improvement or clean-performance gate"}


class ModelRegistry:
    def __init__(self, path: str | Path) -> None: self.path = Path(path)
    def load(self) -> dict[str, object]:
        return json.loads(self.path.read_text()) if self.path.exists() else {"models": []}
    def register(self, entry: dict[str, object]) -> None:
        required = {"version", "training_date", "data_count", "clean_validation_metrics", "robustness_validation_metrics", "status"}
        if required - set(entry): raise ValueError(f"Missing registry fields: {sorted(required - set(entry))}")
        payload = self.load(); payload["models"] = [item for item in payload["models"] if item.get("version") != entry["version"]] + [entry]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
