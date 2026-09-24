"""Run the offline external URL and production-preprocessing reliability audit."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import tldextract
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.inference import MODEL_VERSION, URLInference
from src.features import extract_url_features
from src.url_normalization import canonicalize_url

EXTRACTOR = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)
SUITE_PATH = ROOT / "tests/data/external_url_regression.json"
OUTPUT_PATH = ROOT / "results/external_url_regression.csv"
SUMMARY_PATH = ROOT / "results/external_url_regression_summary.json"
DEPLOYED_PATH = ROOT / "results/deployed_url_regression.csv"


def registered_domain(url: str) -> str:
    extracted = EXTRACTOR(canonicalize_url(url))
    return ".".join(part for part in (extracted.domain, extracted.suffix) if part)


def membership_context() -> tuple[set[str], dict[str, set[str]]]:
    dataset = pd.read_csv(ROOT / "data/processed/dataset.csv")
    exact = set(dataset.url.astype(str))
    domains = {}
    for split in ("train", "val", "test"):
        indices = np.load(ROOT / f"data/processed/{split}_idx.npy")
        domains[split] = {registered_domain(url) for url in dataset.iloc[indices].url.astype(str)}
    return exact, domains


def load_cases() -> list[dict[str, object]]:
    payload = json.loads(SUITE_PATH.read_text(encoding="utf-8"))
    cases = payload["cases"]
    ids = [case["id"] for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("External regression case IDs must be unique")
    return cases


def audit_local() -> tuple[pd.DataFrame, dict[str, object]]:
    service = URLInference()
    service.load_models()
    exact_urls, split_domains = membership_context()
    rows = []
    for case in load_cases():
        prediction = service.predict(str(case["url"]))
        domain = registered_domain(prediction["url"])
        expected = str(case["expected_label"])
        row = {
            "case_id": case["id"], "category": case["category"], "url": case["url"],
            "expected_label": expected, "equivalence_group": case.get("equivalence_group", ""),
            "exact_url_seen_in_dataset": "YES" if str(case["url"]) in exact_urls else "NO",
            "registered_domain": domain,
            "registered_domain_seen_in_train": "YES" if domain in split_domains["train"] else "NO",
            "registered_domain_seen_in_validation": "YES" if domain in split_domains["val"] else "NO",
            "registered_domain_seen_in_test": "YES" if domain in split_domains["test"] else "NO",
            "normalized_url": prediction["url"], "prediction": prediction["verdict"],
            "probability": prediction["phishing_probability"], "cnn_probability": prediction["cnn_probability"],
            "lightgbm_probability": prediction["lightgbm_probability"], "model_version": prediction["model_version"],
            "result": "PASS" if prediction["verdict"] == expected else "FAIL",
        }
        rows.append(row)
    frame = pd.DataFrame(rows)
    benign = frame[frame.expected_label == "LEGITIMATE"]
    synthetic = frame[frame.category == "synthetic_phishing"]
    unseen_benign = benign[benign.registered_domain_seen_in_train == "NO"]
    unseen_synthetic = synthetic[synthetic.registered_domain_seen_in_train == "NO"]
    groups = {}
    for name, group in frame[frame.equivalence_group != ""].groupby("equivalence_group"):
        stable = group.normalized_url.nunique() == 1 and group.prediction.nunique() == 1 and np.allclose(group.probability, group.probability.iloc[0], atol=1e-12)
        groups[name] = {"passed": bool(stable), "cases": int(len(group)), "normalized_values": sorted(group.normalized_url.unique()), "verdicts": sorted(group.prediction.unique())}
    false_positives = []
    for _, row in benign[benign.prediction == "PHISHING"].iterrows():
        features = extract_url_features(row.normalized_url)
        false_positives.append({"url": row.url, "normalized_url": row.normalized_url, "probability": row.probability,
                                "cnn_probability": row.cnn_probability, "lightgbm_probability": row.lightgbm_probability,
                                "likely_reason": "CNN sequence behavior" if row.cnn_probability >= row.lightgbm_probability else "engineered-feature behavior",
                                "features": {key: features[key] for key in ("url_length", "host_length", "hostname_entropy", "num_dots", "hyphen_count", "digit_letter_ratio", "suspicious_tld", "token_login", "token_verify", "token_secure", "token_account")}})
    robustness = json.loads((ROOT / "results/robustness_final_metrics.json").read_text())
    summary = {
        "model_version": MODEL_VERSION, "total_urls": int(len(frame)), "known_good_cases": int(len(benign)),
        "known_good_passed": int((benign.result == "PASS").sum()), "known_good_false_positives": int((benign.prediction == "PHISHING").sum()),
        "synthetic_phishing_cases": int(len(synthetic)), "synthetic_phishing_passed": int((synthetic.result == "PASS").sum()),
        "synthetic_phishing_false_negatives": int((synthetic.prediction == "LEGITIMATE").sum()),
        "format_invariance_groups": groups, "format_invariance_passed": sum(item["passed"] for item in groups.values()),
        "format_invariance_failed": sum(not item["passed"] for item in groups.values()),
        "exact_urls_not_in_dataset": int((frame.exact_url_seen_in_dataset == "NO").sum()),
        "registered_domains_unseen_in_training": int((frame.registered_domain_seen_in_train == "NO").sum()),
        "unseen_domain_benign_accuracy": float((unseen_benign.result == "PASS").mean()) if len(unseen_benign) else None,
        "unseen_domain_benign_false_positives": int((unseen_benign.prediction == "PHISHING").sum()),
        "unseen_domain_synthetic_recall": float((unseen_synthetic.prediction == "PHISHING").mean()) if len(unseen_synthetic) else None,
        "false_positives": false_positives,
        "saved_robustness_metrics": robustness,
    }
    return frame, summary


def production_normalized_metrics(service: URLInference) -> dict[str, object]:
    dataset = pd.read_csv(ROOT / "data/processed/dataset.csv")
    test_indices = np.load(ROOT / "data/processed/test_idx.npy")
    labels = dataset.iloc[test_indices].label.to_numpy(dtype=int)
    probabilities = np.array([service.predict(url)["phishing_probability"] for url in dataset.iloc[test_indices].url.astype(str)])
    predictions = (probabilities >= 0.5).astype(int)
    stored = json.loads((ROOT / "results/final_test_metrics.json").read_text())
    result = {"rows": int(len(labels)), "accuracy": float(accuracy_score(labels, predictions)),
              "precision": float(precision_score(labels, predictions, zero_division=0)),
              "recall": float(recall_score(labels, predictions, zero_division=0)),
              "roc_auc": float(roc_auc_score(labels, probabilities)),
              "false_positives": int(((predictions == 1) & (labels == 0)).sum()),
              "false_negatives": int(((predictions == 0) & (labels == 1)).sum()),
              "historical_research_metrics_unchanged": stored}
    (ROOT / "results/production_normalized_test_metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def audit_deployed(local: pd.DataFrame, base_url: str) -> pd.DataFrame:
    rows = []
    for _, local_row in local.iterrows():
        request = urllib.request.Request(base_url.rstrip("/") + "/predict", data=json.dumps({"url": local_row.url}).encode(), headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                deployed = json.loads(response.read())
                status = response.status
        except urllib.error.HTTPError as exc:
            deployed = json.loads(exc.read()); status = exc.code
        probability = deployed.get("phishing_probability")
        match = status == 200 and deployed.get("url") == local_row.normalized_url and deployed.get("verdict") == local_row.prediction and abs(float(probability) - float(local_row.probability)) <= 1e-6
        rows.append({"case_id": local_row.case_id, "category": local_row.category, "url": local_row.url,
                     "local_normalized_url": local_row.normalized_url, "deployed_normalized_url": deployed.get("url", ""),
                     "local_prediction": local_row.prediction, "deployed_prediction": deployed.get("verdict", "ERROR"),
                     "local_probability": local_row.probability, "deployed_probability": probability,
                     "http_status": status, "match": "YES" if match else "NO"})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment-url")
    parser.add_argument("--skip-heldout", action="store_true")
    args = parser.parse_args()
    frame, summary = audit_local()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT_PATH, index=False)
    if not args.skip_heldout:
        summary["production_normalized_test"] = production_normalized_metrics(URLInference())
    if args.deployment_url:
        deployed = audit_deployed(frame, args.deployment_url)
        deployed.to_csv(DEPLOYED_PATH, index=False)
        summary["deployed_regression_mismatches"] = int((deployed.match == "NO").sum())
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
