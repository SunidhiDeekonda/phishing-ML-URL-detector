"""Compare the previous production ensemble with the CharCNN-BiLSTM model under identical conditions.

Each code version is scored through its own ``app.inference.URLInference.predict``
(the previous version from a git worktree), on the same URLs, threshold 0.5.
URLs are processed as text only; nothing is fetched.

Usage:
    git worktree add tmp/wt_previous f16143e
    python scripts/compare_production_models.py build-sets --out-dir tmp/compare
    python scripts/compare_production_models.py score --worktree tmp/wt_previous --label previous --out-dir tmp/compare
    python scripts/compare_production_models.py score --worktree . --label updated --out-dir tmp/compare
    python scripts/compare_production_models.py compare --out-dir tmp/compare
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
LATENCY_SAMPLE = 1000


def loose_key(url: str) -> str:
    text = str(url).strip().lower()
    if "://" in text:
        text = text.split("://", 1)[1]
    if text.startswith("www."):
        text = text[4:]
    return text.rstrip("/")


def build_sets(out_dir: Path) -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    from train_bilstm import prepare_splits

    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = pd.read_csv(ROOT / "data/processed/dataset.csv")
    hard = json.loads((ROOT / "tests/data/hard_legitimate_urls.json").read_text(encoding="utf-8"))
    previous_train = {loose_key(u) for u in dataset.iloc[np.load(ROOT / "data/processed/train_idx.npy")].url}
    previous_train |= {loose_key(c["url"]) for c in hard if c["hard_negative_split"] == "train"}
    splits, _ = prepare_splits()
    updated_train = {loose_key(u) for u in splits["train"].clean_url}

    frames = []

    def add(name: str, frame: pd.DataFrame) -> None:
        frame = frame[["url", "label"]].copy()
        frame["dataset"] = name
        keys = frame.url.map(loose_key)
        frame["in_previous_train"] = keys.isin(previous_train)
        frame["in_updated_train"] = keys.isin(updated_train)
        frames.append(frame)

    add("previous_heldout_test", dataset.iloc[np.load(ROOT / "data/processed/test_idx.npy")])
    add("previous_adversarial_test", pd.read_csv(ROOT / "data/processed/adversarial_test.csv").rename(columns={"mutated_url": "url"}))
    for split in ("validation", "test"):
        add(f"hard_negative_{split}", pd.DataFrame([{"url": c["url"], "label": 0} for c in hard if c["hard_negative_split"] == split]))
    regression = json.loads((ROOT / "tests/data/external_url_regression.json").read_text(encoding="utf-8"))["cases"]
    add("external_regression_suite", pd.DataFrame([{"url": c["url"], "label": int(c["expected_label"] == "PHISHING")} for c in regression]))
    add("updated_heldout_test", splits["test"].rename(columns={"url": "raw_url", "clean_url": "url"}))
    external = pd.read_csv(ROOT / "results/external_eval_sets.csv")
    for name, frame in external.groupby("dataset", sort=False):
        add(name, frame)

    sets = pd.concat(frames, ignore_index=True)
    sets.to_csv(out_dir / "eval_sets.csv", index=False)
    print(sets.groupby("dataset", sort=False).agg(rows=("label", "size"), phishing=("label", "sum"),
                                                   in_previous_train=("in_previous_train", "sum"),
                                                   in_updated_train=("in_updated_train", "sum")).to_string())


def score(worktree: Path, label: str, out_dir: Path) -> None:
    worktree = worktree.resolve()
    sys.path.insert(0, str(worktree))
    os.chdir(worktree)
    start = time.perf_counter()
    from app.inference import URLInference

    service = URLInference()
    service.load_models()
    load_seconds = time.perf_counter() - start
    sets = pd.read_csv(out_dir / "eval_sets.csv")
    rows = []
    for url in sets.url.astype(str):
        try:
            result = service.predict(url)
            rows.append({"probability": float(result["phishing_probability"]), "model_version": result["model_version"], "error": ""})
        except Exception as exc:
            rows.append({"probability": np.nan, "model_version": "", "error": f"{type(exc).__name__}: {exc}"})
    pd.concat([sets, pd.DataFrame(rows)], axis=1).to_csv(out_dir / f"scores_{label}.csv", index=False)

    sample = sets.url.sample(n=LATENCY_SAMPLE, random_state=SEED).astype(str).tolist()
    for url in sample[:50]:
        service.predict(url)
    timings = []
    for url in sample:
        t0 = time.perf_counter()
        service.predict(url)
        timings.append((time.perf_counter() - t0) * 1000)
    runtime = {
        "label": label,
        "model_version": rows[0]["model_version"],
        "model_load_seconds": load_seconds,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024),
        "predict_latency_ms_p50": float(np.percentile(timings, 50)),
        "predict_latency_ms_p95": float(np.percentile(timings, 95)),
        "errors": int(sum(bool(r["error"]) for r in rows)),
    }
    (out_dir / f"runtime_{label}.json").write_text(json.dumps(runtime, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(runtime, indent=2))


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, object]:
    from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    both = len(np.unique(y)) == 2
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "n": int(len(y)),
        "accuracy": float((tp + tn) / len(y)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
        "fpr": float(fp / (fp + tn)) if fp + tn else None,
        "fnr": float(fn / (fn + tp)) if fn + tp else None,
        "roc_auc": float(roc_auc_score(y, p)) if both else None,
        "pr_auc": float(average_precision_score(y, p)) if both else None,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def compare(out_dir: Path, results_path: Path) -> None:
    previous = pd.read_csv(out_dir / "scores_previous.csv", keep_default_na=False, na_values=[""])
    updated = pd.read_csv(out_dir / "scores_updated.csv", keep_default_na=False, na_values=[""])
    output: dict[str, object] = {
        "runtime": {label: json.loads((out_dir / f"runtime_{label}.json").read_text()) for label in ("previous", "updated")},
        "metrics": {},
    }

    def evaluate(mask: pd.Series) -> dict[str, object]:
        valid = mask & previous.probability.notna() & updated.probability.notna()
        y = previous.loc[valid, "label"].to_numpy(int)
        return {
            "excluded_due_to_errors": int((mask & ~valid).sum()),
            "previous": metrics(y, previous.loc[valid, "probability"].to_numpy(float)),
            "updated": metrics(y, updated.loc[valid, "probability"].to_numpy(float)),
        }

    for name in dict.fromkeys(previous.dataset):
        mask = previous.dataset == name
        output["metrics"][name] = evaluate(mask)
        unseen = mask & ~previous.in_previous_train & ~previous.in_updated_train
        if 0 < unseen.sum() < mask.sum():
            output["metrics"][f"{name}__unseen_by_both"] = evaluate(unseen)
    output["metrics"]["external_overall"] = evaluate(previous.dataset.isin(pd.read_csv(ROOT / "results/external_eval_sets.csv").dataset.unique()))
    results_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {results_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build-sets")
    build.add_argument("--out-dir", type=Path, required=True)
    run = sub.add_parser("score")
    run.add_argument("--worktree", type=Path, required=True)
    run.add_argument("--label", choices=["previous", "updated"], required=True)
    run.add_argument("--out-dir", type=Path, required=True)
    cmp = sub.add_parser("compare")
    cmp.add_argument("--out-dir", type=Path, required=True)
    cmp.add_argument("--results", type=Path, default=ROOT / "results/production_model_comparison.json")
    args = parser.parse_args()
    if args.command == "build-sets":
        build_sets(args.out_dir)
    elif args.command == "score":
        score(args.worktree, args.label, args.out_dir.resolve())
    else:
        compare(args.out_dir, args.results)


if __name__ == "__main__":
    main()
