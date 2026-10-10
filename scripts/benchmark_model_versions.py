"""Benchmark two code versions of the URL detector under identical conditions.

Each version is scored through its own ``app.inference.URLInference.predict``
pipeline (checked out as a git worktree), so the comparison measures exactly
what each application would serve.  URLs are only processed as text; nothing
is fetched.

The frozen evaluation set used for results/model_version_benchmark.json is
results/model_version_benchmark_eval_sets.csv (live feeds such as OpenPhish
rotate, so ``build-sets`` will not regenerate identical rows later).  Copy it to
``<out-dir>/eval_sets.csv`` to rerun ``score`` and ``compare`` without rebuilding.

Usage:
    python scripts/benchmark_model_versions.py build-sets --external-dir tmp/external --out-dir tmp/benchmark
    python scripts/benchmark_model_versions.py score --worktree tmp/wt_main --label previous --out-dir tmp/benchmark
    python scripts/benchmark_model_versions.py score --worktree tmp/wt_mod --label modified --out-dir tmp/benchmark
    python scripts/benchmark_model_versions.py compare --out-dir tmp/benchmark --results results/model_version_benchmark.json
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
EXTERNAL_SAMPLE = 3000
LATENCY_SAMPLE = 1000


def loose_key(url: str) -> str:
    text = str(url).strip().lower()
    if "://" in text:
        text = text.split("://", 1)[1]
    if text.startswith("www."):
        text = text[4:]
    return text.rstrip("/")


def _bilstm_split(balanced: pd.DataFrame, hard: list[dict]) -> dict[str, pd.DataFrame]:
    """Reproduce the train/val/test partition performed by scripts/train_bilstm.py."""
    from sklearn.model_selection import train_test_split

    hard_split = {case["url"]: case["hard_negative_split"] for case in hard}
    parts: dict[str, list[tuple[str, int]]] = {"train": [], "validation": [], "test": []}
    remaining = []
    for _, row in balanced.iterrows():
        if row["url"] in hard_split:
            parts[hard_split[row["url"]]].append((row["url"], int(row["label"])))
        else:
            remaining.append(row)
    rem = pd.DataFrame(remaining)
    rem_train, rem_temp = train_test_split(rem, test_size=0.30, random_state=SEED, stratify=rem["label"])
    rem_val, rem_test = train_test_split(rem_temp, test_size=0.50, random_state=SEED, stratify=rem_temp["label"])
    out = {}
    for name, extra in (("train", rem_train), ("validation", rem_val), ("test", rem_test)):
        frame = pd.DataFrame(parts[name], columns=["url", "label"])
        out[name] = pd.concat([frame, extra[["url", "label"]]], ignore_index=True)
    return out


def build_sets(external_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {}

    dataset = pd.read_csv(ROOT / "data/processed/dataset.csv")
    idx = {name: np.load(ROOT / f"data/processed/{name}_idx.npy") for name in ("train", "val", "test")}
    hard = json.loads((ROOT / "tests/data/hard_legitimate_urls.json").read_text(encoding="utf-8"))
    balanced = pd.read_csv(ROOT / "data/processed/balanced_dataset.csv")
    bilstm = _bilstm_split(balanced, hard)

    previous_train_keys = {loose_key(u) for u in dataset.iloc[idx["train"]].url}
    previous_train_keys |= {loose_key(c["url"]) for c in hard if c["hard_negative_split"] == "train"}
    modified_train_keys = {loose_key(u) for u in bilstm["train"].url}

    sets: list[pd.DataFrame] = []

    def add(name: str, frame: pd.DataFrame, source: str) -> None:
        frame = frame[["url", "label"]].copy()
        frame["dataset"] = name
        keys = frame.url.map(loose_key)
        frame["in_previous_train"] = keys.isin(previous_train_keys)
        frame["in_modified_train"] = keys.isin(modified_train_keys)
        report[name] = {
            "source": source,
            "rows": int(len(frame)),
            "class_counts": {str(k): int(v) for k, v in frame.label.value_counts().sort_index().items()},
            "overlap_with_previous_train": int(frame.in_previous_train.sum()),
            "overlap_with_modified_train": int(frame.in_modified_train.sum()),
        }
        sets.append(frame)

    add("previous_heldout_test", dataset.iloc[idx["test"]], "data/processed/dataset.csv[test_idx.npy] (PhishX sample)")
    adversarial = pd.read_csv(ROOT / "data/processed/adversarial_test.csv").rename(columns={"mutated_url": "url"})
    add("previous_adversarial_test", adversarial, "data/processed/adversarial_test.csv (mutated held-out test)")
    hard_test = pd.DataFrame([{"url": c["url"], "label": 0} for c in hard if c["hard_negative_split"] == "test"])
    add("hard_negative_test", hard_test, "tests/data/hard_legitimate_urls.json split=test")
    hard_val = pd.DataFrame([{"url": c["url"], "label": 0} for c in hard if c["hard_negative_split"] == "validation"])
    add("hard_negative_validation", hard_val, "tests/data/hard_legitimate_urls.json split=validation")
    regression = json.loads((ROOT / "tests/data/external_url_regression.json").read_text(encoding="utf-8"))["cases"]
    add(
        "external_regression_suite",
        pd.DataFrame([{"url": c["url"], "label": int(c["expected_label"] == "PHISHING")} for c in regression]),
        "tests/data/external_url_regression.json",
    )
    add("modified_heldout_test", bilstm["test"], "reconstructed scripts/train_bilstm.py test split of balanced_dataset.csv")

    # Everything either model could have seen during training or that already lives in the repo.
    seen = set(dataset.url.map(loose_key)) | set(balanced.url.map(loose_key))
    for raw in ("legit_urls.csv", "phishing_urls.csv"):
        seen |= set(pd.read_csv(ROOT / "data/raw" / raw).url.astype(str).map(loose_key))
    seen |= set(adversarial.url.astype(str).map(loose_key))
    seen |= {loose_key(c["url"]) for c in hard} | {loose_key(c["url"]) for c in regression}
    for name in ("alex_train.csv", "alex_test.csv"):
        seen |= set(pd.read_csv(external_dir / name).text.dropna().astype(str).map(loose_key))
    for name in ("piro_train.parquet", "piro_test.parquet"):
        seen |= set(pd.read_parquet(external_dir / name).url.dropna().astype(str).map(loose_key))

    def unseen(urls: pd.Series) -> pd.Series:
        urls = urls.dropna().astype(str).str.strip()
        urls = urls[(urls.str.len() > 0) & (urls.str.len() <= 2048)]
        keys = urls.map(loose_key)
        urls = urls[~keys.isin(seen) & ~keys.duplicated()]
        return urls

    external: dict[str, pd.DataFrame] = {}
    openphish = pd.Series((external_dir / "openphish_feed.txt").read_text().splitlines())
    external["external_openphish"] = pd.DataFrame({"url": unseen(openphish), "label": 1})
    phishdb = pd.Series((external_dir / "pdb_phishing-links-ACTIVE.txt").read_text(errors="ignore").splitlines())
    phishdb = unseen(phishdb[phishdb.str.match(r"^https?://", na=False)])
    external["external_phishing_database"] = pd.DataFrame(
        {"url": phishdb.sample(n=min(EXTERNAL_SAMPLE, len(phishdb)), random_state=SEED), "label": 1}
    )
    tranco = pd.read_csv(external_dir / "top-1m.csv", header=None, names=["rank", "domain"]).head(20000)
    tranco_urls = unseen("https://" + tranco.domain.astype(str) + "/")
    external["external_tranco_top20k"] = pd.DataFrame(
        {"url": tranco_urls.sample(n=min(EXTERNAL_SAMPLE, len(tranco_urls)), random_state=SEED), "label": 0}
    )
    mitake = pd.read_csv(external_dir / "mitake.csv")
    for label, name in ((0, "external_mitake_benign"), (1, "external_mitake_phishing")):
        urls = unseen(mitake[mitake.label == label].url)
        external[name] = pd.DataFrame({"url": urls.sample(n=min(EXTERNAL_SAMPLE, len(urls)), random_state=SEED), "label": label})

    combined = pd.concat(external.values(), ignore_index=True)
    combined_keys = combined.url.map(loose_key)
    conflicts = set(combined_keys[combined.label == 0]) & set(combined_keys[combined.label == 1])
    report["external_label_conflicts_removed"] = len(conflicts)
    sources = {
        "external_openphish": "https://openphish.com/feed.txt (community feed snapshot)",
        "external_phishing_database": "https://github.com/mitchellkrogza/Phishing.Database phishing-links-ACTIVE.txt (seeded sample)",
        "external_tranco_top20k": "https://tranco-list.eu top-1m list 8PX5V, ranks 1-20000 as https://<domain>/ (seeded sample)",
        "external_mitake_benign": "https://huggingface.co/datasets/Mitake/PhishingURLsANDBenignURLs label=0 (seeded sample)",
        "external_mitake_phishing": "https://huggingface.co/datasets/Mitake/PhishingURLsANDBenignURLs label=1 (seeded sample)",
    }
    for name, frame in external.items():
        frame = frame[~frame.url.map(loose_key).isin(conflicts)]
        add(name, frame, sources[name])

    pd.concat(sets, ignore_index=True).to_csv(out_dir / "eval_sets.csv", index=False)
    (out_dir / "eval_sets_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


def score(worktree: Path, label: str, out_dir: Path) -> None:
    worktree = worktree.resolve()
    sys.path.insert(0, str(worktree))
    import os

    os.chdir(worktree)
    start = time.perf_counter()
    from app.inference import URLInference

    service = URLInference()
    bundle = service.load_models()
    load_seconds = time.perf_counter() - start
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)

    bilstm_raw = None
    if getattr(bundle, "bilstm_loaded", False):
        from src.charcnn_bilstm import MAX_SEQUENCE_LENGTH as BILSTM_LEN, encode_url_bilstm
        from src.domain_reputation import is_authentic_domain

        def bilstm_raw(url: str) -> tuple[float, bool]:
            sequence = encode_url_bilstm(url, bundle.bilstm_vocab, max_len=BILSTM_LEN)
            logit = float(bundle.bilstm_model.run(["logits"], {"input": sequence[np.newaxis, :]})[0][0])
            return float(1.0 / (1.0 + np.exp(-logit))), bool(is_authentic_domain(url))

    sets = pd.read_csv(out_dir / "eval_sets.csv")
    rows = []
    for url in sets.url.astype(str):
        record: dict[str, object] = {"probability": np.nan, "error": ""}
        try:
            t0 = time.perf_counter()
            result = service.predict(url)
            record["latency_ms"] = (time.perf_counter() - t0) * 1000
            record["probability"] = float(result["phishing_probability"])
            record["model_version"] = result["model_version"]
            if bilstm_raw is not None:
                record["bilstm_raw_probability"], record["allowlisted_domain"] = bilstm_raw(str(result["url"]))
        except Exception as exc:  # recorded and reported, never silently dropped
            record["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(record)
    scored = pd.concat([sets, pd.DataFrame(rows)], axis=1)
    scored.to_csv(out_dir / f"scores_{label}.csv", index=False)

    latency_urls = sets.url.sample(n=LATENCY_SAMPLE, random_state=SEED).astype(str).tolist()
    for url in latency_urls[:50]:
        service.predict(url)
    timings = []
    for url in latency_urls:
        t0 = time.perf_counter()
        service.predict(url)
        timings.append((time.perf_counter() - t0) * 1000)
    runtime = {
        "label": label,
        "worktree": str(worktree),
        "model_load_seconds": load_seconds,
        "peak_rss_mb_after_load": rss_mb,
        "predict_latency_ms_p50": float(np.percentile(timings, 50)),
        "predict_latency_ms_p95": float(np.percentile(timings, 95)),
        "predict_latency_ms_mean": float(np.mean(timings)),
        "errors": int((scored.error != "").sum()),
    }
    (out_dir / f"runtime_{label}.json").write_text(json.dumps(runtime, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(runtime, indent=2))


def metrics(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict[str, object]:
    from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    both = len(np.unique(y)) == 2
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "n": int(len(y)),
        "positives": int(y.sum()),
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
    modified = pd.read_csv(out_dir / "scores_modified.csv", keep_default_na=False, na_values=[""])
    report = json.loads((out_dir / "eval_sets_report.json").read_text(encoding="utf-8"))
    output: dict[str, object] = {"datasets": report, "metrics": {}, "runtime": {}}
    for label in ("previous", "modified"):
        output["runtime"][label] = json.loads((out_dir / f"runtime_{label}.json").read_text(encoding="utf-8"))

    def evaluate(mask: pd.Series) -> dict[str, object]:
        valid = mask & previous.probability.notna() & modified.probability.notna()
        y = previous.loc[valid, "label"].to_numpy(int)
        entry = {
            "excluded_due_to_errors": int((mask & ~valid).sum()),
            "previous": metrics(y, previous.loc[valid, "probability"].to_numpy(float)),
            "modified": metrics(y, modified.loc[valid, "probability"].to_numpy(float)),
        }
        if "bilstm_raw_probability" in modified:
            entry["modified_without_domain_allowlist"] = metrics(y, modified.loc[valid, "bilstm_raw_probability"].to_numpy(float))
            entry["allowlisted_rows"] = int(modified.loc[valid, "allowlisted_domain"].astype(str).eq("True").sum())
        return entry

    names = list(dict.fromkeys(previous.dataset))
    for name in names:
        mask = previous.dataset == name
        output["metrics"][name] = evaluate(mask)
        clean = mask & ~previous.in_previous_train & ~previous.in_modified_train
        if clean.sum() != mask.sum() and clean.sum() > 0:
            output["metrics"][f"{name}__unseen_by_both"] = evaluate(clean)
    external = previous.dataset.str.startswith("external_") & (previous.dataset != "external_regression_suite")
    output["metrics"]["external_overall"] = evaluate(external)

    phishing_on_allowlist = modified[(modified.label == 1) & modified.allowlisted_domain.astype(str).eq("True")]
    output["allowlisted_phishing_examples"] = {
        "count": int(len(phishing_on_allowlist)),
        "missed_by_modified": int((phishing_on_allowlist.probability < 0.5).sum()),
        "caught_by_previous": int(
            (previous.loc[phishing_on_allowlist.index, "probability"] >= 0.5).sum()
        ),
        "registered_domains": phishing_on_allowlist.url.map(
            lambda u: u.split("/")[2] if "://" in u else u.split("/")[0]
        ).str.split(".").str[-2:].str.join(".").value_counts().head(15).to_dict(),
    }
    results_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {results_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build-sets")
    build.add_argument("--external-dir", type=Path, required=True)
    build.add_argument("--out-dir", type=Path, required=True)
    run = sub.add_parser("score")
    run.add_argument("--worktree", type=Path, required=True)
    run.add_argument("--label", choices=["previous", "modified"], required=True)
    run.add_argument("--out-dir", type=Path, required=True)
    cmp = sub.add_parser("compare")
    cmp.add_argument("--out-dir", type=Path, required=True)
    cmp.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build-sets":
        build_sets(args.external_dir, args.out_dir)
    elif args.command == "score":
        score(args.worktree, args.label, args.out_dir.resolve())
    else:
        compare(args.out_dir, args.results)


if __name__ == "__main__":
    main()
