"""Score the served models on a fresh external snapshot that no model or decision has seen.

Plan and decision rule: results/hostsplit_preregistration.json (written before
the snapshot was downloaded). URLs are processed as text only; nothing is fetched.

Usage:
    python scripts/evaluate_fresh_snapshot.py build --evidence-dir tmp/fresh --external-dir tmp/external
    git worktree add tmp/wt_previous f16143e
    python scripts/evaluate_fresh_snapshot.py score-app --worktree tmp/wt_previous --label previous_ensemble
    python scripts/evaluate_fresh_snapshot.py score-onnx --config models/charcnn_bilstm_config.json --label bilstm_v3_1
    python scripts/evaluate_fresh_snapshot.py score-onnx --config models/charcnn_bilstm_hostsplit_config.json --label bilstm_v3_2_hostsplit
    python scripts/evaluate_fresh_snapshot.py compare
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
SETS_PATH = ROOT / "results/fresh_eval_sets.csv"
SCORES_DIR = ROOT / "tmp/fresh_scores"
RESULTS_PATH = ROOT / "results/fresh_snapshot_evaluation.json"
LABELS = ("previous_ensemble", "bilstm_v3_1", "bilstm_v3_2_hostsplit")


def safe_key(url: str) -> str | None:
    sys.path.insert(0, str(ROOT))
    from src.url_normalization import canonicalize_url

    try:
        text = canonicalize_url(str(url)).lower()
    except Exception:
        return None
    text = text.split("://", 1)[-1]
    return text[4:] if text.startswith("www.") else text


def seen_keys() -> set[str]:
    urls = [
        pd.read_csv(ROOT / "data/processed/dataset.csv").url,
        pd.read_csv(ROOT / "data/processed/balanced_dataset.csv").url,
        pd.read_csv(ROOT / "results/external_eval_sets.csv").url,
    ]
    for name in ("adversarial_test.csv", "adversarial_validation.csv"):
        frame = pd.read_csv(ROOT / "data/processed" / name)
        urls += [frame.original_url, frame.mutated_url]
    keys = {safe_key(u) for u in pd.concat(urls).astype(str)}
    keys.discard(None)
    return keys


def build(evidence_dir: Path, external_dir: Path) -> None:
    seen = seen_keys()
    frames = []

    def add(name: str, urls: list[str], label: int, limit: int | None = None) -> None:
        frame = pd.DataFrame({"url": [u.strip() for u in urls if u.strip()]})
        frame["key"] = frame.url.map(safe_key)
        before = len(frame)
        frame = frame[frame.key.notna() & ~frame.key.isin(seen)].drop_duplicates("key")
        if limit and len(frame) > limit:
            frame = frame.sample(n=limit, random_state=SEED)
        print(f"{name}: {before} candidates, {len(frame)} kept after removing previously seen or duplicate URLs")
        frames.append(frame.assign(label=label, dataset=name)[["url", "label", "dataset"]])

    add("fresh_openphish", (evidence_dir / "openphish_feed.txt").read_text().splitlines(), 1)
    add("fresh_phishing_database", (evidence_dir / "pdb_links_active.txt").read_text(errors="replace").splitlines(), 1, 3000)
    tranco = pd.read_csv(external_dir / "top-1m.csv", header=None, names=["rank", "domain"])
    tranco = tranco[(tranco["rank"] > 20_000) & (tranco["rank"] <= 100_000)]
    add("fresh_tranco_rank_20k_100k_roots", [f"https://{d}/" for d in tranco.domain], 0, 2000)
    mitake = pd.read_csv(external_dir / "mitake.csv")
    add("fresh_mitake_benign", mitake.loc[mitake.label == 0, "url"].astype(str).tolist(), 0, 3000)
    add("fresh_mitake_phishing", mitake.loc[mitake.label == 1, "url"].astype(str).tolist(), 1, 3000)

    sets = pd.concat(frames, ignore_index=True)
    sets.to_csv(SETS_PATH, index=False)
    manifest = {
        "sets_sha256": hashlib.sha256(SETS_PATH.read_bytes()).hexdigest(),
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (evidence_dir / "openphish_feed.txt", evidence_dir / "pdb_links_active.txt")},
        "rows": {k: int(v) for k, v in sets.dataset.value_counts(sort=False).items()},
    }
    (ROOT / "results/fresh_eval_sets_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def frozen_sets() -> pd.DataFrame:
    manifest = json.loads((ROOT / "results/fresh_eval_sets_manifest.json").read_text())
    if hashlib.sha256(SETS_PATH.read_bytes()).hexdigest() != manifest["sets_sha256"]:
        raise RuntimeError("results/fresh_eval_sets.csv changed after it was frozen")
    return pd.read_csv(SETS_PATH)


def save_scores(label: str, sets: pd.DataFrame, probabilities: list[float]) -> None:
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    sets.assign(probability=probabilities).to_csv(SCORES_DIR / f"{label}.csv", index=False)
    print(f"{label}: scored {len(sets)} rows, {int(np.isnan(probabilities).sum())} errors")


def score_app(worktree: Path, label: str) -> None:
    sets = frozen_sets()
    worktree = worktree.resolve()
    sys.path.insert(0, str(worktree))
    os.chdir(worktree)
    from app.inference import URLInference

    service = URLInference()
    service.load_models()
    probabilities = []
    for url in sets.url.astype(str):
        try:
            probabilities.append(float(service.predict(url)["phishing_probability"]))
        except Exception:
            probabilities.append(float("nan"))
    save_scores(label, sets, probabilities)


def score_onnx(config_path: Path, label: str) -> None:
    import onnxruntime as ort

    sys.path.insert(0, str(ROOT))
    from src.charcnn_bilstm import encode_url_bilstm

    sets = frozen_sets()
    config = json.loads(config_path.read_text())
    onnx_path = config_path.parent / config_path.name.replace("_config.json", ".onnx")
    if hashlib.sha256(onnx_path.read_bytes()).hexdigest() != config["onnx_sha256"]:
        raise RuntimeError(f"{onnx_path} does not match {config_path}")
    vocab = json.loads((ROOT / "models/charcnn_bilstm_vocab.json").read_text())
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    probabilities = []
    for url in sets.url.astype(str):
        try:
            logit = session.run(["logits"], {"input": encode_url_bilstm(url, vocab, config["max_sequence_length"])[None, :]})[0]
            probabilities.append(float(1.0 / (1.0 + np.exp(-np.asarray(logit).ravel()[0]))))
        except Exception:
            probabilities.append(float("nan"))
    save_scores(label, sets, probabilities)


def mcnemar_exact(a_correct: np.ndarray, b_correct: np.ndarray) -> dict[str, object]:
    from scipy.stats import binomtest

    only_a = int((a_correct & ~b_correct).sum())
    only_b = int((~a_correct & b_correct).sum())
    p = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    return {"only_first_correct": only_a, "only_second_correct": only_b, "p_value": float(p)}


def compare() -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    from compare_production_models import metrics

    scores = {label: pd.read_csv(SCORES_DIR / f"{label}.csv", keep_default_na=False, na_values=[""]) for label in LABELS}
    base = scores[LABELS[0]]
    valid = np.logical_and.reduce([s.probability.notna().to_numpy() for s in scores.values()])
    groups = {name: base.dataset.eq(name).to_numpy() for name in dict.fromkeys(base.dataset)}
    groups = {name: mask for name, mask in groups.items() if mask.any()}
    groups["fresh_phishing_openphish_plus_phishing_database"] = base.dataset.isin(["fresh_openphish", "fresh_phishing_database"]).to_numpy()
    groups["fresh_benign_tranco_plus_mitake"] = base.dataset.isin(["fresh_tranco_rank_20k_100k_roots", "fresh_mitake_benign"]).to_numpy()
    groups["fresh_all"] = np.ones(len(base), dtype=bool)

    output: dict[str, object] = {"excluded_due_to_errors": int((~valid).sum()), "metrics": {}, "mcnemar": {}}
    for name, mask in groups.items():
        rows = mask & valid
        y = base.label.to_numpy(int)[rows]
        output["metrics"][name] = {label: metrics(y, s.probability.to_numpy(float)[rows]) for label, s in scores.items()}
        correct = {label: ((s.probability.to_numpy(float)[rows] >= 0.5).astype(int) == y) for label, s in scores.items()}
        output["mcnemar"][name] = {
            "bilstm_v3_2_hostsplit_vs_bilstm_v3_1": mcnemar_exact(correct["bilstm_v3_2_hostsplit"], correct["bilstm_v3_1"]),
            "bilstm_v3_2_hostsplit_vs_previous_ensemble": mcnemar_exact(correct["bilstm_v3_2_hostsplit"], correct["previous_ensemble"]),
            "bilstm_v3_1_vs_previous_ensemble": mcnemar_exact(correct["bilstm_v3_1"], correct["previous_ensemble"]),
        }

    m = output["metrics"]
    cand, cur, prev = "bilstm_v3_2_hostsplit", "bilstm_v3_1", "previous_ensemble"
    primary = "fresh_openphish" if "fresh_openphish" in m else "fresh_phishing_database"
    primary_gain = m[primary][cand]["recall"] - m[primary][cur]["recall"]
    phish = m["fresh_phishing_openphish_plus_phishing_database"]
    benign = m["fresh_benign_tranco_plus_mitake"]
    checks = {
        f"{primary}_recall_gain_at_least_2pp": primary_gain >= 0.02,
        "combined_phishing_recall_not_lower": phish[cand]["recall"] >= phish[cur]["recall"],
        "benign_fpr_increase_at_most_1pp": benign[cand]["fpr"] - benign[cur]["fpr"] <= 0.01,
    }
    served = cand if all(checks.values()) else cur
    output["decision"] = {
        "checks": checks,
        "primary_set": primary,
        "primary_recall_gain": primary_gain,
        "selected_bilstm": served,
        "selected_bilstm_within_2pp_of_previous_ensemble_phishing_recall": bool(phish[served]["recall"] >= phish[prev]["recall"] - 0.02),
    }
    RESULTS_PATH.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output["decision"], indent=2))
    print(f"Wrote {RESULTS_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--evidence-dir", type=Path, required=True)
    b.add_argument("--external-dir", type=Path, required=True)
    a = sub.add_parser("score-app")
    a.add_argument("--worktree", type=Path, required=True)
    a.add_argument("--label", choices=LABELS, required=True)
    o = sub.add_parser("score-onnx")
    o.add_argument("--config", type=Path, required=True)
    o.add_argument("--label", choices=LABELS, required=True)
    sub.add_parser("compare")
    args = parser.parse_args()
    if args.command == "build":
        build(args.evidence_dir.resolve(), args.external_dir.resolve())
    elif args.command == "score-app":
        score_app(args.worktree, args.label)
    elif args.command == "score-onnx":
        score_onnx(args.config.resolve(), args.label)
    else:
        compare()


if __name__ == "__main__":
    main()
