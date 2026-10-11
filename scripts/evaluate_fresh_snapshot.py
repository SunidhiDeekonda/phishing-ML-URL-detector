"""Score the served models on a fresh external snapshot that no model or decision has seen.

Each snapshot ``--name`` has its own frozen set (results/{name}_eval_sets.csv),
scores (tmp/{name}_scores/) and results (results/{name}_snapshot_evaluation.json).
Plans and decision rules: results/hostsplit_preregistration.json (``fresh``) and
results/v3_3_preregistration.json (``fresh2``). URLs are processed as text only;
nothing is fetched.

Usage:
    python scripts/evaluate_fresh_snapshot.py --name fresh build --evidence-dir tmp/fresh --external-dir tmp/external
    git worktree add tmp/wt_previous f16143e
    python scripts/evaluate_fresh_snapshot.py --name fresh score-app --worktree tmp/wt_previous --label previous_ensemble
    python scripts/evaluate_fresh_snapshot.py --name fresh score-onnx --config models/charcnn_bilstm_config.json --label bilstm_v3_1
    python scripts/evaluate_fresh_snapshot.py --name fresh score-onnx --config models/charcnn_bilstm_hostsplit_config.json --label bilstm_v3_2_hostsplit
    python scripts/evaluate_fresh_snapshot.py --name fresh compare
    python scripts/evaluate_fresh_snapshot.py --name fresh2 build --evidence-dir tmp/fresh2 --external-dir tmp/external
    ... score the same way with --label bilstm_v3_3 for models/charcnn_bilstm_v3_3_config.json, then
    python scripts/evaluate_fresh_snapshot.py --name fresh2 compare
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
CANDIDATE_POOL = 40_000
RULES = {
    "fresh": ("previous_ensemble", "bilstm_v3_1", "bilstm_v3_2_hostsplit"),
    "fresh2": ("previous_ensemble", "bilstm_v3_2_hostsplit", "bilstm_v3_3"),
}


def paths(name: str) -> dict[str, Path]:
    return {
        "sets": ROOT / f"results/{name}_eval_sets.csv",
        "manifest": ROOT / f"results/{name}_eval_sets_manifest.json",
        "scores": ROOT / f"tmp/{name}_scores",
        "results": ROOT / f"results/{name}_snapshot_evaluation.json",
    }


def safe_key(url: str) -> str | None:
    sys.path.insert(0, str(ROOT))
    from src.url_normalization import canonicalize_url

    try:
        text = canonicalize_url(str(url)).lower()
    except Exception:
        return None
    text = text.split("://", 1)[-1]
    return text[4:] if text.startswith("www.") else text


def seen_keys(extra: list[Path] = ()) -> set[str]:
    urls = [
        pd.read_csv(ROOT / "data/processed/dataset.csv").url,
        pd.read_csv(ROOT / "data/processed/balanced_dataset.csv").url,
        pd.read_csv(ROOT / "results/external_eval_sets.csv").url,
    ]
    for name in ("adversarial_test.csv", "adversarial_validation.csv"):
        frame = pd.read_csv(ROOT / "data/processed" / name)
        urls += [frame.original_url, frame.mutated_url]
    urls += [pd.read_csv(path).url for path in extra if path.exists()]
    keys = {safe_key(u) for u in pd.concat(urls).astype(str)}
    keys.discard(None)
    return keys


def build(name: str, evidence_dir: Path, external_dir: Path) -> None:
    out = paths(name)
    earlier = [paths(n)["sets"] for n in RULES if n != name] + [ROOT / "data/processed/augmentation_v3_3.csv"]
    seen = seen_keys(earlier)
    rng = np.random.default_rng(SEED)
    frames = []

    def add(set_name: str, urls: list[str], label: int, limit: int | None = None) -> None:
        urls = [u.strip() for u in urls if u.strip()]
        if limit and len(urls) > CANDIDATE_POOL:
            urls = list(rng.choice(np.asarray(urls, dtype=object), CANDIDATE_POOL, replace=False))
        frame = pd.DataFrame({"url": urls})
        frame["key"] = frame.url.map(safe_key)
        before = len(frame)
        frame = frame[frame.key.notna() & ~frame.key.isin(seen)].drop_duplicates("key")
        if limit and len(frame) > limit:
            frame = frame.sample(n=limit, random_state=SEED)
        print(f"{set_name}: {before} candidates, {len(frame)} kept after removing previously seen or duplicate URLs")
        frames.append(frame.assign(label=label, dataset=set_name)[["url", "label", "dataset"]])

    add(f"{name}_openphish", (evidence_dir / "openphish_feed.txt").read_text().splitlines(), 1)
    add(f"{name}_phishing_database", (evidence_dir / "pdb_links_active.txt").read_text(errors="replace").splitlines(), 1, 3000)
    tranco = pd.read_csv(external_dir / "top-1m.csv", header=None, names=["rank", "domain"])
    tranco = tranco[(tranco["rank"] > 20_000) & (tranco["rank"] <= 100_000)]
    add(f"{name}_tranco_rank_20k_100k_roots", [f"https://{d}/" for d in tranco.domain], 0, 2000)
    mitake = pd.read_csv(external_dir / "mitake.csv")
    add(f"{name}_mitake_benign", mitake.loc[mitake.label == 0, "url"].astype(str).tolist(), 0, 3000)
    add(f"{name}_mitake_phishing", mitake.loc[mitake.label == 1, "url"].astype(str).tolist(), 1, 3000)

    sets = pd.concat(frames, ignore_index=True)
    sets.to_csv(out["sets"], index=False)
    manifest = {
        "sets_sha256": hashlib.sha256(out["sets"].read_bytes()).hexdigest(),
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (evidence_dir / "openphish_feed.txt", evidence_dir / "pdb_links_active.txt")},
        "rows": {k: int(v) for k, v in sets.dataset.value_counts(sort=False).items()},
    }
    out["manifest"].write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def frozen_sets(name: str) -> pd.DataFrame:
    out = paths(name)
    manifest = json.loads(out["manifest"].read_text())
    if hashlib.sha256(out["sets"].read_bytes()).hexdigest() != manifest["sets_sha256"]:
        raise RuntimeError(f"{out['sets']} changed after it was frozen")
    return pd.read_csv(out["sets"])


def save_scores(name: str, label: str, sets: pd.DataFrame, probabilities: list[float]) -> None:
    scores_dir = paths(name)["scores"]
    scores_dir.mkdir(parents=True, exist_ok=True)
    sets.assign(probability=probabilities).to_csv(scores_dir / f"{label}.csv", index=False)
    print(f"{label}: scored {len(sets)} rows, {int(np.isnan(probabilities).sum())} errors")


def score_app(name: str, worktree: Path, label: str) -> None:
    sets = frozen_sets(name)
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
    save_scores(name, label, sets, probabilities)


def onnx_scorer(config_path: Path):
    import onnxruntime as ort

    sys.path.insert(0, str(ROOT))
    from src.charcnn_bilstm import encode_url_bilstm

    config = json.loads(config_path.read_text())
    onnx_path = config_path.parent / config_path.name.replace("_config.json", ".onnx")
    if hashlib.sha256(onnx_path.read_bytes()).hexdigest() != config["onnx_sha256"]:
        raise RuntimeError(f"{onnx_path} does not match {config_path}")
    vocab = json.loads((ROOT / "models/charcnn_bilstm_vocab.json").read_text())
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    def score(url: str) -> float:
        try:
            logit = session.run(["logits"], {"input": encode_url_bilstm(url, vocab, config["max_sequence_length"])[None, :]})[0]
            return float(1.0 / (1.0 + np.exp(-np.asarray(logit).ravel()[0])))
        except Exception:
            return float("nan")

    return score


def score_onnx(name: str, config_path: Path, label: str) -> None:
    sets = frozen_sets(name)
    score = onnx_scorer(config_path)
    save_scores(name, label, sets, [score(u) for u in sets.url.astype(str)])


def mcnemar_exact(a_correct: np.ndarray, b_correct: np.ndarray) -> dict[str, object]:
    from scipy.stats import binomtest

    only_a = int((a_correct & ~b_correct).sum())
    only_b = int((~a_correct & b_correct).sum())
    p = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    return {"only_first_correct": only_a, "only_second_correct": only_b, "p_value": float(p)}


def hard_negative_test_false_positives(config_path: Path) -> int:
    score = onnx_scorer(config_path)
    hard = json.loads((ROOT / "tests/data/hard_legitimate_urls.json").read_text(encoding="utf-8"))
    return int(sum(score(c["url"]) >= 0.5 for c in hard if c["hard_negative_split"] == "test"))


def compare(name: str) -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    from compare_production_models import metrics

    out = paths(name)
    prev, cur, cand = RULES[name]
    labels = (prev, cur, cand)
    scores = {label: pd.read_csv(out["scores"] / f"{label}.csv", keep_default_na=False, na_values=[""]) for label in labels}
    base = scores[prev]
    valid = np.logical_and.reduce([s.probability.notna().to_numpy() for s in scores.values()])
    groups = {n: base.dataset.eq(n).to_numpy() for n in dict.fromkeys(base.dataset)}
    groups = {n: mask for n, mask in groups.items() if mask.any()}
    groups[f"{name}_phishing_openphish_plus_phishing_database"] = base.dataset.isin([f"{name}_openphish", f"{name}_phishing_database"]).to_numpy()
    groups[f"{name}_all_phishing"] = base.label.eq(1).to_numpy()
    groups[f"{name}_benign_tranco_plus_mitake"] = base.label.eq(0).to_numpy()
    groups[f"{name}_all"] = np.ones(len(base), dtype=bool)

    output: dict[str, object] = {"excluded_due_to_errors": int((~valid).sum()), "metrics": {}, "mcnemar": {}}
    for group, mask in groups.items():
        rows = mask & valid
        y = base.label.to_numpy(int)[rows]
        output["metrics"][group] = {label: metrics(y, s.probability.to_numpy(float)[rows]) for label, s in scores.items()}
        correct = {label: ((s.probability.to_numpy(float)[rows] >= 0.5).astype(int) == y) for label, s in scores.items()}
        output["mcnemar"][group] = {
            f"{cand}_vs_{cur}": mcnemar_exact(correct[cand], correct[cur]),
            f"{cand}_vs_{prev}": mcnemar_exact(correct[cand], correct[prev]),
            f"{cur}_vs_{prev}": mcnemar_exact(correct[cur], correct[prev]),
        }

    m = output["metrics"]
    benign = m[f"{name}_benign_tranco_plus_mitake"]
    if name == "fresh":
        primary = "fresh_openphish" if "fresh_openphish" in m else "fresh_phishing_database"
        primary_gain = m[primary][cand]["recall"] - m[primary][cur]["recall"]
        phish = m["fresh_phishing_openphish_plus_phishing_database"]
        checks = {
            f"{primary}_recall_gain_at_least_2pp": primary_gain >= 0.02,
            "combined_phishing_recall_not_lower": phish[cand]["recall"] >= phish[cur]["recall"],
            "benign_fpr_increase_at_most_1pp": benign[cand]["fpr"] - benign[cur]["fpr"] <= 0.01,
        }
        decision = {"primary_set": primary, "primary_recall_gain": primary_gain}
        phish_for_prev = phish
    else:
        configs = {cur: ROOT / "models/charcnn_bilstm_hostsplit_config.json", cand: ROOT / "models/charcnn_bilstm_v3_3_config.json"}
        hard_fp = {label: hard_negative_test_false_positives(path) for label, path in configs.items()}
        phish = m[f"{name}_all_phishing"]
        openphish = m[f"{name}_openphish"]
        checks = {
            "openphish_recall_not_lower": openphish[cand]["recall"] >= openphish[cur]["recall"],
            "all_phishing_recall_gain_at_least_2pp": phish[cand]["recall"] - phish[cur]["recall"] >= 0.02,
            "benign_fpr_increase_at_most_1pp": benign[cand]["fpr"] - benign[cur]["fpr"] <= 0.01,
            "hard_negative_test_false_positives_at_most_plus_1": hard_fp[cand] <= hard_fp[cur] + 1,
        }
        decision = {"hard_negative_test_false_positives": hard_fp}
        phish_for_prev = phish
    served = cand if all(checks.values()) else cur
    output["decision"] = {
        "checks": checks,
        **decision,
        "selected_bilstm": served,
        "selected_bilstm_within_2pp_of_previous_ensemble_phishing_recall": bool(phish_for_prev[served]["recall"] >= phish_for_prev[prev]["recall"] - 0.02),
    }
    out["results"].write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output["decision"], indent=2))
    print(f"Wrote {out['results']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", choices=tuple(RULES), default="fresh")
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--evidence-dir", type=Path, required=True)
    b.add_argument("--external-dir", type=Path, required=True)
    a = sub.add_parser("score-app")
    a.add_argument("--worktree", type=Path, required=True)
    a.add_argument("--label", required=True)
    o = sub.add_parser("score-onnx")
    o.add_argument("--config", type=Path, required=True)
    o.add_argument("--label", required=True)
    sub.add_parser("compare")
    args = parser.parse_args()
    if args.command == "build":
        build(args.name, args.evidence_dir.resolve(), args.external_dir.resolve())
    elif args.command == "score-app":
        score_app(args.name, args.worktree, args.label)
    elif args.command == "score-onnx":
        score_onnx(args.name, args.config.resolve(), args.label)
    else:
        compare(args.name)


if __name__ == "__main__":
    main()
