"""Build data/processed/augmentation_v3_3.csv: extra real training URLs for the BiLSTM.

Why: the V2 dataset labels 9,600 generated ``<reputable root> + /login``-style
URLs as legitimate, and real phishing with short clean paths or bare
random-looking domains is scarce, so the model learned short login paths as a
legitimate cue (results/v3_3_preregistration.json).

Sources (text only, nothing is fetched):
* phishing: Phishing.Database ACTIVE links, minus bare roots of Tranco top-100k domains;
* legitimate: unused alexkstern legitimate URLs with a path, and Tranco rank
  100k-1M domains as roots, both minus hosts in the Phishing.Database domain list.
Every URL in any evaluation set and every V2 row is excluded.

Usage:
    python scripts/build_augmentation_data.py --evidence-dir tmp/fresh2 --external-dir tmp/external
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_fresh_snapshot import RULES, paths, safe_key, seen_keys  # noqa: E402

SEED = 42
OUTPUT = ROOT / "data/processed/augmentation_v3_3.csv"
COUNTS = {"phishing_database": 12_000, "alexkstern_legitimate": 9_000, "tranco_rank_100k_1m_roots": 3_000}


def host(key: str) -> str:
    return key.split("/", 1)[0].split("?", 1)[0].split(":", 1)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--external-dir", type=Path, required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(SEED)

    seen = seen_keys([paths(name)["sets"] for name in RULES])
    tranco = pd.read_csv(args.external_dir / "top-1m.csv", header=None, names=["rank", "domain"])
    top100k = set(tranco.loc[tranco["rank"] <= 100_000, "domain"].str.lower())
    phishing_domains = {d.strip().lower() for d in (args.external_dir / "pdb_phishing-domains-ACTIVE.txt").read_text(errors="replace").splitlines() if d.strip()}

    def take(source: str, urls: list[str], label: int, keep) -> pd.DataFrame:
        urls = [u.strip() for u in urls if isinstance(u, str) and u.strip()]
        urls = list(rng.permutation(np.asarray(urls, dtype=object))[: COUNTS[source] * 4])
        frame = pd.DataFrame({"url": urls})
        frame["key"] = frame.url.map(safe_key)
        frame = frame[frame.key.notna() & ~frame.key.isin(seen)].drop_duplicates("key")
        frame = frame[frame.key.map(keep)].head(COUNTS[source])
        seen.update(frame.key)
        print(f"{source}: {len(frame)} rows")
        return frame.assign(label=label, source=source)[["url", "label", "source"]]

    pdb = (args.evidence_dir / "pdb_links_active.txt").read_text(errors="replace").splitlines()
    phishing = take("phishing_database", pdb, 1, lambda k: not ("/" not in k.rstrip("/") and host(k) in top100k))
    alex = pd.read_csv(args.external_dir / "alex_train.csv")
    alex_legit = alex.loc[alex.label == 0, "text"].astype(str).tolist()
    legit_paths = take("alexkstern_legitimate", alex_legit, 0, lambda k: "/" in k.rstrip("/") and host(k) not in phishing_domains)
    ranked = tranco.loc[tranco["rank"] > 100_000, "domain"]
    legit_roots = take("tranco_rank_100k_1m_roots", [f"https://{d}/" for d in ranked], 0, lambda k: host(k) not in phishing_domains)

    data = pd.concat([phishing, legit_paths, legit_roots], ignore_index=True)
    data.to_csv(OUTPUT, index=False)
    print(json.dumps({"rows": int(len(data)), "class_counts": data.label.value_counts().sort_index().to_dict(),
                      "sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest()}, indent=2))


if __name__ == "__main__":
    main()
