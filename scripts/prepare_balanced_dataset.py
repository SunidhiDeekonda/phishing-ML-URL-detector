"""Prepare a balanced, path-aware dataset for training the modern CharCNN-BiLSTM model.

This addresses the critical dataset flaw in the original PhishX dataset where
legitimate URLs had 0 paths, causing models to correlate any URL path with phishing.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

SEED = 42
TARGET_LEGIT = 20_000
TARGET_PHISH = 20_000


def has_valid_path(url: str) -> bool:
    target = url if "://" in url else f"http://{url}"
    try:
        parsed = urlparse(target)
        path = parsed.path.strip("/")
        return len(path) > 0
    except Exception:
        return False


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)

    print("Step 1: Loading curated hard-negative legitimate URLs...")
    hard_cases_path = ROOT / "tests/data/hard_legitimate_urls.json"
    hard_legit_urls = []
    if hard_cases_path.exists():
        hard_cases = json.loads(hard_cases_path.read_text(encoding="utf-8"))
        hard_legit_urls = [c["url"] for c in hard_cases]
        print(f"Loaded {len(hard_legit_urls)} curated hard-negative URLs.")

    print("Step 2: Loading alexkstern/phishing_urls from Hugging Face...")
    ds_alex = load_dataset("alexkstern/phishing_urls", split="train")
    df_alex = pd.DataFrame(ds_alex)

    alex_legit_raw = df_alex[df_alex["label"] == 0]["text"].dropna().unique().tolist()
    alex_phish_raw = df_alex[df_alex["label"] == 1]["text"].dropna().unique().tolist()
    print(f"alexkstern raw: {len(alex_legit_raw)} legit, {len(alex_phish_raw)} phish")

    # Filter legit URLs: prioritize those with paths
    alex_legit_with_paths = [u for u in alex_legit_raw if has_valid_path(u)]
    alex_legit_roots = [u for u in alex_legit_raw if not has_valid_path(u)]
    print(f"alexkstern legit: {len(alex_legit_with_paths)} with paths, {len(alex_legit_roots)} roots")

    print("Step 3: Loading pirocheto/phishing-url from Hugging Face...")
    ds_piro = load_dataset("pirocheto/phishing-url", split="train")
    df_piro = pd.DataFrame(ds_piro)
    piro_legit_raw = df_piro[df_piro["status"] == "legitimate"]["url"].dropna().unique().tolist()
    piro_phish_raw = df_piro[df_piro["status"] == "phishing"]["url"].dropna().unique().tolist()
    piro_legit_with_paths = [u for u in piro_legit_raw if has_valid_path(u)]
    print(f"pirocheto raw: {len(piro_legit_with_paths)} legit with paths, {len(piro_phish_raw)} phish")

    print("Step 4: Loading existing local raw datasets...")
    local_legit_file = RAW_DIR / "legit_urls.csv"
    local_phish_file = RAW_DIR / "phishing_urls.csv"

    local_legit = pd.read_csv(local_legit_file)["url"].dropna().unique().tolist() if local_legit_file.exists() else []
    local_phish = pd.read_csv(local_phish_file)["url"].dropna().unique().tolist() if local_phish_file.exists() else []
    print(f"local raw: {len(local_legit)} legit, {len(local_phish)} phish")

    # Generate realistic legitimate auth/security paths on top reputable domains
    COMMON_BENIGN_PATHS = [
        "/login",
        "/signin",
        "/sign-in",
        "/account",
        "/accounts/login",
        "/auth/login",
        "/security",
        "/docs/security",
        "/users/login",
        "/support",
        "/dashboard",
        "/verify",
        "/settings/security",
        "/help",
        "/privacy",
        "/download",
    ]
    legit_auth_urls = []
    top_reputable_roots = local_legit[:600]
    for root in top_reputable_roots:
        clean_root = root.rstrip("/")
        for path in COMMON_BENIGN_PATHS:
            legit_auth_urls.append(f"{clean_root}{path}")
    print(f"Generated {len(legit_auth_urls)} legitimate auth/security URLs on top reputable domains.")

    # Combine Legitimate URLs:
    selected_legit = set()
    for u in hard_legit_urls:
        selected_legit.add(u)

    # Add legitimate auth URLs
    for u in legit_auth_urls:
        selected_legit.add(u)

    random.shuffle(alex_legit_with_paths)
    for u in alex_legit_with_paths:
        if len(selected_legit) >= 16_000:
            break
        selected_legit.add(u)

    random.shuffle(piro_legit_with_paths)
    for u in piro_legit_with_paths:
        if len(selected_legit) >= 18_000:
            break
        selected_legit.add(u)

    # Add root domains to complete 20,000
    random.shuffle(local_legit)
    for u in local_legit:
        if len(selected_legit) >= TARGET_LEGIT:
            break
        selected_legit.add(u)

    selected_legit_list = list(selected_legit)[:TARGET_LEGIT]
    print(f"Total curated legitimate URLs: {len(selected_legit_list)}")
    legit_paths_count = sum(1 for u in selected_legit_list if has_valid_path(u))
    legit_login_count = sum(1 for u in selected_legit_list if "login" in u.lower() or "sign-in" in u.lower())
    print(f"Legitimate URLs with deep paths: {legit_paths_count} / {len(selected_legit_list)} ({legit_paths_count / len(selected_legit_list):.1%})")
    print(f"Legitimate URLs with login/sign-in: {legit_login_count} / {len(selected_legit_list)} ({legit_login_count / len(selected_legit_list):.1%})")

    # Combine Phishing URLs:
    selected_phish = set()
    random.shuffle(alex_phish_raw)
    for u in alex_phish_raw:
        if len(selected_phish) >= 12_000:
            break
        selected_phish.add(u)

    random.shuffle(local_phish)
    for u in local_phish:
        if len(selected_phish) >= TARGET_PHISH:
            break
        selected_phish.add(u)

    selected_phish_list = list(selected_phish)[:TARGET_PHISH]
    print(f"Total curated phishing URLs: {len(selected_phish_list)}")
    phish_paths_count = sum(1 for u in selected_phish_list if has_valid_path(u))
    print(f"Phishing URLs with deep paths: {phish_paths_count} / {len(selected_phish_list)} ({phish_paths_count / len(selected_phish_list):.1%})")

    # Build final balanced dataframe
    records = []
    for u in selected_legit_list:
        records.append({"url": u, "label": 0})
    for u in selected_phish_list:
        records.append({"url": u, "label": 1})

    df = pd.DataFrame(records)
    # Shuffle
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    out_path = PROCESSED_DIR / "balanced_dataset.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved balanced dataset to {out_path} with {len(df)} rows.")
    print("Class distribution:\n", df["label"].value_counts())


if __name__ == "__main__":
    main()
