#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import re
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

from src.features import build_features_dataframe
from src.char_tokenizer import build_vocab, encode_urls, save_vocab

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
APP_DIR = BASE_DIR / "app"
TEST_DIR = BASE_DIR / "tests"
SCRIPTS_DIR = BASE_DIR / "scripts"

RUN_LOG = BASE_DIR / "RUN_LOG.md"
DATASET_PATH = PROCESSED_DIR / "dataset.csv"
FEATURES_PATH = PROCESSED_DIR / "features.csv"
VOCAB_PATH = PROCESSED_DIR / "char_vocab.json"
SEQUENCES_PATH = PROCESSED_DIR / "char_sequences.npy"
TRAIN_IDX_PATH = PROCESSED_DIR / "train_idx.npy"
VAL_IDX_PATH = PROCESSED_DIR / "val_idx.npy"
TEST_IDX_PATH = PROCESSED_DIR / "test_idx.npy"

PHISHING_URL = "https://raw.githubusercontent.com/dubeyrudra-1808/PhishX/main/data/raw/phishing_urls.csv"
LEGIT_URL = "https://raw.githubusercontent.com/dubeyrudra-1808/PhishX/main/data/raw/legit_urls.csv"
SEED = 42
SAMPLE_SIZE = 10_000


def log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    with RUN_LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"[{timestamp}] {message}\n")
    print(message)


def get_system_snapshot() -> dict:
    snapshot = {
        "os": platform.platform(),
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
        "pip_version": __import__("pip").__version__,
    }

    try:
        disk = shutil.disk_usage(BASE_DIR)
        snapshot["disk_gb"] = {
            "total": round(disk.total / (1024 ** 3), 2),
            "used": round(disk.used / (1024 ** 3), 2),
            "free": round(disk.free / (1024 ** 3), 2),
        }
    except Exception:
        snapshot["disk_gb"] = {"total": None, "used": None, "free": None}

    if hasattr(os, "sysconf"):
        try:
            if sys.platform == "darwin":
                mem_bytes = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).decode().strip())
                snapshot["ram_gb"] = round(mem_bytes / (1024 ** 3), 2)
            else:
                with open("/proc/meminfo", "r", encoding="utf-8") as fp:
                    for line in fp:
                        if line.startswith("MemTotal:"):
                            kb = int(line.split()[1])
                            snapshot["ram_gb"] = round(kb / 1024 / 1024, 2)
                            break
        except Exception:
            snapshot["ram_gb"] = None
    else:
        snapshot["ram_gb"] = None

    try:
        git_version = subprocess.check_output(["git", "--version"], text=True).strip()
        snapshot["git_version"] = git_version
    except Exception:
        snapshot["git_version"] = "git not available"

    snapshot["torch_version"] = torch.__version__
    snapshot["mps_available"] = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    snapshot["cuda_available"] = bool(torch.cuda.is_available())

    if snapshot["mps_available"]:
        snapshot["preferred_device"] = "mps"
    elif snapshot["cuda_available"]:
        snapshot["preferred_device"] = "cuda"
    else:
        snapshot["preferred_device"] = "cpu"

    return snapshot


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_dataset(url: str, destination: Path) -> None:
    import urllib.request

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()

    log(f"Downloading dataset snapshot from {url}")
    with urllib.request.urlopen(url, timeout=120) as response, destination.open("wb") as out:
        out.write(response.read())


def select_class_rows(path: Path, class_label: int, sample_size: int = SAMPLE_SIZE) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(path)
    original_rows = len(df)

    url_column = None
    for candidate in ["url", "URL", "link", "url_text", "urls", "link_url", "full_url"]:
        if candidate in df.columns:
            url_column = candidate
            break

    if url_column is None:
        # choose first object-like column as fallback
        object_cols = [col for col in df.columns if df[col].dtype == object]
        if not object_cols:
            raise ValueError(f"No object URL column found in {path}")
        url_column = object_cols[0]

    cleaned = df[[url_column]].copy()
    cleaned = cleaned.rename(columns={url_column: "url"})
    null_count = int(cleaned["url"].isna().sum())
    cleaned = cleaned.dropna(subset=["url"])
    cleaned["url"] = cleaned["url"].astype(str).str.strip()

    duplicate_count = int(cleaned.duplicated(subset=["url"]).sum())
    cleaned = cleaned.drop_duplicates(subset=["url"]).copy()
    cleaned_rows = len(cleaned)

    if cleaned_rows < sample_size:
        raise RuntimeError(
            f"Insufficient unique cleaned rows in {path.name}: {cleaned_rows} < {sample_size}"
        )

    sampled = cleaned.sample(n=sample_size, random_state=SEED).copy()
    sampled["label"] = int(class_label)

    return sampled.reset_index(drop=True), {
        "source_file": path.name,
        "original_rows": original_rows,
        "null_rows": null_count,
        "duplicate_rows": duplicate_count,
        "cleaned_unique_rows": cleaned_rows,
        "sampled_rows": len(sampled),
        "sha256": sha256_file(path),
    }


def run_pipeline() -> dict:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    report = {
        "system": get_system_snapshot(),
        "dataset": {},
        "features": {},
        "char": {},
        "split": {},
        "tests": {
            "passed": 0,
            "failed": 0,
        },
        "warnings": [],
    }

    phishing_raw = RAW_DIR / "phishing_urls.csv"
    legit_raw = RAW_DIR / "legit_urls.csv"
    download_dataset(PHISHING_URL, phishing_raw)
    download_dataset(LEGIT_URL, legit_raw)

    phishing_df, phishing_meta = select_class_rows(phishing_raw, class_label=1)
    legitimate_df, legit_meta = select_class_rows(legit_raw, class_label=0)

    dataset = pd.concat([phishing_df, legitimate_df], axis=0, ignore_index=True)
    dataset = dataset.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    if len(dataset) != 2 * SAMPLE_SIZE:
        raise RuntimeError(f"Expected 20,000 records, got {len(dataset)}")

    dataset.to_csv(DATASET_PATH, index=False)

    dataset_counts = dataset["label"].value_counts().sort_index().to_dict()

    feature_frame = build_features_dataframe(dataset["url"])
    feature_frame = feature_frame.astype("float64")
    feature_frame.to_csv(FEATURES_PATH, index=False)

    vocab = build_vocab(dataset["url"])
    save_vocab(vocab, VOCAB_PATH)
    sequences = encode_urls(dataset["url"], vocab, max_len=200)
    np.save(SEQUENCES_PATH, sequences)

    if sequences.shape != (len(dataset), 200):
        raise RuntimeError(f"Unexpected sequence shape: {sequences.shape}")

    X = dataset.index.to_numpy()
    y = dataset["label"].to_numpy()

    train_idx, temp_idx, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.3,
        stratify=y,
        random_state=SEED,
    )

    val_idx, test_idx, y_val, y_test = train_test_split(
        temp_idx,
        y_temp,
        test_size=2 / 3,
        stratify=y_temp,
        random_state=SEED,
    )

    np.save(TRAIN_IDX_PATH, train_idx.astype(np.int64))
    np.save(VAL_IDX_PATH, val_idx.astype(np.int64))
    np.save(TEST_IDX_PATH, test_idx.astype(np.int64))

    split_sizes = {
        "train": int(len(train_idx)),
        "val": int(len(val_idx)),
        "test": int(len(test_idx)),
    }

    if len(set(train_idx).intersection(set(val_idx))) or len(set(train_idx).intersection(set(test_idx))) or len(set(val_idx).intersection(set(test_idx))):
        raise RuntimeError("Splits are not disjoint")

    try:
        import lightgbm  # noqa: F401
    except Exception as exc:
        report["warnings"].append(f"LightGBM runtime issue detected in environment: {exc}")

    # write data provenance
    provenance_lines = [
        "# Data Provenance",
        "",
        "Reference repository: https://github.com/dubeyrudra-1808/PhishX.git",
        f"Phishing source URL: {PHISHING_URL}",
        f"Legitimate source URL: {LEGIT_URL}",
        f"Download timestamp (UTC): {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Raw phishing source snapshot",
        f"Original rows: {phishing_meta['original_rows']}",
        f"Null rows: {phishing_meta['null_rows']}",
        f"Duplicate rows (after null drop): {phishing_meta['duplicate_rows']}",
        f"Unique cleaned rows: {phishing_meta['cleaned_unique_rows']}",
        f"SHA256: {phishing_meta['sha256']}",
        "",
        "## Raw legitimate source snapshot",
        f"Original rows: {legit_meta['original_rows']}",
        f"Null rows: {legit_meta['null_rows']}",
        f"Duplicate rows (after null drop): {legit_meta['duplicate_rows']}",
        f"Unique cleaned rows: {legit_meta['cleaned_unique_rows']}",
        f"SHA256: {legit_meta['sha256']}",
        "",
        "## Sampling strategy",
        "Deterministic sampling: pandas.DataFrame.sample with random_state=42",
        "Stratified by source class before merge.",
        "Phishing sampled rows: 10000",
        "Legitimate sampled rows: 10000",
        "Final rows: 20000",
        "",
        "The complete source dataset contains substantially more URLs, but a balanced deterministic subset of 20,000 URLs was selected for this local reproduction experiment to reduce computational requirements while preserving both classes.",
    ]
    (BASE_DIR / "DATA_PROVENANCE.md").write_text("\n".join(provenance_lines), encoding="utf-8")

    report["dataset"] = {
        "phishing_source_rows": phishing_meta["original_rows"],
        "legit_source_rows": legit_meta["original_rows"],
        "phishing_unique_rows": phishing_meta["cleaned_unique_rows"],
        "legit_unique_rows": legit_meta["cleaned_unique_rows"],
        "phishing_selected": phishing_meta["sampled_rows"],
        "legit_selected": legit_meta["sampled_rows"],
        "final_rows": int(len(dataset)),
        "class_balance": dataset_counts,
        "phishing_sha256": phishing_meta["sha256"],
        "legit_sha256": legit_meta["sha256"],
        "dataset_file": str(DATASET_PATH),
    }

    report["features"] = {
        "feature_count": int(feature_frame.shape[1]),
        "features_shape": tuple(feature_frame.shape),
    }

    char_file_size = SEQUENCES_PATH.stat().st_size / (1024 ** 2)
    report["char"] = {
        "vocab_size": len(vocab),
        "max_seq_len": 200,
        "sequences_shape": tuple(sequences.shape),
        "char_file_size_mb": round(char_file_size, 4),
        "vocab_file": str(VOCAB_PATH),
    }

    report["split"] = {
        "train": split_sizes["train"],
        "val": split_sizes["val"],
        "test": split_sizes["test"],
    }

    # run tests
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests"],
        cwd=BASE_DIR,
        text=True,
        capture_output=True,
    )
    test_stdout = result.stdout.strip()
    test_stderr = result.stderr.strip()
    passed_match = re.search(r"(\d+) passed", test_stdout)
    failed_match = re.search(r"(\d+) failed", test_stdout)
    error_match = re.search(r"(\d+) errors", test_stdout)
    passed_count = int(passed_match.group(1)) if passed_match else 0
    failed_count = int(failed_match.group(1)) if failed_match else 0
    error_count = int(error_match.group(1)) if error_match else 0
    if result.returncode != 0 and failed_count == 0 and error_count == 0:
        failed_count = 1
    report["tests"]["passed"] = passed_count
    report["tests"]["failed"] = failed_count + error_count
    report["tests"]["return_code"] = result.returncode
    report["tests"]["stdout"] = test_stdout
    report["tests"]["stderr"] = test_stderr

    log("Pytest completed")
    log(test_stdout)
    if test_stderr:
        log(test_stderr)
    if result.returncode != 0:
        report["warnings"].append(f"Pytest non-zero return code: {result.returncode}")

    return report


def print_report(report: dict) -> None:
    system = report["system"]
    dataset = report["dataset"]
    features = report["features"]
    split = report["split"]
    char = report["char"]
    tests = report["tests"]

    print("PHASE 1 REPORT")
    print("Machine")
    print("-------")
    print(f"OS: {system['os']}")
    print(f"CPU: {system['architecture']}")
    print(f"RAM: {system.get('ram_gb', 'unknown')} GB")
    print(f"Python: {system['python_version']}")
    print(f"Torch: {system['torch_version']}")
    print(f"MPS: {system['mps_available']}")
    print(f"CUDA: {system['cuda_available']}")
    print(f"Selected future training device: {system['preferred_device']}")

    print("Dependencies")
    print("------------")
    try:
        import lightgbm

        print("LightGBM import: OK")
    except Exception as exc:
        print(f"LightGBM import: FAIL ({exc})")

    try:
        import torch

        print("Torch import: OK")
    except Exception as exc:
        print(f"Torch import: FAIL ({exc})")

    print("Other issues: see warnings")

    print("\nDataset")
    print("-------")
    print(f"Original phishing rows: {dataset.get('phishing_source_rows')}")
    print(f"Original legitimate rows: {dataset.get('legit_source_rows')}")
    print(f"Unique phishing available: {dataset.get('phishing_unique_rows')}")
    print(f"Unique legitimate available: {dataset.get('legit_unique_rows')}")
    print(f"Selected phishing: {dataset.get('phishing_selected')}")
    print(f"Selected legitimate: {dataset.get('legit_selected')}")
    print(f"Final dataset size: {dataset.get('final_rows')}")
    print(f"Class balance: {dataset.get('class_balance')}")
    print("Dataset SHA / provenance saved: Yes")

    print("\nFeatures")
    print("--------")
    print(f"Final numeric feature count: {features.get('feature_count')}")
    print(f"features.csv shape: {features.get('features_shape')}")

    print("\nCharacter encoding")
    print("------------------")
    print(f"Vocabulary size: {char.get('vocab_size')}")
    print(f"Sequence max length: {char.get('max_seq_len')}")
    print(f"char_sequences.npy shape: {char.get('sequences_shape')}")
    print(f"file size: {char.get('char_file_size_mb')} MB")

    print("\nSplit")
    print("-----")
    print(f"Train: {split.get('train')}")
    print(f"Validation: {split.get('val')}")
    print(f"Test: {split.get('test')}")

    print("\nTests")
    print("-----")
    print(f"Passed: {tests.get('passed', 0)}")
    print(f"Failed: {tests.get('failed', 0)}")
    if tests.get("return_code") != 0:
        print(tests.get("stderr"))

    print("\nFiles")
    print("-----")
    created = [
        str(DATASET_PATH),
        str(FEATURES_PATH),
        str(VOCAB_PATH),
        str(SEQUENCES_PATH),
        str(TRAIN_IDX_PATH),
        str(VAL_IDX_PATH),
        str(TEST_IDX_PATH),
    ]
    print("List important created files:")
    for path in created:
        print(f"- {path}")

    print("\nWarnings / blockers")
    print("-------------------")
    if report.get("warnings"):
        for warning in report["warnings"]:
            print(f"- {warning}")
    else:
        print("No blockers detected")


if __name__ == "__main__":
    log("Starting Phase 1 pipeline")
    report = run_pipeline()
    print_report(report)
