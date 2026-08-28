"""URL feature extraction utilities for Phase 1."""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, Iterable, List
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
import tldextract

SUSPICIOUS_TOKENS: List[str] = [
    "login",
    "signin",
    "secure",
    "webscr",
    "bank",
    "verify",
    "update",
    "account",
    "confirm",
]

SUSPICIOUS_TLDS = {
    "tk",
    "ml",
    "ga",
    "cf",
    "top",
    "gq",
    "info",
    "xyz",
    "club",
    "loan",
    "work",
    "buzz",
    "country",
    "kim",
}

VOWELS = set("aeiou")


def _extract_text(url: str) -> str:
    return "" if url is None else str(url).strip()


def _to_int(value: bool | int | float) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and float(value).is_integer():
        return int(value)
    return int(value)


def _safe_entropy(text: str) -> float:
    text = _extract_text(text)
    if not text:
        return 0.0
    counts = Counter(text)
    length = len(text)
    entropy = 0.0
    for count in counts.values():
        p = count / length
        entropy -= p * math.log2(p)
    return float(entropy)


def _url_parse_target(url: str) -> str:
    normalized = _extract_text(url)
    if "://" in normalized:
        return normalized
    return f"http://{normalized}"


def _length_bucket(value: int) -> List[int]:
    bins = [
        (0, 20, "0_20"),
        (21, 40, "21_40"),
        (41, 60, "41_60"),
        (61, 80, "61_80"),
        (81, 100, "81_100"),
        (101, None, "101_plus"),
    ]
    result = [0] * len(bins)
    for idx, (low, high, _name) in enumerate(bins):
        if high is None:
            result[idx] = int(value >= low)
        else:
            result[idx] = int(low <= value <= high)
    return result


def extract_url_features(url: str) -> Dict[str, float]:
    normalized = _extract_text(url)
    lower_text = normalized.lower()
    parse_target = _url_parse_target(normalized)
    parsed = urlparse(parse_target)
    host = parsed.hostname or ""
    path = parsed.path or ""

    digits = sum(ch.isdigit() for ch in lower_text)
    letters = sum(ch.isalpha() for ch in lower_text)
    specials = sum(not ch.isalnum() for ch in lower_text)
    hyphens = lower_text.count("-")
    vowels = sum(ch in VOWELS for ch in lower_text)
    percent_encoded = len(__import__("re").findall(r"%[0-9a-fA-F]{2}", lower_text))
    host_entropy = _safe_entropy(host)
    path_entropy = _safe_entropy(path)
    num_queries = len(parse_qs(parsed.query, keep_blank_values=True))
    num_segments = 0
    if path:
        num_segments = len([segment for segment in path.split("/") if segment])

    extracted_tld = tldextract.extract(lower_text)
    tld = extracted_tld.suffix.lower() if extracted_tld else ""

    bucket_features = _length_bucket(len(normalized))
    features: Dict[str, float] = {
        "url_length": float(len(normalized)),
        "host_length": float(len(host)),
        "path_length": float(len(path)),
        "num_dots": float(normalized.count(".")),
        "num_path_segments": float(num_segments),
        "num_query_params": float(num_queries),
        "digit_count": float(digits),
        "letter_count": float(letters),
        "digit_letter_ratio": float(digits / letters) if letters else 0.0,
        "special_char_count": float(specials),
        "hyphen_count": float(hyphens),
        "has_at_symbol": float(int("@" in normalized)),
        "is_https": float(int(str(parsed.scheme).lower() == "https")),
        "has_port": float(int(parsed.port is not None)),
        "has_fragment": float(int(bool(parsed.fragment))),
        "percent_encoded_fraction": float(percent_encoded / len(normalized)) if normalized else 0.0,
        "vowel_fraction": float(vowels / len(normalized)) if normalized else 0.0,
        "is_ip_host": float(int(__import__("re").match(r"^(?:\d{1,3}\.){3}\d{1,3}$", host) is not None)),
        "suspicious_tld": float(int(tld in SUSPICIOUS_TLDS)),
        "hostname_entropy": float(host_entropy),
        "path_entropy": float(path_entropy),
        "bucket_0_20": float(bucket_features[0]),
        "bucket_21_40": float(bucket_features[1]),
        "bucket_41_60": float(bucket_features[2]),
        "bucket_61_80": float(bucket_features[3]),
        "bucket_81_100": float(bucket_features[4]),
        "bucket_101_plus": float(bucket_features[5]),
    }

    for token in SUSPICIOUS_TOKENS:
        features[f"token_{token}"] = float(int(token in lower_text))

    # guarantee purely numeric feature values and deterministic types
    for key, value in list(features.items()):
        features[key] = float(value)

    return features


def build_features_dataframe(urls: Iterable[str]) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for url in urls:
        rows.append(extract_url_features(url))

    feature_frame = pd.DataFrame(rows)
    feature_frame = feature_frame.reindex(sorted(feature_frame.columns), axis=1)
    feature_frame = feature_frame.astype(np.float64)
    return feature_frame
