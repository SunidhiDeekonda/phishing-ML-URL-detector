import re
from pathlib import Path
import sys
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.features import extract_url_features
from src.char_tokenizer import (
    MAX_SEQUENCE_LENGTH,
    build_vocab,
    PAD_INDEX,
    REFERENCE_CHARACTERS,
    encode_url,
    encode_urls,
    normalize_url,
    UNK_INDEX,
)
DATASET_PATH = ROOT / "data" / "processed" / "dataset.csv"
FEATURES_PATH = ROOT / "data" / "processed" / "features.csv"
SEQUENCES_PATH = ROOT / "data" / "processed" / "char_sequences.npy"
TRAIN_IDX = ROOT / "data" / "processed" / "train_idx.npy"
VAL_IDX = ROOT / "data" / "processed" / "val_idx.npy"
TEST_IDX = ROOT / "data" / "processed" / "test_idx.npy"


def test_url_normalization():
    assert normalize_url("  HTTP://Example.COM/Test?A=1  ") == "http://example.com/test?a=1"


def test_feature_extraction_fields_and_types():
    url = "https://login.webscr.bank.example-secure.com/path/login?user=1&token=abc"
    features = extract_url_features(url)
    required = [
        "url_length",
        "host_length",
        "path_length",
        "num_dots",
        "num_path_segments",
        "num_query_params",
        "digit_count",
        "letter_count",
        "digit_letter_ratio",
        "special_char_count",
        "hyphen_count",
        "has_at_symbol",
        "is_https",
        "has_port",
        "has_fragment",
        "percent_encoded_fraction",
        "vowel_fraction",
        "is_ip_host",
        "suspicious_tld",
        "hostname_entropy",
        "path_entropy",
    ]

    for token in [
        "token_login",
        "token_signin",
        "token_secure",
        "token_webscr",
        "token_bank",
        "token_verify",
        "token_update",
        "token_account",
        "token_confirm",
    ]:
        required.append(token)

    for bucket in [
        "bucket_0_20",
        "bucket_21_40",
        "bucket_41_60",
        "bucket_61_80",
        "bucket_81_100",
        "bucket_101_plus",
    ]:
        required.append(bucket)

    for key in required:
        assert key in features
        assert isinstance(features[key], float)


def test_dataset_size_and_balance():
    df = pd.read_csv(DATASET_PATH)
    assert len(df) == 20_000
    counts = df["label"].value_counts().to_dict()
    assert counts.get(0) == 10_000
    assert counts.get(1) == 10_000


def test_no_network_calls_during_feature_tokenize(monkeypatch):
    import socket
    import urllib.request

    def fail_network(*args, **kwargs):
        raise AssertionError("Network call attempted during preprocessing")

    monkeypatch.setattr(socket.socket, "connect", fail_network)
    monkeypatch.setattr(urllib.request, "urlopen", fail_network)

    assert all(token in "https://example.com" for token in ["https", "example"])  # no-op
    assert extract_url_features("https://example.com/path")
    vocab = build_vocab(["abc", "xyz"])
    _ = encode_urls(["abc", "xyz", "http://example.com"], vocab)


def test_tokenizer_output_length_and_padding():
    vocab = build_vocab(["abc", "def", "http://example.com/long"])
    seq = encode_url("a", vocab)
    assert seq.shape == (MAX_SEQUENCE_LENGTH,)
    assert int(seq[-1]) == 0
    assert int(seq[0]) != 0


def test_long_url_truncation_rightmost():
    long_url = "https://" + ("a" * 200) + ("b" * 120)
    vocab = build_vocab([long_url])
    seq_long = encode_url(long_url, vocab)
    # rightmost 200 characters kept after normalization
    expected = [vocab[ch] for ch in normalize_url(long_url)[-MAX_SEQUENCE_LENGTH:]]
    assert seq_long.tolist() == expected


def test_padding_index_is_zero():
    vocab = build_vocab([])
    assert vocab["<PAD>"] == PAD_INDEX == 0


def test_known_char_mapping_is_deterministic():
    vocab = build_vocab([])
    assert vocab["a"] == 1
    assert vocab["z"] == 26
    assert vocab["0"] == 27
    assert vocab["9"] == 36
    assert vocab[":"] == 37
    assert vocab["/"] == 38
    assert vocab["?"] == 39
    assert vocab["&"] == 40
    assert vocab["="] == 41
    assert vocab["."] == 42
    assert vocab["%"] == 43
    assert vocab["-"] == 44
    assert vocab["_"] == 45
    assert vocab["+"] == 46
    assert vocab["#"] == 47
    assert vocab["@"] == 48
    assert vocab["~"] == 49


def test_unknown_character_maps_to_unk():
    vocab = build_vocab([])
    unknown_encoded = encode_url("€", vocab)
    assert int(unknown_encoded[0]) == UNK_INDEX


def test_padding_index_zero_and_unk_index_fifty():
    vocab = build_vocab([])
    assert vocab["<PAD>"] == 0
    assert vocab["<UNK>"] == 50


def test_explicit_character_count_is_49():
    vocab = build_vocab([])
    assert len(REFERENCE_CHARACTERS) == 49
    assert len(vocab) == 51


def test_explicit_tilde_index_49():
    vocab = build_vocab([])
    assert vocab["~"] == 49


def test_max_sequence_length_is_200():
    assert MAX_SEQUENCE_LENGTH == 200


def test_features_csv_is_numeric():
    features = pd.read_csv(FEATURES_PATH)
    non_numeric = list(features.select_dtypes(exclude=[np.number]).columns)
    assert len(non_numeric) == 0


def test_sequences_shape():
    sequences = np.load(SEQUENCES_PATH)
    assert sequences.shape == (20_000, 200)


def test_splits_are_disjoint_and_cover_all_rows():
    train = np.load(TRAIN_IDX)
    val = np.load(VAL_IDX)
    test = np.load(TEST_IDX)

    assert len(train) + len(val) + len(test) == 20_000
    assert 13_500 <= len(train) <= 14_500
    assert 1_500 <= len(val) <= 2_500
    assert 3_500 <= len(test) <= 4_500

    train_set, val_set, test_set = set(train), set(val), set(test)
    assert not (train_set & val_set)
    assert not (train_set & test_set)
    assert not (val_set & test_set)

    df = pd.read_csv(DATASET_PATH)
    all_indices = set(range(len(df)))
    union = train_set | val_set | test_set
    assert union == all_indices


def test_splits_class_balance_approximately_equal():
    df = pd.read_csv(DATASET_PATH)
    y = df["label"].to_numpy()
    for idx_path, name in [(TRAIN_IDX, "train"), (VAL_IDX, "val"), (TEST_IDX, "test")]:
        idx = np.load(idx_path)
        split_ratio = y[idx].mean()
        assert 0.45 <= split_ratio <= 0.55


def _extract_root_domain(url: str) -> str:
    parsed = urlparse(normalize_url(url))
    host = parsed.hostname or ""
    if not host:
        return ""
    if host.startswith("www."):
        host = host[4:]
    if host.startswith("[") and host.endswith("]"):
        return host[1:-1]
    if all(part.isdigit() for part in host.split(".")):
        return host
    parts = host.split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else host


def test_domain_overlap_not_across_splits():
    df = pd.read_csv(DATASET_PATH)
    urls = df["url"].tolist()

    train_idx = np.load(TRAIN_IDX)
    val_idx = np.load(VAL_IDX)
    test_idx = np.load(TEST_IDX)

    train_domains = { _extract_root_domain(urls[i]) for i in train_idx }
    val_domains = { _extract_root_domain(urls[i]) for i in val_idx }
    test_domains = { _extract_root_domain(urls[i]) for i in test_idx }

    assert not (train_domains & val_domains)
    assert not (train_domains & test_domains)
    assert not (val_domains & test_domains)
