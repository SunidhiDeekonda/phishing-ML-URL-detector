"""Deterministic, network-free URL mutations for defensive testing."""

from __future__ import annotations

import random
import re
from dataclasses import asdict, dataclass
from typing import Iterable
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

SEED = 42
MUTATION_TYPES = (
    "benign_path_token", "redundant_separator", "percent_encoding",
    "subdomain_variation", "keyword_splitting", "character_perturbation",
    "query_noise", "length_noise",
)
SUSPICIOUS_WORDS = ("login", "verify", "secure", "account", "update", "signin", "bank")


@dataclass(frozen=True)
class MutationRecord:
    source_row_index: int
    original_url: str
    mutated_url: str
    mutation_type: str
    label: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _parts(url: str):
    text = str(url).strip()
    return text, urlsplit(text if "://" in text else f"http://{text}")


def _restore(original: str, parts) -> str:
    value = urlunsplit(parts)
    return value if "://" in original else value.split("://", 1)[-1]


def mutate_url(url: str, mutation_type: str, seed: int = SEED, source_row_index: int = 0) -> str:
    """Return one deterministic synthetic mutation. No I/O occurs."""
    if mutation_type not in MUTATION_TYPES:
        raise ValueError(f"Unknown mutation type: {mutation_type}")
    original, p = _parts(url)
    rng = random.Random(f"{seed}:{source_row_index}:{mutation_type}:{original}")
    scheme, netloc, path, query, fragment = p
    if mutation_type == "benign_path_token":
        token = rng.choice(("help", "docs", "support", "portal", "home"))
        path = f"/{token}{path if path.startswith('/') else '/' + path}".rstrip("/") or "/help"
    elif mutation_type == "redundant_separator":
        path = (path or "/index").replace("/", "//", 1)
    elif mutation_type == "percent_encoding":
        target = path or "/login"
        positions = [i for i, c in enumerate(target) if c.isalpha()]
        if not positions:
            target = target.rstrip("/") + "/login"
            positions = [i for i, c in enumerate(target) if c.isalpha()]
        position = positions[0]
        path = target[:position] + f"%{ord(target[position]):02X}" + target[position + 1:]
    elif mutation_type == "subdomain_variation":
        host, sep, port = netloc.partition(":")
        prefix = rng.choice(("cdn", "auth", "portal", "static", "support"))
        netloc = f"{prefix}-{source_row_index % 97}.{host}{sep}{port}" if host else netloc
    elif mutation_type == "keyword_splitting":
        word = next((w for w in SUSPICIOUS_WORDS if w in f"{netloc}{path}{query}".lower()), None)
        if word:
            cut = max(1, len(word) // 2)
            replacement = f"{word[:cut]}-{word[cut:]}"
            netloc = re.sub(word, replacement, netloc, count=1, flags=re.I)
            path = re.sub(word, replacement, path, count=1, flags=re.I)
            query = re.sub(word, replacement, query, count=1, flags=re.I)
        else:
            path = f"{path.rstrip('/')}/account-check"
    elif mutation_type == "character_perturbation":
        target = path or "/verify"
        positions = [i for i, c in enumerate(target) if c.isalpha()]
        position = rng.choice(positions) if positions else len(target)
        path = target[:position] + rng.choice(("-", "_", ".")) + target[position:]
    elif mutation_type == "query_noise":
        pairs = parse_qsl(query, keep_blank_values=True)
        pairs.append((rng.choice(("ref", "source", "session", "lang")), f"r{source_row_index % 1000:03d}"))
        query = urlencode(pairs)
    elif mutation_type == "length_noise":
        noise = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz0123456789") for _ in range(18))
        path = f"{path.rstrip('/')}/resource/{noise}"
    mutated = _restore(original, (scheme, netloc, path, query, fragment))
    if mutated == original:
        mutated += ("&" if "?" in mutated else "?") + f"research_variant={source_row_index % 997}"
    return mutated


def mutate_records(rows: Iterable[tuple[int, str, int]], seed: int = SEED) -> list[MutationRecord]:
    records = []
    for ordinal, (source_index, url, label) in enumerate(rows):
        kind = MUTATION_TYPES[ordinal % len(MUTATION_TYPES)]
        records.append(MutationRecord(int(source_index), str(url), mutate_url(str(url), kind, seed, int(source_index)), kind, int(label)))
    return records
