"""Independent label and data-quality audit for the URL datasets.

Neither dataset is treated as ground truth for the other.  Labels are checked
against independent public evidence that is only processed as text (no URL is
fetched):

* phishing evidence: OpenPhish community feed and Phishing.Database
  (``phishing-links-ACTIVE.txt`` exact URLs, ``phishing-domains-ACTIVE.txt`` hosts)
* popularity evidence: Tranco top-1M ranks.  Popularity is NOT treated as proof
  of legitimacy; it is only used to flag phishing-labelled *bare roots* of
  top-ranked sites, which are very unlikely to be phishing pages.

Usage:
    python scripts/audit_dataset_labels.py --evidence-dir tmp/external
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tldextract

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.url_normalization import canonicalize_url

EXTRACT = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None, include_psl_private_domains=True)
SYNTHETIC_PATHS = [
    "/login", "/signin", "/sign-in", "/account", "/accounts/login", "/auth/login", "/security",
    "/docs/security", "/users/login", "/support", "/dashboard", "/verify", "/settings/security",
    "/help", "/privacy", "/download",
]


def loose_key(url: str) -> str:
    text = str(url).strip().lower()
    if "://" in text:
        text = text.split("://", 1)[1]
    if text.startswith("www."):
        text = text[4:]
    return text.rstrip("/")


def safe_canonical(url: str) -> str:
    try:
        return canonicalize_url(url)
    except ValueError:
        return str(url).strip()


def is_malformed(url: str) -> bool:
    try:
        canonicalize_url(url)
        return False
    except ValueError:
        return True


def host_of(url: str) -> str:
    text = safe_canonical(url)
    rest = text.split("://", 1)[-1]
    host = rest.split("/", 1)[0].split("?", 1)[0].split("#", 1)[0]
    return host.rsplit("@", 1)[-1].split(":", 1)[0].lower()


def registered_domain(url: str) -> str:
    parts = EXTRACT(safe_canonical(url))
    return ".".join(p for p in (parts.domain, parts.suffix) if p).lower()


def is_bare_root(url: str) -> bool:
    rest = safe_canonical(url).split("://", 1)[-1]
    return "/" not in rest.rstrip("/") and "?" not in rest and "#" not in rest


def structure_profile(urls: pd.Series) -> dict[str, float]:
    s = urls.astype(str)
    stripped = s.str.replace(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", "", regex=True)
    return {
        "rows": int(len(s)),
        "has_scheme": float(s.str.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://").mean()),
        "https": float(s.str.lower().str.startswith("https://").mean()),
        "www_prefix": float(stripped.str.lower().str.startswith("www.").mean()),
        "has_path_or_query": float((~s.map(is_bare_root)).mean()),
        "trailing_slash": float(s.str.endswith("/").mean()),
        "median_length": float(s.str.len().median()),
        "quote_or_escape_artifacts": float(s.str.contains(r"^['\"]|\\\\", regex=True).mean()),
    }


def load_evidence(evidence_dir: Path) -> dict[str, object]:
    phishing_urls = {loose_key(u) for u in (evidence_dir / "openphish_feed.txt").read_text().splitlines() if u.strip()}
    phishing_urls |= {
        loose_key(u)
        for u in (evidence_dir / "pdb_phishing-links-ACTIVE.txt").read_text(errors="ignore").splitlines()
        if u.strip()
    }
    phishing_hosts = {
        h.strip().lower()
        for h in (evidence_dir / "pdb_phishing-domains-ACTIVE.txt").read_text(errors="ignore").splitlines()
        if h.strip()
    }
    tranco = pd.read_csv(evidence_dir / "top-1m.csv", header=None, names=["rank", "domain"])
    return {
        "phishing_urls": phishing_urls,
        "phishing_hosts": phishing_hosts,
        "tranco_rank": dict(zip(tranco.domain.str.lower(), tranco["rank"])),
    }


def annotate(frame: pd.DataFrame, evidence: dict[str, object]) -> pd.DataFrame:
    out = frame.copy()
    out["key"] = out.url.map(loose_key)
    out["host"] = out.url.map(host_of)
    out["registered_domain"] = out.url.map(registered_domain)
    out["bare_root"] = out.url.map(is_bare_root)
    ranks = evidence["tranco_rank"]
    out["tranco_rank"] = out.registered_domain.map(lambda d: ranks.get(d, np.nan))
    out["url_in_phishing_feed"] = out.key.isin(evidence["phishing_urls"])
    out["host_in_phishing_feed"] = out.host.isin(evidence["phishing_hosts"]) | out.host.str.removeprefix("www.").isin(
        evidence["phishing_hosts"]
    )
    return out


def evidence_rates(frame: pd.DataFrame) -> dict[str, float]:
    return {
        "rows": int(len(frame)),
        "exact_url_in_phishing_feed": float(frame.url_in_phishing_feed.mean()),
        "host_in_phishing_domain_feed": float(frame.host_in_phishing_feed.mean()),
        "registered_domain_in_tranco_top10k": float((frame.tranco_rank <= 10_000).mean()),
        "registered_domain_in_tranco_top1m": float(frame.tranco_rank.notna().mean()),
    }


def flag_rows(frame: pd.DataFrame) -> pd.DataFrame:
    flags = []
    for row in frame.itertuples(index=False):
        reasons = []
        if row.label == 0 and row.url_in_phishing_feed:
            reasons.append("legit_label_but_exact_url_in_phishing_feed")
        if row.label == 0 and row.host_in_phishing_feed and not (row.tranco_rank <= 100_000):
            reasons.append("legit_label_but_host_in_phishing_domain_feed_and_not_tranco_top100k")
        if row.label == 1 and row.bare_root and row.tranco_rank <= 10_000 and row.host in {
            row.registered_domain, f"www.{row.registered_domain}"
        }:
            reasons.append("phishing_label_but_bare_root_of_tranco_top10k_site")
        flags.append(";".join(reasons))
    out = frame.copy()
    out["audit_flags"] = flags
    return out


def duplicate_report(frame: pd.DataFrame) -> dict[str, int]:
    by_key = frame.groupby("key").label.nunique()
    return {
        "exact_duplicate_urls": int(frame.url.duplicated().sum()),
        "loose_duplicate_urls": int(frame.key.duplicated().sum()),
        "loose_keys_with_conflicting_labels": int((by_key > 1).sum()),
        "empty_or_whitespace_urls": int(frame.url.astype(str).str.strip().eq("").sum()),
        "null_urls": int(frame.url.isna().sum()),
        "malformed_urls_rejected_by_canonicalize_url": int(frame.url.map(is_malformed).sum()),
    }


def v2_provenance(balanced: pd.DataFrame, evidence_dir: Path) -> pd.Series:
    """Attribute every V2 row to the source scripts/prepare_balanced_dataset.py drew it from."""
    hard = {c["url"] for c in json.loads((ROOT / "tests/data/hard_legitimate_urls.json").read_text())}
    local_legit = pd.read_csv(ROOT / "data/raw/legit_urls.csv").url.dropna().unique().tolist()
    local_phish = set(pd.read_csv(ROOT / "data/raw/phishing_urls.csv").url.dropna())
    synthetic = {f"{root.rstrip('/')}{path}" for root in local_legit[:600] for path in SYNTHETIC_PATHS}
    alex = pd.concat([pd.read_csv(evidence_dir / "alex_train.csv"), pd.read_csv(evidence_dir / "alex_test.csv")])
    alex_legit, alex_phish = set(alex[alex.label == 0].text.dropna()), set(alex[alex.label == 1].text.dropna())
    piro = pd.concat([pd.read_parquet(evidence_dir / "piro_train.parquet"), pd.read_parquet(evidence_dir / "piro_test.parquet")])
    piro_legit = set(piro[piro.status == "legitimate"].url.dropna())

    def source(row: pd.Series) -> str:
        url, label = row.url, row.label
        if label == 0:
            if url in hard:
                return "hard_negative_suite"
            if url in synthetic:
                return "synthetic_benign_path_on_phishx_legit_root"
            if url in alex_legit:
                return "alexkstern_label0"
            if url in piro_legit:
                return "pirocheto_legitimate"
            if url in set(local_legit):
                return "phishx_legit_root"
        else:
            if url in alex_phish:
                return "alexkstern_label1"
            if url in local_phish:
                return "phishx_phishing"
        return "unattributed"

    return balanced.apply(source, axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "results/dataset_label_audit.json")
    parser.add_argument("--flags-output", type=Path, default=ROOT / "results/dataset_label_audit_flags.csv")
    args = parser.parse_args()
    random.seed(42)
    evidence = load_evidence(args.evidence_dir)
    report: dict[str, object] = {
        "evidence_sources": {
            "phishing_urls": "openphish.com/feed.txt + mitchellkrogza/Phishing.Database phishing-links-ACTIVE.txt",
            "phishing_hosts": "mitchellkrogza/Phishing.Database phishing-domains-ACTIVE.txt",
            "popularity": "Tranco top-1M (list 8PX5V)",
            "phishtank": "not used: data.phishtank.com returned HTTP 429 (rate limited) at audit time",
        },
        "label_convention": "All project datasets use 1 = phishing, 0 = legitimate.",
    }

    previous = pd.read_csv(ROOT / "data/processed/dataset.csv")
    balanced = pd.read_csv(ROOT / "data/processed/balanced_dataset.csv")
    balanced["source"] = v2_provenance(balanced, args.evidence_dir)
    alex = pd.concat([pd.read_csv(args.evidence_dir / "alex_train.csv"), pd.read_csv(args.evidence_dir / "alex_test.csv")])
    alex = alex.rename(columns={"text": "url"}).dropna(subset=["url"])

    all_flags = []
    for name, frame in (("previous_phishx_dataset", previous), ("v2_balanced_dataset", balanced)):
        annotated = annotate(frame, evidence)
        flagged = flag_rows(annotated)
        flagged["dataset"] = name
        all_flags.append(flagged[flagged.audit_flags != ""])
        section: dict[str, object] = {
            "columns": list(frame.columns),
            "label_values": sorted(int(v) for v in frame.label.unique()),
            "class_counts": {str(k): int(v) for k, v in frame.label.value_counts().sort_index().items()},
            "duplicates": duplicate_report(annotated),
            "structure_by_label": {str(l): structure_profile(frame[frame.label == l].url) for l in (0, 1)},
            "evidence_by_label": {str(l): evidence_rates(annotated[annotated.label == l]) for l in (0, 1)},
            "flag_counts": flagged.audit_flags.str.split(";").explode().loc[lambda s: s != ""].value_counts().to_dict(),
        }
        if "source" in frame:
            section["rows_by_source"] = frame.groupby(["source", "label"]).size().rename("rows").reset_index().to_dict("records")
            section["evidence_by_source"] = {
                src: evidence_rates(annotated[annotated.source == src]) for src in sorted(frame.source.unique())
            }
            section["structure_by_source"] = {
                src: structure_profile(frame[frame.source == src].url) for src in sorted(frame.source.unique())
            }
            section["flag_counts_by_source"] = (
                flagged[flagged.audit_flags != ""].groupby("source").size().to_dict()
            )
        report[name] = section

    alex_sample = annotate(alex.sample(n=40_000, random_state=42), evidence)
    report["alexkstern_label_mapping_check"] = {
        "note": "Seeded 40k-row sample of the full alexkstern dataset; used to confirm what label 0/1 mean.",
        **{f"label_{l}": evidence_rates(alex_sample[alex_sample.label == l]) for l in (0, 1)},
    }

    hard = json.loads((ROOT / "tests/data/hard_legitimate_urls.json").read_text())
    hard_test_domains = {registered_domain(c["url"]) for c in hard if c["hard_negative_split"] == "test"}
    synthetic = balanced[balanced.source == "synthetic_benign_path_on_phishx_legit_root"]
    report["v2_synthetic_rows"] = {
        "rows": int(len(synthetic)),
        "share_of_legit_class": float(len(synthetic) / (balanced.label == 0).sum()),
        "distinct_roots": int(synthetic.url.map(registered_domain).nunique()),
        "rows_on_hard_negative_test_registered_domains": int(synthetic.url.map(registered_domain).isin(hard_test_domains).sum()),
        "note": "Generated as <PhishX legit root> + fixed benign path; existence of these URLs was never verified.",
    }

    flags = pd.concat(all_flags, ignore_index=True)
    cols = ["dataset", "url", "label", "audit_flags", "host", "registered_domain", "tranco_rank"] + (["source"] if "source" in flags else [])
    flags[cols].to_csv(args.flags_output, index=False)
    args.output.write_text(json.dumps(report, indent=2, default=float) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    main()
