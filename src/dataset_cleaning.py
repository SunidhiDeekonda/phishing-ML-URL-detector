"""Training-time repair and rejection of confirmed URL dataset artefacts.

The rules below come from scripts/audit_dataset_labels.py.  They target
formatting artefacts that appear almost exclusively in one class of the V2
dataset (scraping escapes in the alexkstern phishing rows), which a model would
otherwise learn as a label shortcut.  The source CSV is never modified.
"""

from __future__ import annotations

import re
from typing import Tuple

from src.url_normalization import canonicalize_url

_HOST_PATTERN = re.compile(r"^[a-z0-9_-]+(\.[a-z0-9_-]+)+\.?$|^\[[0-9a-f:.]+\]$", re.IGNORECASE)


def clean_training_url(url: object) -> Tuple[str | None, str]:
    """Return ``(cleaned_url, reason)``; ``cleaned_url`` is None when the row is rejected."""
    if url is None:
        return None, "rejected_null"
    text = str(url).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        text = text[1:-1].strip()
    text = text.replace("\\%", "%").replace("&amp;", "&")
    if not text:
        return None, "rejected_empty"
    if re.search(r"\s", text):
        return None, "rejected_internal_whitespace"
    try:
        canonical = canonicalize_url(text)
    except ValueError:
        return None, "rejected_unparseable"
    rest = canonical.split("://", 1)[-1]
    authority = rest.split("/", 1)[0].split("?", 1)[0].split("#", 1)[0]
    if "://" not in text and re.fullmatch(r"[^/@?#]+@[^/@?#]+", text):
        return None, "rejected_email_address"
    host = authority.rsplit("@", 1)[-1]
    if host.startswith("["):
        host = host.split("]", 1)[0] + "]"
    else:
        host = host.split(":", 1)[0]
    if not _HOST_PATTERN.match(host):
        return None, "rejected_invalid_host"
    return text, "kept" if text == str(url).strip() else "repaired"
