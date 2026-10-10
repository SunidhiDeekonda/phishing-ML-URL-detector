"""Domain authority and reputation calibrator.

Uses tldextract to identify the true registered base domain under the Public Suffix List,
ensuring that authentic services (e.g., dash.cloudflare.com, platform.openai.com, github.com/login)
are not misclassified due to login path keywords, while strictly protecting against spoofing
(e.g., cloudflare.com.evil-phish.xyz is correctly identified as evil-phish.xyz).
"""

from __future__ import annotations

from typing import Optional, Set
import tldextract

# network-free PSL snapshot
_EXTRACTOR = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)

# Top authoritative institutions and public services
TOP_AUTHENTIC_DOMAINS: Set[str] = {
    "google.com",
    "apple.com",
    "microsoft.com",
    "github.com",
    "gitlab.com",
    "openai.com",
    "cloudflare.com",
    "vercel.com",
    "vercel.app",
    "amazon.com",
    "mozilla.org",
    "wikipedia.org",
    "python.org",
    "rust-lang.org",
    "docker.com",
    "kubernetes.io",
    "stackoverflow.com",
    "npmjs.com",
    "auth0.com",
    "atlassian.com",
    "bitbucket.org",
    "cisa.gov",
    "login.gov",
    "nist.gov",
    "ncsc.gov.uk",
    "ftc.gov",
    "owasp.org",
    "harvard.edu",
    "oracle.com",
    "facebook.com",
    "instagram.com",
    "twitter.com",
    "x.com",
    "linkedin.com",
    "netflix.com",
    "dropbox.com",
    "slack.com",
    "spotify.com",
    "zoom.us",
    "chase.com",
    "bankofamerica.com",
    "wellsfargo.com",
    "paypal.com",
    "adobe.com",
}


def get_registered_domain(url: str) -> str:
    """Extract the true registered base domain under the Public Suffix List."""
    ext = _EXTRACTOR(url)
    if ext.domain and ext.suffix:
        return f"{ext.domain.lower()}.{ext.suffix.lower()}"
    return ext.domain.lower() if ext.domain else ""


def is_authentic_domain(url: str) -> bool:
    """Check if the URL belongs to an authentic, verified top institution domain."""
    reg = get_registered_domain(url)
    return reg in TOP_AUTHENTIC_DOMAINS


def calibrate_probability_with_domain(url: str, raw_prob: float) -> float:
    """Apply domain authority calibration.
    
    If the URL's true registered domain is verified authentic:
    - Normal login and security paths on authentic domains are bounded below the phishing threshold.
    - An attacker using subdomains (e.g. apple.com.phish.xyz) is unaffected because their registered
      domain is 'phish.xyz', NOT 'apple.com'.
    """
    if is_authentic_domain(url):
        # Even with 'login' or 'security', authentic domains should have low phishing probability
        return min(raw_prob * 0.1, 0.25)
    return raw_prob
