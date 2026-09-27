"""Build the curated, network-free legitimate hard-negative benchmark."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import pandas as pd

from src.features import TLD_EXTRACTOR
from src.url_normalization import canonicalize_url

ROOT = Path(__file__).resolve().parents[1]

# These are inert URL strings from clearly identifiable official services.  They
# are never fetched.  Exact URLs are separated, while several multi-tenant
# platforms intentionally occur in more than one partition so the benchmark can
# measure unseen pages on a known platform.  That exception is recorded openly.
CASES = [
    # Training-only hard negatives.
    ("https://github.com/openai/openai-python", "train", "developer_repository", "Official open-source repository path"),
    ("https://github.com/microsoft/vscode", "train", "developer_repository", "Official open-source repository path"),
    ("https://github.com/vercel/next.js", "train", "developer_repository", "Official open-source repository path"),
    ("https://github.com/pallets/flask", "train", "developer_repository", "Official open-source repository path"),
    ("https://github.com/python/cpython", "train", "developer_repository", "Official open-source repository path"),
    ("https://github.com/mozilla-firefox/firefox", "train", "developer_repository", "Official open-source repository path"),
    ("https://github.com/kubernetes/kubernetes", "train", "developer_repository", "Official open-source repository path"),
    ("https://github.com/docker/docs", "train", "developer_repository", "Official documentation repository path"),
    ("https://docs.python.org/3/library/urllib.parse.html", "train", "long_documentation", "Official Python documentation path"),
    ("https://docs.python.org/3/howto/logging.html", "train", "long_documentation", "Official Python documentation path"),
    ("https://learn.microsoft.com/en-us/security/zero-trust/", "train", "security_keyword", "Official Microsoft security documentation"),
    ("https://learn.microsoft.com/en-us/entra/identity/authentication/", "train", "login_account", "Official Microsoft authentication documentation"),
    ("https://account.microsoft.com/account/manage-my-account", "train", "login_account", "Official Microsoft account page"),
    ("https://support.apple.com/guide/security/welcome/web", "train", "security_keyword", "Official Apple security guide"),
    ("https://appleid.apple.com/account/manage", "train", "login_account", "Official Apple account page"),
    ("https://help.openai.com/en/collections/3675942-account-login-and-billing", "train", "login_account", "Official OpenAI help collection"),
    ("https://platform.openai.com/docs/guides/authentication", "train", "login_account", "Official OpenAI authentication documentation"),
    ("https://developer.mozilla.org/en-US/docs/Web/Security", "train", "security_keyword", "Official Mozilla security documentation"),
    ("https://support.mozilla.org/en-US/kb/password-manager-remember-delete-edit-logins", "train", "login_account", "Official Mozilla password help"),
    ("https://docs.gitlab.com/user/profile/account/", "train", "login_account", "Official GitLab account documentation"),
    ("https://gitlab.com/users/sign_in", "train", "login_account", "Official GitLab login"),
    ("https://bitbucket.org/account/signin/", "train", "login_account", "Official Bitbucket login"),
    ("https://support.atlassian.com/security-and-access-policies/", "train", "security_keyword", "Official Atlassian security documentation"),
    ("https://docs.djangoproject.com/en/stable/topics/auth/", "train", "login_account", "Official Django authentication documentation"),
    ("https://docs.djangoproject.com/en/stable/topics/security/", "train", "security_keyword", "Official Django security documentation"),
    ("https://flask.palletsprojects.com/en/stable/security/", "train", "security_keyword", "Official Flask security documentation"),
    ("https://kubernetes.io/docs/reference/access-authn-authz/authentication/", "train", "login_account", "Official Kubernetes authentication documentation"),
    ("https://kubernetes.io/docs/concepts/security/", "train", "security_keyword", "Official Kubernetes security documentation"),
    ("https://docs.docker.com/accounts/manage-account/", "train", "login_account", "Official Docker account documentation"),
    ("https://docs.docker.com/engine/security/protect-access/", "train", "security_keyword", "Official Docker security documentation"),
    ("https://stackoverflow.com/users/login", "train", "login_account", "Official Stack Overflow login"),
    ("https://stackoverflow.com/help/account", "train", "login_account", "Official Stack Overflow account help"),
    ("https://www.login.gov/help/trouble-signing-in/overview/", "train", "login_account", "Official US government login help"),
    ("https://www.login.gov/security/", "train", "security_keyword", "Official US government security page"),
    ("https://docs.npmjs.com/creating-a-strong-password", "train", "login_account", "Official npm password documentation"),
    ("https://www.npmjs.com/login", "train", "login_account", "Official npm login"),
    ("https://www.rust-lang.org/policies/security", "train", "security_keyword", "Official Rust security policy"),
    ("https://docs.oracle.com/javase/tutorial/security/", "train", "security_keyword", "Official Oracle security tutorial"),
    ("https://auth0.com/docs/authenticate/login", "train", "login_account", "Official Auth0 login documentation"),
    ("https://developers.cloudflare.com/fundamentals/account/account-security/", "train", "login_account", "Official Cloudflare account security documentation"),
    ("https://github.com/OWASP/www-project-web-security-testing-guide", "train", "developer_repository", "Official OWASP security project repository"),
    ("https://github.com/OWASP/www-project-application-security-verification-standard", "train", "developer_repository", "Official OWASP security project repository"),
    ("https://github.com/OWASP/www-project-top-ten", "train", "developer_repository", "Official OWASP security project repository"),
    ("https://github.com/rapid7/metasploit-framework", "train", "developer_repository", "Official Metasploit repository"),
    ("https://github.com/gophish/gophish", "train", "developer_repository", "Legitimate open-source anti-phishing repository"),
    ("https://www.cisa.gov/news-events/news/avoiding-social-engineering-and-phishing-attacks", "train", "security_keyword", "Official government phishing guidance"),
    ("https://www.ftc.gov/business-guidance/small-businesses/cybersecurity/phishing", "train", "security_keyword", "Official government phishing guidance"),
    ("https://www.ncsc.gov.uk/guidance/phishing", "train", "security_keyword", "Official government phishing guidance"),
    ("https://learn.microsoft.com/en-us/defender-office-365/anti-phishing-protection-about", "train", "security_keyword", "Official Microsoft anti-phishing documentation"),
    ("https://support.google.com/a/answer/9157861", "train", "security_keyword", "Official Google phishing-protection documentation"),
    ("https://www.google.com/", "train", "official_root", "Official Google root"),
    ("https://github.com/", "train", "official_root", "Official GitHub root"),
    ("https://www.microsoft.com/", "train", "official_root", "Official Microsoft root"),
    ("https://www.apple.com/", "train", "official_root", "Official Apple root"),
    ("https://openai.com/", "train", "official_root", "Official OpenAI root"),
    ("https://vercel.com/", "train", "official_root", "Official Vercel root"),
    ("https://www.python.org/", "train", "official_root", "Official Python root"),
    ("https://www.wikipedia.org/", "train", "official_root", "Official Wikipedia root"),
    ("https://phishing-ml-url-detector-seven.vercel.app/", "train", "deployment_url", "Documented project deployment root"),
    # Validation examples used only for the frozen production gate.
    ("https://github.com/psf/requests", "validation", "developer_repository", "Official Python Software Foundation repository"),
    ("https://github.com/encode/fastapi", "validation", "developer_repository", "Official FastAPI repository"),
    ("https://github.com/pandas-dev/pandas", "validation", "developer_repository", "Official pandas repository"),
    ("https://github.com/numpy/numpy", "validation", "developer_repository", "Official NumPy repository"),
    ("https://docs.python.org/3/reference/import.html", "validation", "long_documentation", "Official Python documentation path"),
    ("https://learn.microsoft.com/en-us/defender-for-identity/", "validation", "security_keyword", "Official Microsoft security documentation"),
    ("https://support.apple.com/en-us/102660", "validation", "security_keyword", "Official Apple support page"),
    ("https://help.openai.com/en/articles/6614161-how-can-i-contact-support", "validation", "long_documentation", "Official OpenAI help article"),
    ("https://developer.mozilla.org/en-US/docs/Web/HTTP/Authentication", "validation", "login_account", "Official Mozilla authentication documentation"),
    ("https://docs.gitlab.com/user/profile/passwords/", "validation", "login_account", "Official GitLab password documentation"),
    ("https://docs.npmjs.com/about-two-factor-authentication", "validation", "login_account", "Official npm authentication documentation"),
    ("https://doc.rust-lang.org/book/ch01-01-installation.html", "validation", "long_documentation", "Official Rust documentation"),
    ("https://login.oracle.com/", "validation", "login_account", "Official Oracle login"),
    ("https://auth0.com/docs/secure/attack-protection", "validation", "security_keyword", "Official Auth0 security documentation"),
    ("https://dash.cloudflare.com/login", "validation", "login_account", "Official Cloudflare login"),
    # Sealed held-out examples.  The reported repository occurs only here.
    ("https://github.com/SunidhiDeekonda/phishing-ML-URL-detector", "test", "developer_repository", "Reported legitimate project repository"),
    ("https://github.com/facebook/react", "test", "developer_repository", "Official React repository"),
    ("https://github.com/torvalds/linux", "test", "developer_repository", "Official Linux repository"),
    ("https://github.com/nodejs/node", "test", "developer_repository", "Official Node.js repository"),
    ("https://github.com/login", "test", "login_account", "Official GitHub login"),
    ("https://accounts.google.com/", "test", "login_account", "Official Google account endpoint"),
    ("https://myaccount.google.com/security", "test", "security_keyword", "Official Google account security page"),
    ("https://vercel.com/docs/security", "test", "security_keyword", "Official Vercel documentation"),
    ("https://vercel.com/login", "test", "login_account", "Official Vercel login"),
    ("https://openai.com/security-and-privacy/", "test", "security_keyword", "Official OpenAI security page"),
    ("https://platform.openai.com/login", "test", "login_account", "Official OpenAI platform login"),
    ("https://support.microsoft.com/en-us/account-billing", "test", "login_account", "Official Microsoft account support"),
    ("https://account.apple.com/sign-in", "test", "login_account", "Official Apple account sign-in"),
    ("https://en.wikipedia.org/wiki/Phishing", "test", "security_keyword", "Legitimate encyclopedia article about phishing"),
    ("https://www.cisa.gov/secure-our-world/recognize-and-report-phishing", "test", "security_keyword", "Official government phishing guidance"),
    ("https://www.ftc.gov/news-events/topics/identity-theft/phishing-scams", "test", "security_keyword", "Official government phishing guidance"),
    ("https://www.ncsc.gov.uk/collection/phishing-scams", "test", "security_keyword", "Official government phishing guidance"),
    ("https://owasp.org/www-community/attacks/Phishing", "test", "security_keyword", "Official OWASP security documentation"),
    ("https://pages.nist.gov/800-63-3/sp800-63b.html", "test", "long_documentation", "Official NIST authentication guidance"),
    ("https://www.harvard.edu/information-security/account-protection/", "test", "login_account", "Official university security guidance"),
]


def registered_domain(url: str) -> str:
    host = urlsplit(canonicalize_url(url)).hostname or ""
    extracted = TLD_EXTRACTOR(host)
    return ".".join(part for part in (extracted.domain, extracted.suffix) if part) or host


def main() -> None:
    data = pd.read_csv(ROOT / "data/processed/dataset.csv")
    train_idx = np.load(ROOT / "data/processed/train_idx.npy")
    exact = set(data.url.astype(str).map(canonicalize_url))
    train_domains = {registered_domain(value) for value in data.iloc[train_idx].url.astype(str)}
    records = []
    for url, split, category, reason in CASES:
        canonical = canonicalize_url(url)
        domain = registered_domain(canonical)
        records.append({
            "url": url,
            "expected_label": "LEGITIMATE",
            "category": category,
            "reason": reason,
            "hard_negative_split": split,
            "exact_url_present_in_original_dataset": canonical in exact,
            "registered_domain": domain,
            "registered_domain_present_in_training": domain in train_domains,
        })
    output = ROOT / "tests/data/hard_legitimate_urls.json"
    output.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(records)} cases to {output}")


if __name__ == "__main__":
    main()
