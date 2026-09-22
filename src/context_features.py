"""Network-free extraction of optional caller-supplied HTML/email signals."""

from __future__ import annotations

import re
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit

RISK_TERMS = ("login", "verify", "urgent", "account", "password", "credential", "confirm", "suspend")
CTA_TERMS = ("click here", "act now", "sign in", "verify now", "confirm now", "open link")
URL_RE = re.compile(r"https?://[^\s<>\"']+", re.I)


class _SafeHTMLParser(HTMLParser):
    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base_host = (urlsplit(base_url if "://" in base_url else f"http://{base_url}").hostname or "").lower()
        self.form_count = self.password_inputs = self.iframe_count = 0
        self.script_count = self.meta_refresh = self.external_targets = 0
        self.text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {k.lower(): (v or "") for k, v in attrs}
        tag = tag.lower()
        self.form_count += int(tag == "form")
        self.password_inputs += int(tag == "input" and values.get("type", "").lower() == "password")
        self.iframe_count += int(tag == "iframe")
        self.script_count += int(tag == "script")
        self.meta_refresh += int(tag == "meta" and values.get("http-equiv", "").lower() == "refresh")
        target = values.get("action") or values.get("href") or values.get("src")
        if target and not target.lower().startswith(("data:", "javascript:", "#")):
            host = (urlsplit(urljoin(f"http://{self.base_host}/", target)).hostname or "").lower()
            self.external_targets += int(bool(host and self.base_host and host != self.base_host))

    def handle_data(self, data: str) -> None:
        self.text.append(data)


def extract_html_context(url: str, html: str | None) -> dict[str, int]:
    parser = _SafeHTMLParser(url)
    parser.feed((html or "")[:200_000])
    text = " ".join(parser.text).lower()
    return {"form_count": parser.form_count, "password_input_count": parser.password_inputs,
            "iframe_count": parser.iframe_count, "script_count": parser.script_count,
            "external_target_count": parser.external_targets, "meta_refresh_count": parser.meta_refresh,
            "credential_term_count": sum(text.count(term) for term in RISK_TERMS)}


def extract_email_context(email_text: str | None) -> dict[str, int]:
    text = (email_text or "")[:200_000].lower()
    return {"url_count": len(URL_RE.findall(text)), "risk_term_count": sum(text.count(term) for term in RISK_TERMS),
            "credential_request_count": len(re.findall(r"(?:enter|send|provide|confirm)\s+(?:your\s+)?(?:password|credentials)", text)),
            "call_to_action_count": sum(text.count(term) for term in CTA_TERMS)}


def analyze_context(url: str, html: str | None = None, email_text: str | None = None) -> dict[str, object]:
    html_signals, email_signals = extract_html_context(url, html), extract_email_context(email_text)
    flags: list[str] = []
    if html_signals["password_input_count"] and html_signals["external_target_count"]: flags.append("password form references an external host")
    if html_signals["meta_refresh_count"]: flags.append("HTML contains meta refresh")
    if html_signals["credential_term_count"] >= 3: flags.append("HTML contains repeated credential-related language")
    if email_signals["credential_request_count"]: flags.append("email asks for credentials")
    if email_signals["risk_term_count"] >= 3 and email_signals["call_to_action_count"]: flags.append("email combines urgency/account language with a call to action")
    return {"html": html_signals, "email": email_signals, "context_risk_flags": flags,
            "included_in_validated_probability": False, "automatic_url_fetching": False}
