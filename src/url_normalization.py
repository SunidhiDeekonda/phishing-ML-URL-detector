"""Conservative, network-free URL canonicalization for production inference."""

from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit


DEFAULT_SCHEME = "https"


def canonicalize_url(url: str | None, default_scheme: str = DEFAULT_SCHEME) -> str:
    """Canonicalize only representations that are safely equivalent.

    This function performs string processing only. It never resolves DNS,
    follows redirects, or fetches a webpage. Path, query, fragment, and percent
    encoding are preserved. Only a root ``/`` is removed.
    """
    text = "" if url is None else str(url).strip()
    if not text:
        return ""

    candidate = text if "://" in text else f"{default_scheme}://{text}"
    parsed = urlsplit(candidate)
    if not parsed.hostname:
        return candidate

    scheme = parsed.scheme.lower()
    host = parsed.hostname.lower()
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"

    userinfo, separator, _host_port = parsed.netloc.rpartition("@")
    try:
        port = parsed.port
    except ValueError:
        return candidate
    include_port = port is not None and not (
        (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
    )
    netloc = f"{userinfo}{separator if userinfo else ''}{host}"
    if include_port:
        netloc += f":{port}"

    path = "" if parsed.path in {"", "/"} else parsed.path
    return urlunsplit((scheme, netloc, path, parsed.query, parsed.fragment))
