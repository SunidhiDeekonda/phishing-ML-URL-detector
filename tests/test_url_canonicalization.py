"""Dedicated edge-case tests for src.url_normalization.canonicalize_url.

These tests cover boundary inputs (empty, None, IP hosts, non-default ports)
that are not explicitly tested elsewhere, improving coverage of this
safety-critical production function.
"""

from src.url_normalization import canonicalize_url


def test_empty_string_returns_empty():
    assert canonicalize_url("") == ""


def test_none_returns_empty():
    assert canonicalize_url(None) == ""


def test_whitespace_only_returns_empty():
    assert canonicalize_url("   ") == ""


def test_scheme_is_lowercased():
    assert canonicalize_url("HTTP://Example.COM") == "http://example.com"


def test_missing_scheme_defaults_to_https():
    result = canonicalize_url("example.com")
    assert result.startswith("https://")


def test_ip_address_host_preserved():
    result = canonicalize_url("http://192.168.1.1/path")
    assert "192.168.1.1" in result
    assert "/path" in result


def test_non_default_port_preserved():
    result = canonicalize_url("https://example.com:8443/api")
    assert ":8443" in result


def test_default_https_port_removed():
    result = canonicalize_url("https://example.com:443/page")
    assert ":443" not in result


def test_default_http_port_removed():
    result = canonicalize_url("http://example.com:80/page")
    assert ":80" not in result


def test_root_slash_removed():
    assert canonicalize_url("https://example.com/") == "https://example.com"


def test_non_root_trailing_slash_preserved():
    result = canonicalize_url("https://example.com/account/")
    assert result.endswith("/account/")


def test_query_string_preserved():
    result = canonicalize_url("https://example.com/search?q=test")
    assert "?q=test" in result
