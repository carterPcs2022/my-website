"""Tests for zane.tools.fetch_page. HTTP behavior is tested against a real
httpx transport (httpx.MockTransport), same pattern as
tests/test_wolfram_tool.py. The SSRF guard (`_resolves_to_public_address`)
is tested directly against IP literals — `socket.getaddrinfo` resolves
those without any real DNS/network call, so these stay fully offline and
deterministic; the one DNS-failure path is exercised via monkeypatching
`socket.getaddrinfo` itself rather than relying on a real failed lookup.
"""
import socket

import httpx
import pytest

from zane.tools.fetch_page import (
    FETCH_PAGE_TOOL_SCHEMA,
    PageFetcher,
    _is_safe_url,
    _resolves_to_public_address,
    _TextExtractor,
    fetch_page,
)


def _allow_all_hosts(monkeypatch):
    monkeypatch.setattr("zane.tools.fetch_page._resolves_to_public_address", lambda hostname: True)


# --- _TextExtractor ---


def test_text_extractor_strips_script_and_style():
    html = (
        "<html><head><style>body{color:red}</style></head>"
        "<body><script>alert(1)</script><p>Hello world</p></body></html>"
    )
    extractor = _TextExtractor()
    extractor.feed(html)
    text = extractor.get_text()
    assert "Hello world" in text
    assert "alert" not in text
    assert "color:red" not in text


def test_text_extractor_captures_title_separately_from_body_text():
    html = "<html><head><title>My Page</title></head><body><p>Body text</p></body></html>"
    extractor = _TextExtractor()
    extractor.feed(html)
    assert extractor.title == "My Page"
    assert "My Page" not in extractor.get_text()
    assert "Body text" in extractor.get_text()


def test_text_extractor_collapses_whitespace():
    html = "<p>Hello\n\n   world</p>"
    extractor = _TextExtractor()
    extractor.feed(html)
    assert extractor.get_text() == "Hello world"


# --- SSRF guard: _resolves_to_public_address / _is_safe_url ---


def test_resolves_to_public_address_rejects_loopback_ip_literal():
    assert _resolves_to_public_address("127.0.0.1") is False
    assert _resolves_to_public_address("::1") is False


def test_resolves_to_public_address_rejects_private_ip_literals():
    assert _resolves_to_public_address("10.0.0.5") is False
    assert _resolves_to_public_address("192.168.1.1") is False
    assert _resolves_to_public_address("172.16.0.1") is False


def test_resolves_to_public_address_rejects_link_local_ip_literal():
    # 169.254.169.254 is the AWS/GCP/Azure cloud metadata endpoint — the
    # canonical real-world SSRF target this guard exists to block.
    assert _resolves_to_public_address("169.254.169.254") is False


def test_resolves_to_public_address_accepts_public_ip_literal():
    assert _resolves_to_public_address("8.8.8.8") is True


def test_resolves_to_public_address_handles_dns_failure_gracefully(monkeypatch):
    def _raise(*args, **kwargs):
        raise socket.gaierror("simulated DNS failure")

    monkeypatch.setattr("zane.tools.fetch_page.socket.getaddrinfo", _raise)
    assert _resolves_to_public_address("example.com") is False


async def test_is_safe_url_rejects_non_http_scheme():
    assert await _is_safe_url("ftp://example.com/file") is False


async def test_is_safe_url_rejects_missing_hostname():
    assert await _is_safe_url("http:///path") is False


async def test_is_safe_url_rejects_private_ip_literal_url():
    assert await _is_safe_url("http://127.0.0.1/admin") is False


# --- fetch_page: fetch/parse behavior ---


async def test_fetch_page_returns_extracted_text_and_title(monkeypatch):
    _allow_all_hosts(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        html = (
            "<html><head><title>Example Title</title></head>"
            "<body><p>Some article text.</p></body></html>"
        )
        return httpx.Response(200, text=html, headers={"content-type": "text/html"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    result = await fetch_page("http://example.com/article", client)
    assert "Example Title" in result
    assert "Some article text." in result
    assert "http://example.com/article" in result
    await client.aclose()


async def test_fetch_page_rejects_non_text_content_type(monkeypatch):
    _allow_all_hosts(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"\x89PNG", headers={"content-type": "image/png"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    result = await fetch_page("http://example.com/image.png", client)
    assert result.startswith("FETCH_ERROR")
    assert "content type" in result.lower()
    await client.aclose()


async def test_fetch_page_returns_error_on_http_failure_status(monkeypatch):
    _allow_all_hosts(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="not found")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    result = await fetch_page("http://example.com/missing", client)
    assert result.startswith("FETCH_ERROR")
    await client.aclose()


async def test_fetch_page_returns_error_on_network_failure(monkeypatch):
    _allow_all_hosts(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("simulated connection failure", request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    result = await fetch_page("http://example.com/", client)
    assert result.startswith("FETCH_ERROR")
    await client.aclose()


async def test_fetch_page_reports_no_readable_text_honestly(monkeypatch):
    _allow_all_hosts(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        html = "<html><body><script>x = 1;</script></body></html>"
        return httpx.Response(200, text=html, headers={"content-type": "text/html"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    result = await fetch_page("http://example.com/", client)
    assert result.startswith("FETCH_ERROR")
    assert "no readable text" in result.lower()
    await client.aclose()


async def test_fetch_page_truncates_long_content(monkeypatch):
    _allow_all_hosts(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        long_text = "word " * 5000
        return httpx.Response(200, text=f"<p>{long_text}</p>", headers={"content-type": "text/html"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    result = await fetch_page("http://example.com/", client, max_chars=100)
    assert "[Content truncated.]" in result
    await client.aclose()


# --- SSRF guard, end to end through fetch_page (no network occurs) ---


async def test_fetch_page_refuses_url_resolving_to_private_address():
    result = await fetch_page("http://10.0.0.5/secret")
    assert result.startswith("FETCH_ERROR")
    assert "disallowed network address" in result.lower()


async def test_fetch_page_refuses_cloud_metadata_endpoint():
    result = await fetch_page("http://169.254.169.254/latest/meta-data/")
    assert result.startswith("FETCH_ERROR")


# --- redirect guard: re-validated on every hop ---


async def test_fetch_page_follows_redirect_to_safe_url(monkeypatch):
    _allow_all_hosts(monkeypatch)
    visited = []

    def handler(request: httpx.Request) -> httpx.Response:
        visited.append(str(request.url))
        if str(request.url) == "http://example.com/start":
            return httpx.Response(302, headers={"location": "http://example.com/final"})
        return httpx.Response(200, text="<p>Final content.</p>", headers={"content-type": "text/html"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    result = await fetch_page("http://example.com/start", client)
    assert "Final content." in result
    assert len(visited) == 2
    await client.aclose()


async def test_fetch_page_blocks_redirect_to_disallowed_address(monkeypatch):
    def _resolve(hostname: str) -> bool:
        return hostname != "internal.example"

    monkeypatch.setattr("zane.tools.fetch_page._resolves_to_public_address", _resolve)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "http://internal.example/secret"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    result = await fetch_page("http://example.com/start", client)
    assert result.startswith("FETCH_ERROR")
    assert "disallowed" in result.lower()
    await client.aclose()


async def test_fetch_page_gives_up_after_too_many_redirects(monkeypatch):
    _allow_all_hosts(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "http://example.com/loop"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    result = await fetch_page("http://example.com/loop", client)
    assert result.startswith("FETCH_ERROR")
    assert "too many redirects" in result.lower()
    await client.aclose()


# --- PageFetcher ---


async def test_page_fetcher_fetch_delegates_to_fetch_page(monkeypatch):
    _allow_all_hosts(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, text="<p>Fetched via PageFetcher.</p>", headers={"content-type": "text/html"}
        )

    fetcher = PageFetcher()
    fetcher._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    result = await fetcher.fetch("http://example.com/")
    assert "Fetched via PageFetcher." in result
    await fetcher.close()


# --- tool schema ---


def test_tool_schema_name_and_required_param():
    assert FETCH_PAGE_TOOL_SCHEMA["function"]["name"] == "fetch_page"
    assert FETCH_PAGE_TOOL_SCHEMA["function"]["parameters"]["required"] == ["url"]
