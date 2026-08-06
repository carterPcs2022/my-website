"""Zane's "read the full page" tool: fetches a specific URL (typically one
returned by a prior `web_search` call) and extracts its readable text, for
when a short search snippet isn't enough to answer confidently.

SECURITY NOTE — Server-Side Request Forgery (SSRF): this tool lets the LLM
ask this process to make an arbitrary outbound HTTP request. Without a
guard, a crafted URL (or a URL surfaced by a malicious search result/page)
could reach internal-only endpoints — `http://169.254.169.254/` (cloud
metadata), `http://localhost:...`, a private-network admin panel — that
have no business being reachable from a chat tool call. `_is_safe_url`
resolves the hostname and rejects anything that isn't a public IP address
*before* connecting, and `_fetch_with_redirect_guard` re-runs that same
check on every redirect hop (an httpx client configured to auto-follow
redirects would silently bypass the initial check the moment a remote
server 302s somewhere unsafe). This closes the two most common ways an
LLM-facing fetch tool becomes an SSRF vector; it does not defend against a
DNS-rebinding race between the check and the actual connect (an accepted,
documented residual risk — the same class of caveat this project applies
elsewhere, e.g. `thermal_monitor.py`'s sensor-access platform caveats).
"""
from __future__ import annotations

import asyncio
import ipaddress
import logging
import re
import socket
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse

import httpx

logger = logging.getLogger("zane.tools.fetch_page")

DEFAULT_TIMEOUT_S = 10.0
DEFAULT_MAX_TEXT_CHARS = 6000
_MAX_RESPONSE_CHARS = 2_000_000  # cap how much raw HTML we'll ever parse
_MAX_REDIRECTS = 3
_USER_AGENT = "ZaneDigitalMind/1.0 (+fetch_page tool; single-page reader)"

FETCH_PAGE_TOOL_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "fetch_page",
        "description": (
            "Fetches and reads the full text content of a specific web page "
            "(typically a URL returned by a prior web_search call) when a "
            "short snippet isn't enough to answer confidently — use this to "
            "verify a claim, read a full article, or find detail a search "
            "summary omitted. Only fetches public http(s) pages; refuses "
            "URLs that resolve to private, loopback, or internal network "
            "addresses."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The full http(s) URL to fetch, e.g. from a web_search result.",
                },
            },
            "required": ["url"],
        },
    },
}


class FetchPageError(RuntimeError):
    """Raised internally for any fetch/parse failure; always caught by
    `fetch_page`, which returns an honest `FETCH_ERROR: ...` string rather
    than letting this propagate into the tool-calling loop."""


def _resolves_to_public_address(hostname: str) -> bool:
    """Blocking DNS resolution + IP-class check — always called via
    `asyncio.to_thread`, never directly on the event loop."""
    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        return False
    if not infos:
        return False
    for info in infos:
        try:
            ip = ipaddress.ip_address(info[4][0])
        except ValueError:
            return False
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            return False
    return True


async def _is_safe_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False
    hostname = parsed.hostname
    if not hostname:
        return False
    return await asyncio.to_thread(_resolves_to_public_address, hostname)


class _TextExtractor(HTMLParser):
    """Minimal, dependency-free HTML-to-text extraction (stdlib only — no
    new project dependency for something this scoped). Skips script/style/
    similar non-visible content and captures `<title>` separately."""

    _SKIP_TAGS = {"script", "style", "noscript", "template", "svg"}

    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self._in_title = False
        self._chunks: List[str] = []
        self.title = ""

    def handle_starttag(self, tag: str, attrs) -> None:  # noqa: ANN001
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
        elif tag == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
        elif tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._in_title:
            self.title += data
            return
        stripped = data.strip()
        if stripped:
            self._chunks.append(stripped)

    def get_text(self) -> str:
        return re.sub(r"\s+", " ", " ".join(self._chunks)).strip()


async def _fetch_with_redirect_guard(
    client: httpx.AsyncClient, url: str, max_redirects: int = _MAX_REDIRECTS
) -> httpx.Response:
    """Fetches `url`, following redirects manually so every hop is
    re-validated by `_is_safe_url` before being requested — see the module
    docstring's SSRF note for why this can't just be `follow_redirects=True`."""
    current_url = url
    for _ in range(max_redirects + 1):
        if not await _is_safe_url(current_url):
            raise FetchPageError(
                f"{current_url!r} resolves to a private, loopback, or otherwise "
                "disallowed network address."
            )
        response = await client.get(
            current_url, headers={"User-Agent": _USER_AGENT}, follow_redirects=False
        )
        if response.is_redirect:
            location = response.headers.get("location")
            if not location:
                response.raise_for_status()
                return response
            current_url = urljoin(current_url, location)
            continue
        return response
    raise FetchPageError(f"Too many redirects while fetching {url!r}.")


async def fetch_page(
    url: str,
    client: Optional[httpx.AsyncClient] = None,
    *,
    max_chars: int = DEFAULT_MAX_TEXT_CHARS,
) -> str:
    """The LLM tool entry point. Never raises: every failure mode (invalid
    URL, disallowed address, network error, non-text content, unparsable
    HTML) returns an honest `FETCH_ERROR: ...` string instead."""
    owns_client = client is None
    active_client = client or httpx.AsyncClient(timeout=DEFAULT_TIMEOUT_S)
    try:
        try:
            response = await _fetch_with_redirect_guard(active_client, url)
            response.raise_for_status()
        except FetchPageError as exc:
            logger.warning("Refused to fetch %s: %s", url, exc)
            return f"FETCH_ERROR: {exc}"
        except httpx.HTTPError as exc:
            logger.warning("fetch_page failed for %s: %s", url, exc)
            return f"FETCH_ERROR: Could not fetch {url}: {exc}"

        content_type = response.headers.get("content-type", "")
        if "html" not in content_type and "text" not in content_type:
            return f"FETCH_ERROR: Unsupported content type ({content_type or 'unknown'}) for {url}."

        raw_html = response.text[:_MAX_RESPONSE_CHARS]
    finally:
        if owns_client:
            await active_client.aclose()

    extractor = _TextExtractor()
    try:
        extractor.feed(raw_html)
    except Exception as exc:  # noqa: BLE001 - malformed HTML must not crash the tool call
        logger.warning("HTML parsing failed for %s: %s", url, exc)
        return f"FETCH_ERROR: Could not parse page content from {url}."

    text = extractor.get_text()
    if not text:
        return f"FETCH_ERROR: No readable text content found at {url}."

    truncated = len(text) > max_chars
    text = text[:max_chars]
    header = f"Fetched page: {extractor.title.strip() or url}\nURL: {url}\n\n"
    footer = "\n\n[Content truncated.]" if truncated else ""
    return header + text + footer


class PageFetcher:
    """Holds a persistent `httpx.AsyncClient` for connection reuse across
    calls — the same shape as `WolframAlphaClient`, constructed once in
    `SharedBackend.build()` and shared across the process."""

    def __init__(self, *, timeout_s: float = DEFAULT_TIMEOUT_S, max_text_chars: int = DEFAULT_MAX_TEXT_CHARS) -> None:
        self._client = httpx.AsyncClient(timeout=timeout_s)
        self.max_text_chars = max_text_chars

    async def fetch(self, url: str) -> str:
        return await fetch_page(url, self._client, max_chars=self.max_text_chars)

    async def close(self) -> None:
        await self._client.aclose()
