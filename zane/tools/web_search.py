"""Zane's "infinite information" pipeline: a live web search tool exposed to
the LLM as a function call.

Backend selection is automatic ("auto") across four tiers, in priority
order: Tavily (if `TAVILY_API_KEY` is set — purpose-built for LLM
tool-calling, clean pre-summarized snippets) > SerpAPI (if
`SERPAPI_API_KEY` is set — real-time Google search results) > Serper (if
`SERPER_API_KEY` is set — a different, cheaper real-time Google search
results provider) > DuckDuckGo via the `ddgs` package, which requires no
API key at all and is always the final fallback. A configured but failing
backend falls through to the next tier rather than raising, so a single
backend's outage never breaks the tool. All four paths are wrapped to
return the same `SearchResultItem` shape.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger("zane.tools.web_search")


class WebSearchError(RuntimeError):
    """Raised when neither search backend can fulfill a request."""


@dataclass
class SearchResultItem:
    title: str
    url: str
    snippet: str

    def to_dict(self) -> Dict[str, str]:
        return {"title": self.title, "url": self.url, "snippet": self.snippet}


WEB_SEARCH_TOOL_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the live internet for current, factual information that may "
            "not be present in the model's training data (recent events, "
            "specific figures, up-to-date documentation, etc). Returns a short "
            "list of titled results with URLs and text snippets."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query, phrased as concise keywords.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default 5).",
                    "minimum": 1,
                    "maximum": 10,
                },
            },
            "required": ["query"],
        },
    },
}


class WebSearchTool:
    def __init__(
        self,
        tavily_api_key: Optional[str] = None,
        backend: str = "auto",
        default_max_results: int = 5,
        serpapi_api_key: Optional[str] = None,
        serper_api_key: Optional[str] = None,
    ) -> None:
        self.tavily_api_key = tavily_api_key
        self.serpapi_api_key = serpapi_api_key
        self.serper_api_key = serper_api_key
        self.backend = backend
        self.default_max_results = default_max_results
        self._tavily_client = None
        # Lazily created on first use, not eagerly here — no reason to
        # open an httpx client for a backend that's never actually called
        # (e.g. backend="tavily" with SerpAPI/Serper keys also configured).
        self._serpapi_http_client: Optional[httpx.AsyncClient] = None
        self._serper_http_client: Optional[httpx.AsyncClient] = None

        if self._wants_tavily():
            try:
                from tavily import AsyncTavilyClient

                self._tavily_client = AsyncTavilyClient(api_key=self.tavily_api_key)
            except ImportError:
                logger.warning(
                    "TAVILY_API_KEY is set but the `tavily-python` package is not "
                    "installed; falling back to SerpAPI/DuckDuckGo. Run "
                    "`pip install tavily-python` to enable Tavily."
                )
                self._tavily_client = None

    def _wants_tavily(self) -> bool:
        if self.backend in ("duckduckgo", "serpapi", "serper"):
            return False
        if self.backend == "tavily":
            return True
        return bool(self.tavily_api_key)  # "auto"

    def _wants_serpapi(self) -> bool:
        if self.backend in ("duckduckgo", "tavily", "serper"):
            return False
        if self.backend == "serpapi":
            return True
        return bool(self.serpapi_api_key)  # "auto"

    def _wants_serper(self) -> bool:
        if self.backend in ("duckduckgo", "tavily", "serpapi"):
            return False
        if self.backend == "serper":
            return True
        return bool(self.serper_api_key)  # "auto"

    async def search(self, query: str, max_results: Optional[int] = None) -> List[SearchResultItem]:
        n = max_results or self.default_max_results
        if self._tavily_client is not None:
            try:
                return await self._search_tavily(query, n)
            except Exception as exc:  # noqa: BLE001 - fall through to the next tier
                logger.warning("Tavily search failed (%s); trying next backend.", exc)

        if self._wants_serpapi():
            try:
                return await self._search_serpapi(query, n)
            except Exception as exc:  # noqa: BLE001 - fall through to the next tier
                logger.warning("SerpAPI search failed (%s); trying next backend.", exc)

        if self._wants_serper():
            try:
                return await self._search_serper(query, n)
            except Exception as exc:  # noqa: BLE001 - fall back to DuckDuckGo
                logger.warning("Serper search failed (%s); falling back to DuckDuckGo.", exc)

        return await self._search_duckduckgo(query, n)

    async def _search_tavily(self, query: str, max_results: int) -> List[SearchResultItem]:
        response = await self._tavily_client.search(
            query=query, max_results=max_results, search_depth="basic"
        )
        results = []
        for item in response.get("results", []):
            results.append(
                SearchResultItem(
                    title=item.get("title", "Untitled"),
                    url=item.get("url", ""),
                    snippet=item.get("content", ""),
                )
            )
        return results

    async def _search_serpapi(self, query: str, max_results: int) -> List[SearchResultItem]:
        if self._serpapi_http_client is None:
            self._serpapi_http_client = httpx.AsyncClient(timeout=10.0)

        try:
            response = await self._serpapi_http_client.get(
                "https://serpapi.com/search.json",
                params={
                    "engine": "google",
                    "q": query,
                    "api_key": self.serpapi_api_key,
                    "num": max_results,
                },
            )
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError as exc:
            raise WebSearchError(f"SerpAPI request failed: {exc}") from exc

        if "error" in data:
            raise WebSearchError(f"SerpAPI returned an error: {data['error']}")

        results = []
        for item in data.get("organic_results", [])[:max_results]:
            results.append(
                SearchResultItem(
                    title=item.get("title", "Untitled"),
                    url=item.get("link", ""),
                    snippet=item.get("snippet", ""),
                )
            )
        return results

    async def _search_serper(self, query: str, max_results: int) -> List[SearchResultItem]:
        if self._serper_http_client is None:
            self._serper_http_client = httpx.AsyncClient(timeout=10.0)

        try:
            response = await self._serper_http_client.post(
                "https://google.serper.dev/search",
                headers={"X-API-KEY": self.serper_api_key or "", "Content-Type": "application/json"},
                json={"q": query, "num": max_results},
            )
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError as exc:
            raise WebSearchError(f"Serper request failed: {exc}") from exc

        if "error" in data:
            raise WebSearchError(f"Serper returned an error: {data['error']}")

        results = []
        for item in data.get("organic", [])[:max_results]:
            results.append(
                SearchResultItem(
                    title=item.get("title", "Untitled"),
                    url=item.get("link", ""),
                    snippet=item.get("snippet", ""),
                )
            )
        return results

    async def _search_duckduckgo(self, query: str, max_results: int) -> List[SearchResultItem]:
        def _sync_search() -> List[SearchResultItem]:
            try:
                from ddgs import DDGS
            except ImportError:
                try:
                    from duckduckgo_search import DDGS  # type: ignore
                except ImportError as exc:
                    raise WebSearchError(
                        "No search backend available: install `ddgs` "
                        "(or `duckduckgo-search`) or configure TAVILY_API_KEY."
                    ) from exc

            with DDGS() as ddgs:
                raw_results = list(ddgs.text(query, max_results=max_results))

            return [
                SearchResultItem(
                    title=r.get("title", "Untitled"),
                    url=r.get("href", r.get("url", "")),
                    snippet=r.get("body", r.get("snippet", "")),
                )
                for r in raw_results
            ]

        try:
            return await asyncio.to_thread(_sync_search)
        except WebSearchError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise WebSearchError(f"DuckDuckGo search failed: {exc}") from exc

    async def search_as_tool_result(self, query: str, max_results: Optional[int] = None) -> str:
        """Formats results as a compact string suitable for feeding straight
        back into the LLM as a tool response message."""
        try:
            results = await self.search(query, max_results)
        except WebSearchError as exc:
            return f"SEARCH_ERROR: {exc}"

        if not results:
            return f"No results found for query: {query!r}"

        lines = [f"Search results for {query!r}:"]
        for i, item in enumerate(results, start=1):
            lines.append(f"{i}. {item.title}\n   URL: {item.url}\n   Snippet: {item.snippet}")
        return "\n".join(lines)

    async def close(self) -> None:
        """Releases the SerpAPI/Serper httpx clients, if either was ever
        opened. Tavily's `AsyncTavilyClient` manages its own session
        lifecycle and needs no explicit close here."""
        if self._serpapi_http_client is not None:
            await self._serpapi_http_client.aclose()
        if self._serper_http_client is not None:
            await self._serper_http_client.aclose()
