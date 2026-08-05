"""Zane's "infinite information" pipeline: a live web search tool exposed to
the LLM as a function call.

Backend selection is automatic ("auto"): if a Tavily API key is configured,
Tavily is used (it is purpose-built for LLM tool-calling and returns clean,
pre-summarized snippets); otherwise the pipeline falls back to DuckDuckGo
via the `ddgs` package, which requires no API key at all. Both paths are
wrapped to look identical to the rest of the system.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

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
    ) -> None:
        self.tavily_api_key = tavily_api_key
        self.backend = backend
        self.default_max_results = default_max_results
        self._tavily_client = None

        if self._wants_tavily():
            try:
                from tavily import AsyncTavilyClient

                self._tavily_client = AsyncTavilyClient(api_key=self.tavily_api_key)
            except ImportError:
                logger.warning(
                    "TAVILY_API_KEY is set but the `tavily-python` package is not "
                    "installed; falling back to DuckDuckGo. Run "
                    "`pip install tavily-python` to enable Tavily."
                )
                self._tavily_client = None

    def _wants_tavily(self) -> bool:
        if self.backend == "duckduckgo":
            return False
        if self.backend == "tavily":
            return True
        return bool(self.tavily_api_key)  # "auto"

    async def search(self, query: str, max_results: Optional[int] = None) -> List[SearchResultItem]:
        n = max_results or self.default_max_results
        if self._tavily_client is not None:
            try:
                return await self._search_tavily(query, n)
            except Exception as exc:  # noqa: BLE001 - fall back on any Tavily failure
                logger.warning("Tavily search failed (%s); falling back to DuckDuckGo.", exc)

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
