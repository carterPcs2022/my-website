"""Tests for zane.tools.web_search, focused on the SerpAPI/Serper backends
and the four-tier auto priority (Tavily > SerpAPI > Serper > DuckDuckGo).
Both HTTP-backed backends are tested against a real httpx transport
(httpx.MockTransport), same pattern as tests/test_wolfram_tool.py.
"""
import httpx
import pytest

from zane.tools.web_search import SearchResultItem, WebSearchError, WebSearchTool


def _serpapi_tool(handler, **kwargs) -> WebSearchTool:
    tool = WebSearchTool(serpapi_api_key="test-serpapi-key", backend="serpapi", **kwargs)
    tool._serpapi_http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return tool


def _serper_tool(handler, **kwargs) -> WebSearchTool:
    tool = WebSearchTool(serper_api_key="test-serper-key", backend="serper", **kwargs)
    tool._serper_http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return tool


# --- backend priority selection ---


def test_explicit_serpapi_backend_disables_tavily_and_serper():
    tool = WebSearchTool(tavily_api_key="tk", serpapi_api_key="sk", serper_api_key="pk", backend="serpapi")
    assert tool._wants_tavily() is False
    assert tool._wants_serpapi() is True
    assert tool._wants_serper() is False


def test_explicit_serper_backend_disables_tavily_and_serpapi():
    tool = WebSearchTool(tavily_api_key="tk", serpapi_api_key="sk", serper_api_key="pk", backend="serper")
    assert tool._wants_tavily() is False
    assert tool._wants_serpapi() is False
    assert tool._wants_serper() is True


def test_auto_backend_prioritizes_tavily_over_serpapi_and_serper():
    tool = WebSearchTool(tavily_api_key="tk", serpapi_api_key="sk", serper_api_key="pk", backend="auto")
    assert tool._wants_tavily() is True


def test_auto_backend_prioritizes_serpapi_over_serper_when_no_tavily_key():
    tool = WebSearchTool(tavily_api_key=None, serpapi_api_key="sk", serper_api_key="pk", backend="auto")
    assert tool._wants_tavily() is False
    assert tool._wants_serpapi() is True


def test_auto_backend_uses_serper_when_only_serper_key_present():
    tool = WebSearchTool(tavily_api_key=None, serpapi_api_key=None, serper_api_key="pk", backend="auto")
    assert tool._wants_tavily() is False
    assert tool._wants_serpapi() is False
    assert tool._wants_serper() is True


def test_auto_backend_uses_duckduckgo_when_no_keys_present():
    tool = WebSearchTool(tavily_api_key=None, serpapi_api_key=None, serper_api_key=None, backend="auto")
    assert tool._wants_tavily() is False
    assert tool._wants_serpapi() is False
    assert tool._wants_serper() is False


def test_duckduckgo_explicit_backend_disables_all_others():
    tool = WebSearchTool(tavily_api_key="tk", serpapi_api_key="sk", serper_api_key="pk", backend="duckduckgo")
    assert tool._wants_tavily() is False
    assert tool._wants_serpapi() is False
    assert tool._wants_serper() is False


def test_explicit_tavily_backend_disables_serpapi_and_serper():
    tool = WebSearchTool(tavily_api_key="tk", serpapi_api_key="sk", serper_api_key="pk", backend="tavily")
    assert tool._wants_serpapi() is False
    assert tool._wants_serper() is False


# --- _search_serpapi ---


async def test_search_serpapi_returns_parsed_results():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["q"] == "ninjago"
        assert request.url.params["api_key"] == "test-serpapi-key"
        assert request.url.params["engine"] == "google"
        return httpx.Response(
            200,
            json={
                "organic_results": [
                    {
                        "title": "Ninjago Wiki",
                        "link": "https://ninjago.fandom.com",
                        "snippet": "A wiki about Ninjago.",
                    },
                ]
            },
        )

    tool = _serpapi_tool(handler)
    results = await tool.search("ninjago")
    assert results == [
        SearchResultItem(
            title="Ninjago Wiki", url="https://ninjago.fandom.com", snippet="A wiki about Ninjago."
        )
    ]
    await tool.close()


async def test_search_serpapi_respects_max_results():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "organic_results": [
                    {"title": f"Result {i}", "link": f"https://example.com/{i}", "snippet": "s"}
                    for i in range(10)
                ]
            },
        )

    tool = _serpapi_tool(handler)
    results = await tool.search("query", max_results=3)
    assert len(results) == 3
    await tool.close()


async def test_search_serpapi_raises_on_error_field_in_response():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"error": "Invalid API key"})

    tool = _serpapi_tool(handler)
    with pytest.raises(WebSearchError):
        await tool._search_serpapi("query", 5)
    await tool.close()


async def test_search_serpapi_raises_on_http_error_status():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="server error")

    tool = _serpapi_tool(handler)
    with pytest.raises(WebSearchError):
        await tool._search_serpapi("query", 5)
    await tool.close()


async def test_search_falls_back_to_duckduckgo_when_serpapi_fails(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="server error")

    tool = _serpapi_tool(handler)
    called = {}

    async def _fake_ddg(query, max_results):
        called["used"] = True
        return [SearchResultItem(title="DDG result", url="https://ddg.example", snippet="fallback")]

    monkeypatch.setattr(tool, "_search_duckduckgo", _fake_ddg)
    results = await tool.search("query")
    assert called.get("used") is True
    assert results[0].title == "DDG result"
    await tool.close()


# --- _search_serper ---


async def test_search_serper_returns_parsed_results():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["x-api-key"] == "test-serper-key"
        assert request.content  # POST body carries the query, not query params
        return httpx.Response(
            200,
            json={
                "organic": [
                    {
                        "title": "Ninjago Fandom",
                        "link": "https://ninjago.fandom.com",
                        "snippet": "The Ninjago wiki.",
                    },
                ]
            },
        )

    tool = _serper_tool(handler)
    results = await tool.search("ninjago")
    assert results == [
        SearchResultItem(
            title="Ninjago Fandom", url="https://ninjago.fandom.com", snippet="The Ninjago wiki."
        )
    ]
    await tool.close()


async def test_search_serper_raises_on_error_field_in_response():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"error": "Invalid API key"})

    tool = _serper_tool(handler)
    with pytest.raises(WebSearchError):
        await tool._search_serper("query", 5)
    await tool.close()


async def test_search_serper_raises_on_http_error_status():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="server error")

    tool = _serper_tool(handler)
    with pytest.raises(WebSearchError):
        await tool._search_serper("query", 5)
    await tool.close()


async def test_search_falls_through_serpapi_then_serper_then_duckduckgo(monkeypatch):
    # backend="auto" with both keys configured: SerpAPI fails -> Serper is
    # tried next (not skipped straight to DuckDuckGo) -> Serper succeeds.
    def serpapi_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="serpapi down")

    tool = WebSearchTool(serpapi_api_key="sk", serper_api_key="pk", backend="auto")
    tool._serpapi_http_client = httpx.AsyncClient(transport=httpx.MockTransport(serpapi_handler))

    def serper_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"organic": [{"title": "Serper result", "link": "https://s.example", "snippet": "s"}]}
        )

    tool._serper_http_client = httpx.AsyncClient(transport=httpx.MockTransport(serper_handler))

    results = await tool.search("query")
    assert results[0].title == "Serper result"
    await tool.close()


# --- close() ---


async def test_close_closes_serpapi_client_if_one_was_opened():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"organic_results": []})

    tool = _serpapi_tool(handler)
    await tool.close()
    assert tool._serpapi_http_client.is_closed


async def test_close_closes_serper_client_if_one_was_opened():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"organic": []})

    tool = _serper_tool(handler)
    await tool.close()
    assert tool._serper_http_client.is_closed


async def test_close_is_a_safe_no_op_when_neither_was_ever_used():
    tool = WebSearchTool()
    await tool.close()  # must not raise
