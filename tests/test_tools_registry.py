import json

import pytest

from zane.analytics_bridge import AnalyticsEngine
from zane.tools.registry import ToolRegistry
from zane.tools.web_search import WebSearchTool


class _StubWebSearchTool(WebSearchTool):
    """Avoids any real network call in tests."""

    def __init__(self):
        super().__init__(tavily_api_key=None, backend="duckduckgo")

    async def search_as_tool_result(self, query, max_results=None):
        return f"stubbed results for {query!r}"


class _StubPageFetcher:
    """Avoids any real network call in tests."""

    async def fetch(self, url):
        return f"stubbed page content for {url!r}"


class _StubGroqClient:
    """Avoids any real network call to Groq in tests."""

    def __init__(self, reply: str = "stubbed translation", fail: bool = False):
        self.reply = reply
        self.fail = fail
        self.last_messages = None

    async def chat_completion(self, messages, **kwargs):
        self.last_messages = messages
        if self.fail:
            from zane.groq_client import GroqUnavailableError

            raise GroqUnavailableError("simulated Groq outage")

        class _Msg:
            def __init__(self, content):
                self.content = content

        class _Choice:
            def __init__(self, content):
                self.message = _Msg(content)

        class _Completion:
            def __init__(self, content):
                self.choices = [_Choice(content)]

        return _Completion(self.reply)


@pytest.fixture
def registry():
    analytics = AnalyticsEngine(worker_threads=1, seed=99)
    web_search = _StubWebSearchTool()
    groq_client = _StubGroqClient()
    reg = ToolRegistry(analytics, web_search, groq_client, page_fetcher=_StubPageFetcher())
    yield reg
    analytics.shutdown()


async def test_dispatch_web_search(registry):
    result = await registry.dispatch("web_search", json.dumps({"query": "Ninjago lore"}))
    assert "Ninjago lore" in result


async def test_dispatch_calculate_success_probability(registry):
    args = json.dumps(
        {
            "danger_level": 40,
            "team_synergy": 90,
            "resource_availability": 80,
            "historical_success_rate": 75,
            "complexity_index": 30,
        }
    )
    result = await registry.dispatch("calculate_success_probability", args)
    assert "success_probability=" in result
    assert "risk_index=" in result


async def test_dispatch_translate_text(registry):
    args = json.dumps({"text": "Hello", "target_language": "French"})
    result = await registry.dispatch("translate_text", args)
    assert result == "stubbed translation"
    assert registry._groq_client.last_messages[0]["role"] == "system"
    assert "French" in registry._groq_client.last_messages[0]["content"]
    assert registry._groq_client.last_messages[1]["content"] == "Hello"


async def test_dispatch_translate_text_propagates_groq_failure_as_tool_error():
    analytics = AnalyticsEngine(worker_threads=1, seed=1)
    reg = ToolRegistry(analytics, _StubWebSearchTool(), _StubGroqClient(fail=True))
    try:
        result = await reg.dispatch(
            "translate_text", json.dumps({"text": "Hello", "target_language": "French"})
        )
        assert "TOOL_ERROR" in result
        assert "translate_text" in result
    finally:
        analytics.shutdown()


async def test_dispatch_fetch_page(registry):
    result = await registry.dispatch("fetch_page", json.dumps({"url": "https://example.com"}))
    assert "example.com" in result


async def test_dispatch_fetch_page_without_configured_page_fetcher_still_works():
    # No page_fetcher passed at all -> dispatch falls back to the plain
    # fetch_page() function with an ephemeral client. Using a URL that
    # fails the SSRF check means this never makes a real network call.
    analytics = AnalyticsEngine(worker_threads=1, seed=2)
    reg = ToolRegistry(analytics, _StubWebSearchTool(), _StubGroqClient())
    try:
        result = await reg.dispatch("fetch_page", json.dumps({"url": "http://127.0.0.1/secret"}))
        assert "FETCH_ERROR" in result
    finally:
        analytics.shutdown()


async def test_dispatch_unknown_tool(registry):
    result = await registry.dispatch("nonexistent_tool", "{}")
    assert "TOOL_ERROR" in result
    assert "unknown tool" in result


async def test_dispatch_malformed_arguments(registry):
    result = await registry.dispatch("web_search", "{not valid json")
    assert "TOOL_ERROR" in result


def test_schemas_include_all_tools(registry):
    names = {schema["function"]["name"] for schema in registry.schemas()}
    assert names == {
        "web_search",
        "calculate_success_probability",
        "translate_text",
        "query_wolfram_alpha",
        "fetch_page",
    }
