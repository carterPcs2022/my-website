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


@pytest.fixture
def registry():
    analytics = AnalyticsEngine(worker_threads=1, seed=99)
    web_search = _StubWebSearchTool()
    reg = ToolRegistry(analytics, web_search)
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


async def test_dispatch_unknown_tool(registry):
    result = await registry.dispatch("nonexistent_tool", "{}")
    assert "TOOL_ERROR" in result
    assert "unknown tool" in result


async def test_dispatch_malformed_arguments(registry):
    result = await registry.dispatch("web_search", "{not valid json")
    assert "TOOL_ERROR" in result


def test_schemas_include_both_tools(registry):
    names = {schema["function"]["name"] for schema in registry.schemas()}
    assert names == {"web_search", "calculate_success_probability"}
