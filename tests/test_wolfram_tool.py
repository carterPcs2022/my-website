"""Tests for zane.tools.wolfram_tool. WolframAlphaClient's HTTP layer is
tested against a real httpx transport (httpx.MockTransport) rather than a
hand-rolled fake — this exercises the actual httpx request/response
machinery, just without a real network call or a real Wolfram API key.
"""
import httpx
import pytest

from zane.tools.wolfram_tool import (
    SYSTEM_LOG_FALLBACK_NOTICE,
    WOLFRAM_TOOL_SCHEMA,
    WolframAlphaClient,
    WolframQueryError,
    query_wolfram_alpha,
    safe_local_eval,
)


def _client_with_transport(handler, app_id="test-app-id"):
    client = WolframAlphaClient(app_id=app_id)
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return client


# --- safe_local_eval: the local fallback ---


def test_safe_local_eval_basic_arithmetic():
    assert safe_local_eval("2 + 2 * 3") == 8
    assert safe_local_eval("(200 * 9.81) / 0.0005") == pytest.approx(3924000.0)
    assert safe_local_eval("2 ** 10") == 1024


def test_safe_local_eval_supports_whitelisted_functions_and_constants():
    assert safe_local_eval("sqrt(16)") == 4.0
    assert safe_local_eval("round(pi, 2)") == 3.14


def test_safe_local_eval_rejects_natural_language():
    assert safe_local_eval("torque required to shear a bolt") is None


def test_safe_local_eval_rejects_code_injection_attempts():
    assert safe_local_eval('__import__("os").system("echo pwned")') is None
    assert safe_local_eval("open('/etc/passwd').read()") is None
    assert safe_local_eval("[x for x in range(10)]") is None


def test_safe_local_eval_rejects_disallowed_names():
    assert safe_local_eval("os.getcwd()") is None


def test_safe_local_eval_handles_division_by_zero_gracefully():
    assert safe_local_eval("1 / 0") is None


# --- WolframAlphaClient: real httpx request/response handling ---


async def test_client_returns_stripped_text_on_success():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["input"] == "mass of the sun"
        assert request.url.params["appid"] == "test-app-id"
        return httpx.Response(200, text="  1.989 x 10^30 kg  \n")

    client = _client_with_transport(handler)
    result = await client.query("mass of the sun")
    assert result == "1.989 x 10^30 kg"
    await client.close()


async def test_client_raises_on_non_200_status():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(501, text="Wolfram could not understand the query.")

    client = _client_with_transport(handler)
    with pytest.raises(WolframQueryError):
        await client.query("gibberish query")
    await client.close()


async def test_client_raises_on_empty_response():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="")

    client = _client_with_transport(handler)
    with pytest.raises(WolframQueryError):
        await client.query("anything")
    await client.close()


async def test_client_raises_on_network_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("simulated connection failure", request=request)

    client = _client_with_transport(handler)
    with pytest.raises(WolframQueryError):
        await client.query("anything")
    await client.close()


async def test_client_raises_without_app_id():
    client = WolframAlphaClient(app_id=None)
    with pytest.raises(WolframQueryError):
        await client.query("anything")
    await client.close()


# --- query_wolfram_alpha: the full tool entry point + fallback chain ---


async def test_query_wolfram_alpha_uses_real_api_result_on_success():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="42")

    client = _client_with_transport(handler)
    result = await query_wolfram_alpha("the answer to everything", client)
    assert result == "42"
    await client.close()


async def test_query_wolfram_alpha_falls_back_on_api_failure():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="internal error")

    client = _client_with_transport(handler)
    result = await query_wolfram_alpha("2 + 2", client)
    assert SYSTEM_LOG_FALLBACK_NOTICE in result
    assert "4" in result
    await client.close()


async def test_query_wolfram_alpha_falls_back_with_no_client():
    result = await query_wolfram_alpha("3 * 7", None)
    assert SYSTEM_LOG_FALLBACK_NOTICE in result
    assert "21" in result


async def test_query_wolfram_alpha_falls_back_without_app_id():
    client = WolframAlphaClient(app_id=None)
    result = await query_wolfram_alpha("5 + 5", client)
    assert SYSTEM_LOG_FALLBACK_NOTICE in result
    assert "10" in result
    await client.close()


async def test_query_wolfram_alpha_unresolvable_nl_query_is_honest_not_fabricated():
    result = await query_wolfram_alpha("what is the meaning of life", None)
    assert SYSTEM_LOG_FALLBACK_NOTICE in result
    assert "unable to verify an exact numeric answer" in result


def test_tool_schema_has_required_description_phrase():
    description = WOLFRAM_TOOL_SCHEMA["function"]["description"]
    assert (
        "Use this tool for exact structural calculations, material stresses, "
        "torque requirements, or verifying mathematical realities."
    ) in description


def test_tool_schema_name_and_required_param():
    assert WOLFRAM_TOOL_SCHEMA["function"]["name"] == "query_wolfram_alpha"
    assert WOLFRAM_TOOL_SCHEMA["function"]["parameters"]["required"] == ["query"]
