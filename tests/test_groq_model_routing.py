from zane.groq_client import AsyncGroqClient


def make_client(monkeypatch):
    monkeypatch.setenv("GROQ_FAST_MODEL", "openai/gpt-oss-20b")
    monkeypatch.setenv("GROQ_DEEP_MODEL", "openai/gpt-oss-120b")
    return AsyncGroqClient(api_key="test-key", max_retries=0)


def test_short_turn_uses_fast_model(monkeypatch):
    client = make_client(monkeypatch)
    model, effort = client._select_model(
        [{"role": "user", "content": "Hello Zane!"}]
    )
    assert model == "openai/gpt-oss-20b"
    assert effort == "low"


def test_complex_turn_uses_deep_model(monkeypatch):
    client = make_client(monkeypatch)
    model, effort = client._select_model(
        [{"role": "user", "content": "Analyze and debug this architecture."}]
    )
    assert model == "openai/gpt-oss-120b"
    assert effort == "medium"


def test_retired_default_is_mapped_to_fast(monkeypatch):
    monkeypatch.setenv("GROQ_FAST_MODEL", "openai/gpt-oss-20b")
    monkeypatch.setenv("GROQ_DEEP_MODEL", "openai/gpt-oss-120b")
    client = AsyncGroqClient(
        api_key="test-key",
        model="llama-3.3-70b-versatile",
        max_retries=0,
    )
    assert client.model == "openai/gpt-oss-20b"
