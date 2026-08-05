import pytest

from zane.voice.switch import VoiceSwitch
from zane.voice.tts import AsyncElevenLabsClient, TextToSpeechError, synthesize_with_fallback


class _FakeApiError(Exception):
    def __init__(self, status_code: int, message: str = "fake api error"):
        super().__init__(f"{message} ({status_code})")
        self.status_code = status_code


class _FakeConvertEndpoint:
    """Stands in for the real SDK's `client.text_to_speech.convert(...)`,
    which our AsyncElevenLabsClient treats as returning an async iterator
    of byte chunks (not awaited itself)."""

    def __init__(self, chunks=(b"chunk-a", b"chunk-b"), fail_times=0, status_code=500):
        self.calls = 0
        self.chunks = chunks
        self.fail_times = fail_times
        self.status_code = status_code

    def convert(self, **kwargs):
        call_index = self.calls
        self.calls += 1
        self.last_kwargs = kwargs
        if call_index < self.fail_times:
            raise _FakeApiError(self.status_code)

        chunks = self.chunks

        async def _gen():
            for c in chunks:
                yield c

        return _gen()


class _FakeSDKClient:
    def __init__(self, **convert_kwargs):
        self.text_to_speech = _FakeConvertEndpoint(**convert_kwargs)
        self.closed = False

    async def aclose(self):
        self.closed = True


def make_client(fast_backoff=True, **convert_kwargs) -> AsyncElevenLabsClient:
    fake_sdk = _FakeSDKClient(**convert_kwargs)
    kwargs = dict(voice_id="test-voice-id", client=fake_sdk)
    if fast_backoff:
        kwargs.update(base_backoff_s=0.001, max_backoff_s=0.01)
    return AsyncElevenLabsClient(**kwargs)


async def test_synthesize_success_returns_concatenated_bytes():
    client = make_client(chunks=(b"hello ", b"world"))
    audio = await client.synthesize("Hello, world.")
    assert audio == b"hello world"
    assert client._client.text_to_speech.calls == 1


async def test_synthesize_passes_voice_and_model_through():
    client = make_client()
    await client.synthesize("Hi")
    kwargs = client._client.text_to_speech.last_kwargs
    assert kwargs["voice_id"] == "test-voice-id"
    assert kwargs["model_id"] == "eleven_multilingual_v2"


async def test_synthesize_retries_transient_failure_then_succeeds():
    client = make_client(fail_times=2, status_code=429, chunks=(b"ok",))
    audio = await client.synthesize("Retry me")
    assert audio == b"ok"
    assert client._client.text_to_speech.calls == 3  # 2 failures + 1 success


async def test_synthesize_raises_after_retries_exhausted():
    client = make_client(fail_times=99, status_code=500)
    client.max_retries = 2
    with pytest.raises(TextToSpeechError):
        await client.synthesize("Always fails")
    assert client._client.text_to_speech.calls == 3  # initial + 2 retries


async def test_synthesize_does_not_retry_non_retryable_status():
    client = make_client(fail_times=99, status_code=400)
    with pytest.raises(TextToSpeechError):
        await client.synthesize("Bad request")
    assert client._client.text_to_speech.calls == 1  # no retry attempted


async def test_synthesize_empty_text_raises_without_calling_backend():
    client = make_client()
    with pytest.raises(TextToSpeechError):
        await client.synthesize("   ")
    assert client._client.text_to_speech.calls == 0


def test_missing_voice_id_raises_value_error():
    with pytest.raises(ValueError):
        AsyncElevenLabsClient(api_key="sk_fake", voice_id="", client=_FakeSDKClient())


def test_missing_api_key_without_injected_client_raises_value_error():
    with pytest.raises(ValueError):
        AsyncElevenLabsClient(api_key=None, voice_id="some-voice-id")


async def test_close_delegates_to_injected_client():
    fake_sdk = _FakeSDKClient()
    client = AsyncElevenLabsClient(voice_id="v", client=fake_sdk)
    await client.close()
    assert fake_sdk.closed is True


# --- synthesize_with_fallback: the actual seam ZaneMind.respond() calls ---


async def test_fallback_returns_none_when_voice_disabled():
    client = make_client()
    audio, fmt = await synthesize_with_fallback(client, enabled=False, text="Hello")
    assert (audio, fmt) == (None, None)
    assert client._client.text_to_speech.calls == 0


async def test_fallback_returns_none_when_no_client_configured():
    audio, fmt = await synthesize_with_fallback(None, enabled=True, text="Hello")
    assert (audio, fmt) == (None, None)


async def test_fallback_returns_audio_and_format_on_success():
    client = make_client(chunks=(b"spoken",))
    audio, fmt = await synthesize_with_fallback(client, enabled=True, text="Hello")
    assert audio == b"spoken"
    assert fmt == client.output_format


async def test_fallback_degrades_silently_on_synthesis_failure():
    client = make_client(fail_times=99, status_code=400)
    audio, fmt = await synthesize_with_fallback(client, enabled=True, text="Hello")
    assert (audio, fmt) == (None, None)  # never raises, falls back to text-only


# --- VoiceSwitch: mirrors HumorSwitch behavior ---


def test_voice_switch_defaults_off_and_toggles():
    switch = VoiceSwitch()
    assert switch.enabled is False
    assert switch.toggle() is True
    assert switch.enabled is True
    switch.off()
    assert switch.enabled is False
    switch.on()
    assert switch.enabled is True
