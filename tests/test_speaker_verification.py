"""Tests for zane/voice/speaker_verification.py against a fake
Resemblyzer (no real torch/resemblyzer install required, same spirit as
test_tts.py's fake ElevenLabs SDK). The fake maps the first byte of the
uploaded audio to a deterministic 8-dim "direction" vector, standing in
for a real speaker embedding: identical first bytes -> identical
embeddings (same speaker), different first bytes -> effectively
uncorrelated embeddings (different speakers)."""
import numpy as np
import pytest

from zane.voice import speaker_verification as sv


class _FakeEncoder:
    def embed_utterance(self, wav):
        return np.asarray(wav[:8], dtype=np.float32)


def _fake_preprocess_wav(file_like):
    data = file_like.read()
    marker = data[0] if data else 0
    direction = np.random.RandomState(marker).rand(8).astype(np.float32)
    # Longer input -> longer "decoded" audio, so a real too-short clip
    # still fails sv._MIN_AUDIO_SAMPLES the same way it would for real.
    padding = np.ones(max(len(data) * 200, 0), dtype=np.float32)
    return np.concatenate([direction, padding])


@pytest.fixture(autouse=True)
def fake_resemblyzer(monkeypatch):
    monkeypatch.setattr(sv, "_RESEMBLYZER_AVAILABLE", True)
    monkeypatch.setattr(sv, "preprocess_wav", _fake_preprocess_wav)
    monkeypatch.setattr(sv, "VoiceEncoder", lambda device=None: _FakeEncoder())


AUDIO_A = b"A" * 100
AUDIO_A_ALT = b"A" * 150  # same speaker (same first byte), different "utterance"
AUDIO_B = b"B" * 100
AUDIO_TOO_SHORT = b"x"


def make_verifier(tmp_path, threshold=0.99, max_samples=10):
    return sv.SpeakerVerifier(
        profiles_path=str(tmp_path / "profiles.json"),
        threshold=threshold,
        max_samples_per_speaker=max_samples,
    )


def test_available_reflects_resemblyzer_import(tmp_path):
    assert make_verifier(tmp_path).available is True


def test_unavailable_raises_before_touching_encoder(tmp_path, monkeypatch):
    monkeypatch.setattr(sv, "_RESEMBLYZER_AVAILABLE", False)
    verifier = make_verifier(tmp_path)
    with pytest.raises(sv.SpeakerVerificationUnavailableError):
        verifier.embed_audio(AUDIO_A)


def test_audio_too_short_raises(tmp_path):
    with pytest.raises(sv.SpeakerVerificationAudioError):
        make_verifier(tmp_path).embed_audio(AUDIO_TOO_SHORT)


def test_enroll_creates_profile_and_reports_sample_count(tmp_path):
    result = make_verifier(tmp_path).enroll("carter", AUDIO_A)
    assert result == {"speaker_id": "carter", "samples": 1}


def test_enroll_caps_at_max_samples(tmp_path):
    verifier = make_verifier(tmp_path, max_samples=2)
    verifier.enroll("carter", AUDIO_A)
    verifier.enroll("carter", AUDIO_A)
    result = verifier.enroll("carter", AUDIO_A)
    assert result["samples"] == 2


def test_enroll_persists_to_disk_and_reloads(tmp_path):
    profiles_path = tmp_path / "profiles.json"
    sv.SpeakerVerifier(profiles_path=str(profiles_path), threshold=0.99).enroll("carter", AUDIO_A)
    assert profiles_path.exists()

    reloaded = sv.SpeakerVerifier(profiles_path=str(profiles_path), threshold=0.99)
    assert reloaded.list_speakers() == [{"speaker_id": "carter", "samples": 1}]


def test_verify_matches_enrolled_speaker(tmp_path):
    verifier = make_verifier(tmp_path)
    verifier.enroll("carter", AUDIO_A)
    result = verifier.verify(AUDIO_A_ALT)
    assert result["verified"] is True
    assert result["speaker_id"] == "carter"
    assert result["similarity"] == pytest.approx(1.0, abs=1e-4)


def test_verify_rejects_different_speaker(tmp_path):
    verifier = make_verifier(tmp_path)
    verifier.enroll("carter", AUDIO_A)
    result = verifier.verify(AUDIO_B)
    assert result["verified"] is False
    assert result["speaker_id"] is None


def test_verify_against_specific_speaker_id(tmp_path):
    verifier = make_verifier(tmp_path)
    verifier.enroll("carter", AUDIO_A)
    verifier.enroll("someone_else", AUDIO_B)
    result = verifier.verify(AUDIO_A_ALT, speaker_id="carter")
    assert result["verified"] is True
    assert result["speaker_id"] == "carter"


def test_verify_unknown_speaker_id_returns_error(tmp_path):
    result = make_verifier(tmp_path).verify(AUDIO_A, speaker_id="nobody")
    assert result["verified"] is False
    assert "error" in result


def test_verify_no_enrolled_speakers_returns_error(tmp_path):
    result = make_verifier(tmp_path).verify(AUDIO_A)
    assert result["verified"] is False
    assert "error" in result


def test_delete_speaker(tmp_path):
    verifier = make_verifier(tmp_path)
    verifier.enroll("carter", AUDIO_A)
    assert verifier.delete_speaker("carter") is True
    assert verifier.list_speakers() == []
    assert verifier.delete_speaker("carter") is False


def test_list_speakers(tmp_path):
    verifier = make_verifier(tmp_path)
    verifier.enroll("carter", AUDIO_A)
    verifier.enroll("carter", AUDIO_A)
    verifier.enroll("dana", AUDIO_B)
    speakers = {s["speaker_id"]: s["samples"] for s in verifier.list_speakers()}
    assert speakers == {"carter": 2, "dana": 1}
