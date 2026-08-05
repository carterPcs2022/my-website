"""Best-effort local audio playback for the CLI interface.

Deliberately avoids adding an audio-processing dependency (pydub,
simpleaudio, etc.) just to play back a short MP3 clip: it writes the
synthesized bytes to a temp file and shells out to whichever locally
installed player it can find. If none are available, playback is skipped
with a warning — this must never crash the CLI turn, since it runs after
the text response has already been printed.
"""
from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger("zane.voice.playback")

# Ordered by preference; each entry is (binary_name, args-builder).
_CANDIDATE_PLAYERS: List[str] = ["ffplay", "mpv", "mpg123", "afplay", "paplay", "aplay"]


def _format_to_suffix(audio_format: Optional[str]) -> str:
    """Best-effort mapping from an ElevenLabs `output_format` string (e.g.
    "mp3_44100_128", "pcm_16000", "ulaw_8000") to a file suffix a local
    player can sniff."""
    fmt = (audio_format or "").lower()
    if fmt.startswith("mp3"):
        return ".mp3"
    if fmt.startswith("pcm") or fmt.startswith("wav"):
        return ".wav"
    if fmt.startswith("ulaw") or fmt.startswith("alaw"):
        return ".wav"
    return ".mp3"


def _build_command(player: str, path: str) -> List[str]:
    if player == "ffplay":
        return ["ffplay", "-autoexit", "-nodisp", "-loglevel", "quiet", path]
    if player == "mpv":
        return ["mpv", "--no-video", "--really-quiet", path]
    if player == "mpg123":
        return ["mpg123", "-q", path]
    if player == "afplay":
        return ["afplay", path]
    if player == "paplay":
        return ["paplay", path]
    if player == "aplay":
        return ["aplay", "-q", path]
    raise ValueError(f"Unknown player: {player}")


def _find_player() -> Optional[str]:
    for candidate in _CANDIDATE_PLAYERS:
        if shutil.which(candidate):
            return candidate
    return None


async def play_audio_bytes(data: bytes, audio_format: Optional[str] = None) -> bool:
    """Writes `data` to a temp file and plays it with the first available
    local player. Returns True if playback was attempted successfully,
    False if skipped (no player found) or if it failed — either way this
    never raises, since it's a CLI convenience, not a core function."""
    if not data:
        return False

    player = _find_player()
    if player is None:
        logger.warning(
            "Voice is on but no local audio player was found (tried: %s). "
            "Skipping playback; install one of these to hear Zane speak.",
            ", ".join(_CANDIDATE_PLAYERS),
        )
        return False

    suffix = _format_to_suffix(audio_format)
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(data)
            temp_path = f.name
    except OSError as exc:
        logger.warning("Failed to write temp audio file for playback: %s", exc)
        return False

    try:
        command = _build_command(player, temp_path)
        proc = await asyncio.create_subprocess_exec(
            *command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        await proc.wait()
        return True
    except OSError as exc:
        logger.warning("Failed to play audio via %s: %s", player, exc)
        return False
    finally:
        Path(temp_path).unlink(missing_ok=True)
