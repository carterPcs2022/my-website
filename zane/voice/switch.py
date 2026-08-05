"""Per-session voice (text-to-speech) toggle. Mirrors HumorSwitch in
zane/personality.py — simple stateful boolean so interfaces (CLI/API) can
flip it without threading a bool through every call site. Defaults OFF:
text-only behavior must be identical whether or not TTS is even
configured until a session explicitly opts in.
"""
from __future__ import annotations


class VoiceSwitch:
    def __init__(self, enabled: bool = False) -> None:
        self._enabled = enabled

    @property
    def enabled(self) -> bool:
        return self._enabled

    def on(self) -> None:
        self._enabled = True

    def off(self) -> None:
        self._enabled = False

    def toggle(self) -> bool:
        self._enabled = not self._enabled
        return self._enabled
