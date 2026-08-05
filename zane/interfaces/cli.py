"""CLI deployment surface for Zane's digital mind.

Run with:  python -m zane --mode cli
"""
from __future__ import annotations

import asyncio
import logging

from zane.core import ZaneMind
from zane.interfaces.base import ZaneInterface
from zane.voice.playback import play_audio_bytes

logger = logging.getLogger("zane.interfaces.cli")

_BANNER = """\
==================================================================
  ZANE — Nindroid Digital Mind (CLI Interface)
  Type your message and press Enter. Commands: /humor /voice /reset /help /exit
==================================================================\
"""


class CLIInterface(ZaneInterface):
    def __init__(self, mind: ZaneMind, prompt_name: str = "You") -> None:
        super().__init__(mind)
        self.prompt_name = prompt_name
        self._running = False

    async def start(self) -> None:
        print(_BANNER)
        self._running = True
        loop = asyncio.get_event_loop()

        while self._running:
            try:
                line = await loop.run_in_executor(None, input, f"{self.prompt_name}: ")
            except (EOFError, KeyboardInterrupt):
                print()
                break

            line = line.strip()
            if not line:
                continue

            if line.startswith("/"):
                if line[1:].strip().lower() in ("exit", "quit"):
                    break
                reply = await self.handle_command(line[1:].strip())
                print(f"Zane: {reply if reply is not None else 'Unrecognized command. Try /help.'}")
                continue

            try:
                result = await self.handle_message(line, display_name=self.prompt_name)
            except Exception:
                logger.exception("Unhandled error while producing a response")
                print("Zane: My apologies — an internal fault occurred while processing that.")
                continue

            print(f"Zane: {result.text}")
            if result.tool_calls_made:
                logger.debug(
                    "[diagnostics] tools used: %s | %.1fms",
                    ", ".join(result.tool_calls_made),
                    result.elapsed_ms,
                )

            if result.audio:
                await play_audio_bytes(result.audio, result.audio_format)

        await self.stop()

    async def stop(self) -> None:
        self._running = False
        await self.mind.aclose()
        print("Zane's digital mind has been powered down gracefully.")


async def run_cli(mind: ZaneMind) -> None:
    interface = CLIInterface(mind)
    await interface.start()
