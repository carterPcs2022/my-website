"""ZaneInterface: the abstract contract every deployment surface implements.

The same `ZaneMind` engine (Groq + memory + tools + the C++ analytics
bridge) powers every surface; only how messages arrive and get delivered
changes. Concrete adapters in this package (CLI, API) are thin — new hosts
(a desktop app, a web chat widget, a driving-sim control panel) just need
to subclass this and implement `start`/`stop`, reusing `handle_message` and
`handle_command` as-is.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from zane.core import ZaneMind, TurnResult

# Slash/bang commands shared across every interface, so "/humor" in the CLI
# behaves identically to a REST call to /command.
_COMMAND_HELP = (
    "Available commands:\n"
    "  humor   — toggle Zane's dad-joke / literal-humor subroutine\n"
    "  reset   — clear the current conversation memory\n"
    "  help    — show this message"
)


class ZaneInterface(ABC):
    """Base class for all Zane deployment surfaces."""

    def __init__(self, mind: ZaneMind) -> None:
        self.mind = mind

    @abstractmethod
    async def start(self) -> None:
        """Begins accepting input on this surface (blocking loop, bot
        gateway connection, or ASGI server, depending on subclass)."""

    @abstractmethod
    async def stop(self) -> None:
        """Gracefully shuts the surface down, releasing the shared
        ZaneMind's resources (Groq client, analytics engine threads)."""

    async def handle_message(
        self,
        text: str,
        *,
        display_name: Optional[str] = None,
        mission_context: Optional[str] = None,
    ) -> TurnResult:
        """Routes a single inbound message through the shared engine."""
        return await self.mind.respond(
            text, addressed_by=display_name, mission_context=mission_context
        )

    async def handle_command(self, command: str, args: str = "") -> Optional[str]:
        """Handles the small set of interface-agnostic slash/bang commands.
        Returns None if `command` isn't recognized so subclasses can extend
        the command set with surface-specific ones."""
        command = command.strip().lower()

        if command in ("humor", "joke", "joke_mode"):
            enabled = self.mind.humor.toggle()
            state = "ENGAGED" if enabled else "DISENGAGED"
            return f"Humor subroutine {state}. I shall adjust my dialogue accordingly."

        if command == "reset":
            self.mind.reset_conversation()
            return "My conversational memory has been cleared. How may I assist you?"

        if command == "help":
            return _COMMAND_HELP

        return None
