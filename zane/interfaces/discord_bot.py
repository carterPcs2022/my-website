"""Discord deployment surface for Zane's digital mind.

Requires `discord.py` (`pip install discord.py`) and a DISCORD_BOT_TOKEN.
Run with:  python -m zane --mode discord

Zane responds when mentioned or when messaged in a DM, and supports
`!zane humor` / `!zane reset` / `!zane help` commands. Each Discord channel
gets its own ZaneMind (own memory + humor toggle) sharing one backend.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

from zane.config import Settings
from zane.core import SharedBackend, ZaneMind
from zane.groq_client import GroqUnavailableError
from zane.interfaces.base import ZaneInterface

logger = logging.getLogger("zane.interfaces.discord")

try:
    import discord
    from discord.ext import commands

    _DISCORD_AVAILABLE = True
except ImportError:  # pragma: no cover - discord.py is an optional dependency
    discord = None  # type: ignore
    commands = None  # type: ignore
    _DISCORD_AVAILABLE = False


COMMAND_PREFIX = "!zane "
MAX_DISCORD_MESSAGE_LEN = 2000


class DiscordInterface(ZaneInterface):
    """Adapter used per-channel; `start`/`stop` are no-ops here because the
    actual Discord gateway connection lifecycle is owned by ZaneDiscordBot
    below (one gateway connection serves every channel's ZaneMind)."""

    async def start(self) -> None:  # pragma: no cover - lifecycle owned by the bot client
        return None

    async def stop(self) -> None:
        await self.mind.aclose()


def _chunk_message(text: str, limit: int = MAX_DISCORD_MESSAGE_LEN):
    for i in range(0, len(text), limit):
        yield text[i : i + limit]


def build_discord_bot(settings_obj: Settings):
    """Constructs the discord.py Bot instance. Kept as a factory function
    (rather than module-level) so importing this module doesn't require
    discord.py or a token unless a caller actually wants the bot."""
    if not _DISCORD_AVAILABLE:
        raise RuntimeError(
            "discord.py is not installed. Run `pip install discord.py` to enable "
            "the Discord deployment surface."
        )
    if not settings_obj.discord_bot_token:
        raise RuntimeError("DISCORD_BOT_TOKEN is not set.")

    intents = discord.Intents.default()
    intents.message_content = True

    bot = commands.Bot(command_prefix=COMMAND_PREFIX, intents=intents, help_command=None)

    backend: Optional[SharedBackend] = None
    channel_minds: Dict[int, ZaneMind] = {}

    def _mind_for(channel_id: int) -> ZaneMind:
        assert backend is not None
        if channel_id not in channel_minds:
            channel_minds[channel_id] = ZaneMind(settings_override=settings_obj, shared=backend)
        return channel_minds[channel_id]

    @bot.event
    async def on_ready():
        nonlocal backend
        backend = SharedBackend.build(settings_obj)
        logger.info("Zane's digital mind is online as %s (Discord interface).", bot.user)

    @bot.event
    async def on_message(message):
        if message.author.bot:
            return

        await bot.process_commands(message)

        is_dm = isinstance(message.channel, discord.DMChannel)
        is_mentioned = bot.user in message.mentions if bot.user else False
        if not (is_dm or is_mentioned):
            return

        content = message.content
        if bot.user is not None:
            content = content.replace(f"<@{bot.user.id}>", "").replace(f"<@!{bot.user.id}>", "")
        content = content.strip()
        if not content or content.startswith(COMMAND_PREFIX):
            return

        mind = _mind_for(message.channel.id)
        async with message.channel.typing():
            try:
                result = await mind.respond(content, addressed_by=message.author.display_name)
                reply = result.text
            except GroqUnavailableError as exc:
                reply = f"My apologies — my cognitive backend is unreachable right now. ({exc})"
            except Exception:
                logger.exception("Unhandled error responding to Discord message")
                reply = "My apologies — an internal fault occurred while processing that."

        for chunk in _chunk_message(reply):
            await message.channel.send(chunk)

    @bot.command(name="zane")
    async def zane_command(ctx, subcommand: str = "help", *_args):
        mind = _mind_for(ctx.channel.id)
        interface = DiscordInterface(mind)
        reply = await interface.handle_command(subcommand)
        await ctx.send(reply if reply is not None else f"Unrecognized command: {subcommand!r}")

    return bot


async def run_discord_bot(settings_obj: Settings) -> None:
    bot = build_discord_bot(settings_obj)
    await bot.start(settings_obj.discord_bot_token)
