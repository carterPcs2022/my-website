"""Universal entry point for every Zane deployment surface.

    python -m zane --mode cli
    python -m zane --mode api      # then: uvicorn zane.interfaces.api:app
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from zane.config import settings
from zane.core import ZaneMind


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


async def _main_async(mode: str) -> int:
    if mode == "cli":
        from zane.interfaces.cli import run_cli

        mind = ZaneMind(settings_override=settings)
        await run_cli(mind)
        return 0

    if mode == "api":
        print(
            "The API surface runs under an ASGI server, not `python -m zane`.\n"
            "Start it with:\n"
            "    uvicorn zane.interfaces.api:app --host 0.0.0.0 --port 8000",
            file=sys.stderr,
        )
        return 1

    print(f"Unknown mode: {mode!r}", file=sys.stderr)
    return 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Zane's digital mind.")
    parser.add_argument(
        "--mode", choices=["cli", "api"], default="cli",
        help="Which deployment surface to launch (default: cli).",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()

    _configure_logging(args.verbose)

    try:
        exit_code = asyncio.run(_main_async(args.mode))
    except KeyboardInterrupt:
        exit_code = 130
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
