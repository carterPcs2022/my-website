# Zane — Digital Mind

A hybrid C++/Python cognitive architecture for **Zane Julien**, the Nindroid
Master of Ice: ultra-fast LLM responses via **Groq**, a live web-search
pipeline for "infinite information," and a thread-safe C++ engine for his
analytical/probability readouts — all wrapped behind one interface so the
exact same engine runs as a CLI, a Discord bot, or a REST API.

## Architecture

```
cpp/                      Low-level processing (C++)
  include/zane/analytics.hpp   ZaneAnalytics: thread-safe probability engine
  src/analytics.cpp            Implementation (worker pool, mutex-guarded state)
  bindings/bindings.cpp        pybind11 bridge -> `zane_cpp` Python extension

zane/                     High-level AI layer (Python)
  config.py                    Environment-driven settings
  personality.py                System-prompt injector + Humor Switch
  groq_client.py                Async Groq client w/ retry & backoff
  memory.py                     Rolling conversational memory
  analytics_bridge.py            zane_cpp wrapper w/ pure-Python fallback
  tools/
    web_search.py                Tavily/DuckDuckGo search tool
    registry.py                   LLM function-calling schema + dispatch
  core.py                        ZaneMind: the orchestrator
  interfaces/
    base.py                      ZaneInterface abstract base class
    cli.py                        CLI adapter
    api.py                        FastAPI adapter
    discord_bot.py                discord.py adapter
  __main__.py                    `python -m zane --mode {cli,api,discord}`
```

### Why C++ + Python?

Zane's *voice* (the Groq-backed conversational layer, the search pipeline,
persona/state management) is naturally expressed in async Python. His
*analytical mode* — the exact, specific-sounding success percentages he
quotes — is modeled as a small, genuinely multi-threaded numerical engine in
C++, bridged into Python with pybind11 (`cpp/bindings/bindings.cpp`). All
shared state in the C++ engine (the job queue, results map, RNG, and Zane's
internal state map) is guarded by its own `std::mutex`/`lock_guard`, and
blocking calls release the Python GIL so asyncio keeps making progress while
the worker pool computes.

If the extension hasn't been built for the current platform,
`zane/analytics_bridge.py` transparently falls back to a pure-Python
reimplementation with an identical interface and identical math — the rest
of the system never needs to know which backend is active.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Build the C++ extension (optional but recommended):
pip install -e .

cp .env.example .env
# then edit .env and set GROQ_API_KEY at minimum
```

## Running

```bash
# CLI
python -m zane --mode cli

# REST API (FastAPI + uvicorn)
uvicorn zane.interfaces.api:app --host 0.0.0.0 --port 8000

# Discord bot (requires discord.py + DISCORD_BOT_TOKEN)
python -m zane --mode discord
```

CLI/Discord commands: `/humor` (or `!zane humor`) toggles the awkward
dad-joke/literal-humor switch, `/reset` clears conversation memory, `/help`
lists commands. The API exposes the same via `POST /command`.

## Testing

```bash
pytest
```

Tests exercise the personality prompt builder, the analytics engine (using
whichever backend — native or pure-Python fallback — is available in the
current environment), conversational memory trimming, and tool dispatch.

## Extending to a new surface

Every deployment surface subclasses `zane.interfaces.base.ZaneInterface`
and reuses `handle_message` / `handle_command` against a shared `ZaneMind`.
To add a new host (a desktop app, a Slack bot, a web chat widget):

```python
from zane.core import ZaneMind
from zane.interfaces.base import ZaneInterface

class MyInterface(ZaneInterface):
    async def start(self): ...   # wire up your transport
    async def stop(self): await self.mind.aclose()
```
