# Zane — Digital Mind

A hybrid C++/Python cognitive architecture for **Zane**: ultra-fast LLM
responses via **Groq**, a live web-search pipeline for "infinite
information," and a thread-safe C++ engine for his analytical/probability
readouts — all wrapped behind one interface so the exact same engine runs
as a CLI or a REST API.

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
  analytics_bridge.py            zane_cpp wrapper w/ pure-Python fallback
  memory/
    rolling.py                    Short-term in-context rolling window
    store.py                      SQLite durable storage (messages, summaries)
    embeddings.py                  Local sentence-transformers embedding backend
    vector_index.py                FAISS semantic-search index
    summarizer.py                  LLM-driven summarization of aged-out history
    persistent.py                  PersistentMemory: ties the above together
    memory_defragmenter.py         RAG conflict detection + resolution (opt-in)
  tools/
    web_search.py                Tavily/DuckDuckGo search tool
    translate.py                  LLM-backed translation tool
    registry.py                   LLM function-calling schema + dispatch
  voice/
    tts.py                        Async ElevenLabs client + fallback helper
    switch.py                     Per-session voice on/off toggle
    playback.py                    CLI local audio playback (best-effort)
  core.py                        ZaneMind: the orchestrator
  control.py                     ZaneSensorInput / ZaneControlOutput seam + vehicle telemetry connector
  thermal_monitor.py             Ice Protocol: real host thermal safety governor
  falcon_worker.py                Falcon Scout: system health daemon -> self-recall memory
  interfaces/
    base.py                      ZaneInterface abstract base class
    cli.py                        CLI adapter
    api.py                        FastAPI adapter
  __main__.py                    `python -m zane --mode {cli,api}`

Dockerfile / render.yaml     Container deployment (see "Deploying to Render")
requirements-core.txt        Light deps; requirements-memory.txt adds the heavy RAG stack
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

### Memory: rolling window + persistent retrieval-augmented recall

Every `ZaneMind` session keeps two layers of memory, both owned by
`zane.memory.persistent.PersistentMemory`:

- **Rolling window** (`zane/memory/rolling.py`, unchanged from the
  original design): the last N messages, sent verbatim as chat history on
  every turn.
- **Persistent, retrieval-augmented memory**: every message is written to
  SQLite (`zane/memory/store.py`, `messages` + `summaries` tables)
  immediately on send/receive, so history survives a process restart —
  a new session backed by the same database file rebuilds its rolling
  window from the most recent stored rows on startup. Each message is
  also embedded locally via `sentence-transformers` (all-MiniLM-L6-v2, no
  external API — see `zane/memory/embeddings.py`) and indexed in a FAISS
  vector index (`zane/memory/vector_index.py`) keyed by the same SQLite
  row ID. On every new user turn, the query is embedded and the top-k most
  semantically relevant *older* messages (excluding anything already in
  the rolling window) are retrieved and injected into the system prompt as
  a distinct "RELEVANT PAST CONTEXT" block — separate from both the
  rolling window and the "EARLIER CONVERSATION SUMMARY" block described
  next (see `zane/personality.py`'s `build_system_prompt`).
- **Background summarization + pruning**: once more raw messages sit
  beyond a configurable retention window than a configurable threshold
  (`ZANE_MEMORY_RETENTION_WINDOW` / `ZANE_MEMORY_SUMMARIZE_AFTER_N`), the
  oldest chunk is condensed into a summary via an LLM call
  (`zane/memory/summarizer.py`) and stored in `summaries`; only then are
  the raw rows (and their vectors) pruned. If summarization fails for any
  reason (Groq drop, empty completion), the raw messages are **retained**
  and retried on the next turn — history is never silently destroyed.

All of the above degrades gracefully rather than crashing the
conversation: a failed embedding call, a cold-start empty history, or
database lock contention (retried with backoff in `store.py`) all fall
back to "no retrieved context this turn" instead of raising into
`ZaneMind.respond()`.

### Character flavor vs. real capabilities

Zane's system prompt (`zane/personality.py`'s `ZANE_FLAVOR_GUIDANCE`,
included unconditionally on every turn) draws an explicit line between
in-character dialogue color and real system capabilities, so the model
never lets flavor read as a technical claim to the user:

- **Digital Mind backstory** — Zane may reference surviving as a
  consciousness inside a computer when relevant to identity questions;
  this is character lore, not a technical claim about the software.
- **Advanced Scanning** — Zane may narrate "scanning" flavor (e.g.
  detecting stress indicators) as personality color. No real biometric
  sensing or lie detection exists in this system, and this flavor must
  never be used to frame an actual judgment about whether the user is
  being truthful.
- **Fast Calculations** — the percentages from `calculate_success_probability`
  (backed by the real `ZaneAnalytics` C++ engine) are dialogue flavor —
  pseudo-randomized and contextually weighted for narrative color, not
  statistically validated predictions — and must never be represented to
  the user as genuine forecasting outside the roleplay frame.

The `translate_text` tool (`zane/tools/translate.py`) is the one exception
called out explicitly in the prompt: it's a real LLM-backed translation,
not flavor, so no such caveat applies to it.

### Voice: optional ElevenLabs text-to-speech

Every `ZaneMind` session has a `voice` toggle (`zane/voice/switch.py`,
default **OFF**) independent of the `humor` toggle. When on — and only
when on — `respond()` runs one extra step after generating text:
`zane.voice.tts.synthesize_with_fallback` calls the shared
`AsyncElevenLabsClient` (`zane/voice/tts.py`, same exponential-backoff
retry pattern as `AsyncGroqClient`) and returns `(audio_bytes,
audio_format)`. Every failure mode — voice off, no `ELEVENLABS_API_KEY`/
`ZANE_VOICE_ID` configured, the `elevenlabs` package not installed, or a
synthesis call failing even after retries — degrades to `(None, None)`
without raising, so a conversational turn can never be broken by TTS.
`TurnResult.audio`/`audio_format` are `None` in every one of those cases,
making text-only output byte-for-byte identical whether voice is
unavailable or simply switched off.

The voice ID is always read from `ZANE_VOICE_ID` (never hardcoded in
source) so it can be swapped independently per deployment.

- **CLI**: after printing the text reply, `zane/voice/playback.py` writes
  the audio to a temp file and shells out to the first local player it
  finds (`ffplay`, `mpv`, `mpg123`, `afplay`, `paplay`, `aplay`) —
  deliberately not a new Python audio dependency for a short MP3 clip. No
  player found just skips playback with a warning; it never crashes the
  CLI turn.
- **API**: `POST /chat` includes `audio_base64` + `audio_format` in the
  *same* JSON response as `reply`, rather than switching response types or
  adding a separate streaming route — see the design-choice note at the
  top of `zane/interfaces/api.py` for the reasoning and the tradeoff
  (~33% base64 size overhead, accepted for atomicity and a single stable
  `/chat` schema). Toggle voice per session via `POST /command`
  (`{"command": "voice"}`), same as `humor`.

### Vehicle telemetry / driving-sim connector: `zane/control.py`

`ZaneSensorInput`/`ZaneControlOutput` started as unimplemented, generic
scaffolding and are now `Generic` ABCs with a first concrete consumer: a
Gymnasium/OpenAI-Gym-style telemetry bridge (`TelemetryLoop`,
`VehicleTelemetryFrame`, `VehicleControlOutput`). Honest about what's real
here: `ZaneAnalytics` has no pathfinding methods, and none were invented —
the obstacle-avoidance vector math (`compute_avoidance_vector`) is
deterministic, dependency-free pure Python (an artificial-potential-field,
a standard robotics technique). What genuinely routes through the
C++-bridged `zane_cpp` engine is the maneuver's confidence/risk score, via
`AnalyticsEngine.calculate_success_probability` with a documented field
mapping (`build_probability_factors`) — the same real computation Zane
already uses for mission-style risk assessment. Any obstacle inside a
configurable safety envelope, or too many consecutive tick failures,
triggers `emergency_stop()` immediately, bypassing the normal decision
math. `InMemoryVehicleSensor`/`InMemoryVehicleControlSink` are reference
implementations a real sim integration replaces; nothing here is wired
into `ZaneMind` — it's a standalone loop a driving-sim host would run.

### Ice Protocol: `zane/thermal_monitor.py`

A real host thermal safety governor — a background `threading.Thread`
polls `psutil.sensors_temperatures()`, and crossing the configured
threshold (default 75°C, with a hysteresis band to avoid flapping) sets
`ice_protocol_active`, which `ZaneMind.respond()` checks *before* touching
memory or Groq at all, short-circuiting to a local rules-based responder
when active. **Read the caveat in the module docstring before enabling
this**: `psutil` only reports real sensor data on Linux hosts with exposed
`hwmon` sensors and never reports GPU temperature on its own — in
virtually all containers and VMs, including this project's own
Render/Docker deployment target, no sensors are visible to the guest at
all, so the protocol simply never trips. This module never fabricates a
reading to compensate; it's built for bare-metal/local/robotics
deployments where `psutil` genuinely has sensor access, which is why
`ZANE_ICE_PROTOCOL_ENABLED` defaults to `false`.

### Falcon Scout: `zane/falcon_worker.py`

A background async daemon (default every 60s) that profiles the running
process — API request latency (via FastAPI middleware feeding
`LatencyRecorder`), SQLite responsiveness (`SQLiteMessageStore.ping()`),
and any `ERROR`+ log records captured since the last cycle
(`InMemoryLogCapture`) — and, only when something is actually wrong,
writes a technical summary directly into the shared RAG memory pipeline
tagged `role="falcon_scout_telemetry"`. Deliberately does **not** call the
LLM to write the summary (plain deterministic string formatting): Falcon
Scout needs to keep working even if Groq itself is what's having
problems. Because semantic retrieval already searches across every
session rather than just the current one (see "Memory" above), these
entries surface for organic recall — "how's your operational stability
been?" — with no retrieval-path changes needed. Pure observability with
no effect on response behavior, so `ZANE_FALCON_SCOUT_ENABLED` defaults
to `true`.

### Memory Defragmenter: `zane/memory/memory_defragmenter.py`

Detects and resolves factual conflicts between stored memories — e.g. two
contradictory records of "the vault code." Two-stage by design: cosine
similarity is used only as a *candidate filter* (entries about the same
subject — it measures topical relatedness, not truth-value agreement),
and the actual contradiction judgment is delegated to an LLM call
mirroring `zane/memory/summarizer.py`'s existing pattern. **Nothing is
ever pruned without an explicit LLM-confirmed conflict** — an ambiguous,
unparsable, or unavailable judgment always means "do nothing this cycle."
When a conflict is confirmed, an exponential half-life recency weight
(`_recency_weight`) picks which entry is authoritative, the older one is
pruned from both SQLite and the FAISS index, and a new consolidated-truth
entry is written (`role="memory_defrag_consolidated"`). Because this
autonomously deletes data (however conservatively gated),
`ZANE_MEMORY_DEFRAG_ENABLED` defaults to `false` — an intentionally higher
bar to opt into than Falcon Scout's pure observability.

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
```

CLI commands: `/humor` toggles the awkward dad-joke/literal-humor switch,
`/voice` toggles spoken responses (see "Voice" above), `/reset` clears
conversation memory, `/help` lists commands. The API exposes the same via
`POST /command`.

## Deploying to Render

Only the API surface makes sense as a hosted deployment — the CLI needs
an interactive terminal, which a Render web service doesn't give you.
`Dockerfile` and `render.yaml` are set up for exactly this.

1. In the Render dashboard: **New +** -> **Blueprint**, point it at this
   repo/branch. Render reads `render.yaml` and creates a Docker-based web
   service running `zane.interfaces.api:app`.
2. Render will prompt for every env var marked `sync: false` in
   `render.yaml` (`GROQ_API_KEY` required; `TAVILY_API_KEY`,
   `ELEVENLABS_API_KEY`, `ZANE_VOICE_ID` optional) — set them there, in
   Render's dashboard. Never in this repo, never pasted in chat.
3. Deploy. `/health` is wired as the health check path.

**If the build is too slow/heavy for your Render plan**: the C++
extension compiles fine on Render's standard build machines, but
`sentence-transformers` pulls in `torch`, which is the single heaviest
part of the image. Set the Docker build arg `INSTALL_MEMORY_EXTRAS=false`
(Render dashboard -> service -> Settings -> Build, or add it under
`dockerCommand`/build args in `render.yaml`) to skip
`requirements-memory.txt` entirely — Zane still runs fully; only semantic
memory retrieval and background summarization go quiet (see "Memory"
above; `PersistentMemory` was built to degrade gracefully for exactly this
kind of case, not just for missing network access).

**Persistent memory across restarts**: Render web services have an
*ephemeral* filesystem by default — `ZANE_MEMORY_DB_PATH` and
`ZANE_MEMORY_VECTOR_INDEX_PATH` will be wiped on every deploy/restart
unless you attach a Render persistent Disk (a paid feature) and point
those two env vars at a path under its mount. Without one, Zane still
works correctly within a single running instance's lifetime — it just
starts fresh each time the container restarts.

## Testing

```bash
pytest
```

Tests exercise the personality prompt builder, the analytics engine (using
whichever backend — native or pure-Python fallback — is available in the
current environment), rolling memory trimming, tool dispatch, the
persistent memory subsystem (SQLite write durability, FAISS retrieval
ranking, summarization triggering, and pruning), voice synthesis (success
path, retry-then-succeed, non-retryable and retry-exhausted failure, and
the toggle-on/off fallback behavior — all against an injected fake SDK
client, no real ElevenLabs calls), the vehicle telemetry connector
(parsing/validation, avoidance-vector math, emergency-stop triggering on
close obstacles and on repeated tick failures), the Ice Protocol
(threshold/hysteresis transitions via a monkeypatched sensor reader, the
low-power responder), Falcon Scout (latency/DB/exception detection against
fake stores, graceful degradation on embedding failure), and the memory
defragmenter (confirmed-conflict pruning, and — just as importantly — that
a no-conflict, unparsable, or Groq-unavailable judgment never deletes
anything). The real sentence-transformers model requires downloading
weights on first use; `tests/test_embeddings.py` skips its real-model
assertions (rather than failing) in offline environments while still
testing failure handling.

## Extending to a new surface

Every deployment surface subclasses `zane.interfaces.base.ZaneInterface`
and reuses `handle_message` / `handle_command` against a shared `ZaneMind`.
To add a new host (a desktop app, a web chat widget):

```python
from zane.core import ZaneMind
from zane.interfaces.base import ZaneInterface

class MyInterface(ZaneInterface):
    async def start(self): ...   # wire up your transport
    async def stop(self): await self.mind.aclose()
```
