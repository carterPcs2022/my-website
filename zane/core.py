"""ZaneMind: the orchestrator that ties the Groq LLM layer, conversational
memory, the C++/Python analytics bridge, and the web-search tool pipeline
into a single coherent "digital mind" for Zane.

This is the one class every deployment surface (CLI, API) talks
to — see zane/interfaces/ for the thin adapters built on top of it.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass
from typing import List, Optional

from zane.analytics_bridge import AnalyticsEngine, DataStreamFrame
from zane.config import Settings, settings as default_settings
from zane.falcon_worker import FalconWorker, FalconWorkerConfig, InMemoryLogCapture, LatencyRecorder
from zane.groq_client import AsyncGroqClient, GroqUnavailableError
from zane.memory.embeddings import EmbeddingBackend
from zane.memory.memory_defragmenter import MemoryDefragmenter
from zane.memory.persistent import PersistentMemory, PersistentMemoryConfig
from zane.memory.store import SQLiteMessageStore
from zane.memory.summarizer import ConversationSummarizer
from zane.memory.vector_index import FaissVectorIndex
from zane.personality import HumorSwitch, PersonaContext, build_system_prompt
from zane.thermal_monitor import IceProtocolState, ThermalMonitor, maybe_handle_ice_protocol
from zane.tools.registry import ToolRegistry
from zane.tools.web_search import WebSearchTool
from zane.voice.switch import VoiceSwitch
from zane.voice.tts import AsyncElevenLabsClient, TextToSpeechError, synthesize_with_fallback

logger = logging.getLogger("zane.core")

_OFFLINE_FALLBACK = (
    "My apologies — I am currently unable to reach my higher cognitive functions; "
    "the connection to my analytical network appears to be interrupted. ({detail}) "
    "Please try again shortly, or let me know if there is anything I can assist "
    "with using only my local systems."
)


@dataclass
class TurnResult:
    """Everything about a single response, useful for interfaces that want
    to display more than just the final text (e.g. which tools fired)."""

    text: str
    tool_calls_made: List[str]
    elapsed_ms: float
    # Populated only when the session's voice toggle is on and synthesis
    # succeeded; None in every other case (voice off, no TTS backend
    # configured, or synthesis failed) — i.e. identical to text-only.
    audio: Optional[bytes] = None
    audio_format: Optional[str] = None


@dataclass
class SharedBackend:
    """The expensive, stateless-per-user resources: the Groq HTTP client,
    the C++/Python analytics engine (with its own thread pool), the web
    search tool, and the persistent-memory backing (SQLite store, local
    embedding model, FAISS index, and LLM summarizer). One of these is
    built per process and shared across every concurrent ZaneMind session
    in multi-user hosts; per-session state (rolling window, humor toggle)
    still lives on each ZaneMind."""

    analytics: AnalyticsEngine
    web_search: WebSearchTool
    tools: ToolRegistry
    groq: AsyncGroqClient
    memory_store: SQLiteMessageStore
    embeddings: EmbeddingBackend
    vector_index: FaissVectorIndex
    summarizer: ConversationSummarizer
    # None whenever ELEVENLABS_API_KEY/ZANE_VOICE_ID aren't configured, or
    # the `elevenlabs` package isn't installed — voice then silently no-ops
    # even if a session's VoiceSwitch is on (see synthesize_with_fallback).
    tts: Optional[AsyncElevenLabsClient]
    # Ice Protocol: state is always present (cheap, just a flag+lock) so
    # ZaneMind.respond() can check it unconditionally; the background
    # thread that could ever set it active only runs if enabled.
    ice_protocol_state: IceProtocolState
    ice_protocol_threshold_c: float
    thermal_monitor: Optional[ThermalMonitor]
    # Falcon Scout: latency_recorder is always present so API middleware
    # always has somewhere to record into, even if the worker's disabled.
    latency_recorder: LatencyRecorder
    log_capture: InMemoryLogCapture
    falcon_worker: Optional[FalconWorker]
    falcon_task: Optional["asyncio.Task"]
    memory_defragmenter: Optional[MemoryDefragmenter]
    memory_defrag_task: Optional["asyncio.Task"]

    @classmethod
    def build(cls, settings_obj: Settings) -> "SharedBackend":
        settings_obj.validate_for_groq()
        analytics = AnalyticsEngine(
            worker_threads=settings_obj.analytics_worker_threads,
            seed=settings_obj.analytics_seed,
        )
        web_search = WebSearchTool(
            tavily_api_key=settings_obj.tavily_api_key,
            backend=settings_obj.search_backend,
            default_max_results=settings_obj.search_max_results,
        )
        groq = AsyncGroqClient(
            api_key=settings_obj.groq_api_key,
            model=settings_obj.groq_model,
            temperature=settings_obj.groq_temperature,
            max_tokens=settings_obj.groq_max_tokens,
            timeout_s=settings_obj.groq_timeout_s,
            max_retries=settings_obj.groq_max_retries,
        )
        tools = ToolRegistry(analytics, web_search, groq)
        memory_store = SQLiteMessageStore(settings_obj.memory_db_path)
        embeddings = EmbeddingBackend(settings_obj.memory_embedding_model)
        vector_index = FaissVectorIndex(settings_obj.memory_vector_index_path)
        summarizer = ConversationSummarizer(groq)

        tts: Optional[AsyncElevenLabsClient] = None
        if settings_obj.elevenlabs_api_key and settings_obj.elevenlabs_voice_id:
            try:
                tts = AsyncElevenLabsClient(
                    api_key=settings_obj.elevenlabs_api_key,
                    voice_id=settings_obj.elevenlabs_voice_id,
                    model_id=settings_obj.elevenlabs_model_id,
                    output_format=settings_obj.elevenlabs_output_format,
                    max_retries=settings_obj.elevenlabs_max_retries,
                )
            except (TextToSpeechError, ValueError) as exc:
                logger.warning(
                    "Voice synthesis unavailable, sessions will run text-only: %s", exc
                )
                tts = None

        ice_protocol_state = IceProtocolState()
        thermal_monitor: Optional[ThermalMonitor] = None
        if settings_obj.ice_protocol_enabled:
            thermal_monitor = ThermalMonitor(
                ice_protocol_state,
                threshold_c=settings_obj.ice_protocol_threshold_c,
                hysteresis_c=settings_obj.ice_protocol_hysteresis_c,
                poll_interval_s=settings_obj.ice_protocol_poll_interval_s,
            )
            thermal_monitor.start()

        latency_recorder = LatencyRecorder()
        log_capture = InMemoryLogCapture()
        log_capture.install("zane")

        falcon_worker: Optional[FalconWorker] = None
        falcon_task: Optional[asyncio.Task] = None
        if settings_obj.falcon_scout_enabled:
            falcon_worker = FalconWorker(
                memory_store, embeddings, vector_index, latency_recorder, log_capture,
                FalconWorkerConfig(
                    interval_s=settings_obj.falcon_scout_interval_s,
                    latency_threshold_ms=settings_obj.falcon_scout_latency_threshold_ms,
                    db_latency_threshold_ms=settings_obj.falcon_scout_db_latency_threshold_ms,
                ),
            )
            falcon_task = asyncio.create_task(falcon_worker.run_forever())

        memory_defragmenter: Optional[MemoryDefragmenter] = None
        memory_defrag_task: Optional[asyncio.Task] = None
        if settings_obj.memory_defrag_enabled:
            memory_defragmenter = MemoryDefragmenter(
                memory_store, embeddings, vector_index, groq,
                similarity_threshold=settings_obj.memory_defrag_similarity_threshold,
                half_life_s=settings_obj.memory_defrag_half_life_hours * 3600.0,
                interval_s=settings_obj.memory_defrag_interval_s,
            )
            memory_defrag_task = asyncio.create_task(memory_defragmenter.run_forever())

        return cls(
            analytics=analytics,
            web_search=web_search,
            tools=tools,
            groq=groq,
            memory_store=memory_store,
            embeddings=embeddings,
            vector_index=vector_index,
            summarizer=summarizer,
            tts=tts,
            ice_protocol_state=ice_protocol_state,
            ice_protocol_threshold_c=settings_obj.ice_protocol_threshold_c,
            thermal_monitor=thermal_monitor,
            latency_recorder=latency_recorder,
            log_capture=log_capture,
            falcon_worker=falcon_worker,
            falcon_task=falcon_task,
            memory_defragmenter=memory_defragmenter,
            memory_defrag_task=memory_defrag_task,
        )

    async def aclose(self) -> None:
        await self.groq.close()
        self.analytics.shutdown()
        self.memory_store.close()
        if self.tts is not None:
            await self.tts.close()
        if self.thermal_monitor is not None:
            self.thermal_monitor.stop()
        for worker, task in (
            (self.falcon_worker, self.falcon_task),
            (self.memory_defragmenter, self.memory_defrag_task),
        ):
            if worker is not None:
                worker.stop()
            if task is not None:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass


class ZaneMind:
    """One conversational session's worth of state (memory + humor toggle),
    optionally sharing the expensive backend resources — the Groq client,
    the C++ analytics engine, and the web search tool — with sibling
    sessions via `shared`. Multi-user hosts (e.g. the API adapter) build
    one `SharedBackend` and pass it to every per-session ZaneMind; the CLI
    just lets each instance build its own."""

    def __init__(
        self,
        settings_override: Optional[Settings] = None,
        shared: Optional["SharedBackend"] = None,
        session_id: Optional[str] = None,
    ) -> None:
        self.settings = settings_override or default_settings

        if shared is not None:
            self._shared = shared
            self._owns_shared = False
        else:
            self.settings.validate_for_groq()
            self._shared = SharedBackend.build(self.settings)
            self._owns_shared = True

        self.session_id = session_id or str(uuid.uuid4())
        self.humor = HumorSwitch(self.settings.humor_enabled_default)
        self.voice = VoiceSwitch(self.settings.voice_enabled_default)
        self.memory = PersistentMemory(
            session_id=self.session_id,
            store=self._shared.memory_store,
            embeddings=self._shared.embeddings,
            vector_index=self._shared.vector_index,
            summarizer=self._shared.summarizer,
            config=PersistentMemoryConfig(
                rolling_max_messages=self.settings.memory_max_messages,
                rolling_max_chars=self.settings.memory_max_chars,
                retrieval_top_k=self.settings.memory_retrieval_top_k,
                retention_window=self.settings.memory_retention_window,
                summarize_after_n=self.settings.memory_summarize_after_n,
            ),
        )

    @property
    def analytics(self) -> AnalyticsEngine:
        return self._shared.analytics

    @property
    def web_search(self) -> WebSearchTool:
        return self._shared.web_search

    @property
    def tools(self) -> ToolRegistry:
        return self._shared.tools

    @property
    def groq(self) -> AsyncGroqClient:
        return self._shared.groq

    @property
    def tts(self) -> Optional[AsyncElevenLabsClient]:
        return self._shared.tts

    async def respond(
        self,
        user_input: str,
        *,
        addressed_by: Optional[str] = None,
        mission_context: Optional[str] = None,
        max_tool_rounds: int = 4,
    ) -> TurnResult:
        """Runs one full conversational turn, including any number of tool
        calls the model requests, and returns Zane's synthesized reply."""
        start = time.monotonic()

        # Ice Protocol interception: if the host is thermally throttled,
        # completely bypass Groq and the RAG memory stack (including this
        # turn's own memory write) for a local, rules-based reply instead.
        ice_protocol_text = await maybe_handle_ice_protocol(
            user_input,
            self._shared.ice_protocol_state,
            self.analytics,
            self._shared.ice_protocol_threshold_c,
        )
        if ice_protocol_text is not None:
            elapsed_ms = (time.monotonic() - start) * 1000
            return TurnResult(text=ice_protocol_text, tool_calls_made=[], elapsed_ms=elapsed_ms)

        tool_calls_made: List[str] = []

        await self.memory.add_user_message(user_input)
        relevant_memories = await self.memory.retrieve_relevant(user_input)
        conversation_summary = self.memory.get_latest_summary()
        context = PersonaContext(
            addressed_by=addressed_by,
            mission_context=mission_context,
            relevant_memories=relevant_memories,
            conversation_summary=conversation_summary,
        )
        system_prompt = build_system_prompt(
            humor_enabled=self.humor.enabled, tools_enabled=True, context=context
        )

        for _round in range(max_tool_rounds):
            messages = [{"role": "system", "content": system_prompt}] + self.memory.get_messages()

            try:
                completion = await self.groq.chat_completion(messages, tools=self.tools.schemas())
            except GroqUnavailableError as exc:
                logger.error("Groq unavailable: %s", exc)
                fallback_text = _OFFLINE_FALLBACK.format(detail=str(exc))
                audio, audio_format = await synthesize_with_fallback(
                    self.tts, self.voice.enabled, fallback_text
                )
                elapsed_ms = (time.monotonic() - start) * 1000
                return TurnResult(
                    text=fallback_text,
                    tool_calls_made=tool_calls_made,
                    elapsed_ms=elapsed_ms,
                    audio=audio,
                    audio_format=audio_format,
                )

            message = completion.choices[0].message

            if message.tool_calls:
                # A tool call means Zane needs data beyond his training
                # weights — trigger the C++ analysis phase to time this
                # round (feeds his "internal diagnostics" flavor text and
                # real telemetry alike), then execute every requested tool.
                round_start = time.monotonic()

                assistant_tool_calls = [tc.model_dump() for tc in message.tool_calls]
                tool_result_messages = []
                for tc in message.tool_calls:
                    tool_calls_made.append(tc.function.name)
                    result_str = await self.tools.dispatch(tc.function.name, tc.function.arguments)
                    tool_result_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "name": tc.function.name,
                            "content": result_str,
                        }
                    )

                round_elapsed_ms = (time.monotonic() - round_start) * 1000
                frame = DataStreamFrame(label="TOOL_ROUND_LATENCY_MS", values=[round_elapsed_ms])
                logger.debug(self.analytics.format_data_stream(frame))
                self.analytics.update_internal_state("last_tool_round_latency_ms", round_elapsed_ms)

                await self.memory.add_tool_exchange(assistant_tool_calls, tool_result_messages)
                continue  # let the model synthesize the tool results next round

            final_text = message.content or ""
            await self.memory.add_assistant_message(final_text)
            audio, audio_format = await synthesize_with_fallback(
                self.tts, self.voice.enabled, final_text
            )
            elapsed_ms = (time.monotonic() - start) * 1000
            return TurnResult(
                text=final_text,
                tool_calls_made=tool_calls_made,
                elapsed_ms=elapsed_ms,
                audio=audio,
                audio_format=audio_format,
            )

        exhausted_text = (
            "I apologize; I was unable to resolve this within my allotted "
            "analytical cycles. Might I suggest rephrasing your request?"
        )
        audio, audio_format = await synthesize_with_fallback(
            self.tts, self.voice.enabled, exhausted_text
        )
        elapsed_ms = (time.monotonic() - start) * 1000
        return TurnResult(
            text=exhausted_text,
            tool_calls_made=tool_calls_made,
            elapsed_ms=elapsed_ms,
            audio=audio,
            audio_format=audio_format,
        )

    async def stream_respond(self, user_input: str, *, addressed_by: Optional[str] = None):
        """Low-latency streaming path with no tool-calling — used by
        interfaces that want token-by-token output for simple exchanges."""
        await self.memory.add_user_message(user_input)
        context = PersonaContext(addressed_by=addressed_by)
        system_prompt = build_system_prompt(
            humor_enabled=self.humor.enabled, tools_enabled=False, context=context
        )
        messages = [{"role": "system", "content": system_prompt}] + self.memory.get_messages()

        collected = []
        async for chunk in self.groq.stream_chat_completion(messages):
            collected.append(chunk)
            yield chunk

        await self.memory.add_assistant_message("".join(collected))

    def reset_conversation(self) -> None:
        """Clears the in-context rolling window only; durable history in
        SQLite and the vector index is untouched, so long-term recall via
        `retrieve_relevant` still works across a reset."""
        self.memory.reset_rolling_window()

    async def aclose(self) -> None:
        """Releases resources this instance owns. When constructed with a
        shared backend, the backend's owner (not this session) is
        responsible for closing it — call `SharedBackend.aclose()` instead."""
        if self._owns_shared:
            await self._shared.aclose()
