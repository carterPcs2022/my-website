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
from typing import List, Optional, Tuple

from zane.amphibious_bounty import BOUNTY_MAX_STRUCTURAL_DEPTH_M
from zane.analytics_bridge import AnalyticsEngine, DataStreamFrame
from zane.companion_bridge import NeuralBridge, PixalSystemsCore
from zane.config import Settings, settings as default_settings
from zane.falcon_worker import (
    FalconWorker,
    FalconWorkerConfig,
    InMemoryLogCapture,
    LatencyRecorder,
    SystemFaultState,
)
from zane.groq_client import AsyncGroqClient, GroqUnavailableError
from zane.hardware.hal import HardwareAbstractionLayer, get_hal
from zane.hardware.hardware_state_controller import HardwareStateController
from zane.hardware.peripheral_io import PeripheralManager
from zane.hardware.sensory_localization import AcousticLocalizer, HeadTrackingState
from zane.hardware.vision_processor import VisionPipeline
from zane.knowledge_manager import KnowledgeManager
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
from zane.tools.wolfram_tool import WolframAlphaClient
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
    # Populated only when P.I.X.A.L. had a pending anomaly notice this turn
    # AND voice is on AND PIXAL_VOICE_ID is configured — her own
    # independent audio stream, distinct from Zane's `audio` above (see
    # zane/companion_bridge.py and synthesize_with_fallback's `voice_id`
    # override). None in every other case.
    companion_audio: Optional[bytes] = None
    companion_audio_format: Optional[str] = None


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
    # Physical hardware layer (zane/hardware/). `hal` is always present
    # (MockHAL if hardware isn't enabled/available — see get_hal); the
    # rest are None unless ZANE_HARDWARE_ENABLED.
    hal: HardwareAbstractionLayer
    system_fault_state: SystemFaultState
    peripheral_manager: Optional[PeripheralManager]
    vision_pipeline: Optional[VisionPipeline]
    vision_task: Optional["asyncio.Task"]
    head_tracking_state: Optional[HeadTrackingState]
    acoustic_localizer: Optional[AcousticLocalizer]
    localizer_task: Optional["asyncio.Task"]
    hardware_state_controller: Optional[HardwareStateController]
    # One physical robot body has one humor state, not one per chat
    # session. Only set when hardware is enabled — see ZaneMind.__init__,
    # which falls back to a private per-session HumorSwitch otherwise.
    shared_humor_switch: Optional[HumorSwitch]
    # Multi-namespace RAG (zane/knowledge_manager.py): always present,
    # searches organic chat memory alongside the static archival lore
    # index (empty/harmless until compile_lore_database() has been run).
    knowledge_manager: KnowledgeManager
    # Wolfram|Alpha analytical math tool: always present so the tool stays
    # advertised even without WOLFRAM_APP_ID, degrading to a local
    # restricted-arithmetic fallback (see zane/tools/wolfram_tool.py).
    wolfram_client: WolframAlphaClient
    # P.I.X.A.L. companion bridge (zane/companion_bridge.py): always
    # present — cheap, no external dependencies. `neural_bridge` carries
    # her anomaly notices into every turn's system prompt regardless of
    # whether anything has ever flagged one.
    neural_bridge: NeuralBridge
    pixal: PixalSystemsCore

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

        hal = get_hal(
            prefer_physical=settings_obj.hardware_enabled and settings_obj.hardware_use_real_gpio
        )
        system_fault_state = SystemFaultState()

        peripheral_manager: Optional[PeripheralManager] = None
        vision_pipeline: Optional[VisionPipeline] = None
        vision_task: Optional[asyncio.Task] = None
        head_tracking_state: Optional[HeadTrackingState] = None
        acoustic_localizer: Optional[AcousticLocalizer] = None
        localizer_task: Optional[asyncio.Task] = None
        hardware_state_controller: Optional[HardwareStateController] = None
        # One physical robot body has one humor state, not one per chat
        # session — see ZaneMind.__init__, which uses this shared instance
        # instead of a private per-session HumorSwitch whenever hardware is
        # enabled. None (and unused) otherwise.
        shared_humor_switch: Optional[HumorSwitch] = None

        if settings_obj.hardware_enabled:
            peripheral_manager = PeripheralManager(
                hal,
                cryo_relay_pin=settings_obj.hardware_cryo_relay_pin,
                neopixel_pin=settings_obj.hardware_neopixel_pin,
                neopixel_count=settings_obj.hardware_neopixel_count,
                armed=settings_obj.hardware_cryo_armed,
                max_discharge_s=settings_obj.hardware_cryo_max_discharge_s,
                min_cooldown_s=settings_obj.hardware_cryo_cooldown_s,
            )

            vision_pipeline = VisionPipeline(
                hal,
                camera_index=settings_obj.hardware_camera_index,
                detection_interval_s=settings_obj.hardware_vision_interval_s,
            )
            vision_task = asyncio.create_task(vision_pipeline.run_forever())

            head_tracking_state = HeadTrackingState()
            acoustic_localizer = AcousticLocalizer(
                hal,
                head_tracking_state,
                servo_pin=settings_obj.hardware_neck_servo_pin,
                smoothing_alpha=settings_obj.hardware_doa_smoothing_alpha,
            )
            localizer_task = asyncio.create_task(acoustic_localizer.run_forever())

            shared_humor_switch = HumorSwitch(settings_obj.humor_enabled_default)
            hardware_state_controller = HardwareStateController(
                hal,
                shared_humor_switch,
                fault_state=system_fault_state,
                gpio_pin=settings_obj.hardware_humor_gpio_pin,
                socket_port=settings_obj.hardware_override_socket_port,
            )
            hardware_state_controller.start()

        memory_store = SQLiteMessageStore(settings_obj.memory_db_path)
        embeddings = EmbeddingBackend(settings_obj.memory_embedding_model)
        vector_index = FaissVectorIndex(settings_obj.memory_vector_index_path)
        summarizer = ConversationSummarizer(groq)

        knowledge_manager = KnowledgeManager(
            embeddings,
            lore_index_path=settings_obj.lore_index_path,
            lore_metadata_path=settings_obj.lore_metadata_path,
        )
        wolfram_client = WolframAlphaClient(
            app_id=settings_obj.wolfram_app_id, timeout_s=settings_obj.wolfram_timeout_s
        )

        neural_bridge = NeuralBridge()
        try:
            neural_bridge.bind_loop(asyncio.get_running_loop())
        except RuntimeError:
            # Constructed outside a running event loop (e.g. a sync test
            # harness) — cross-thread flagging is simply unavailable until
            # something calls bind_loop() itself; the async-native
            # flag_anomaly path still works fine either way.
            logger.debug(
                "SharedBackend.build() called outside a running event loop; "
                "NeuralBridge loop binding deferred."
            )
        pixal = PixalSystemsCore(
            groq,
            neural_bridge,
            latency_threshold_ms=settings_obj.falcon_scout_latency_threshold_ms,
            battery_voltage_threshold_v=settings_obj.pixal_battery_voltage_threshold_v,
            critical_depth_threshold_m=BOUNTY_MAX_STRUCTURAL_DEPTH_M,
        )

        tools = ToolRegistry(analytics, web_search, groq, peripheral_manager, wolfram_client)

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
                fault_state=system_fault_state,
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
            hal=hal,
            system_fault_state=system_fault_state,
            peripheral_manager=peripheral_manager,
            vision_pipeline=vision_pipeline,
            vision_task=vision_task,
            head_tracking_state=head_tracking_state,
            acoustic_localizer=acoustic_localizer,
            localizer_task=localizer_task,
            hardware_state_controller=hardware_state_controller,
            shared_humor_switch=shared_humor_switch,
            knowledge_manager=knowledge_manager,
            wolfram_client=wolfram_client,
            neural_bridge=neural_bridge,
            pixal=pixal,
        )

    async def aclose(self) -> None:
        await self.groq.close()
        self.analytics.shutdown()
        self.memory_store.close()
        await self.wolfram_client.close()
        if self.tts is not None:
            await self.tts.close()
        if self.thermal_monitor is not None:
            self.thermal_monitor.stop()
        for worker, task in (
            (self.falcon_worker, self.falcon_task),
            (self.memory_defragmenter, self.memory_defrag_task),
            (self.vision_pipeline, self.vision_task),
            (self.acoustic_localizer, self.localizer_task),
        ):
            if worker is not None:
                worker.stop()
            if task is not None:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        if self.peripheral_manager is not None:
            self.peripheral_manager.stop()
        if self.hardware_state_controller is not None:
            await self.hardware_state_controller.stop()


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
        # One physical robot body has one humor state: if hardware is
        # enabled, every session on this process shares the same
        # HumorSwitch (so a hardware/Falcon-triggered lock affects the one
        # real Zane, not just whichever chat session happened to trigger
        # it). Otherwise each session gets its own, exactly as before.
        self.humor = self._shared.shared_humor_switch or HumorSwitch(
            self.settings.humor_enabled_default
        )
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

    @property
    def knowledge_manager(self) -> KnowledgeManager:
        return self._shared.knowledge_manager

    @property
    def neural_bridge(self) -> NeuralBridge:
        return self._shared.neural_bridge

    @property
    def pixal(self) -> PixalSystemsCore:
        return self._shared.pixal

    async def _set_led_state(self, state: str) -> None:
        """Best-effort reflection of Zane's operational mode on the
        NeoPixel ring — never allowed to break a conversational turn. A
        system-driven state transition (see zane/hardware/peripheral_io.py),
        not an LLM tool: the model doesn't decide when it's "thinking,"
        the orchestrator does. IDLE is only ever the ring's resting state
        before the first request of the process; after that it settles
        into THINKING/SPEAKING/ERROR as those transitions occur, which is
        more informative than round-tripping through IDLE every turn with
        no way to know when TTS playback has actually finished."""
        peripheral_manager = self._shared.peripheral_manager
        if peripheral_manager is None:
            return
        try:
            await peripheral_manager.update_led_state(state)
        except Exception:  # noqa: BLE001
            logger.exception("Failed to update peripheral LED state to %s.", state)

    def _get_sensory_hud(self) -> Optional[str]:
        vision_pipeline = self._shared.vision_pipeline
        if vision_pipeline is None:
            return None
        return vision_pipeline.get_latest_hud()

    async def _maybe_synthesize_companion_notice(
        self, notice_text: Optional[str]
    ) -> Tuple[Optional[bytes], Optional[str]]:
        """P.I.X.A.L.'s dual-voice routing: reuses Zane's own
        AsyncElevenLabsClient (see synthesize_with_fallback's `voice_id`
        override) rather than a second TTS client, so there is no
        duplicate retry/backoff/auth plumbing to maintain. No-ops (None,
        None) whenever there's nothing pending, PIXAL_VOICE_ID isn't
        configured, or voice is off — identical fallback shape to Zane's
        own synthesis path."""
        if not notice_text:
            return None, None
        pixal_voice_id = self.settings.pixal_voice_id
        if not pixal_voice_id:
            return None, None
        return await synthesize_with_fallback(
            self.tts, self.voice.enabled, notice_text, voice_id=pixal_voice_id
        )

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

        await self._set_led_state("THINKING")
        tool_calls_made: List[str] = []

        await self.memory.add_user_message(user_input)
        # Unified RAG lookup: embeds user_input once, searches organic chat
        # memory and the static archival lore database concurrently, and
        # returns both as tagged entries in the same list — see
        # zane/knowledge_manager.py.
        relevant_memories = await self._shared.knowledge_manager.query_all_knowledge_sources(
            user_input, self.memory, top_k=self.settings.memory_retrieval_top_k
        )
        conversation_summary = self.memory.get_latest_summary()
        context = PersonaContext(
            addressed_by=addressed_by,
            mission_context=mission_context,
            relevant_memories=relevant_memories,
            conversation_summary=conversation_summary,
            sensory_hud=self._get_sensory_hud(),
        )
        system_prompt = build_system_prompt(
            humor_enabled=self.humor.enabled, tools_enabled=True, context=context
        )
        # P.I.X.A.L. companion bridge: fold in any anomaly notices she's
        # flagged since the last turn before Zane's Groq client fires —
        # see zane/companion_bridge.py. companion_notice_text is kept
        # separately so it can also be synthesized on her own voice
        # stream below, independent of Zane's own response audio.
        system_prompt, companion_notice_text = await self._shared.neural_bridge.inject_into_prompt(
            system_prompt
        )

        for _round in range(max_tool_rounds):
            messages = [{"role": "system", "content": system_prompt}] + self.memory.get_messages()

            try:
                completion = await self.groq.chat_completion(messages, tools=self.tools.schemas())
            except GroqUnavailableError as exc:
                logger.error("Groq unavailable: %s", exc)
                await self._set_led_state("ERROR")
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
            await self._set_led_state("SPEAKING")
            # Zane's own response and P.I.X.A.L.'s pending notice (if any)
            # are synthesized concurrently — her audio stream never blocks
            # or delays Zane's main terminal output, and vice versa.
            (audio, audio_format), (companion_audio, companion_audio_format) = await asyncio.gather(
                synthesize_with_fallback(self.tts, self.voice.enabled, final_text),
                self._maybe_synthesize_companion_notice(companion_notice_text),
            )
            elapsed_ms = (time.monotonic() - start) * 1000
            return TurnResult(
                text=final_text,
                tool_calls_made=tool_calls_made,
                elapsed_ms=elapsed_ms,
                audio=audio,
                audio_format=audio_format,
                companion_audio=companion_audio,
                companion_audio_format=companion_audio_format,
            )

        exhausted_text = (
            "I apologize; I was unable to resolve this within my allotted "
            "analytical cycles. Might I suggest rephrasing your request?"
        )
        await self._set_led_state("SPEAKING")
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
