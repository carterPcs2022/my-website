"""Digital-to-analog hardware interface: the cryo-discharge solenoid relay
and the WS2812B NeoPixel status ring.

SAFETY DESIGN NOTE — read before wiring this to a real relay: the cryo
tool (`engage_cryo_discharge`) drives a solenoid controlling a 12V CO2
valve, i.e. a real pressurized-gas actuator, and it is reachable from the
LLM's own tool-calling loop. An LLM's judgment is deliberately **not**
trusted as the sole safety boundary for that:

- `PeripheralManager` starts **disarmed** (`armed=False`). Nothing the LLM
  says can fire the solenoid until something outside the LLM's control —
  a human, or `hardware_state_controller.py`'s hardware/socket path —
  calls `arm()`. This is a hard gate, not a suggestion in the prompt.
- The requested duration is always clamped to `max_discharge_s`,
  regardless of what the LLM asks for.
- A `min_cooldown_s` window after every discharge rejects rapid re-fire
  requests (protects the canister/relay and avoids a runaway loop).
- The relay is switched off in a `finally` block, so a mid-discharge
  exception (or task cancellation) can never leave it stuck on.
"""
from __future__ import annotations

import asyncio
import logging
import math
import threading
import time
from typing import Any, Callable, Dict, List, Tuple

from zane.hardware.hal import HardwareAbstractionLayer, RGBColor

logger = logging.getLogger("zane.hardware.peripheral_io")

# LLM tool schema for engage_cryo_discharge, only ever added to
# ToolRegistry.schemas() when a PeripheralManager is actually configured
# (see zane/tools/registry.py) — a text-only session on Render has no
# business advertising this tool exists at all.
CRYO_DISCHARGE_TOOL_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "engage_cryo_discharge",
        "description": (
            "Physically fires Zane's cryo-discharge solenoid (a real 12V CO2 "
            "valve relay) for the requested duration. Requires the peripheral "
            "to be armed by an operator; will be refused otherwise. The "
            "requested duration is always clamped to a safe maximum "
            "regardless of what is requested here."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "duration_seconds": {
                    "type": "number",
                    "description": "Requested discharge duration in seconds (will be clamped).",
                },
            },
            "required": ["duration_seconds"],
        },
    },
}

# --- NeoPixel operational-mode patterns -----------------------------------

_ICE_BLUE: RGBColor = (30, 140, 255)
_SOLID_BLUE: RGBColor = (20, 60, 255)
_RED: RGBColor = (255, 0, 0)
_DIM_WHITE: RGBColor = (40, 40, 45)
_OFF: RGBColor = (0, 0, 0)


def _scale(color: RGBColor, factor: float) -> RGBColor:
    factor = max(0.0, min(1.0, factor))
    return (int(color[0] * factor), int(color[1] * factor), int(color[2] * factor))


def _idle_pattern(t: float, count: int) -> List[RGBColor]:
    # Slow breathing dim white — "awake, at rest".
    brightness = 0.15 + 0.15 * (0.5 + 0.5 * math.sin(t * (2 * math.pi / 6.0)))
    return [_scale(_DIM_WHITE, brightness)] * count


def _thinking_pattern(t: float, count: int) -> List[RGBColor]:
    # Pulsing ice-blue — matches Zane's element, distinct cadence from idle.
    brightness = 0.35 + 0.65 * (0.5 + 0.5 * math.sin(t * (2 * math.pi / 1.2)))
    return [_scale(_ICE_BLUE, brightness)] * count


def _speaking_pattern(t: float, count: int) -> List[RGBColor]:
    # Solid blue, steady — no animation needed while actively speaking.
    return [_SOLID_BLUE] * count


def _error_pattern(t: float, count: int) -> List[RGBColor]:
    # Flashing red square wave at 2Hz.
    on = int(t * 2.0) % 2 == 0
    return [(_RED if on else _OFF)] * count


_PATTERNS: Dict[str, Callable[[float, int], List[RGBColor]]] = {
    "IDLE": _idle_pattern,
    "THINKING": _thinking_pattern,
    "SPEAKING": _speaking_pattern,
    "ERROR": _error_pattern,
}


class PeripheralManager:
    def __init__(
        self,
        hal: HardwareAbstractionLayer,
        *,
        cryo_relay_pin: int = 17,
        neopixel_pin: int = 18,
        neopixel_count: int = 16,
        armed: bool = False,
        max_discharge_s: float = 2.0,
        min_cooldown_s: float = 5.0,
        animation_fps: float = 20.0,
    ) -> None:
        self._hal = hal
        self._cryo_relay_pin = cryo_relay_pin
        self._neopixel_pin = neopixel_pin
        self._neopixel_count = neopixel_count
        self._max_discharge_s = max_discharge_s
        self._min_cooldown_s = min_cooldown_s
        self._animation_fps = animation_fps

        self._lock = threading.Lock()
        self._armed = armed
        self._last_discharge_end = 0.0
        self._current_led_state = "IDLE"
        self._animation_task: "asyncio.Task | None" = None

    # --- arming (never controllable by the LLM tool itself) ---

    def arm(self) -> None:
        with self._lock:
            self._armed = True
        logger.warning("[PERIPHERAL::CRYO] Solenoid ARMED. Discharge tool is now live.")

    def disarm(self) -> None:
        with self._lock:
            self._armed = False
        logger.warning("[PERIPHERAL::CRYO] Solenoid DISARMED. Discharge tool will refuse all requests.")

    @property
    def armed(self) -> bool:
        with self._lock:
            return self._armed

    # --- cryo-discharge tool ---

    async def engage_cryo_discharge(self, duration_seconds: float) -> str:
        """LLM tool entry point. See the module docstring for the safety
        gating this applies before ever touching the relay."""
        if not self.armed:
            logger.warning(
                "Cryo discharge requested (%.2fs) but the solenoid is not armed; refusing.",
                duration_seconds,
            )
            return (
                "Cryo-discharge solenoid is not armed; a physical arm switch or "
                "operator command is required before this system will fire."
            )

        with self._lock:
            now = time.monotonic()
            elapsed_since_last = now - self._last_discharge_end
            if elapsed_since_last < self._min_cooldown_s:
                remaining = self._min_cooldown_s - elapsed_since_last
                with_lock_remaining = remaining
            else:
                with_lock_remaining = None

        if with_lock_remaining is not None:
            logger.warning(
                "Cryo discharge requested during cooldown (%.1fs remaining); refusing.",
                with_lock_remaining,
            )
            return (
                f"Cryo-discharge solenoid is cooling down "
                f"({with_lock_remaining:.1f}s remaining); request denied."
            )

        clamped_duration = max(0.05, min(duration_seconds, self._max_discharge_s))
        if clamped_duration != duration_seconds:
            logger.warning(
                "Requested discharge duration %.2fs clamped to safe maximum %.2fs.",
                duration_seconds, clamped_duration,
            )

        logger.info("[PERIPHERAL::CRYO] Engaging discharge for %.2fs.", clamped_duration)
        try:
            self._hal.digital_write(self._cryo_relay_pin, True)
            await asyncio.sleep(clamped_duration)
        finally:
            # Always throttles off, even on cancellation or a HAL exception
            # mid-discharge — a solenoid must never be left energized.
            try:
                self._hal.digital_write(self._cryo_relay_pin, False)
            except Exception:  # noqa: BLE001 - shutoff must not raise past this point
                logger.exception("Failed to command cryo relay off; treat hardware as unsafe.")
            with self._lock:
                self._last_discharge_end = time.monotonic()
            logger.info("[PERIPHERAL::CRYO] Discharge relay throttled off.")

        return f"Cryo-discharge engaged for {clamped_duration:.2f} seconds."

    # --- NeoPixel operational-mode ring ---

    async def update_led_state(self, state: str) -> None:
        """Reflects Zane's current operational mode on the status ring.
        This is a system-driven state transition (see ZaneMind.respond's
        THINKING/SPEAKING/ERROR/IDLE hooks) — not an LLM tool; the model
        doesn't decide when it's "thinking," the orchestrator does."""
        normalized = state.upper()
        if normalized not in _PATTERNS:
            logger.warning("Unknown LED state %r; defaulting to IDLE.", state)
            normalized = "IDLE"

        with self._lock:
            changed = normalized != self._current_led_state
            self._current_led_state = normalized

        if changed:
            logger.info("[PERIPHERAL::LED] Operational mode -> %s", normalized)
        self._ensure_animation_running()

    def _ensure_animation_running(self) -> None:
        if self._animation_task is None or self._animation_task.done():
            self._animation_task = asyncio.create_task(self._animate_forever())

    async def _animate_forever(self) -> None:
        start = time.monotonic()
        try:
            while True:
                with self._lock:
                    state = self._current_led_state
                pattern_fn = _PATTERNS[state]
                t = time.monotonic() - start
                colors = pattern_fn(t, self._neopixel_count)
                try:
                    self._hal.set_neopixel_frame(self._neopixel_pin, self._neopixel_count, colors)
                except Exception:  # noqa: BLE001 - a bad frame write must not kill the animation loop
                    logger.exception("NeoPixel frame write failed; will retry next frame.")
                await asyncio.sleep(1.0 / self._animation_fps)
        except asyncio.CancelledError:
            raise

    def stop(self) -> None:
        if self._animation_task is not None:
            self._animation_task.cancel()
        try:
            self._hal.digital_write(self._cryo_relay_pin, False)
        except Exception:  # noqa: BLE001
            logger.exception("Failed to command cryo relay off during shutdown.")
