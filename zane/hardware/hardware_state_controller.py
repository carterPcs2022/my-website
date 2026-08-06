"""Master hardware daemon: connects the physical Humor Switch (a GPIO
toggle, or a local network socket command) and Falcon Scout's critical-
fault signal to `zane.personality.HumorSwitch`'s hardware lock, so either
one can force Zane into a purely analytical, strict utility-preservation
state — instantly, and in a way normal `/humor` commands cannot override
until the condition clears.

Two independent trigger paths, both real and testable without physical
hardware:
- GPIO interrupt (`HardwareAbstractionLayer.watch_digital_pin`) — a real
  edge-callback on real hardware; on `MockHAL`, triggered via
  `MockHAL.simulate_pin_change` (exactly what the tests do).
- A local asyncio socket server accepting simple newline-terminated text
  commands (`HUMOR_SWITCH:OFF` / `HUMOR_SWITCH:ON` / `OVERRIDE:CLEAR`) —
  this is the "dedicated network socket command" path from the original
  spec, and arguably the more relevant one given this project's actual
  cloud deployment has no GPIO pins to interrupt on at all.

Recovery is automatic where it can be: if the hardware switch is restored
to ON, or Falcon Scout's fault clears, the override lifts — *unless* the
other trigger is still active, in which case it stays engaged (both must
clear). There is no physical "switch back on" position for the socket
path, so `OVERRIDE:CLEAR` is the explicit reset command for that source.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

from zane.falcon_worker import SystemFaultState
from zane.hardware.hal import HardwareAbstractionLayer
from zane.personality import HumorSwitch

logger = logging.getLogger("zane.hardware.hardware_state_controller")


class HardwareStateController:
    def __init__(
        self,
        hal: HardwareAbstractionLayer,
        humor_switch: HumorSwitch,
        *,
        fault_state: Optional[SystemFaultState] = None,
        gpio_pin: Optional[int] = None,
        socket_host: str = "127.0.0.1",
        socket_port: Optional[int] = None,
        fault_poll_interval_s: float = 5.0,
    ) -> None:
        self._hal = hal
        self._humor_switch = humor_switch
        self._fault_state = fault_state
        self._gpio_pin = gpio_pin
        self._socket_host = socket_host
        self._socket_port = socket_port
        self._fault_poll_interval_s = fault_poll_interval_s

        # Independent trigger flags: the override stays engaged as long as
        # *either* is true, and only both being false clears it.
        self._hardware_switch_off = False
        self._fault_active = False

        self._socket_server: Optional[asyncio.AbstractServer] = None
        self._socket_task: Optional[asyncio.Task] = None
        self._fault_monitor_task: Optional[asyncio.Task] = None

    def start(self) -> None:
        """Synchronous by design (schedules tasks via `asyncio.create_task`
        but never itself needs to await) so it can be called from a sync
        context that's merely running inside an event loop — exactly how
        `SharedBackend.build()` constructs the rest of the hardware
        stack."""
        if self._gpio_pin is not None:
            self._hal.watch_digital_pin(self._gpio_pin, self._on_gpio_edge)
            logger.info(
                "[HARDWARE_OVERRIDE] Watching GPIO pin %d for humor-switch state.", self._gpio_pin
            )
        if self._socket_port is not None:
            self._socket_task = asyncio.create_task(self._run_socket_server())
        if self._fault_state is not None:
            self._fault_monitor_task = asyncio.create_task(self._watch_fault_state())

    # --- trigger sources ---

    def _on_gpio_edge(self, pin: int, value: bool) -> None:
        # Convention (matches GPIOHAL.watch_digital_pin): value=True is the
        # switch's resting/ON position, value=False is flipped/OFF.
        self._hardware_switch_off = not value
        if self._hardware_switch_off:
            self._engage_override(f"hardware humor switch (GPIO pin {pin}) flipped OFF")
        else:
            self._maybe_clear_override(source="hardware switch restored to ON")

    async def _run_socket_server(self) -> None:
        async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            try:
                data = await reader.readline()
                command = data.decode(errors="replace").strip().upper()
                if command == "HUMOR_SWITCH:OFF":
                    self._hardware_switch_off = True
                    self._engage_override("network socket command HUMOR_SWITCH:OFF")
                    writer.write(b"OK\n")
                elif command == "HUMOR_SWITCH:ON":
                    self._hardware_switch_off = False
                    self._maybe_clear_override(source="network socket command HUMOR_SWITCH:ON")
                    writer.write(b"OK\n")
                elif command == "OVERRIDE:CLEAR":
                    self._hardware_switch_off = False
                    self._fault_active = False
                    self._humor_switch.unlock()
                    logger.info("[HARDWARE_OVERRIDE] Override force-cleared via OVERRIDE:CLEAR.")
                    writer.write(b"OK\n")
                else:
                    writer.write(b"ERR unknown command\n")
            except Exception:  # noqa: BLE001 - a bad client must not kill the server
                logger.exception("Hardware override socket handler failed.")
                writer.write(b"ERR internal error\n")
            finally:
                await writer.drain()
                writer.close()

        server = await asyncio.start_server(handle, self._socket_host, self._socket_port)
        self._socket_server = server
        logger.info(
            "[HARDWARE_OVERRIDE] Listening for humor-switch commands on %s:%d.",
            self._socket_host, self._socket_port,
        )
        async with server:
            await server.serve_forever()

    async def _watch_fault_state(self) -> None:
        try:
            while True:
                assert self._fault_state is not None
                snapshot = self._fault_state.snapshot()
                self._fault_active = snapshot.active
                if snapshot.active:
                    self._engage_override(f"Falcon Scout critical fault: {snapshot.reason}")
                else:
                    self._maybe_clear_override(source="Falcon Scout fault cleared")
                await asyncio.sleep(self._fault_poll_interval_s)
        except asyncio.CancelledError:
            raise

    # --- override engagement ---

    def _engage_override(self, reason: str) -> None:
        was_locked = self._humor_switch.locked
        self._humor_switch.lock()
        if not was_locked:
            logger.warning(
                "[HARDWARE_OVERRIDE] Priority override engaged: %s. Humor protocol discarded; "
                "locking into strict analytical utility-preservation state.", reason,
            )

    def _maybe_clear_override(self, source: str) -> None:
        if self._hardware_switch_off or self._fault_active:
            logger.info(
                "[HARDWARE_OVERRIDE] %s, but another trigger is still active; override remains engaged.",
                source,
            )
            return
        if self._humor_switch.locked:
            self._humor_switch.unlock()
            logger.info(
                "[HARDWARE_OVERRIDE] Override cleared (%s); humor protocol control restored "
                "to normal operation.", source,
            )

    async def stop(self) -> None:
        if self._socket_task is not None:
            self._socket_task.cancel()
            try:
                await self._socket_task
            except asyncio.CancelledError:
                pass
        if self._fault_monitor_task is not None:
            self._fault_monitor_task.cancel()
            try:
                await self._fault_monitor_task
            except asyncio.CancelledError:
                pass
        if self._socket_server is not None:
            self._socket_server.close()
            await self._socket_server.wait_closed()
