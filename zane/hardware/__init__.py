"""Zane's physical-integration layer: camera vision, GPIO peripherals
(cryo-discharge solenoid, NeoPixel status ring), acoustic servo tracking,
and hardware-driven system overrides.

All four modules take a `zane.hardware.hal.HardwareAbstractionLayer`
instance via dependency injection and degrade to `MockHAL` (a terminal
log, not real hardware) whenever real hardware isn't available — which is
always, in this project's actual Render/Docker deployment. Nothing here
is wired into `ZaneMind` unless `ZANE_HARDWARE_ENABLED=true`.
"""
from zane.hardware.hal import GPIOHAL, HardwareAbstractionLayer, MockHAL, get_hal

__all__ = ["HardwareAbstractionLayer", "MockHAL", "GPIOHAL", "get_hal"]
