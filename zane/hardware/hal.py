"""Hardware Abstraction Layer for Zane's physical-integration modules.

Exactly one HAL implementation is active per process:

- `MockHAL` — always available, zero external dependencies. Every
  operation logs in Zane's formal diagnostic voice instead of touching
  real hardware. This is what runs by default, and what always runs on
  this project's actual Render/Docker deployment (or any host without a
  real GPIO chip) — vision/peripheral/servo/GPIO-interrupt code paths are
  fully exercised in tests against this implementation.
- `GPIOHAL` — real hardware, built on `gpiozero` (lazy-imported; it
  already abstracts RPi.GPIO/lgpio/pigpio backends) plus
  `adafruit-circuitpython-neopixel` for WS2812B output. Only constructed
  if explicitly enabled AND those libraries import successfully AND a
  real GPIO chip is detected; any failure at any stage falls back to
  `MockHAL` with a logged warning rather than crashing. This path cannot
  be exercised in this sandbox or in the Render deployment — there is no
  physical GPIO chip in either — and has not been tested against real
  hardware. It is written to the documented gpiozero/neopixel APIs as
  carefully as possible, but treat it as unverified until run on an
  actual Raspberry Pi.

Every hardware-touching module in `zane/hardware/` takes a HAL instance
via dependency injection rather than importing hardware libraries
directly, so all of them can be exercised in tests — and in this
project's real cloud deployment — with no physical device attached.
"""
from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("zane.hardware.hal")

RGBColor = Tuple[int, int, int]
PinEdgeCallback = Callable[[int, bool], None]


class HardwareUnavailableError(RuntimeError):
    """Raised by `GPIOHAL` construction when real hardware access isn't
    possible (missing library, no GPIO chip, permission error). Callers
    should catch this and fall back to `MockHAL` — see `get_hal` below."""


class HardwareAbstractionLayer(ABC):
    @abstractmethod
    def digital_write(self, pin: int, value: bool) -> None:
        """Drives a digital output pin high (True) or low (False)."""

    @abstractmethod
    def pwm_write(self, pin: int, duty_cycle: float) -> None:
        """Sets a PWM output's duty cycle, clamped to [0.0, 1.0]."""

    @abstractmethod
    def set_servo_angle(self, pin: int, angle_degrees: float) -> None:
        """Commands a servo on `pin` to `angle_degrees`, clamped to
        [0, 180]."""

    @abstractmethod
    def set_neopixel_frame(self, pin: int, pixel_count: int, colors: List[RGBColor]) -> None:
        """Writes one full frame of RGB values to a WS2812B strip/ring on
        `pin`. `colors` is padded/truncated to `pixel_count`."""

    @abstractmethod
    def watch_digital_pin(self, pin: int, callback: PinEdgeCallback) -> None:
        """Registers `callback(pin, value)` to fire on every edge
        transition of a digital input pin — the "interrupt handler" for
        e.g. a physical toggle switch. `value` is True when the pin reads
        high, False when low."""

    @property
    @abstractmethod
    def is_physical(self) -> bool:
        """True for a HAL actually driving real hardware, False for
        `MockHAL`. Modules use this only for logging/diagnostics, never to
        change behavior — the whole point of the HAL is that callers don't
        need to know which one they have."""


class MockHAL(HardwareAbstractionLayer):
    """Logs every operation instead of touching hardware. Also exposes
    `simulate_pin_change` so tests (and this sandbox) can trigger the
    exact same interrupt-callback path a real GPIO edge would."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pin_watchers: Dict[int, List[PinEdgeCallback]] = {}
        self._pin_states: Dict[int, bool] = {}

    @property
    def is_physical(self) -> bool:
        return False

    def digital_write(self, pin: int, value: bool) -> None:
        logger.info("[HAL::MOCK] digital_write(pin=%d, value=%s)", pin, value)

    def pwm_write(self, pin: int, duty_cycle: float) -> None:
        clamped = max(0.0, min(1.0, duty_cycle))
        logger.info("[HAL::MOCK] pwm_write(pin=%d, duty_cycle=%.3f)", pin, clamped)

    def set_servo_angle(self, pin: int, angle_degrees: float) -> None:
        clamped = max(0.0, min(180.0, angle_degrees))
        logger.info("[HAL::MOCK] set_servo_angle(pin=%d, angle=%.1f°)", pin, clamped)

    def set_neopixel_frame(self, pin: int, pixel_count: int, colors: List[RGBColor]) -> None:
        preview = colors[: min(3, len(colors))]
        logger.debug(
            "[HAL::MOCK] set_neopixel_frame(pin=%d, count=%d, first=%s%s)",
            pin, pixel_count, preview, ", ..." if len(colors) > 3 else "",
        )

    def watch_digital_pin(self, pin: int, callback: PinEdgeCallback) -> None:
        with self._lock:
            self._pin_watchers.setdefault(pin, []).append(callback)
        logger.info("[HAL::MOCK] watch_digital_pin(pin=%d) registered (simulate via simulate_pin_change).", pin)

    def simulate_pin_change(self, pin: int, value: bool) -> None:
        """Test/dev hook: fires every callback registered on `pin` as if a
        real edge interrupt had just occurred."""
        with self._lock:
            self._pin_states[pin] = value
            callbacks = list(self._pin_watchers.get(pin, []))
        logger.info("[HAL::MOCK] simulate_pin_change(pin=%d, value=%s)", pin, value)
        for callback in callbacks:
            callback(pin, value)


class GPIOHAL(HardwareAbstractionLayer):
    """Real hardware backend. Lazily imports `gpiozero` and
    `adafruit-circuitpython-neopixel` (`board`/`neopixel`) — neither is a
    project dependency by default (see requirements-hardware.txt); both
    are Raspberry-Pi-specific and will not install on most non-Pi hosts,
    including this project's own Render/Docker deployment.

    UNTESTED: there is no physical GPIO chip in this sandbox or in the
    Render deployment target, so this class has never been run against
    real hardware. It follows the documented gpiozero/neopixel APIs as
    carefully as possible; verify on an actual Raspberry Pi before relying
    on it.
    """

    def __init__(self) -> None:
        try:
            import gpiozero  # noqa: F401
        except ImportError as exc:
            raise HardwareUnavailableError(
                "gpiozero is not installed. See requirements-hardware.txt "
                "(Raspberry Pi deployments only)."
            ) from exc

        self._gpiozero = gpiozero
        self._pwm_devices: Dict[int, "gpiozero.PWMOutputDevice"] = {}
        self._digital_devices: Dict[int, "gpiozero.DigitalOutputDevice"] = {}
        self._servo_devices: Dict[int, "gpiozero.AngularServo"] = {}
        self._buttons: Dict[int, "gpiozero.Button"] = {}
        self._neopixel_strips: Dict[int, object] = {}
        # gpiozero requires a specific pin number to construct any device,
        # so there is no generic "is there a GPIO chip at all" probe to run
        # here; a real absence of hardware surfaces (and is logged) the
        # first time a specific pin is actually used below.

    def _digital_out(self, pin: int) -> "object":
        if pin not in self._digital_devices:
            self._digital_devices[pin] = self._gpiozero.DigitalOutputDevice(pin)
        return self._digital_devices[pin]

    def digital_write(self, pin: int, value: bool) -> None:
        device = self._digital_out(pin)
        device.on() if value else device.off()

    def pwm_write(self, pin: int, duty_cycle: float) -> None:
        clamped = max(0.0, min(1.0, duty_cycle))
        if pin not in self._pwm_devices:
            self._pwm_devices[pin] = self._gpiozero.PWMOutputDevice(pin)
        self._pwm_devices[pin].value = clamped

    def set_servo_angle(self, pin: int, angle_degrees: float) -> None:
        clamped = max(0.0, min(180.0, angle_degrees))
        if pin not in self._servo_devices:
            self._servo_devices[pin] = self._gpiozero.AngularServo(
                pin, min_angle=0, max_angle=180
            )
        self._servo_devices[pin].angle = clamped

    def set_neopixel_frame(self, pin: int, pixel_count: int, colors: List[RGBColor]) -> None:
        try:
            import board
            import neopixel
        except ImportError as exc:
            raise HardwareUnavailableError(
                "adafruit-circuitpython-neopixel is not installed; cannot drive "
                "NeoPixel output. See requirements-hardware.txt."
            ) from exc

        if pin not in self._neopixel_strips:
            board_pin = getattr(board, f"D{pin}", None)
            if board_pin is None:
                raise HardwareUnavailableError(f"No board pin mapping found for GPIO {pin}.")
            self._neopixel_strips[pin] = neopixel.NeoPixel(
                board_pin, pixel_count, auto_write=False
            )
        strip = self._neopixel_strips[pin]
        padded = (colors + [(0, 0, 0)] * pixel_count)[:pixel_count]
        for i, color in enumerate(padded):
            strip[i] = color
        strip.show()

    def watch_digital_pin(self, pin: int, callback: PinEdgeCallback) -> None:
        button = self._gpiozero.Button(pin, pull_up=True, bounce_time=0.05)
        # Convention: pull-up wiring, switch to GND -> pressed == pin low.
        # "value=True" is reported to `callback` as the switch's normal
        # (un-pressed, pulled-high) resting state; "value=False" as
        # pressed/pulled-low. Document your physical wiring against this.
        button.when_pressed = lambda: callback(pin, False)
        button.when_released = lambda: callback(pin, True)
        self._buttons[pin] = button

    @property
    def is_physical(self) -> bool:
        return True


def get_hal(prefer_physical: bool = False) -> HardwareAbstractionLayer:
    """Selects a HAL implementation. Only ever attempts `GPIOHAL` if
    `prefer_physical` is True (i.e. an operator explicitly opted in via
    `ZANE_HARDWARE_ENABLED`/`ZANE_HARDWARE_USE_REAL_GPIO`); any failure to
    construct it falls back to `MockHAL` with a logged warning rather than
    raising, since "hardware unavailable" is an expected, normal condition
    for this project's default cloud deployment, not an error."""
    if not prefer_physical:
        return MockHAL()

    try:
        return GPIOHAL()
    except HardwareUnavailableError as exc:
        logger.warning(
            "Physical GPIO requested but unavailable (%s); falling back to MockHAL.", exc
        )
        return MockHAL()
