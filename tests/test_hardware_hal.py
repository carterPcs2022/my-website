from zane.hardware.hal import GPIOHAL, HardwareUnavailableError, MockHAL, get_hal


def test_mock_hal_is_not_physical():
    hal = MockHAL()
    assert hal.is_physical is False


def test_mock_hal_operations_do_not_raise():
    hal = MockHAL()
    hal.digital_write(17, True)
    hal.pwm_write(18, 0.5)
    hal.pwm_write(18, 5.0)  # out of range, must clamp not raise
    hal.set_servo_angle(22, 45.0)
    hal.set_servo_angle(22, 999.0)  # out of range, must clamp not raise
    hal.set_neopixel_frame(18, 3, [(0, 0, 255)] * 3)


def test_mock_hal_watch_digital_pin_and_simulate():
    hal = MockHAL()
    events = []
    hal.watch_digital_pin(4, lambda pin, value: events.append((pin, value)))

    hal.simulate_pin_change(4, False)
    hal.simulate_pin_change(4, True)

    assert events == [(4, False), (4, True)]


def test_mock_hal_multiple_watchers_on_same_pin():
    hal = MockHAL()
    calls_a, calls_b = [], []
    hal.watch_digital_pin(4, lambda pin, value: calls_a.append(value))
    hal.watch_digital_pin(4, lambda pin, value: calls_b.append(value))

    hal.simulate_pin_change(4, False)

    assert calls_a == [False]
    assert calls_b == [False]


def test_get_hal_defaults_to_mock():
    hal = get_hal(prefer_physical=False)
    assert isinstance(hal, MockHAL)


def test_get_hal_falls_back_to_mock_when_gpiozero_unavailable():
    # gpiozero is not installed in this environment (by design — see
    # requirements-hardware.txt), so this exercises the real fallback path.
    hal = get_hal(prefer_physical=True)
    assert isinstance(hal, MockHAL)


def test_gpiohal_raises_hardware_unavailable_without_gpiozero():
    try:
        import gpiozero  # noqa: F401

        import pytest

        pytest.skip("gpiozero is installed in this environment; fallback path not exercised.")
    except ImportError:
        pass

    import pytest

    with pytest.raises(HardwareUnavailableError):
        GPIOHAL()
