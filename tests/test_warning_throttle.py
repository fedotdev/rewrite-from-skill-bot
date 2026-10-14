
from core.rate_limiter import WarningThrottle


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_first_call_is_allowed() -> None:
    throttle = WarningThrottle(FakeClock())

    assert throttle.should_warn("user:1", 30.0) is True


def test_repeat_inside_window_is_muted() -> None:
    throttle = WarningThrottle(FakeClock())

    assert throttle.should_warn("user:1", 30.0) is True
    assert throttle.should_warn("user:1", 30.0) is False
    assert throttle.should_warn("user:1", 30.0) is False


def test_allowed_again_after_window() -> None:
    clock = FakeClock()
    throttle = WarningThrottle(clock)

    assert throttle.should_warn("user:1", 30.0) is True
    clock.advance(29.9)
    assert throttle.should_warn("user:1", 30.0) is False
    clock.advance(0.2)
    assert throttle.should_warn("user:1", 30.0) is True


def test_keys_are_independent() -> None:
    throttle = WarningThrottle(FakeClock())

    assert throttle.should_warn("user:1", 30.0) is True
    assert throttle.should_warn("user:2", 30.0) is True
    assert throttle.should_warn("chat:-100", 30.0) is True
    assert throttle.should_warn("user:1", 30.0) is False


def test_zero_retry_after_warns_again_immediately() -> None:
    clock = FakeClock()
    throttle = WarningThrottle(clock)

    assert throttle.should_warn("user:1", 0.0) is True
    clock.advance(0.01)
    assert throttle.should_warn("user:1", 0.0) is True


def test_negative_retry_after_is_clamped() -> None:
    clock = FakeClock()
    throttle = WarningThrottle(clock)

    assert throttle.should_warn("user:1", -5.0) is True
    clock.advance(0.01)
    assert throttle.should_warn("user:1", -5.0) is True


def test_window_is_refreshed_on_each_warning() -> None:
    clock = FakeClock()
    throttle = WarningThrottle(clock)

    assert throttle.should_warn("user:1", 10.0) is True
    clock.advance(11)
    assert throttle.should_warn("user:1", 10.0) is True
    clock.advance(5)
    assert throttle.should_warn("user:1", 10.0) is False


def test_reset_single_key() -> None:
    throttle = WarningThrottle(FakeClock())
    throttle.should_warn("user:1", 30.0)

    throttle.reset("user:1")

    assert throttle.should_warn("user:1", 30.0) is True


def test_reset_all() -> None:
    throttle = WarningThrottle(FakeClock())
    throttle.should_warn("user:1", 30.0)
    throttle.should_warn("user:2", 30.0)

    throttle.reset()

    assert throttle.should_warn("user:1", 30.0) is True
    assert throttle.should_warn("user:2", 30.0) is True


def test_default_clock_is_monotonic() -> None:
    throttle = WarningThrottle()

    assert throttle.should_warn("user:1", 30.0) is True
    assert throttle.should_warn("user:1", 30.0) is False
