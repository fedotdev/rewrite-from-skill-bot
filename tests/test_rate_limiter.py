import time

from core.rate_limiter import CompositeRateLimiter, InMemoryRateLimiter


def test_rate_limiter_allows_within_limit():
    limiter = InMemoryRateLimiter()
    for i in range(5):
        result = limiter.check_limit("user:1", 5)
        assert result.allowed is True
        assert result.current_count == i + 1


def test_rate_limiter_blocks_over_limit():
    limiter = InMemoryRateLimiter()
    for _ in range(5):
        limiter.check_limit("user:1", 5)
    result = limiter.check_limit("user:1", 5)
    assert result.allowed is False
    assert result.retry_after_seconds > 0


def test_rate_limiter_resets_after_window():
    limiter = InMemoryRateLimiter()
    for _ in range(5):
        limiter.check_limit("user:1", 5, window_seconds=1)
    time.sleep(1.1)
    result = limiter.check_limit("user:1", 5, window_seconds=1)
    assert result.allowed is True


def test_rate_limiter_separate_keys():
    limiter = InMemoryRateLimiter()
    for _ in range(3):
        limiter.check_limit("user:1", 3)
    result1 = limiter.check_limit("user:1", 3)
    result2 = limiter.check_limit("user:2", 3)
    assert result1.allowed is False
    assert result2.allowed is True


def test_rate_limiter_reset():
    limiter = InMemoryRateLimiter()
    limiter.check_limit("user:1", 1)
    limiter.reset("user:1")
    result = limiter.check_limit("user:1", 1)
    assert result.allowed is True


def test_composite_rate_limiter():
    user_limiter = InMemoryRateLimiter()
    chat_limiter = InMemoryRateLimiter()
    composite = CompositeRateLimiter(user_limiter, chat_limiter, user_limit=2, chat_limit=3)

    user_res, chat_res = composite.check(1, 100)
    assert user_res.allowed is True
    assert chat_res.allowed is True


def test_composite_rate_limiter_user_blocked():
    user_limiter = InMemoryRateLimiter()
    chat_limiter = InMemoryRateLimiter()
    composite = CompositeRateLimiter(user_limiter, chat_limiter, user_limit=1, chat_limit=10)

    composite.check(1, 100)
    user_res, chat_res = composite.check(1, 100)
    assert user_res.allowed is False
    assert chat_res.allowed is True


def test_composite_rate_limiter_chat_blocked():
    user_limiter = InMemoryRateLimiter()
    chat_limiter = InMemoryRateLimiter()
    composite = CompositeRateLimiter(user_limiter, chat_limiter, user_limit=10, chat_limit=1)

    composite.check(1, 100)
    user_res, chat_res = composite.check(2, 100)
    assert user_res.allowed is True
    assert chat_res.allowed is False
