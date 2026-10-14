import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from threading import Lock


@dataclass(slots=True)
class RateLimitResult:
    allowed: bool
    retry_after_seconds: float = 0.0
    current_count: int = 0
    limit: int = 0


class RateLimiter(ABC):
    @abstractmethod
    def check_limit(self, key: str, limit: int, window_seconds: int = 60) -> RateLimitResult:
        pass

    @abstractmethod
    def reset(self, key: str) -> None:
        pass


class InMemoryRateLimiter(RateLimiter):
    def __init__(self) -> None:
        self._buckets: dict[str, list[float]] = defaultdict(list)
        self._lock = Lock()

    def check_limit(self, key: str, limit: int, window_seconds: int = 60) -> RateLimitResult:
        now = time.time()
        window_start = now - window_seconds

        with self._lock:
            bucket = self._buckets[key]
            while bucket and bucket[0] < window_start:
                bucket.pop(0)

            current_count = len(bucket)

            if current_count >= limit:
                retry_after = bucket[0] + window_seconds - now if bucket else float(window_seconds)
                return RateLimitResult(
                    allowed=False,
                    retry_after_seconds=max(0.0, retry_after),
                    current_count=current_count,
                    limit=limit,
                )

            bucket.append(now)
            return RateLimitResult(
                allowed=True,
                current_count=current_count + 1,
                limit=limit,
            )

    def reset(self, key: str) -> None:
        with self._lock:
            self._buckets.pop(key, None)


@dataclass(slots=True)
class CompositeRateLimiter:
    user_limiter: RateLimiter
    chat_limiter: RateLimiter
    user_limit: int
    chat_limit: int

    def check(self, user_id: int, chat_id: int) -> tuple[RateLimitResult, RateLimitResult]:
        user_result = self.user_limiter.check_limit(f"user:{user_id}", self.user_limit)
        chat_result = self.chat_limiter.check_limit(f"chat:{chat_id}", self.chat_limit)
        return user_result, chat_result