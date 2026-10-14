import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from threading import Lock


@dataclass(slots=True, frozen=True)
class DeduplicationResult:
    is_duplicate: bool
    ttl_seconds: float = 0.0


class Deduplicator(ABC):
    @abstractmethod
    def check_and_mark(self, key: str, ttl_seconds: int = 300) -> DeduplicationResult:
        pass

    @abstractmethod
    def cleanup_expired(self) -> int:
        pass


class InMemoryDeduplicator(Deduplicator):
    def __init__(self) -> None:
        self._seen: dict[str, float] = {}
        self._lock = Lock()

    def check_and_mark(self, key: str, ttl_seconds: int = 300) -> DeduplicationResult:
        now = time.time()
        expiry = now + ttl_seconds

        with self._lock:
            self._cleanup_expired_locked(now)
            if key in self._seen:
                return DeduplicationResult(is_duplicate=True, ttl_seconds=self._seen[key] - now)
            self._seen[key] = expiry
            return DeduplicationResult(is_duplicate=False)

    def cleanup_expired(self) -> int:
        now = time.time()
        with self._lock:
            return self._cleanup_expired_locked(now)

    def _cleanup_expired_locked(self, now: float) -> int:
        expired_keys = [k for k, v in self._seen.items() if v <= now]
        for k in expired_keys:
            del self._seen[k]
        return len(expired_keys)