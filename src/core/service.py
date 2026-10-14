from typing import TYPE_CHECKING

from core.deduplication import Deduplicator
from core.message_resolver import MessageParseError, build_rewrite_request, find_bot_mention
from core.models import RewriteRequest, RewriteResult
from core.rate_limiter import CompositeRateLimiter, RateLimitResult, WarningThrottle
from core.style_loader import StyleLoader
from llm.client import LLMClient

if TYPE_CHECKING:
    from aiogram.types import Message, MessageEntity


class RewriteService:
    def __init__(
        self,
        llm_client: LLMClient,
        deduplicator: Deduplicator,
        rate_limiter: CompositeRateLimiter,
        allowed_chat_ids: list[int],
        bot_username: str,
        bot_id: int,
        style_loader: StyleLoader | None = None,
        warning_throttle: WarningThrottle | None = None,
    ) -> None:
        if not bot_username:
            raise ValueError("bot_username is required")
        self._llm = llm_client
        self._dedup = deduplicator
        self._rate_limiter = rate_limiter
        # fail closed: an empty allowlist admits nobody
        self._allowed_chats = frozenset(allowed_chat_ids)
        self._bot_username = bot_username
        self._bot_id = bot_id
        self._style_loader = style_loader or StyleLoader()
        self._warning_throttle = warning_throttle or WarningThrottle()

    @property
    def has_allowed_chats(self) -> bool:
        return bool(self._allowed_chats)

    @property
    def bot_username(self) -> str:
        return self._bot_username

    @property
    def bot_id(self) -> int:
        return self._bot_id

    def find_mention(self, message: "Message") -> "MessageEntity | None":
        return find_bot_mention(message, self._bot_username, self._bot_id)

    def is_allowed_chat(self, chat_id: int) -> bool:
        return chat_id in self._allowed_chats

    def check_rate_limits(self, user_id: int, chat_id: int) -> tuple[RateLimitResult, RateLimitResult]:
        return self._rate_limiter.check(user_id, chat_id)

    def check_deduplication(self, chat_id: int, message_id: int) -> bool:
        result = self._dedup.check_and_mark(f"msg:{chat_id}:{message_id}")
        return result.is_duplicate

    def should_warn_user_limit(self, chat_id: int, user_id: int, retry_after: float) -> bool:
        return self._warning_throttle.should_warn(f"user:{chat_id}:{user_id}", retry_after)

    def should_warn_chat_limit(self, chat_id: int, retry_after: float) -> bool:
        return self._warning_throttle.should_warn(f"chat:{chat_id}", retry_after)

    async def process(self, request: RewriteRequest) -> RewriteResult:
        return await self._llm.rewrite(request, self._style_loader.load())

    async def handle_message(self, message: "Message") -> RewriteResult:
        request = build_rewrite_request(message, self._bot_username, self._bot_id)
        if request is None:
            raise MessageParseError(
                message="Bot not mentioned",
                user_facing="Упоминание бота не найдено.",
            )
        return await self.process(request)
