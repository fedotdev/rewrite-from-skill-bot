from typing import TYPE_CHECKING

from core.deduplication import Deduplicator
from core.message_resolver import build_rewrite_request
from core.models import RewriteRequest, RewriteResult
from core.rate_limiter import CompositeRateLimiter, RateLimitResult
from core.style_loader import load_style_instructions
from llm.client import LLMClient

if TYPE_CHECKING:
    from aiogram.types import Message


class RewriteService:
    def __init__(
        self,
        llm_client: LLMClient,
        deduplicator: Deduplicator,
        rate_limiter: CompositeRateLimiter,
        allowed_chat_ids: list[int],
    ) -> None:
        self._llm = llm_client
        self._dedup = deduplicator
        self._rate_limiter = rate_limiter
        self._allowed_chats = set(allowed_chat_ids)
        self._style = load_style_instructions()
        self._bot_username: str = ""

    def set_bot_username(self, username: str) -> None:
        self._bot_username = username

    def is_allowed_chat(self, chat_id: int) -> bool:
        if not self._allowed_chats:
            return True
        return chat_id in self._allowed_chats

    def check_rate_limits(self, user_id: int, chat_id: int) -> tuple[RateLimitResult, RateLimitResult]:
        return self._rate_limiter.check(user_id, chat_id)

    def check_deduplication(self, chat_id: int, message_id: int) -> bool:
        result = self._dedup.check_and_mark(f"msg:{chat_id}:{message_id}")
        return result.is_duplicate

    async def process(self, request: RewriteRequest) -> RewriteResult:
        return await self._llm.rewrite(request, self._style)

    async def handle_message(self, message: "Message") -> RewriteResult:
        request = build_rewrite_request(message, self._bot_username)
        return await self.process(request)
