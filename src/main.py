import asyncio
import sys

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties

from config import get_settings
from core.deduplication import InMemoryDeduplicator
from core.rate_limiter import CompositeRateLimiter, InMemoryRateLimiter
from core.service import RewriteService
from handlers.rewrite import router as rewrite_router
from llm.client import LLMClient
from logging_config import get_logger, setup_logging

logger = get_logger(__name__)


async def main() -> None:
    settings = get_settings()
    setup_logging()

    bot = Bot(token=settings.bot_token, default=DefaultBotProperties())

    bot_info = await bot.get_me()
    if not bot_info.username:
        await bot.session.close()
        raise RuntimeError("Telegram did not return a username for this bot; cannot detect mentions")

    llm_client = LLMClient()
    deduplicator = InMemoryDeduplicator()
    user_limiter = InMemoryRateLimiter()
    chat_limiter = InMemoryRateLimiter()
    rate_limiter = CompositeRateLimiter(
        user_limiter=user_limiter,
        chat_limiter=chat_limiter,
        user_limit=settings.rate_limit_per_user_per_minute,
        chat_limit=settings.rate_limit_per_chat_per_minute,
    )

    service = RewriteService(
        llm_client=llm_client,
        deduplicator=deduplicator,
        rate_limiter=rate_limiter,
        allowed_chat_ids=settings.allowed_chat_ids,
        bot_username=bot_info.username,
        bot_id=bot_info.id,
    )

    if not service.has_allowed_chats:
        logger.warning(
            "ALLOWED_CHAT_IDS is empty - the bot will not respond anywhere",
            allowed_chats=[],
        )

    dp = Dispatcher()
    dp["service"] = service
    dp.include_router(rewrite_router)

    logger.info("Bot started", username=bot_info.username, allowed_chats=settings.allowed_chat_ids)

    try:
        await dp.start_polling(bot)
    finally:
        await llm_client.close()
        await bot.session.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped")
        sys.exit(0)
