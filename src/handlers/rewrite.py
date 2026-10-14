from aiogram import Router
from aiogram.filters import Filter
from aiogram.types import Message

from core.message_resolver import build_rewrite_request
from core.models import RewriteError
from core.service import RewriteService
from logging_config import get_logger

logger = get_logger(__name__)

router = Router()


class MentionFilter(Filter):
    """Bot reacts only when it is actually mentioned."""

    async def __call__(self, message: Message, service: RewriteService) -> bool:
        return service.find_mention(message) is not None


class AllowedChatFilter(Filter):
    async def __call__(self, message: Message, service: RewriteService) -> bool:
        return service.is_allowed_chat(message.chat.id)


class NotCommandFilter(Filter):
    async def __call__(self, message: Message) -> bool:
        text = message.text or message.caption
        return not (text or "").startswith("/")


@router.message(MentionFilter(), AllowedChatFilter(), NotCommandFilter())
async def handle_rewrite(message: Message, service: RewriteService) -> None:
    # parse first: invalid or empty requests must not consume rate limit
    try:
        request = build_rewrite_request(message, service.bot_username, service.bot_id)
        if request is None:
            return
    except RewriteError as e:
        await message.reply(e.user_facing)
        return

    if service.check_deduplication(message.chat.id, message.message_id):
        logger.info("Duplicate message ignored", chat_id=message.chat.id, message_id=message.message_id)
        return

    user_id = message.from_user.id if message.from_user else 0
    user_result, chat_result = service.check_rate_limits(user_id, message.chat.id)
    if not user_result.allowed:
        if service.should_warn_user_limit(
            message.chat.id, user_id, user_result.retry_after_seconds
        ):
            await message.reply(
                f"Слишком много запросов. Попробуйте через {int(user_result.retry_after_seconds)} сек."
            )
        return
    if not chat_result.allowed:
        if service.should_warn_chat_limit(message.chat.id, chat_result.retry_after_seconds):
            await message.reply(
                f"Чат достиг лимита. Попробуйте через {int(chat_result.retry_after_seconds)} сек."
            )
        return

    try:
        result = await service.process(request)
        reply_text = result.text
        if result.warning:
            reply_text += f"\n\n⚠️ {result.warning}"
        await message.reply(reply_text)
        logger.info(
            "Rewrite successful",
            chat_id=message.chat.id,
            user_id=message.from_user.id if message.from_user else 0,
        )
    except RewriteError as e:
        logger.warning("Rewrite error", error=e.message, chat_id=message.chat.id)
        await message.reply(e.user_facing)
    except Exception:
        logger.exception("Unexpected error in handler", chat_id=message.chat.id)
        await message.reply("Произошла ошибка. Попробуйте позже.")
