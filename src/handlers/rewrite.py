from aiogram import Router
from aiogram.filters import Filter
from aiogram.types import Message

from core.models import RewriteError
from core.service import RewriteService
from logging_config import get_logger

logger = get_logger(__name__)

router = Router()


class AllowedChatFilter(Filter):
    async def __call__(self, message: Message, service: RewriteService) -> bool:
        return service.is_allowed_chat(message.chat.id)


class NotCommandFilter(Filter):
    async def __call__(self, message: Message) -> bool:
        if not message.text:
            return True
        return not message.text.startswith("/")


@router.message(AllowedChatFilter(), NotCommandFilter())
async def handle_rewrite(message: Message, service: RewriteService) -> None:
    if service.check_deduplication(message.chat.id, message.message_id):
        logger.info("Duplicate message ignored", chat_id=message.chat.id, message_id=message.message_id)
        return

    user_result, chat_result = service.check_rate_limits(
        message.from_user.id if message.from_user else 0,
        message.chat.id,
    )

    if not user_result.allowed:
        await message.reply(
            f"Слишком много запросов. Попробуйте через {int(user_result.retry_after_seconds)} сек."
        )
        return

    if not chat_result.allowed:
        await message.reply(
            f"Чат достиг лимита. Попробуйте через {int(chat_result.retry_after_seconds)} сек."
        )
        return

    try:
        result = await service.handle_message(message)
        reply_text = result.text
        if result.warning:
            reply_text += f"\n\n⚠️ {result.warning}"
        await message.reply(reply_text)
        logger.info("Rewrite successful", chat_id=message.chat.id, user_id=message.from_user.id if message.from_user else 0)

    except RewriteError as e:
        logger.warning("Rewrite error", error=e.message, chat_id=message.chat.id)
        await message.reply(e.user_facing)
    except Exception:
        logger.exception("Unexpected error in handler", chat_id=message.chat.id)
        await message.reply("Произошла ошибка. Попробуйте позже.")
