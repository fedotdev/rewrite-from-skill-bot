from aiogram import Router
from aiogram.filters import Filter
from aiogram.types import Message

from core.message_resolver import build_rewrite_request
from core.models import RewriteError
from core.service import RewriteService
from logging_config import get_logger

logger = get_logger(__name__)

router = Router()


class PrivateCommandFilter(Filter):
    async def __call__(self, message: Message, service: RewriteService) -> bool:
        if message.chat.type != "private":
            return False
        text = message.text or message.caption or ""
        command = text.split(maxsplit=1)[0].split("@", 1)[0] if text else ""
        user_id = message.from_user.id if message.from_user else message.chat.id
        return command in {"/start", "/help", "/privacy"} and service.is_allowed_chat(
            user_id, message.chat.type
        )


HELP_TEXT = (
    "Я переписываю текст по вашему запросу. Пришлите текст или ответьте на сообщение "
    "с инструкцией. Текст отправляется стороннему LLM-провайдеру, но в логи не записывается."
)
PRIVACY_TEXT = (
    "Текст вашего запроса отправляется стороннему LLM-провайдеру для обработки. "
    "В логи приложения текст не записывается."
)


@router.message(PrivateCommandFilter())
async def handle_help(message: Message) -> None:
    command = (message.text or message.caption or "").split(maxsplit=1)[0].split("@", 1)[0]
    await message.reply(PRIVACY_TEXT if command == "/privacy" else HELP_TEXT)


class AccessFilter(Filter):
    async def __call__(self, message: Message, service: RewriteService) -> bool:
        if message.chat.type not in {"private", "group", "supergroup"}:
            return False
        identifier = (
            message.from_user.id
            if message.chat.type == "private" and message.from_user
            else message.chat.id
        )
        return service.is_allowed_chat(identifier, message.chat.type)


class MentionFilter(Filter):
    async def __call__(self, message: Message, service: RewriteService) -> bool:
        return message.chat.type == "private" or service.find_mention(message) is not None


class NotCommandFilter(Filter):
    async def __call__(self, message: Message) -> bool:
        text = message.text or message.caption
        return not (text or "").startswith("/")


@router.message(AccessFilter(), MentionFilter(), NotCommandFilter())
async def handle_rewrite(message: Message, service: RewriteService) -> None:
    # parse first: invalid or empty requests must not consume rate limit
    try:
        request = build_rewrite_request(
            message,
            service.bot_username,
            service.bot_id,
            private=message.chat.type == "private",
            max_source_chars=service.max_source_chars,
        )
        if request is None:
            return
    except RewriteError as e:
        await message.reply(e.user_facing)
        return

    if service.check_deduplication(message.chat.id, message.message_id):
        logger.info("Duplicate message ignored", chat_id=message.chat.id, message_id=message.message_id)
        return

    user_id = message.from_user.id if message.from_user else 0
    if message.chat.type == "private":
        user_result, chat_result = service.check_private_rate_limits(user_id)
    else:
        user_result, chat_result = service.check_rate_limits(user_id, message.chat.id)
    if not user_result.allowed:
        throttle_key = (
            f"private-user:{user_id}"
            if message.chat.type == "private"
            else f"user:{message.chat.id}:{user_id}"
        )
        if service.should_warn(throttle_key, user_result.retry_after_seconds):
            await message.reply(
                f"Слишком много запросов. Попробуйте через {int(user_result.retry_after_seconds)} сек."
            )
        return
    if not chat_result.allowed:
        throttle_key = (
            "private-global" if message.chat.type == "private" else f"chat:{message.chat.id}"
        )
        if service.should_warn(throttle_key, chat_result.retry_after_seconds):
            warning = (
                "Чат достиг лимита."
                if message.chat.type != "private"
                else "Лимит запросов исчерпан."
            )
            await message.reply(
                f"{warning} Попробуйте через {int(chat_result.retry_after_seconds)} сек."
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
