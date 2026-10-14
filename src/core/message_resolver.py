from dataclasses import dataclass

from aiogram.types import Message, MessageEntity
from aiogram.utils.text_decorations import add_surrogates, remove_surrogates

from core.models import RewriteError, RewriteRequest


@dataclass(slots=True, frozen=True)
class ParsedMessage:
    source_text: str
    user_instruction: str | None


class MessageParseError(RewriteError):
    pass


def get_text_and_entities(message: Message) -> tuple[str, tuple[MessageEntity, ...]]:
    if message.text is not None:
        return message.text, tuple(message.entities or ())
    return message.caption or "", tuple(message.caption_entities or ())


def get_text_content(message: Message) -> str:
    text, _ = get_text_and_entities(message)
    return text


def find_bot_mention(
    message: Message,
    bot_username: str,
    bot_id: int | None = None,
) -> MessageEntity | None:
    text, entities = get_text_and_entities(message)
    if not text or not entities:
        return None

    expected = f"@{bot_username.lstrip('@').lower()}"
    for entity in entities:
        if entity.type == "mention":
            # entity.offset/length are UTF-16 code units; extract_from handles surrogates
            if entity.extract_from(text).lower() == expected:
                return entity
        elif entity.type == "text_mention" and bot_id is not None:
            if entity.user is not None and entity.user.id == bot_id:
                return entity
    return None


def remove_mention(text: str, entity: MessageEntity) -> str:
    raw = add_surrogates(text)
    start = entity.offset * 2
    end = (entity.offset + entity.length) * 2
    before = remove_surrogates(raw[:start])
    after = remove_surrogates(raw[end:])

    # fix only the double space left at the cut; newlines and other spacing are preserved
    if before.endswith(" ") and after.startswith(" "):
        after = after[1:]
    return (before + after).strip()


def parse_rewrite_request(
    message: Message,
    bot_username: str,
    bot_id: int | None = None,
) -> ParsedMessage | None:
    """Returns None when the bot is not mentioned - the caller must stay silent."""
    mention_entity = find_bot_mention(message, bot_username, bot_id)
    if mention_entity is None:
        return None

    replied = message.reply_to_message

    if replied is not None:
        # the quoted message is always the source; current text is only an instruction
        replied_text = get_text_content(replied)
        if not replied_text:
            raise MessageParseError(
                message="Replied message has no text or caption",
                user_facing="Можно переписывать только текстовые сообщения или подписи к медиа.",
            )
        # a mention inside the quote is not part of the text to rewrite
        quoted_mention = find_bot_mention(replied, bot_username, bot_id)
        if quoted_mention is not None:
            stripped = remove_mention(replied_text, quoted_mention)
            if not stripped:
                raise MessageParseError(
                    message="Replied message contains nothing but a mention",
                    user_facing="В сообщении, на которое вы ответили, есть только упоминание бота.",
                )
            replied_text = stripped

        current_text, _ = get_text_and_entities(message)
        instruction = remove_mention(current_text, mention_entity) if current_text else ""
        return ParsedMessage(
            source_text=replied_text,
            user_instruction=instruction or None,
        )

    own_text, _ = get_text_and_entities(message)
    source_text = remove_mention(own_text, mention_entity) if own_text else ""
    if not source_text:
        raise MessageParseError(
            message="Bare mention without text",
            user_facing="Напишите текст после упоминания или ответьте на сообщение с упоминанием.",
        )
    return ParsedMessage(source_text=source_text, user_instruction=None)


def build_rewrite_request(
    message: Message,
    bot_username: str,
    bot_id: int | None = None,
) -> RewriteRequest | None:
    parsed = parse_rewrite_request(message, bot_username, bot_id)
    if parsed is None:
        return None
    return RewriteRequest(
        source_text=parsed.source_text,
        user_instruction=parsed.user_instruction,
        chat_id=message.chat.id,
        message_id=message.message_id,
        message_thread_id=message.message_thread_id,
        user_id=message.from_user.id if message.from_user else 0,
    )
