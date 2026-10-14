from dataclasses import dataclass

from aiogram.types import Message, MessageEntity

from core.models import RewriteError, RewriteRequest


@dataclass(slots=True, frozen=True)
class ParsedMessage:
    source_text: str
    user_instruction: str | None
    has_reply: bool


class MessageParseError(RewriteError):
    pass


def get_text_and_entities(message: Message) -> tuple[str | None, tuple[MessageEntity, ...]]:
    if message.text is not None:
        return message.text, tuple(message.entities or ())
    return message.caption, tuple(message.caption_entities or ())


def extract_mention_entity(message: Message, bot_username: str) -> MessageEntity | None:
    text, entities = get_text_and_entities(message)
    if not text:
        return None
    expected = f"@{bot_username.lower()}"
    for entity in entities:
        if entity.type == "mention" and text[entity.offset : entity.offset + entity.length].lower() == expected:
            return entity
    return None


def remove_mention(text: str, entity: MessageEntity) -> str:
    before = text[:entity.offset]
    after = text[entity.offset + entity.length:]
    result = before + after
    return " ".join(result.split())


def get_text_content(message: Message) -> str | None:
    if message.text:
        return message.text
    if message.caption:
        return message.caption
    return None


def parse_rewrite_request(
    message: Message,
    bot_username: str,
) -> ParsedMessage:
    mention_entity = extract_mention_entity(message, bot_username)
    has_reply = message.reply_to_message is not None

    if not mention_entity and not has_reply:
        raise MessageParseError(
            message="No mention and no reply",
            user_facing="Напишите @bot <текст> или ответьте на сообщение с упоминанием @bot.",
        )

    if mention_entity and not has_reply:
        # @bot текст — переписываем текст после упоминания
        full_text = get_text_content(message)
        if not full_text:
            raise MessageParseError(
                message="Mention without text content",
                user_facing="После @bot должен быть текст для переписывания.",
            )
        source_text = remove_mention(full_text, mention_entity)
        if not source_text:
            raise MessageParseError(
                message="Empty text after mention",
                user_facing="Текст после @bot не может быть пустым.",
            )
        return ParsedMessage(
            source_text=source_text,
            user_instruction=None,
            has_reply=False,
        )

    # Has reply
    replied = message.reply_to_message
    assert replied is not None
    replied_text = get_text_content(replied)
    if not replied_text:
        raise MessageParseError(
            message="Replied message has no text or caption",
            user_facing="Можно переписывать только текстовые сообщения или подписи к медиа.",
        )

    user_instruction = None
    if mention_entity:
        full_text = get_text_content(message)
        if full_text:
            instruction = remove_mention(full_text, mention_entity)
            if instruction:
                user_instruction = instruction

    return ParsedMessage(
        source_text=replied_text,
        user_instruction=user_instruction,
        has_reply=True,
    )


def build_rewrite_request(message: Message, bot_username: str) -> RewriteRequest:
    parsed = parse_rewrite_request(message, bot_username)
    return RewriteRequest(
        source_text=parsed.source_text,
        user_instruction=parsed.user_instruction,
        chat_id=message.chat.id,
        message_id=message.message_id,
        message_thread_id=message.message_thread_id,
        user_id=message.from_user.id if message.from_user else 0,
    )
