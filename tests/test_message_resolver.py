import pytest
from aiogram.types import MessageEntity, User

from core.message_resolver import (
    MessageParseError,
    find_bot_mention,
    parse_rewrite_request,
    remove_mention,
)

BOT = "testbot"
BOT_ID = 777


class FakeMessage:
    """Minimal stand-in that keeps entity offsets in Telegram's UTF-16 units."""

    def __init__(
        self,
        text: str | None = None,
        entities: list[MessageEntity] | None = None,
        reply_to_message: "FakeMessage | None" = None,
    ) -> None:
        self.text = text
        self.caption = None
        self.entities = entities or []
        self.caption_entities = []
        self.reply_to_message = reply_to_message


class FakeCaptionMessage(FakeMessage):
    def __init__(
        self,
        caption: str | None = None,
        caption_entities: list[MessageEntity] | None = None,
        reply_to_message: "FakeMessage | None" = None,
    ) -> None:
        super().__init__(text=None, reply_to_message=reply_to_message)
        self.caption = caption
        self.caption_entities = caption_entities or []


def u16_offset(text: str, index: int) -> int:
    """Telegram counts entity offsets in UTF-16 code units."""
    return len(text[:index].encode("utf-16-le")) // 2


def mention_at(text: str, needle: str) -> list[MessageEntity]:
    start = text.index(needle)
    return [
        MessageEntity(
            type="mention",
            offset=u16_offset(text, start),
            length=len(needle.encode("utf-16-le")) // 2,
        )
    ]


def bot_message(text: str = "@testbot текст", **kwargs) -> FakeMessage:
    return FakeMessage(text=text, entities=mention_at(text, "@testbot"), **kwargs)


# --- 1. "@bot текст" without reply ---------------------------------------


def test_mention_with_text():
    parsed = parse_rewrite_request(bot_message("@testbot текст"), BOT)
    assert parsed is not None
    assert parsed.source_text == "текст"
    assert parsed.user_instruction is None


def test_mention_in_middle_of_text():
    msg = bot_message("привет @testbot как дела")
    parsed = parse_rewrite_request(msg, BOT)
    assert parsed is not None
    assert parsed.source_text == "привет как дела"


def test_mention_at_end():
    msg = bot_message("перепиши это @testbot")
    parsed = parse_rewrite_request(msg, BOT)
    assert parsed is not None
    assert parsed.source_text == "перепиши это"


def test_bare_mention_without_reply_raises_hint():
    with pytest.raises(MessageParseError) as exc:
        parse_rewrite_request(bot_message("@testbot"), BOT)
    assert "упоминания" in exc.value.user_facing.lower()


# --- 2. reply + mention ----------------------------------------------------


def test_reply_with_bare_mention_uses_quoted_text():
    quoted = FakeMessage(text="текст из reply")
    parsed = parse_rewrite_request(bot_message("@testbot", reply_to_message=quoted), BOT)
    assert parsed is not None
    assert parsed.source_text == "текст из reply"
    assert parsed.user_instruction is None


def test_reply_with_instruction():
    quoted = FakeMessage(text="очень длинный текст из reply")
    parsed = parse_rewrite_request(
        bot_message("@testbot сделай короче", reply_to_message=quoted), BOT
    )
    assert parsed is not None
    assert parsed.source_text == "очень длинный текст из reply"
    assert parsed.user_instruction == "сделай короче"


def test_reply_never_uses_current_text_as_source():
    """Text next to the mention is an instruction, never a replacement for the quote."""
    quoted = FakeMessage(text="оригинал")
    parsed = parse_rewrite_request(bot_message("@testbot новое", reply_to_message=quoted), BOT)
    assert parsed is not None
    assert parsed.source_text == "оригинал"
    assert parsed.user_instruction == "новое"


def test_reply_to_caption_message():
    quoted = FakeCaptionMessage(caption="подпись к медиа")
    parsed = parse_rewrite_request(bot_message("@testbot", reply_to_message=quoted), BOT)
    assert parsed is not None
    assert parsed.source_text == "подпись к медиа"


def test_reply_to_message_without_text_raises():
    empty = FakeMessage(text=None)
    with pytest.raises(MessageParseError) as exc:
        parse_rewrite_request(bot_message("@testbot", reply_to_message=empty), BOT)
    assert "подписи к медиа" in exc.value.user_facing


# --- 3. reply without mention must be ignored ------------------------------


def test_reply_without_mention_is_ignored():
    quoted = FakeMessage(text="что угодно")
    assert parse_rewrite_request(FakeMessage(text="просто текст", reply_to_message=quoted), BOT) is None


def test_reply_to_bot_message_without_mention_is_ignored():
    bot_msg = FakeMessage(text="переписанный текст")
    assert parse_rewrite_request(FakeMessage(text="ок", reply_to_message=bot_msg), BOT) is None


def test_plain_message_is_ignored():
    assert parse_rewrite_request(FakeMessage(text="просто текст"), BOT) is None


# --- 4. UTF-16 offsets -----------------------------------------------------


@pytest.mark.parametrize("prefix", ["🙂 ", "👨‍👩‍👧‍👦 ", "🇷🇺 ", "🏳️‍🌈 ", "❤️ "])
def test_mention_after_astral_emoji(prefix):
    text = f"{prefix}@testbot текст"
    msg = FakeMessage(text=text, entities=mention_at(text, "@testbot"))
    parsed = parse_rewrite_request(msg, BOT)
    assert parsed is not None
    assert parsed.source_text == f"{prefix}текст"
    assert "@testbot" not in parsed.source_text


def test_astral_emoji_before_mention_is_preserved():
    text = "🙂 @testbot привет"
    msg = FakeMessage(text=text, entities=mention_at(text, "@testbot"))
    parsed = parse_rewrite_request(msg, BOT)
    assert parsed is not None
    assert parsed.source_text == "🙂 привет"


def test_remove_mention_keeps_astral_emoji_intact():
    text = "🇷🇺 @testbot да"
    msg = FakeMessage(text=text, entities=mention_at(text, "@testbot"))
    entity = find_bot_mention(msg, BOT)
    assert entity is not None
    assert remove_mention(text, entity) == "🇷🇺 да"


def test_naive_slice_would_be_wrong():
    """Guards the regression: offsets are UTF-16 units, not code points."""
    text = "🙂 @testbot текст"
    entity = mention_at(text, "@testbot")[0]
    assert entity.extract_from(text) == "@testbot"
    assert text[entity.offset : entity.offset + entity.length] != "@testbot"


# --- 5. newlines preserved -------------------------------------------------


def test_multiline_text_keeps_newlines():
    text = "@testbot первая\nвторая\nтретья"
    msg = FakeMessage(text=text, entities=mention_at(text, "@testbot"))
    parsed = parse_rewrite_request(msg, BOT)
    assert parsed is not None
    assert parsed.source_text == "первая\nвторая\nтретья"


def test_multiline_instruction_keeps_newlines():
    quoted = FakeMessage(text="оригинал")
    text = "@testbot сделай\nкороче"
    msg = FakeMessage(text=text, entities=mention_at(text, "@testbot"), reply_to_message=quoted)
    parsed = parse_rewrite_request(msg, BOT)
    assert parsed is not None
    assert parsed.user_instruction == "сделай\nкороче"


def test_paragraphs_not_collapsed():
    text = "@testbot абзац один\n\nабзац два"
    msg = FakeMessage(text=text, entities=mention_at(text, "@testbot"))
    parsed = parse_rewrite_request(msg, BOT)
    assert parsed is not None
    assert parsed.source_text == "абзац один\n\nабзац два"


# --- 6. captions -----------------------------------------------------------


def test_mention_in_caption():
    caption = "@testbot текст подписи"
    msg = FakeCaptionMessage(
        caption=caption,
        caption_entities=[
            MessageEntity(type="mention", offset=0, length=len("@testbot")),
        ],
    )
    parsed = parse_rewrite_request(msg, BOT)
    assert parsed is not None
    assert parsed.source_text == "текст подписи"


def test_mention_in_caption_with_reply():
    caption = "@testbot короче"
    msg = FakeCaptionMessage(
        caption=caption,
        caption_entities=[MessageEntity(type="mention", offset=0, length=len("@testbot"))],
        reply_to_message=FakeMessage(text="оригинал"),
    )
    parsed = parse_rewrite_request(msg, BOT)
    assert parsed is not None
    assert parsed.source_text == "оригинал"
    assert parsed.user_instruction == "короче"


# --- 7. foreign mentions and case -----------------------------------------


def test_foreign_username_is_not_a_mention():
    text = "@otherbot текст"
    msg = FakeMessage(text=text, entities=mention_at(text, "@otherbot"))
    assert parse_rewrite_request(msg, BOT) is None


def test_case_insensitive_username():
    text = "@TeStBoT текст"
    msg = FakeMessage(
        text=text,
        entities=[MessageEntity(type="mention", offset=0, length=len("@TeStBoT"))],
    )
    parsed = parse_rewrite_request(msg, BOT)
    assert parsed is not None
    assert parsed.source_text == "текст"


def test_username_with_at_prefix_in_config():
    msg = bot_message("@testbot текст")
    assert parse_rewrite_request(msg, f"@{BOT}") is not None


def test_text_mention_by_bot_id():
    text = "Переписать вот это"
    mention_len = len("Переписать")
    msg = FakeMessage(
        text=text,
        entities=[
            MessageEntity(
                type="text_mention",
                offset=0,
                length=mention_len,
                user=User(id=BOT_ID, is_bot=True, first_name="Bot"),
            )
        ],
    )
    parsed = parse_rewrite_request(msg, BOT, bot_id=BOT_ID)
    assert parsed is not None
    assert parsed.source_text == "вот это"


def test_text_mention_of_other_user_is_not_a_mention():
    text = "перепиши это"
    msg = FakeMessage(
        text=text,
        entities=[
            MessageEntity(
                type="text_mention",
                offset=0,
                length=len("перепиши это"),
                user=User(id=1, is_bot=False, first_name="Human"),
            )
        ],
    )
    assert parse_rewrite_request(msg, BOT, bot_id=BOT_ID) is None


# --- 8. mention inside the quoted message ----------------------------------


def test_mention_inside_reply_text_is_stripped():
    quoted = FakeMessage(text="@testbot привет", entities=mention_at("@testbot привет", "@testbot"))
    parsed = parse_rewrite_request(bot_message("@testbot", reply_to_message=quoted), BOT)
    assert parsed is not None
    assert parsed.source_text == "привет"
    assert "@testbot" not in parsed.source_text


def test_mention_inside_reply_text_with_instruction():
    quoted = FakeMessage(text="@testbot длинный текст", entities=mention_at("@testbot длинный текст", "@testbot"))
    parsed = parse_rewrite_request(
        bot_message("@testbot короче", reply_to_message=quoted), BOT
    )
    assert parsed is not None
    assert parsed.source_text == "длинный текст"
    assert parsed.user_instruction == "короче"


def test_foreign_mention_inside_reply_text_is_kept():
    text = "@otherbot привет"
    quoted = FakeMessage(text=text, entities=mention_at(text, "@otherbot"))
    parsed = parse_rewrite_request(bot_message("@testbot", reply_to_message=quoted), BOT)
    assert parsed is not None
    assert parsed.source_text == "@otherbot привет"


def test_mention_inside_reply_caption_is_stripped():
    caption = "@testbot подпись"
    quoted = FakeCaptionMessage(
        caption=caption,
        caption_entities=[MessageEntity(type="mention", offset=0, length=len("@testbot"))],
    )
    parsed = parse_rewrite_request(bot_message("@testbot", reply_to_message=quoted), BOT)
    assert parsed is not None
    assert parsed.source_text == "подпись"


def test_reply_containing_only_a_mention_raises():
    quoted = FakeMessage(text="@testbot", entities=mention_at("@testbot", "@testbot"))
    with pytest.raises(MessageParseError) as exc:
        parse_rewrite_request(bot_message("@testbot", reply_to_message=quoted), BOT)
    assert "только упоминание" in exc.value.user_facing


def test_mention_inside_reply_after_astral_emoji_is_stripped():
    text = "🙂 @testbot привет"
    quoted = FakeMessage(text=text, entities=mention_at(text, "@testbot"))
    parsed = parse_rewrite_request(bot_message("@testbot", reply_to_message=quoted), BOT)
    assert parsed is not None
    assert parsed.source_text == "🙂 привет"


# --- 9. whitespace at the seam --------------------------------------------


def test_double_space_at_cut_is_removed():
    text = "привет @testbot текст"
    msg = FakeMessage(text=text, entities=mention_at(text, "@testbot"))
    entity = find_bot_mention(msg, BOT)
    assert entity is not None
    assert remove_mention(text, entity) == "привет текст"


def test_remove_mention_across_bmp_and_non_bmp_chars():
    """Exact result for a mention preceded by both BMP and astral characters."""
    text = "🙂é @testbot текст"
    msg = FakeMessage(text=text, entities=mention_at(text, "@testbot"))
    entity = find_bot_mention(msg, BOT)

    assert entity is not None
    assert entity.extract_from(text) == "@testbot"
    assert remove_mention(text, entity) == "🙂é текст"


def test_remove_mention_with_only_bmp_prefix():
    text = "éж @testbot текст"
    msg = FakeMessage(text=text, entities=mention_at(text, "@testbot"))
    entity = find_bot_mention(msg, BOT)

    assert entity is not None
    assert remove_mention(text, entity) == "éж текст"


def test_remove_mention_with_only_astral_prefix():
    text = "🙂🙂 @testbot текст"
    msg = FakeMessage(text=text, entities=mention_at(text, "@testbot"))
    entity = find_bot_mention(msg, BOT)

    assert entity is not None
    assert remove_mention(text, entity) == "🙂🙂 текст"


def test_remove_mention_at_end_after_astral():
    text = "🙂 @testbot"
    msg = FakeMessage(text=text, entities=mention_at(text, "@testbot"))
    entity = find_bot_mention(msg, BOT)

    assert entity is not None
    assert remove_mention(text, entity) == "🙂"
