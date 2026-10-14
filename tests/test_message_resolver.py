
import pytest

from core.message_resolver import (
    MessageParseError,
    ParsedMessage,
    build_rewrite_request,
    extract_mention_entity,
    get_text_content,
    parse_rewrite_request,
    remove_mention,
)


class MockUser:
    def __init__(self, id: int):
        self.id = id


class MockChat:
    def __init__(self, id: int):
        self.id = id


class MockMessage:
    def __init__(
        self,
        text: str | None = None,
        caption: str | None = None,
        entities: list | None = None,
        caption_entities: list | None = None,
        reply_to_message: "MockMessage | None" = None,
        message_id: int = 1,
        chat_id: int = -100123456,
        user_id: int = 123,
        message_thread_id: int | None = None,
    ):
        self.text = text
        self.caption = caption
        self.entities = entities or []
        self.caption_entities = caption_entities or []
        self.reply_to_message = reply_to_message
        self.message_id = message_id
        self.chat = MockChat(chat_id)
        self.from_user = MockUser(user_id)
        self.message_thread_id = message_thread_id


class MockEntity:
    def __init__(self, type: str, offset: int, length: int):
        self.type = type
        self.offset = offset
        self.length = length


def test_extract_mention_entity_found():
    entities = [MockEntity("mention", 0, 4)]
    msg = MockMessage(text="@bot hello", entities=entities)
    entity = extract_mention_entity(msg, "bot")
    assert entity is not None
    assert entity.offset == 0
    assert entity.length == 4


def test_extract_mention_entity_not_found():
    entities = [MockEntity("mention", 0, 6)]
    msg = MockMessage(text="@other hello", entities=entities)
    entity = extract_mention_entity(msg, "bot")
    assert entity is None


def test_extract_mention_entity_no_entities():
    msg = MockMessage(text="@bot hello", entities=[])
    entity = extract_mention_entity(msg, "bot")
    assert entity is None


def test_remove_mention():
    text = "@bot hello world"
    entity = MockEntity("mention", 0, 4)
    result = remove_mention(text, entity)
    assert result == "hello world"


def test_remove_mention_middle():
    text = "hey @bot hello"
    entity = MockEntity("mention", 4, 4)
    result = remove_mention(text, entity)
    assert result == "hey hello"


def test_get_text_content_text():
    msg = MockMessage(text="hello")
    assert get_text_content(msg) == "hello"


def test_get_text_content_caption():
    msg = MockMessage(caption="caption text")
    assert get_text_content(msg) == "caption text"


def test_get_text_content_none():
    msg = MockMessage()
    assert get_text_content(msg) is None


def test_parse_mention_only_with_text():
    entities = [MockEntity("mention", 0, 4)]
    msg = MockMessage(text="@bot rewrite this", entities=entities)
    result = parse_rewrite_request(msg, "bot")
    assert isinstance(result, ParsedMessage)
    assert result.source_text == "rewrite this"
    assert result.user_instruction is None
    assert result.has_reply is False


def test_parse_mention_only_empty_after_mention():
    entities = [MockEntity("mention", 0, 4)]
    msg = MockMessage(text="@bot", entities=entities)
    with pytest.raises(MessageParseError):
        parse_rewrite_request(msg, "bot")


def test_parse_mention_only_no_text():
    entities = [MockEntity("mention", 0, 4)]
    msg = MockMessage(text="@bot", entities=entities)
    with pytest.raises(MessageParseError):
        parse_rewrite_request(msg, "bot")


def test_parse_reply_only():
    replied = MockMessage(text="original message")
    msg = MockMessage(reply_to_message=replied)
    result = parse_rewrite_request(msg, "bot")
    assert result.source_text == "original message"
    assert result.user_instruction is None
    assert result.has_reply is True


def test_parse_reply_with_mention_and_instruction():
    replied = MockMessage(text="original message")
    entities = [MockEntity("mention", 0, 4)]
    msg = MockMessage(text="@bot make it shorter", entities=entities, reply_to_message=replied)
    result = parse_rewrite_request(msg, "bot")
    assert result.source_text == "original message"
    assert result.user_instruction == "make it shorter"
    assert result.has_reply is True


def test_parse_reply_with_mention_no_instruction():
    replied = MockMessage(text="original message")
    entities = [MockEntity("mention", 0, 4)]
    msg = MockMessage(text="@bot", entities=entities, reply_to_message=replied)
    result = parse_rewrite_request(msg, "bot")
    assert result.source_text == "original message"
    assert result.user_instruction is None
    assert result.has_reply is True


def test_parse_no_mention_no_reply():
    msg = MockMessage(text="just text")
    with pytest.raises(MessageParseError):
        parse_rewrite_request(msg, "bot")


def test_parse_reply_no_text():
    replied = MockMessage()
    msg = MockMessage(reply_to_message=replied)
    with pytest.raises(MessageParseError):
        parse_rewrite_request(msg, "bot")


def test_build_rewrite_request():
    entities = [MockEntity("mention", 0, 4)]
    msg = MockMessage(text="@bot hello world", entities=entities, message_id=42, chat_id=-1001, user_id=999)
    request = build_rewrite_request(msg, "bot")
    assert request.source_text == "hello world"
    assert request.chat_id == -1001
    assert request.message_id == 42
    assert request.user_id == 999
    assert request.message_thread_id is None


def test_build_rewrite_request_with_thread():
    entities = [MockEntity("mention", 0, 4)]
    msg = MockMessage(text="@bot hello", entities=entities, message_id=1, chat_id=-1001, user_id=999, message_thread_id=5)
    request = build_rewrite_request(msg, "bot")
    assert request.message_thread_id == 5


def test_mention_in_caption_is_detected():
    msg = MockMessage(caption="@bot rewrite this", caption_entities=[MockEntity("mention", 0, 4)])
    parsed = parse_rewrite_request(msg, "bot")
    assert parsed.source_text == "rewrite this"
    assert parsed.user_instruction is None


def test_caption_with_mention_and_instruction_on_reply():
    replied = MockMessage(text="original")
    msg = MockMessage(
        caption="@bot make it shorter",
        caption_entities=[MockEntity("mention", 0, 4)],
        reply_to_message=replied,
    )
    parsed = parse_rewrite_request(msg, "bot")
    assert parsed.source_text == "original"
    assert parsed.user_instruction == "make it shorter"


def test_mention_entity_ignored_when_only_in_other_field():
    # text present, mention only registered on caption_entities -> must NOT match
    msg = MockMessage(text="@bot hello", caption="x", caption_entities=[MockEntity("mention", 0, 4)])
    assert extract_mention_entity(msg, "bot") is None


def test_reply_to_caption_is_rewritten():
    replied = MockMessage(caption="photo caption text")
    msg = MockMessage(text="@bot", entities=[MockEntity("mention", 0, 4)], reply_to_message=replied)
    parsed = parse_rewrite_request(msg, "bot")
    assert parsed.source_text == "photo caption text"
