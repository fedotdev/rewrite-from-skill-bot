from collections.abc import Iterator
from datetime import datetime
from typing import Any

import pytest
from aiogram import Bot, Dispatcher
from aiogram.types import Chat, Message, MessageEntity, Update, User

from core.deduplication import InMemoryDeduplicator
from core.models import RewriteError, RewriteResult
from core.rate_limiter import CompositeRateLimiter, InMemoryRateLimiter
from core.service import RewriteService
from handlers.rewrite import router as rewrite_router

BOT_USERNAME = "testbot"
TOKEN = "123456:TEST-FAKE-TOKEN"


class StubLLM:
    def __init__(self, result: RewriteResult | None = None, error: Exception | None = None) -> None:
        self.result = result or RewriteResult(text="rewritten")
        self.error = error
        self.calls: list[tuple[str, str | None]] = []

    async def rewrite(self, request: Any, style: str) -> RewriteResult:
        self.calls.append((request.source_text, request.user_instruction))
        if self.error:
            raise self.error
        return self.result


def make_service(
    llm: StubLLM,
    allowed_chat_ids: list[int] | None = None,
    user_limit: int = 20,
    chat_limit: int = 100,
) -> RewriteService:
    service = RewriteService(
        llm_client=llm,  # type: ignore[arg-type]
        deduplicator=InMemoryDeduplicator(),
        rate_limiter=CompositeRateLimiter(
            user_limiter=InMemoryRateLimiter(),
            chat_limiter=InMemoryRateLimiter(),
            user_limit=user_limit,
            chat_limit=chat_limit,
        ),
        allowed_chat_ids=allowed_chat_ids if allowed_chat_ids is not None else [],
    )
    service.set_bot_username(BOT_USERNAME)
    return service


def make_update(
    text: str | None = None,
    mention: bool = False,
    chat_id: int = -1001,
    user_id: int = 42,
    message_id: int = 1,
    reply_to: Message | None = None,
) -> Update:
    entities = None
    if mention and text:
        entities = [MessageEntity(type="mention", offset=0, length=len(f"@{BOT_USERNAME}"))]
    return Update(
        update_id=message_id,
        message=Message(
            message_id=message_id,
            date=datetime.now(),
            chat=Chat(id=chat_id, type="group"),
            from_user=User(id=user_id, is_bot=False, first_name="u"),
            text=text,
            entities=entities,
            reply_to_message=reply_to,
        ),
    )


@pytest.fixture(autouse=True)
def detached_router() -> Iterator[None]:
    yield
    rewrite_router._parent_router = None


@pytest.fixture
def replies(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    sent: list[str] = []

    async def fake_reply(self: Message, text: str, **kwargs: Any) -> Message:
        sent.append(text)
        return self

    monkeypatch.setattr(Message, "reply", fake_reply)
    return sent


async def feed(dp: Dispatcher, update: Update) -> None:
    bot = Bot(token=TOKEN)
    try:
        await dp.feed_update(bot, update)
    finally:
        await bot.session.close()


def build_dp(service: RewriteService) -> Dispatcher:
    dp = Dispatcher()
    dp["service"] = service
    dp.include_router(rewrite_router)
    return dp


async def test_mention_with_text_is_rewritten(replies: list[str]) -> None:
    llm = StubLLM(RewriteResult(text="Коротко и по делу"))
    dp = build_dp(make_service(llm))

    await feed(dp, make_update(text="@testbot сделай короче", mention=True))

    assert llm.calls == [("сделай короче", None)]
    assert replies == ["Коротко и по делу"]


async def test_warning_is_appended(replies: list[str]) -> None:
    llm = StubLLM(RewriteResult(text="текст", warning="факты не проверены"))
    dp = build_dp(make_service(llm))

    await feed(dp, make_update(text="@testbot перепиши", mention=True))

    assert replies == ["текст\n\n⚠️ факты не проверены"]


async def test_reply_uses_quoted_text_and_instruction(replies: list[str]) -> None:
    quoted = Message(
        message_id=7,
        date=datetime.now(),
        chat=Chat(id=-1001, type="group"),
        from_user=User(id=1, is_bot=False, first_name="a"),
        text="исходный текст",
    )
    llm = StubLLM()
    dp = build_dp(make_service(llm))

    await feed(dp, make_update(text="@testbot короче", mention=True, reply_to=quoted))

    assert llm.calls == [("исходный текст", "короче")]


async def test_no_mention_and_no_reply_returns_hint(replies: list[str]) -> None:
    llm = StubLLM()
    dp = build_dp(make_service(llm))

    await feed(dp, make_update(text="просто текст"))

    assert llm.calls == []
    assert len(replies) == 1
    assert "ответьте на сообщение" in replies[0]


async def test_reply_to_non_text_message_is_refused(replies: list[str]) -> None:
    photo = Message(
        message_id=8,
        date=datetime.now(),
        chat=Chat(id=-1001, type="group"),
        from_user=User(id=1, is_bot=False, first_name="a"),
        photo=[],
    )
    llm = StubLLM()
    dp = build_dp(make_service(llm))

    await feed(dp, make_update(text="@testbot", mention=True, reply_to=photo))

    assert llm.calls == []
    assert "только текстовые" in replies[0]


async def test_foreign_chat_is_ignored(replies: list[str]) -> None:
    llm = StubLLM()
    dp = build_dp(make_service(llm, allowed_chat_ids=[-1002]))

    await feed(dp, make_update(text="@testbot перепиши", mention=True, chat_id=-1001))

    assert llm.calls == []
    assert replies == []


async def test_user_rate_limit_blocks_call(replies: list[str]) -> None:
    llm = StubLLM()
    dp = build_dp(make_service(llm, user_limit=1))

    await feed(dp, make_update(text="@testbot перепиши", mention=True, message_id=1))
    await feed(dp, make_update(text="@testbot перепиши", mention=True, message_id=2))

    assert len(llm.calls) == 1
    assert "Слишком много запросов" in replies[1]


async def test_chat_rate_limit_blocks_call(replies: list[str]) -> None:
    llm = StubLLM()
    dp = build_dp(make_service(llm, chat_limit=1))

    await feed(dp, make_update(text="@testbot перепиши", mention=True, message_id=1))
    await feed(dp, make_update(text="@testbot перепиши", mention=True, message_id=2))

    assert len(llm.calls) == 1
    assert "Чат достиг лимита" in replies[1]


async def test_duplicate_message_is_processed_once(replies: list[str]) -> None:
    llm = StubLLM()
    dp = build_dp(make_service(llm))

    update = make_update(text="@testbot перепиши", mention=True, message_id=5)
    await feed(dp, update)
    await feed(dp, update)

    assert len(llm.calls) == 1
    assert len(replies) == 1


async def test_llm_error_shows_user_facing_message(replies: list[str]) -> None:
    llm = StubLLM(error=RewriteError(message="boom", user_facing="Модель недоступна"))
    dp = build_dp(make_service(llm))

    await feed(dp, make_update(text="@testbot перепиши", mention=True))

    assert replies == ["Модель недоступна"]


async def test_unexpected_error_is_caught(replies: list[str]) -> None:
    llm = StubLLM(error=ValueError("kaboom"))
    dp = build_dp(make_service(llm))

    await feed(dp, make_update(text="@testbot перепиши", mention=True))

    assert replies == ["Произошла ошибка. Попробуйте позже."]


async def test_command_is_not_rewritten(replies: list[str]) -> None:
    llm = StubLLM()
    dp = build_dp(make_service(llm))

    await feed(dp, make_update(text="/start"))

    assert llm.calls == []
    assert replies == []
