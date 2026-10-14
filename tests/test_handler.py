from collections.abc import Iterator
from datetime import datetime
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from aiogram import Bot, Dispatcher
from aiogram.types import Chat, Message, MessageEntity, Update, User
from structlog.testing import capture_logs

from core.deduplication import InMemoryDeduplicator
from core.models import RewriteError, RewriteRequest, RewriteResult
from core.rate_limiter import CompositeRateLimiter, InMemoryRateLimiter, WarningThrottle
from core.service import RewriteService
from handlers.rewrite import router as rewrite_router
from llm.client import LLMClient, LLMClientError

BOT_USERNAME = "testbot"
BOT_ID = 777
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
    throttle: WarningThrottle | None = None,
) -> RewriteService:
    return RewriteService(
        llm_client=llm,  # type: ignore[arg-type]
        deduplicator=InMemoryDeduplicator(),
        rate_limiter=CompositeRateLimiter(
            user_limiter=InMemoryRateLimiter(),
            chat_limiter=InMemoryRateLimiter(),
            user_limit=user_limit,
            chat_limit=chat_limit,
        ),
        allowed_chat_ids=allowed_chat_ids if allowed_chat_ids is not None else [-1001],
        bot_username=BOT_USERNAME,
        bot_id=BOT_ID,
        warning_throttle=throttle,
    )


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def fake_clock() -> FakeClock:
    return FakeClock()


def u16_offset(text: str, index: int) -> int:
    return len(text[:index].encode("utf-16-le")) // 2


def make_update(
    text: str | None = None,
    mention: bool = True,
    chat_id: int = -1001,
    user_id: int = 42,
    message_id: int = 1,
    reply_to: Message | None = None,
    caption: str | None = None,
) -> Update:
    entities = None
    caption_entities = None
    if mention and text:
        start = text.index("@testbot")
        entities = [
            MessageEntity(type="mention", offset=u16_offset(text, start), length=len("@testbot"))
        ]
    if mention and caption:
        start = caption.index("@testbot")
        caption_entities = [
            MessageEntity(type="mention", offset=u16_offset(caption, start), length=len("@testbot"))
        ]
    return Update(
        update_id=message_id,
        message=Message(
            message_id=message_id,
            date=datetime.now(),
            chat=Chat(id=chat_id, type="group"),
            from_user=User(id=user_id, is_bot=False, first_name="u"),
            text=text,
            entities=entities,
            caption=caption,
            caption_entities=caption_entities,
            reply_to_message=reply_to,
        ),
    )


def quoted(text: str | None = None, caption: str | None = None, from_bot: bool = False) -> Message:
    return Message(
        message_id=7,
        date=datetime.now(),
        chat=Chat(id=-1001, type="group"),
        from_user=User(id=BOT_ID, is_bot=True, first_name="bot")
        if from_bot
        else User(id=1, is_bot=False, first_name="a"),
        text=text,
        caption=caption,
        photo=[] if caption else None,
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


@pytest.fixture
def fast_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("llm.client.RETRY_BASE_SECONDS", 0.001)
    monkeypatch.setattr("llm.client.RETRY_MAX_SECONDS", 0.002)


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


def _request(source: str) -> RewriteRequest:
    return RewriteRequest(
        source_text=source,
        user_instruction=None,
        chat_id=-1001,
        message_id=1,
        message_thread_id=None,
        user_id=1,
    )


def _make_real_client(monkeypatch: pytest.MonkeyPatch) -> LLMClient:
    settings = SimpleNamespace(
        llm_api_url="https://openrouter.ai/api/v1/chat/completions",
        llm_api_key="sk-secret-key",
        llm_model="poolside/laguna-s-2.1",
        llm_timeout_seconds=5.0,
        llm_max_tokens=100,
        llm_temperature=0.3,
        llm_json_mode=False,
    )
    monkeypatch.setattr("llm.client.get_settings", lambda: settings)
    return LLMClient()


def _attach(client: LLMClient, handler) -> list[dict]:
    sent: list[dict] = []

    def wrapper(request: httpx.Request) -> httpx.Response:
        sent.append({"url": str(request.url), "auth": request.headers.get("authorization")})
        return handler(request)

    client._client = httpx.AsyncClient(
        transport=httpx.MockTransport(wrapper), headers=dict(client._client.headers)
    )
    return sent


# --- main behaviour: only reacts to mentions ------------------------------


async def test_mention_with_text_is_rewritten(replies: list[str]) -> None:
    llm = StubLLM(RewriteResult(text="Коротко и по делу"))
    dp = build_dp(make_service(llm))

    await feed(dp, make_update(text="@testbot сделай короче"))

    assert llm.calls == [("сделай короче", None)]
    assert replies == ["Коротко и по делу"]


async def test_plain_message_is_ignored_silently(replies: list[str]) -> None:
    llm = StubLLM()
    dp = build_dp(make_service(llm))

    await feed(dp, make_update(text="просто текст", mention=False))

    assert llm.calls == []
    assert replies == []


async def test_reply_without_mention_is_ignored(replies: list[str]) -> None:
    llm = StubLLM()
    dp = build_dp(make_service(llm))

    await feed(dp, make_update(text="ок", mention=False, reply_to=quoted(text="оригинал")))

    assert llm.calls == []
    assert replies == []


async def test_reply_to_bot_message_without_mention_is_ignored(replies: list[str]) -> None:
    bot_reply = quoted(text="переписанный текст", from_bot=True)
    llm = StubLLM()
    dp = build_dp(make_service(llm))

    await feed(dp, make_update(text="спасибо", mention=False, reply_to=bot_reply))

    assert llm.calls == []
    assert replies == []


async def test_foreign_username_is_ignored(replies: list[str]) -> None:
    text = "@otherbot текст"
    update = Update(
        update_id=1,
        message=Message(
            message_id=1,
            date=datetime.now(),
            chat=Chat(id=-1001, type="group"),
            from_user=User(id=42, is_bot=False, first_name="u"),
            text=text,
            entities=[MessageEntity(type="mention", offset=0, length=len("@otherbot"))],
        ),
    )
    llm = StubLLM()
    dp = build_dp(make_service(llm))

    await feed(dp, update)

    assert llm.calls == []
    assert replies == []


async def test_mention_in_caption_is_handled(replies: list[str]) -> None:
    llm = StubLLM(RewriteResult(text="ок"))
    dp = build_dp(make_service(llm))

    await feed(dp, make_update(text=None, caption="@testbot текст подписи"))

    assert llm.calls == [("текст подписи", None)]
    assert replies == ["ок"]


# --- reply + mention -------------------------------------------------------


async def test_reply_with_bare_mention_uses_quoted_text(replies: list[str]) -> None:
    llm = StubLLM()
    dp = build_dp(make_service(llm))

    await feed(dp, make_update(text="@testbot", reply_to=quoted(text="текст из reply")))

    assert llm.calls == [("текст из reply", None)]


async def test_reply_with_instruction(replies: list[str]) -> None:
    llm = StubLLM()
    dp = build_dp(make_service(llm))

    await feed(
        dp,
        make_update(text="@testbot сделай короче", reply_to=quoted(text="очень длинный текст")),
    )

    assert llm.calls == [("очень длинный текст", "сделай короче")]


async def test_reply_to_caption_message(replies: list[str]) -> None:
    llm = StubLLM()
    dp = build_dp(make_service(llm))

    await feed(dp, make_update(text="@testbot", reply_to=quoted(caption="подпись к медиа")))

    assert llm.calls == [("подпись к медиа", None)]


async def test_reply_to_non_text_message_is_refused(replies: list[str]) -> None:
    llm = StubLLM()
    dp = build_dp(make_service(llm))

    await feed(dp, make_update(text="@testbot", reply_to=quoted()))

    assert llm.calls == []
    assert "подписи к медиа" in replies[0]


async def test_bare_mention_without_reply_shows_hint(replies: list[str]) -> None:
    llm = StubLLM()
    dp = build_dp(make_service(llm))

    await feed(dp, make_update(text="@testbot"))

    assert llm.calls == []
    assert len(replies) == 1
    assert "ответьте" in replies[0].lower()


# --- allowlist -------------------------------------------------------------


async def test_foreign_chat_is_ignored(replies: list[str]) -> None:
    llm = StubLLM()
    dp = build_dp(make_service(llm, allowed_chat_ids=[-1002]))

    await feed(dp, make_update(text="@testbot перепиши"))

    assert llm.calls == []
    assert replies == []


async def test_empty_allowlist_admits_nobody(replies: list[str]) -> None:
    llm = StubLLM()
    dp = build_dp(make_service(llm, allowed_chat_ids=[]))

    await feed(dp, make_update(text="@testbot перепиши"))

    assert llm.calls == []
    assert replies == []


async def test_service_rejects_empty_username() -> None:
    with pytest.raises(ValueError, match="bot_username"):
        RewriteService(
            llm_client=StubLLM(),  # type: ignore[arg-type]
            deduplicator=InMemoryDeduplicator(),
            rate_limiter=CompositeRateLimiter(
                InMemoryRateLimiter(), InMemoryRateLimiter(), 20, 100
            ),
            allowed_chat_ids=[-1001],
            bot_username="",
            bot_id=BOT_ID,
        )


# --- rate limit and dedup ordering ----------------------------------------


async def test_invalid_request_does_not_consume_rate_limit(replies: list[str]) -> None:
    llm = StubLLM()
    dp = build_dp(make_service(llm, user_limit=2))

    # invalid: bare mention -> hint, must not spend the limit
    await feed(dp, make_update(text="@testbot", message_id=1))
    await feed(dp, make_update(text="@testbot", message_id=2))
    # no mention -> silent, must not spend the limit either
    await feed(dp, make_update(text="просто текст", mention=False, message_id=3))

    # both remaining slots are still available
    await feed(dp, make_update(text="@testbot перепиши", message_id=4))
    await feed(dp, make_update(text="@testbot перепиши", message_id=5))

    assert len(llm.calls) == 2
    assert "Слишком много запросов" not in " ".join(replies)


async def test_user_rate_limit_warns_once_then_stays_silent(replies: list[str]) -> None:
    llm = StubLLM()
    dp = build_dp(make_service(llm, user_limit=1))

    await feed(dp, make_update(text="@testbot перепиши", message_id=1))
    await feed(dp, make_update(text="@testbot перепиши", message_id=2))
    await feed(dp, make_update(text="@testbot перепиши", message_id=3))

    assert len(llm.calls) == 1
    assert len(replies) == 2, "only the successful reply plus one warning"
    assert "Слишком много запросов" in replies[1]


async def test_user_rate_limit_warning_repeats_after_window(
    replies: list[str], fake_clock: FakeClock
) -> None:
    llm = StubLLM()
    dp = build_dp(make_service(llm, user_limit=1, throttle=WarningThrottle(fake_clock)))

    await feed(dp, make_update(text="@testbot перепиши", message_id=1))
    await feed(dp, make_update(text="@testbot перепиши", message_id=2))
    assert "Слишком много запросов" in replies[1]

    # window expires; a fresh warning is allowed again
    fake_clock.advance(61)
    await feed(dp, make_update(text="@testbot перепиши", message_id=3))
    await feed(dp, make_update(text="@testbot перепиши", message_id=4))

    assert len(replies) == 3
    assert "Слишком много запросов" in replies[2]


async def test_chat_limit_warning_is_per_chat(replies: list[str]) -> None:
    llm = StubLLM()
    dp = build_dp(make_service(llm, chat_limit=1, user_limit=100))

    await feed(dp, make_update(text="@testbot перепиши", message_id=1, user_id=1))
    await feed(dp, make_update(text="@testbot перепиши", message_id=2, user_id=2))
    await feed(dp, make_update(text="@testbot перепиши", message_id=3, user_id=3))

    assert len(llm.calls) == 1
    assert len(replies) == 2, "chat warning must not repeat per user"
    assert "Чат достиг лимита" in replies[1]


async def test_chat_rate_limit_blocks_call(replies: list[str]) -> None:
    llm = StubLLM()
    dp = build_dp(make_service(llm, chat_limit=1))

    await feed(dp, make_update(text="@testbot перепиши", message_id=1))
    await feed(dp, make_update(text="@testbot перепиши", message_id=2))

    assert len(llm.calls) == 1
    assert "Чат достиг лимита" in replies[1]


async def test_duplicate_message_is_processed_once(replies: list[str]) -> None:
    llm = StubLLM()
    dp = build_dp(make_service(llm))

    update = make_update(text="@testbot перепиши", message_id=5)
    await feed(dp, update)
    await feed(dp, update)

    assert len(llm.calls) == 1
    assert len(replies) == 1


# --- errors ----------------------------------------------------------------


async def test_warning_is_appended(replies: list[str]) -> None:
    llm = StubLLM(RewriteResult(text="текст", warning="факты не проверены"))
    dp = build_dp(make_service(llm))

    await feed(dp, make_update(text="@testbot перепиши"))

    assert replies == ["текст\n\n⚠️ факты не проверены"]


async def test_llm_error_shows_user_facing_message(replies: list[str]) -> None:
    llm = StubLLM(error=RewriteError(message="boom", user_facing="Модель недоступна"))
    dp = build_dp(make_service(llm))

    await feed(dp, make_update(text="@testbot перепиши"))

    assert replies == ["Модель недоступна"]


async def test_unexpected_error_is_caught(replies: list[str]) -> None:
    llm = StubLLM(error=ValueError("kaboom"))
    dp = build_dp(make_service(llm))

    await feed(dp, make_update(text="@testbot перепиши"))

    assert replies == ["Произошла ошибка. Попробуйте позже."]


async def test_command_with_mention_is_not_rewritten(replies: list[str]) -> None:
    llm = StubLLM()
    dp = build_dp(make_service(llm))

    await feed(dp, make_update(text="/start @testbot"))

    assert llm.calls == []
    assert replies == []


async def test_command_with_bot_username_suffix_is_not_rewritten(replies: list[str]) -> None:
    llm = StubLLM()
    dp = build_dp(make_service(llm))

    await feed(dp, make_update(text="/rewrite@testbot текст"))

    assert llm.calls == []
    assert replies == []


async def test_plain_command_is_not_rewritten(replies: list[str]) -> None:
    llm = StubLLM()
    dp = build_dp(make_service(llm))

    await feed(dp, make_update(text="/start", mention=False))

    assert llm.calls == []
    assert replies == []


async def test_message_text_never_reaches_the_logs(replies: list[str]) -> None:
    """The capture itself must be non-empty, otherwise the assertion below is vacuous."""
    secret = "очень секретный текст"
    llm = StubLLM()
    dp = build_dp(make_service(llm))

    with capture_logs() as captured:
        await feed(dp, make_update(text=f"@testbot {secret}"))

    assert captured, "structlog capture returned nothing, test would be vacuous"
    assert any(entry.get("event") == "Rewrite successful" for entry in captured)
    assert secret not in str(captured)


async def test_source_text_never_reaches_the_logs_on_error(replies: list[str]) -> None:
    secret = "суперсекретный исходник"
    llm = StubLLM(error=RewriteError(message="boom", user_facing="Модель недоступна"))
    dp = build_dp(make_service(llm))

    with capture_logs() as captured:
        await feed(dp, make_update(text=f"@testbot {secret}"))

    assert captured
    assert any(entry.get("event") == "Rewrite error" for entry in captured)
    assert secret not in str(captured)


async def test_source_text_never_reaches_llm_error_message(monkeypatch: pytest.MonkeyPatch) -> None:
    secret = "суперсекретный исходник"
    client = _make_real_client(monkeypatch)
    body = '{"error": {"message": "bad request"}}'
    sent = _attach(client, lambda r: httpx.Response(400, text=body))

    with pytest.raises(LLMClientError) as exc:
        await client.rewrite(_request(secret), "стиль")

    assert len(sent) == 1
    assert secret not in exc.value.message
    assert secret not in exc.value.user_facing


async def test_logs_do_not_contain_api_key(
    monkeypatch: pytest.MonkeyPatch, replies: list[str], fast_backoff: None
) -> None:
    client = _make_real_client(monkeypatch)
    _attach(client, lambda r: httpx.Response(500, text="boom"))
    dp = build_dp(
        RewriteService(
            llm_client=client,
            deduplicator=InMemoryDeduplicator(),
            rate_limiter=CompositeRateLimiter(
                InMemoryRateLimiter(), InMemoryRateLimiter(), 20, 100
            ),
            allowed_chat_ids=[-1001],
            bot_username=BOT_USERNAME,
            bot_id=BOT_ID,
        )
    )

    with capture_logs() as captured:
        await feed(dp, make_update(text="@testbot текст"))

    assert captured
    assert "sk-secret-key" not in str(captured)
