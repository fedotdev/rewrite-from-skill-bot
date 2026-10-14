# Проект: Telegram-бот переписывания

## Стек

- Python 3.12+, aiogram 3, httpx, tenacity, pytest, pydantic-settings, structlog.
- Рекомендуемый LLM-профиль — OpenRouter через `LLM_API_URL`, `LLM_API_KEY`, `LLM_MODEL`; значения по умолчанию указаны в `.env.example`: `poolside/laguna-s-2.1`.
- Секреты не хардкодить.

## Архитектура

- `handlers/rewrite.py` только фильтрует update и вызывает `core/service`; напрямую LLM не вызывает.
- `core/message_resolver.py` отвечает за parsing.
- `core/service.py` координирует allowlist, parsing, dedup, rate limit, style и LLM.
- `llm/client.py` — единственное место вызова LLM.
- `runtime_skills/rewrite/SKILL.md` и `references/*.md` — runtime-конфигурация. `StyleLoader.load()` возвращает строку и перечитывает изменения по mtime без рестарта.

## Обязательное поведение

- Порядок обработки: mention → allowlist → parse → dedup → rate limit → LLM.
- Отвечать только при реальном mention entity бота: `mention` или `text_mention` в `text` или `caption` с `caption_entities`; username сравнивать без учёта регистра.
- Без mention — полная тишина: без reply, вызова LLM и расхода лимита.
- При reply source всегда брать из `text` или `caption` процитированного сообщения; текущий текст после mention использовать только как instruction.
- Telegram offsets — UTF-16 code units; использовать `MessageEntity.extract_from` и surrogate-safe removal.
- Пустой `ALLOWED_CHAT_IDS` означает fail-closed: никто не допущен.
- Invalid requests не расходуют dedup/rate-limit; предупреждения о rate limit должны быть throttled.

## Runtime style references

- Подключать только `references/<name>.md`, на которые есть ссылка в `SKILL.md`; автоматически подключать все файлы нельзя.
- Учитывать ссылки в raw `SKILL.md`, включая YAML frontmatter; порядок — первое появление, без дублей.
- Разрешены только плоские `.md` внутри `references/` в UTF-8; отклонять traversal, абсолютные пути, nested folders и внешние symlinks.
- Лимиты: `MAX_REFERENCE_BYTES` — 32 KiB, `MAX_REFERENCES_TOTAL_BYTES` — 64 KiB.
- Удалять frontmatter из `SKILL.md` и reference; формат appendix фиксированный.
- При ошибках использовать последнюю успешную версию, не обновлять кэш частично и не логировать содержимое.

## LLM и retry

- Payload OpenRouter/OpenAI-compatible проверять по документации конкретной модели; JSON mode не включать без подтверждения совместимости.
- Повторять только 429, 5xx, timeout/connect; максимум 3 попытки, с соблюдением `Retry-After`.
- Ошибки 400/401/402/403/404 и malformed 200 не повторять.
- Не логировать HTTP body без санации.

## Безопасность и логи

- Секреты брать только из `.env`/environment; `.env` не коммитить.
- Не логировать полный пользовательский текст, исходник, ответ модели, API key/token или unsanitized provider error body.
- Логировать только безопасные метаданные: IDs, статусы, классы ошибок и размеры файлов.

## Тесты

- Использовать реальные aiogram `Message`/`MessageEntity` или реалистичные update fixtures; LLM мокать через `httpx.MockTransport`, реальные сети запрещены.
- Покрывать mention, reply, caption, UTF-16, commands, allowlist, rate-limit, dedup, retry, style references и privacy.
- При изменениях `StyleLoader` добавлять тест, доказывающий попадание reference в LLM prompt.
- Для retry проверять вычисленный backoff, не ожидая секунд.

## Проверка перед завершением

```text
ruff check .
mypy src
pytest
```

Не менять архитектуру слоёв и не добавлять зависимости без необходимости.
