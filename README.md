# rewrite-from-skill-bot

Telegram-бот, переписывающий сообщения в стиле из `runtime_skills/rewrite/SKILL.md`.
Стиль лежит в репозитории и меняется без правки кода.

## Как работает

1. Сообщение с упоминанием бота (или reply на сообщение с упоминанием).
2. `core/message_resolver.py` определяет `source_text` и `user_instruction`
   по правилам из `.opencode/skills/rewrite-request-resolution/SKILL.md`.
3. `llm/client.py` собирает prompt из SKILL-файла и вызывает LLM.
4. Ответ отправляется в чат.

## Запуск

```bash
cp .env.example .env   # заполнить BOT_TOKEN, LLM_API_KEY, ALLOWED_CHAT_IDS
pip install -e ".[dev]"
python src/main.py
```

## Модель

По умолчанию — OpenRouter `poolside/laguna-s-2.1` (free):

```env
LLM_API_URL=https://openrouter.ai/api/v1/chat/completions
LLM_API_KEY=sk-or-...
LLM_MODEL=poolside/laguna-s-2.1
```

Любой OpenAI-совместимый endpoint работает через те же три переменные.

`response_format` не входит в список поддерживаемых полей этой модели и может дать 400,
поэтому `LLM_JSON_MODE=false` по умолчанию: клиент берёт `{"text": ...}`, если модель
вернула JSON, иначе использует текст как есть.

Ошибки OpenRouter: 402 (нет кредитов), 403 (доступ закрыт), 404 (модель недоступна)
не повторяются и показывают пользователю отдельное сообщение; 429 и 5xx — с backoff.

## Использование

| Ввод | Что переписывается |
|---|---|
| `@bot текст` | текст после упоминания |
| reply + `@bot` | процитированное сообщение |
| reply + `@bot сделай короче` | процитированное сообщение + требование |

## Проверка

```bash
ruff check . && mypy src && pytest
```

## Ограничения

Rate limit и дедупликация живут в памяти процесса — при нескольких инстансах
нужен общий Redis (`core/rate_limiter.py`, `core/deduplication.py`).