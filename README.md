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