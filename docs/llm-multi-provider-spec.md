# Multi-Provider LLM — spec изменений

## Мотивация
Сделать pageindex-api независимым от OpenAI: можно подключать DeepSeek, Anthropic Claude, Ollama, vLLM, Groq — любой API, совместимый с OpenAI или имеющий свой SDK.

## Архитектура (1 новый файл, 2 изменённых)

### 1. Новый файл: `llm.py`

Файл-адаптер с двумя классами:

**`LLMProvider`** (enum) — `openai | openai_compatible | anthropic`

**`LLMClient`** — единый интерфейс:
- `chat(messages, model=None, temperature=0.1) → str`
- `embed(text, model=None) → list[float]` — только для OpenAI, иначе fallback
- Внутри выбирает SDK по `provider`

### 2. Изменения в `config.py`

Добавить поля в `Settings`:
- `LLM_PROVIDER` (default `"openai"`)
- `LLM_API_KEY` (default `""`) — fallback на `CHATGPT_API_KEY`
- `LLM_BASE_URL` (default `""`) — для openai_compatible
- `LLM_CHAT_MODEL` (default `"gpt-4o-mini"`)
- `LLM_EMBEDDING_MODEL` (default `"text-embedding-3-small"`)
- `LLM_VISION_MODEL` (default `"gpt-4o-mini"`) — замена `VISION_MODEL`

Метод `get_llm_config() → LLMConfig` — собирает конфиг из настроек.

### 3. Изменения в `app.py`

- Заменить `get_openai_client()` на `get_llm_client() → LLMClient`
- Все `client.chat.completions.create(messages=..., model=...)` → `client.chat(messages, model=...)`
- `client.embeddings.create(...)` → `client.embed(text, model=...)`
- В `score_node_for_query_semantic()` — если провайдер не OpenAI → keyword scoring fallback
- Удалить `get_openai_client()`

### 4. `requirements.txt`

Добавить `anthropic>=0.30.0` (можно опционально — импорт с try/except)

### 5. `.env` — примеры

**OpenAI (как было):**
```
LLM_PROVIDER=openai
CHATGPT_API_KEY=sk-...
```

**DeepSeek:**
```
LLM_PROVIDER=openai_compatible
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_API_KEY=sk-...
LLM_CHAT_MODEL=deepseek-chat
CHATGPT_API_KEY=sk-...  # только для эмбеддингов
```

**Anthropic Claude:**
```
LLM_PROVIDER=anthropic
LLM_API_KEY=sk-ant-...
LLM_CHAT_MODEL=claude-3-5-sonnet-20241022
```

## Что не меняется

- `file_handler.py`, `task_manager.py`, `auth.py` — не трогать
- Эндпоинты (`/upload`, `/search`, `/vision`, `/ask_agent`, `/tasks`, `/registry`) — signature остаётся
- Логика обработки документов (pageindex, PyMuPDF) — не меняется

## Риски

- **Embeddings — уникальная фича OpenAI.** У DeepSeek нет публичного embedding endpoint. У Anthropic нет embeddings. Решение: если провайдер не OpenAI — `embed()` всегда возвращает `None`, scoring падает на keyword.
- **Vision — только OpenAI и Anthropic.** У OpenAI-compatible провайдеров vision через base64 работает, если модель поддерживает. У Anthropic — через SDK. Для остальных — HTTPException с сообщением "провайдер не поддерживает vision".
- **Разные форматы ответов.** Anthropic возвращает `content` как список блоков, а не строку. `LLMClient.chat()` нормализует.

## Порядок реализации

1. Создать `llm.py` (классы + фабрика)
2. Дополнить `config.py` (новые поля + `get_llm_config()`)
3. Заменить в `app.py` все вызовы OpenAI через `LLMClient`
4. Протестировать с OpenAI (регрессия)
5. Протестировать с DeepSeek
6. Обновить `README.md` (секция конфигурации)
7. Закоммитить

---

*Spec сохранён 2026-07-21*
