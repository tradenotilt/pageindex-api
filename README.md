# 🗂 PageIndex API

Human-like Document AI Indexer — твой личный архивариус на базе ИИ.

**Powered by PageIndex**

⚠️ **Дисклеймер:** Этот проект является независимой реализацией API для библиотеки PageIndex и не аффилирован с официальной командой разработчиков.

## 🚀 Быстрый старт

```bash
git clone https://github.com/tradenotilt/pageindex-api.git .
cd pageindex-api

cp .env.example .env
# Отредактируйте .env и добавьте свои API ключи!
```

## 🐳 Установка и запуск (Docker)

После того как вы настроили файл `.env`, выполните следующие шаги в терминале:

### Шаг 1: Сборка образа

```bash
docker compose build
```

### Шаг 2: Запуск сервера

```bash
docker compose up -d
```

### Проверка статуса

```bash
docker ps
```

Если в колонке `STATUS` вы видите `Up`, значит, ваш персональный ИИ-архивариус готов к работе!

## ✅ Как проверить работу?

```bash
# Проверить список документов
curl -s http://localhost:8000/registry -H "X-API-Key: ваш_ключ"

# Загрузить документ
curl -s -X POST http://localhost:8000/upload \
  -H "X-API-Key: ваш_ключ" \
  -F "file=@document.pdf"

# Задать вопрос
curl -s -X POST http://localhost:8000/ask_agent \
  -H "Content-Type: application/json" \
  -H "X-API-Key: ваш_ключ" \
  -d '{"query": "о чем документ"}'
```

Также доступна Swagger-документация:

```
http://IP_ВАШЕГО_СЕРВЕРА:8000/docs
```

⚠️ **ВАЖНО:** Никогда не выкладывайте свой `.env` файл в публичный доступ!

## 🔧 Настройка

### Провайдер LLM

Поддерживаются два режима:

**Вариант 1: OpenAI**
```env
LLM_PROVIDER=openai
CHATGPT_API_KEY=sk-...
```

**Вариант 2: OpenAI-совместимый (OpenCode, DeepSeek, Groq, Llama.cpp и др.)**
```env
LLM_PROVIDER=openai_compatible
LLM_API_KEY=sk-...
LLM_BASE_URL=https://opencode.ai/zen/go/v1
LLM_CHAT_MODEL=deepseek-v4-flash
# CHATGPT_API_KEY нужен только для эмбеддингов (опционально)
```

### Полный список переменных `.env`

```env
# ── LLM ──
LLM_PROVIDER=openai                     # openai | openai_compatible
CHATGPT_API_KEY=                        # OpenAI (чат + эмбеддинги)
LLM_API_KEY=                            # Ключ для OpenAI-совместимого провайдера
LLM_BASE_URL=                           # Базовый URL (для openai_compatible)
LLM_CHAT_MODEL=gpt-4o-mini              # Модель для чата

# ── API защита ──
APP_API_KEY=                            # Ваш личный ключ для доступа к API

# ── Хранилище ──
DATA_DIR=./data
RESULTS_DIR=./results
MAX_FILE_SIZE=52428800                  # 50 MB

# ── PageIndex ──
PAGEINDEX_MODEL=gpt-4o-2024-11-20
PAGEINDEX_TOC_CHECK_PAGES=20
PAGEINDEX_PDF_CHUNK_SIZE=1800
PAGEINDEX_PDF_CHUNK_OVERLAP=200
PAGEINDEX_PDF_CHUNK_THRESHOLD=1200

# ── Keywords & Vision ──
KEYWORDS_MODEL=gpt-4o-2024-11-20
VISION_MODEL=gpt-4o-2024-11-20
FORCE_ENHANCED_PARSING=false
```

> **Важно:** `APP_API_KEY` — это ваш личный пароль для доступа к серверу, а не ключ OpenAI. Используйте надежный пароль.

## 📖 Как это работает

1. Пользователь загружает файл через `POST /upload`
2. Файл сохраняется и передаётся в очередь задач (`TaskManager`)
3. `FileHandler.process_file()` индексирует документ:
   - PDF → PageIndex с обогащением таблицами → Markdown-мост → camelot для таблиц
   - PDF с изображениями → Vision OCR
4. Создаётся document_index с keywords через LLM
5. Результат сохраняется в JSON внутри папки `data/`
6. Поиск по документу через `POST /search` (с указанием doc_id)
7. Интеллектуальный вопрос через `POST /ask_agent` (автовыбор документа)
8. Все эндпоинты защищены API-ключом через заголовок `X-API-Key`
9. **LLM-клиент мульти-провайдерный:** работает с OpenAI и любым OpenAI-совместимым API
   (OpenCode, DeepSeek, Groq, Together, Llama.cpp, vLLM и др.)

## 🛠 Фичи

- **Мульти-провайдер LLM:** OpenAI или OpenAI-совместимые (OpenCode, DeepSeek, Groq и др.)
- **Умный индекс:** сохранение структуры документа (PageIndex)
- **Распознавание таблиц:** автоматическое извлечение таблиц из PDF через camelot
- **Vision OCR:** автоматическое описание графиков и изображений через LLM
- **Document Index:** автоматическое создание индекса документа с keywords
- **Полная приватность:** все данные хранятся локально в папке `data/`
- **Async Task Manager:** фоновая обработка тяжелых файлов
- **Поддержка форматов:** PDF, DOC, DOCX, Markdown
- **Connection-инструменты:** интеграция с Iva-агентом через `pageindex__*`

## 🎯 Эндпоинты

| Метод | Путь | Описание |
|-------|------|---------|
| `GET` | `/registry` | Список всех документов |
| `POST` | `/upload` | Загрузить документ (multipart/form-data) |
| `DELETE` | `/documents/{doc_id}` | Удалить документ |
| `POST` | `/search` | Поиск внутри документа (нужен doc_id) |
| `POST` | `/ask_agent` | Вопрос по всем документам (автовыбор) |
| `POST` | `/vision` | Анализ изображения (multipart: file + query) |
| `GET` | `/tasks` | Список задач индексации |
| `GET` | `/tasks/{job_id}` | Статус конкретной задачи |

### Примеры запросов

```bash
API_KEY="ваш_ключ"

# Загрузить
curl -s -X POST http://localhost:8000/upload \
  -H "X-API-Key: $API_KEY" \
  -F "file=@документ.pdf"

# Поиск в конкретном документе
curl -s -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{"doc_id": "doc_xxx", "query": "ключевой вопрос"}'

# Вопрос по всем документам
curl -s -X POST http://localhost:8000/ask_agent \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{"query": "о чем все документы"}'

# Анализ изображения
curl -s -X POST http://localhost:8000/vision \
  -H "X-API-Key: $API_KEY" \
  -F "file=@график.png" \
  -F "query=опиши график"
```

## 🔒 Безопасность

- API защищен ключом через заголовок `X-API-Key`
- Никогда не выкладывайте `.env` файл в публичный репозиторий
- Используйте сильные пароли для `APP_API_KEY`
- Все данные хранятся локально — self-hosted решение

## ⚙️ Требования

- Docker и Docker Compose
- API-ключ LLM провайдера (OpenAI или OpenAI-совместимый)

> **Примечание:** LibreOffice (`soffice`) для конвертации DOC/DOCX в PDF уже включен в Docker-образ

## 📁 Структура проекта

- `app.py` — FastAPI-приложение и эндпоинты
- `config.py` — настройки окружения (Pydantic Settings)
- `auth.py` — проверка API-ключа
- `file_handler.py` — индексация документов
- `task_manager.py` — управление задачами и реестром
- `llm.py` — мульти-провайдерный LLM-клиент (резервный)
- `Dockerfile` — инструкция по сборке Docker-образа
- `docker-compose.yml` — настройка запуска контейнера
- `data/` — все проиндексированные документы и задачи

## 📝 Примечания

- Без PageIndex сервис продолжит работать в fallback-режиме, но качество индексации PDF будет ниже
- Без `APP_API_KEY` приложение не запустится
- Без `CHATGPT_API_KEY` (режим OpenAI) или `LLM_API_KEY` (режим openai_compatible) поиск работать не будет
- Все данные хранятся локально в папке `data/` — полная приватность
- Для интеграции с Iva-агентом используются connection-инструменты `pageindex__*`
