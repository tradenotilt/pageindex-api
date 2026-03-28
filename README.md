# pageindex-api

FastAPI-сервис для загрузки, индексации и поиска по документам. Поддерживаются PDF, DOC, DOCX и Markdown.

## Как это работает

1. Пользователь загружает файл через [`POST /upload`](app.py:246).
2. Файл сохраняется в рабочую папку и передаётся в очередь задач.
3. [`TaskManager`](task_manager.py:1) вызывает [`FileHandler.process_file()`](file_handler.py:87).
4. Для PDF используется каскадная обработка:
   - сначала пытается [`PageIndex`](file_handler.py:111),
   - затем Markdown-мост,
   - затем резервное извлечение текста по страницам.
5. Результат сохраняется в JSON внутри папки [`data/`](.gitignore:1).
6. Поиск по документу выполняется через [`POST /search`](app.py:311):
   - сначала модель выбирает релевантные узлы,
   - затем система собирает контекст,
   - если структура слабая, используется fallback по релевантным фрагментам.
7. Все эндпоинты защищены API-ключом через заголовок `X-API-Key`.

## Требования

- Python 3.11+
- OpenAI API key
- `APP_API_KEY` для защиты API
- LibreOffice (`soffice`) для конвертации DOC/DOCX в PDF

## Установка

### 1. Клонировать проект

```bash
git clone <repo-url>
cd pageindex-api
```

### 2. Создать виртуальное окружение

```bash
py -3 -m venv .venv
.venv\Scripts\activate
```

### 3. Установить зависимости

```bash
pip install -r requirements.txt
```

### 4. Создать файл `.env`

Скопируйте [`.env.example`](.env.example:1) в [`.env`](.env:1) и заполните значения:

```env
CHATGPT_API_KEY=your_openai_key
APP_API_KEY=your_private_api_key
```

При необходимости задайте и остальные параметры:

```env
DATA_DIR=./data
RESULTS_DIR=./results
MAX_FILE_SIZE=52428800
```

### 5. Запустить сервер

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

## Проверка работы

### Swagger

Откройте:

```text
http://127.0.0.1:8000/docs
```


## Структура проекта

- [`app.py`](app.py:1) — FastAPI-приложение и эндпоинты
- [`auth.py`](auth.py:1) — проверка API-ключа
- [`config.py`](config.py:1) — настройки окружения
- [`file_handler.py`](file_handler.py:1) — индексация документов
- [`task_manager.py`](task_manager.py:1) — управление задачами и реестром
- [`data/`](data/README.md) — сгенерированные индексы и служебные файлы

## Примечания

- Без [`PageIndex`](file_handler.py:39) сервис продолжит работать на fallback-режиме, но качество индексации PDF будет ниже.
- Если не задать [`APP_API_KEY`](.env.example:5), приложение не запустится.
- Если не задать [`CHATGPT_API_KEY`](.env.example:2), поиск и автоописания работать не будут.
