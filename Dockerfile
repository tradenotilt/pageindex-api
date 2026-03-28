FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Системные зависимости
RUN apt-get update && apt-get install -y \
    git \
    libreoffice \
    --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

# Забираем официальный исходный код PageIndex и добавляем его в PYTHONPATH,
# чтобы импорт шёл из исходников, а не из урезанного PyPI-пакета
RUN git clone https://github.com/VectifyAI/PageIndex.git /opt/PageIndex
ENV PYTHONPATH=/opt/PageIndex:${PYTHONPATH}

WORKDIR /app

# Сначала копируем manifest зависимостей для кеширования слоёв
COPY requirements.txt ./requirements.txt

# Ставим зависимости приложения
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Копируем исходники приложения
COPY . .

EXPOSE 8000

# Запускаем API
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]