FROM python:3.12-slim

WORKDIR /app

# Копирование зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование кода и модели
COPY app/ ./app/
COPY models/ ./models/

# Открытие порта
EXPOSE 5000

# Запуск приложения
WORKDIR /app/app
CMD ["python", "api.py"]