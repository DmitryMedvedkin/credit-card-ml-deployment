# Сервис прогнозирования дефолта по кредитным картам

Production-like-среду сервис машинного обучения для прогнозирования дефолта по кредитным картам. Проект охватывает полный цикл: обучение и сохранение моделей -> Flask API -> контейнеризация (Docker) -> A/B-тестирование

**Домен:** финансы / кредитный скоринг

**Датасет:** Default of Credit Card Clients Dataset с UCI Machine Learning Repository

**Docker Hub:** [https://hub.docker.com/repository/docker/dmitrymedvedkin/credit-default-api](https://hub.docker.com/r/dmitrymedvedkin/credit-default-api/tags)

**GitHub:** https://github.com/DmitryMedvedkin/credit-card-ml-deployment

## Что реализовано

- Обучено две версии модели, LogisticRegression(v1) и RandomForest(v2), для A/B-теста
- REST API на Flask с эндпоинтами `/predict` и `/health`
- A/B-роутинг. По умолчанию случайное деление трафика 50/50 между версиями модели. Также версию можно задать явно через параметр `model_version`.
- JSON-логирование запросов и ответов
- Контейнеризация (Docker) и оркестрация (docker-compose)
- Тесты API (pytest)

## Структура репозитория

```
credit-card-ml-deployment/
├── app/
│   ├── api.py # Flask-приложение (эндпоинты, A/B-роутинг, логи)
│   ├── model_handler.py # Загрузка моделей и инференс
│   └── __init__.py
├── models/
│   ├── train_model.py # Скрипт обучения моделей
│   ├── model_v1.joblib # Модель v1 (LogisticRegression)
│   ├── model_v2.joblib # Модель v2 (RandomForest)
│   └── metrics_summary.json # Метрики моделей
├── notebooks/
│   └── train_model.ipynb # EDA и обучение с визуализацией
├── tests/
│   └── test_api.py # Тесты API (pytest)
├── data/
│   └── UCI_Credit_Card.csv # Датасет
├── Dockerfile # Сборка образа сервиса
├── docker-compose.yml # Оркестрация
├── requirements.txt # Зависимости
├── ARCHITECTURE.md # Архитектура и MLOps-концепты
├── ab_test_plan.md # План A/B-тестирования
└── README.md
```

## Модели

| Модель | F1 | Precision | Recall | ROC-AUC |
|--------|------|-----------|--------|---------|
| v1 - LogisticRegression | 0.4706 | 0.3756 | 0.6297 | 0.7162 |
| v2 - RandomForest | 0.5296 | 0.4990 | 0.5641 | 0.7720 |
Метрики по классу - дефолт

## Инструкция по запуску

### Локально

```bash
# 1. Создать и активировать виртуальное окружение
python -m venv venv
venv\Scripts\activate

# 2. Установить зависимости
pip install -r requirements.txt

# 3. Обучить модели (создаст model_v1.joblib, model_v2.joblib)
python models/train_model.py

# 4. Запустить сервис
cd app
python api.py
```

Сервис доступен на `http://localhost:5000`.

### Docker

```bash
# Собрать образ
docker build -t credit-default-api .

# Запустить контейнер
docker run -p 5000:5000 credit-default-api
```

Или скачать готовый образ с Docker Hub:

```bash
docker pull dmitrymedvedkin/credit-default-api:latest
docker run -p 5000:5000 dmitrymedvedkin/credit-default-api:latest
```

### docker-compose

```bash
docker-compose up # запуск
docker-compose down # остановка
```
## API

### `GET /health`

**Ответ:**
```json
{
  "status": "healthy",
  "models_loaded": ["v1", "v2"]
}
```

### `POST /predict`

Прогноз дефолта по признакам клиента.

**Тело запроса** - JSON с 23 признаками. Опциональное поле
`model_version` (`"v1"` или `"v2"`) принудительно выбирает версию, без него версия выбирается случайно (A/B 50/50).

**Пример ответа:**
```json
{
  "prediction": 1,
  "probability": 0.9038,
  "model_version": "v2",
  "routing": "ab_random"
}
```

- `prediction` - 0 (нет дефолта) или 1 (дефолт)
- `probability` - вероятность дефолта
- `model_version` - использованная версия модели
- `routing` - `ab_random` (случайный выбор) или `explicit` (задан вручную)

### Примеры запросов (curl)

**Health:**
```bash
curl http://localhost:5000/health
```

**Предсказание (A/B, случайная версия):**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"LIMIT_BAL":50000,"SEX":1,"EDUCATION":2,"MARRIAGE":1,"AGE":30,"PAY_0":0,"PAY_2":0,"PAY_3":0,"PAY_4":0,"PAY_5":0,"PAY_6":0,"BILL_AMT1":1000,"BILL_AMT2":1000,"BILL_AMT3":1000,"BILL_AMT4":1000,"BILL_AMT5":1000,"BILL_AMT6":1000,"PAY_AMT1":500,"PAY_AMT2":500,"PAY_AMT3":500,"PAY_AMT4":500,"PAY_AMT5":500,"PAY_AMT6":500}'
```

**Предсказание с явной версией (v2):**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"LIMIT_BAL":50000,"SEX":1,"EDUCATION":2,"MARRIAGE":1,"AGE":30,"PAY_0":2,"PAY_2":2,"PAY_3":2,"PAY_4":2,"PAY_5":2,"PAY_6":2,"BILL_AMT1":90000,"BILL_AMT2":90000,"BILL_AMT3":90000,"BILL_AMT4":90000,"BILL_AMT5":90000,"BILL_AMT6":90000,"PAY_AMT1":0,"PAY_AMT2":0,"PAY_AMT3":0,"PAY_AMT4":0,"PAY_AMT5":0,"PAY_AMT6":0,"model_version":"v2"}'
```

## Тестирование

```bash
pytest tests/ -v
```

## Демонстрация работы

Скриншоты работы сервиса (API, Docker, тесты, Docker Hub) находятся в
папке screens:

- `health.jpg` - проверка `/health`
- `random_v1.jpg`, `random_v2.jpg` - A/B-роутинг (случайный выбор версии)
- `explicit_choice.jpg` - явный выбор версии модели
- `bad_user.jpg` - прогноз для клиента с высоким риском дефолта
- `docker.jpg` - работа сервиса в Docker-контейнере
- `compose_healthy.jpg` - запуск через docker-compose (healthcheck)
- `pytest.jpg` - прохождение тестов
- `dockerhub_public.jpg` - публичный образ на Docker Hub


## Признаки модели

23 признака из датасета UCI: `LIMIT_BAL`, `SEX`, `EDUCATION`, `MARRIAGE`,`AGE`, `PAY_0`, `PAY_2`–`PAY_6` (статусы платежей), `BILL_AMT1`–`BILL_AMT6` (суммы счетов), `PAY_AMT1`–`PAY_AMT6` (суммы платежей).

## Документация

- ARCHITECTURE.md - архитектура сервиса, MLOps-концепты (монолит vs микросервисы, RabbitMQ, логирование, DVC/MLflow, бизнес-метрики)
- ab_test_plan.md - план A/B-тестирования моделей

## Технологии

Python, scikit-learn, pandas, Flask, joblib, Docker, docker-compose, pytest.
