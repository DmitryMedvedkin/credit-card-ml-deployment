import os
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")

# Кэш загруженных моделей
_MODELS = {}

def load_models():
    """Загружаем все доступные версии моделей в память"""
    global _MODELS
    if _MODELS:
        return _MODELS
    for version in ("v1", "v2"):
        path = os.path.join(MODELS_DIR, f"model_{version}.joblib")
        if os.path.exists(path):
            _MODELS[version] = joblib.load(path)
            print(f"Загружена модель {version} из {path}")
        else:
            print(f"Не найден файл модели {path}")
    if not _MODELS:
        raise RuntimeError("Не загружено ни одной модели. Запустите train_model.py")
    return _MODELS


def predict(data: dict, version: str):
    """
    Делает предсказание моделью заданной версии.
    data - словарь признаков {"LIMIT_BAL": 50000, "SEX": 1, ...}
    version - "v1" или "v2"
    Возвращает (prediction, probability):
    prediction - 0 или 1 (дефолт)
    probability - вероятность дефолта (класс 1)
    """
    models = load_models()
    if version not in models:
        raise ValueError(f"Модель версии {version} недоступна. Доступны: {list(models.keys())}")

    artifact = models[version]
    model = artifact["model"]
    features = artifact["features"]

    # Проверяем, что все нужные признаки переданы
    missing = [f for f in features if f not in data]
    if missing:
        raise ValueError(f"Отсутствуют признаки: {missing}")

    # Выстраиваем признаки строго в порядке features
    row = pd.DataFrame([[data[f] for f in features]], columns=features)
    prediction = int(model.predict(row)[0])
    probability = float(model.predict_proba(row)[0][1])
    return prediction, probability