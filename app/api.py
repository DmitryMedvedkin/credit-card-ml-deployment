import json
import logging
import random
from datetime import datetime, timezone
from flask import Flask, request, jsonify
from model_handler import load_models, predict

app = Flask(__name__)

# JSON логи
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("credit-api")

load_models()

def log_event(payload: dict):
    """Пишет структурированный JSON-лог"""
    payload["timestamp"] = datetime.now(timezone.utc).isoformat()
    logger.info(json.dumps(payload, ensure_ascii=False))


@app.route("/health", methods=["GET"])
def health():
    """Проверка здоровья сервиса"""
    models = load_models()
    return jsonify({"status": "healthy", "models_loaded": list(models.keys())}), 200


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    """Эндпоинт для предсказания дефолта. В A/B версия случайна, если не задана явно"""
    try:
        data = request.get_json(force=True)
        if data is None:
            return jsonify({"error": "Тело запроса должно быть JSON"}), 400

        version = data.pop("model_version", None)
        if version is None:
            version = random.choice(["v1", "v2"])
            routing = "ab_random"
        else:
            routing = "explicit"

        prediction, probability = predict(data, version)

        response = {
            "prediction": prediction,
            "probability": round(probability, 4),
            "model_version": version,
            "routing": routing,
        }
        log_event({"event": "predict", "request": data, "response": response})
        return jsonify(response), 200

    except ValueError as e:
        # Ошибка входных данных
        log_event({"event": "error", "message": str(e)})
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        # Прочие ошибки
        log_event({"event": "error", "message": str(e)})
        return jsonify({"error": "Внутренняя ошибка сервиса"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)