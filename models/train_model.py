import json
import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "UCI_Credit_Card.csv")
TARGET = "default.payment.next.month"
FEATURES = ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6", "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6", "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    df = df.drop(columns=["ID"], errors="ignore")
    df = df.drop_duplicates()
    X = df[FEATURES]
    y = df[TARGET]
    print(f"Данные: {df.shape}, доля дефолтов: {y.mean():.3f}")
    return X, y

def build_models():
    log_reg = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    random_forest = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(n_estimators=100, max_depth=12, class_weight="balanced", random_state=42, n_jobs=-1)),
    ])
    return {"v1": log_reg, "v2": random_forest}

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "f1": round(f1_score(y_test, y_pred, pos_label=1), 4),
        "precision": round(precision_score(y_test, y_pred, pos_label=1), 4),
        "recall": round(recall_score(y_test, y_pred, pos_label=1), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
    }
    
def main():
    
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    metrics_all = {}
    for version, model in build_models().items():
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_test, y_test)
        metrics_all[version] = metrics
        artifact = {"model": model, "features": FEATURES, "version": version}
        out_path = os.path.join(BASE_DIR, f"model_{version}.joblib")
        joblib.dump(artifact, out_path)
        print(f"{version}: {metrics} сохранено в {os.path.basename(out_path)}")

    with open(os.path.join(BASE_DIR, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_all, f, indent=2, ensure_ascii=False)
    print("Метрики в metrics_summary.json")

if __name__ == "__main__":
    main()