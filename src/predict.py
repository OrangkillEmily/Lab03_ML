"""Inference: gán cụm cho khách hàng mới bằng scaler và K-means đã lưu."""
import csv
import json
import pickle
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "best_kmeans.pkl"
SCALER_PATH = ROOT / "data" / "processed" / "scaler.json"
HISTORY_PATH = (ROOT / "reports" / "customer_segmentation" / "inference_history.csv")


def load_model():
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        raise FileNotFoundError("Chưa có model/scaler. Hãy chạy python -m src.pipeline trước.")
    with MODEL_PATH.open("rb") as f:
        model = pickle.load(f)
    scaler = json.loads(SCALER_PATH.read_text(encoding="utf-8"))
    return model, scaler


def predict_customer(age, annual_income, spending_score, model=None, scaler=None):
    start = time.perf_counter()
    if model is None or scaler is None:
        model, scaler = load_model()

    values = {
        "Age": float(age),
        "Annual Income": float(annual_income),
        "Spending Score": float(spending_score)
    }
    if values["Age"] <= 0 or values["Annual Income"] < 0:
        raise ValueError("Tuổi phải lớn hơn 0 và thu nhập không được âm.")
    if not 0 <= values["Spending Score"] <= 100:
        raise ValueError("Spending Score phải nằm trong khoảng 0–100.")

    features = scaler["features"]
    x_scaled = [[
        (values[col] - scaler["means"][col]) / scaler["stds"][col]
        for col in features
    ]]
    cluster = int(model.predict(x_scaled)[0])
    elapsed = time.perf_counter() - start
    result = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "input": values,
        "output": {
            "cluster": cluster,
            "scaled_features": x_scaled[0],
            "prediction_time_seconds": round(elapsed, 6)
        }
    }
    save_inference_history(result)
    return result


def save_inference_history(result):
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": result["timestamp"],
        **result["input"],
        "Cluster": result["output"]["cluster"],
        "prediction_time_seconds": result["output"]["prediction_time_seconds"]
    }
    exists = HISTORY_PATH.exists() and HISTORY_PATH.stat().st_size > 0
    with HISTORY_PATH.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


if __name__ == "__main__":
    model, scaler = load_model()
    result = predict_customer(30, 75000, 65, model, scaler)
    print(json.dumps(result, ensure_ascii=False, indent=2))
