"""Thử nghiệm K-means với nhiều giá trị K và lưu mô hình tốt nhất."""

from pathlib import Path
import pickle
import time

import pandas as pd
import yaml

from src.models.kmeans import KMeans
from src.utils import silhouette_score_sample


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.yaml"
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
LOG_PATH = ROOT / "docs" / "experiment_logs.csv"


def load_config():
    """Đọc cấu hình pipeline từ config.yaml."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy config: {CONFIG_PATH}")

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("config.yaml phải chứa một mapping hợp lệ.")

    return config


def main(run_id=None):
    """Chạy thực nghiệm chính với các giá trị K trong config.yaml."""
    config = load_config()

    k_values = config.get("k_values", [2, 3, 4, 5, 6, 7, 8])
    max_iter = int(config.get("max_iter", 60))
    random_state = int(config.get("random_state", 42))
    sample_size = int(config.get("silhouette_sample_size", 1000))
    tolerance = float(config.get("tolerance", 1e-4))

    if not k_values or any(int(k) < 2 for k in k_values):
        raise ValueError("k_values phải chứa các giá trị K >= 2.")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    x_path = PROCESSED_DIR / "X_scaled.csv"
    if not x_path.exists():
        raise FileNotFoundError(
            "Chưa có X_scaled.csv. Hãy chạy python -m src.preprocess trước."
        )

    X = pd.read_csv(x_path).to_numpy(dtype=float)

    if X.ndim != 2 or len(X) < 2:
        raise ValueError("X_scaled.csv phải có ít nhất 2 mẫu.")

    results = []
    best_model = None
    best_score = float("-inf")
    best_k = None

    print("=" * 70)
    print(f"THỬ NGHIỆM K-MEANS: K = {k_values}")
    print(
        f"max_iter={max_iter} | random_state={random_state} "
        f"| silhouette_sample_size={sample_size}"
    )
    print("=" * 70)

    for k in k_values:
        k = int(k)

        if k > len(X):
            print(f"Bỏ qua K={k}: K lớn hơn số mẫu ({len(X)}).")
            continue

        start = time.time()

        model = KMeans(
            n_clusters=k,
            init="random",
            max_iter=max_iter,
            tol=tolerance,
            random_state=random_state,
        ).fit(X)

        score = silhouette_score_sample(
            X,
            model.labels_,
            sample_size=sample_size,
            random_state=random_state,
        )

        elapsed = time.time() - start

        results.append({
            "K": k,
            "Inertia": model.inertia_,
            "Silhouette Score (sample)": score,
            "Iterations": model.n_iter_,
            "Time (seconds)": elapsed,
        })

        print(
            f"K={k} | Inertia={model.inertia_:.2f} | "
            f"Silhouette={score:.4f} | Iterations={model.n_iter_} | "
            f"Time={elapsed:.2f}s"
        )

        if pd.notna(score) and score > best_score:
            best_score = score
            best_model = model
            best_k = k

    if best_model is None:
        raise RuntimeError(
            "Không chọn được mô hình. Hãy kiểm tra dữ liệu và Silhouette scores."
        )

    results_df = pd.DataFrame(results)
    results_df.to_csv(
        PROCESSED_DIR / "experiment_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Lưu lịch sử thực nghiệm, không ghi đè các lần chạy trước.
    history = results_df.copy()
    history.insert(0, "Run_ID", run_id or "MANUAL")
    history.insert(1, "Timestamp", pd.Timestamp.now().isoformat(timespec="seconds"))

    if LOG_PATH.exists() and LOG_PATH.stat().st_size > 0:
        old_history = pd.read_csv(LOG_PATH)
        history = pd.concat([old_history, history], ignore_index=True)

    history.to_csv(LOG_PATH, index=False, encoding="utf-8-sig")

    with (MODEL_DIR / "best_kmeans.pkl").open("wb") as file:
        pickle.dump(best_model, file)

    print("\nBẢNG KẾT QUẢ:")
    print(results_df.round(4).to_string(index=False))
    print(f"\nK được chọn theo Silhouette Score: {best_k} (score={best_score:.4f})")
    print(f"Đã lưu mô hình: {MODEL_DIR / 'best_kmeans.pkl'}")


if __name__ == "__main__":
    main()
