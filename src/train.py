"""Huấn luyện K-means với nhiều giá trị K và chọn mô hình tốt nhất."""

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
    """Đọc cấu hình từ config.yaml."""

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy config: {CONFIG_PATH}"
        )

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("config.yaml không hợp lệ.")

    return config


def main(run_id=None):
    """Thử nghiệm nhiều K và lưu mô hình có Silhouette cao nhất."""

    config = load_config()

    k_values = config.get("k_values", [2, 3, 4, 5, 6, 7, 8])
    init = config.get("init", "random")
    max_iter = int(config.get("max_iter", 100))
    random_state = int(config.get("random_state", 42))
    sample_size = int(config.get("silhouette_sample_size", 1000))
    tolerance = float(config.get("tolerance", 1e-4))

    # Kiểm tra cấu hình.
    if not k_values or any(int(k) < 2 for k in k_values):
        raise ValueError("k_values phải chứa các giá trị K >= 2.")

    if init not in ("random", "k-means++"):
        raise ValueError("init phải là 'random' hoặc 'k-means++'.")

    if max_iter < 1:
        raise ValueError("max_iter phải lớn hơn 0.")

    if sample_size < 2:
        raise ValueError("silhouette_sample_size phải >= 2.")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Đọc dữ liệu đã chuẩn hóa.
    x_path = PROCESSED_DIR / "X_scaled.csv"

    if not x_path.exists():
        raise FileNotFoundError(
            "Chưa có X_scaled.csv. "
            "Hãy chạy python -m src.preprocess trước."
        )

    X = pd.read_csv(x_path).to_numpy(dtype=float)

    if X.ndim != 2 or len(X) < 2:
        raise ValueError("Dữ liệu phải có ít nhất 2 mẫu.")

    if not pd.notna(X).all():
        raise ValueError("Dữ liệu chứa giá trị thiếu hoặc không hợp lệ.")

    results = []
    best_model = None
    best_score = float("-inf")
    best_k = None

    print("=" * 75)
    print("K-MEANS TRAINING")
    print(f"Data shape: {X.shape}")
    print(f"K values: {k_values}")
    print(f"Initialization: {init}")
    print(f"Max iterations: {max_iter}")
    print(f"Tolerance: {tolerance}")
    print(f"Random state: {random_state}")
    print("=" * 75)

    # Thử nghiệm từng giá trị K.
    for k in k_values:
        k = int(k)

        if k > len(X):
            print(f"Bỏ qua K={k}: K lớn hơn số mẫu.")
            continue

        start = time.time()

        model = KMeans(
            n_clusters=k,
            init=init,
            max_iter=max_iter,
            tol=tolerance,
            random_state=random_state,
        )

        model.fit(X)

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
            "Converged": model.converged_,
            "Stop reason": model.stop_reason_,
            "Time (seconds)": elapsed,
        })

        print(
            f"K={k} | "
            f"Inertia={model.inertia_:.2f} | "
            f"Silhouette={score:.4f} | "
            f"Iterations={model.n_iter_} | "
            f"Converged={model.converged_} | "
            f"Stop={model.stop_reason_}"
        )

        # Chọn mô hình có Silhouette Score cao nhất.
        if pd.notna(score) and score > best_score:
            best_score = score
            best_model = model
            best_k = k

    if best_model is None:
        raise RuntimeError(
            "Không chọn được mô hình. "
            "Hãy kiểm tra dữ liệu và Silhouette Score."
        )

    # Lưu kết quả thực nghiệm hiện tại.
    results_df = pd.DataFrame(results)

    results_path = PROCESSED_DIR / "experiment_results.csv"

    results_df.to_csv(
        results_path,
        index=False,
        encoding="utf-8-sig",
    )

    # Ghi lịch sử thực nghiệm.
    history = results_df.copy()

    history.insert(0, "Run_ID", run_id or "MANUAL")
    history.insert(
        1,
        "Timestamp",
        pd.Timestamp.now().isoformat(timespec="seconds"),
    )

    if LOG_PATH.exists() and LOG_PATH.stat().st_size > 0:
        old_history = pd.read_csv(LOG_PATH)

        history = pd.concat(
            [old_history, history],
            ignore_index=True,
        )

    history.to_csv(
        LOG_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # Lưu mô hình tốt nhất.
    model_path = MODEL_DIR / "best_kmeans.pkl"

    with model_path.open("wb") as file:
        pickle.dump(best_model, file)

    # Hiển thị kết quả.
    print("\n" + "=" * 75)
    print("EXPERIMENT RESULTS")
    print(results_df.round(4).to_string(index=False))

    print("\nBEST MODEL")
    print(f"Selected K: {best_k}")
    print(f"Silhouette Score: {best_score:.4f}")
    print(f"Inertia: {best_model.inertia_:.4f}")
    print(f"Iterations: {best_model.n_iter_}")
    print(f"Converged: {best_model.converged_}")
    print(f"Stop reason: {best_model.stop_reason_}")

    print(f"\nSaved model: {model_path}")
    print(f"Saved results: {results_path}")
    print(f"Saved history: {LOG_PATH}")
    print("\n[DONE] TRAINING")


if __name__ == "__main__":
    main()
