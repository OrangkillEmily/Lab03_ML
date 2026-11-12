"""Thực nghiệm mở rộng: feature spaces, đa chỉ số và grid search K-means.

Chạy từ thư mục gốc: python -m src.experiments
Không đưa Customer ID vào mô hình. Gender chỉ xuất hiện trong không gian 5D
mang tính khảo sát vì mã hóa nhị phân tạo ra giả định khoảng cách nhân tạo.
"""
from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models.kmeans import KMeans
from src.utils import silhouette_score_sample
from src.preprocess import standardize_features

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "customers_clean.csv"
OUT = ROOT / "data" / "processed"
IMAGES = ROOT / "docs" / "images"


def internal_metrics(X, labels, centers):
    """Tính Calinski-Harabasz và Davies-Bouldin theo Euclidean."""
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    groups = np.unique(labels)
    n, k = len(X), len(groups)
    if k < 2 or k >= n:
        return float("nan"), float("nan")
    overall = X.mean(axis=0)
    within = 0.0
    between = 0.0
    scatters = []
    for g in groups:
        points = X[labels == g]
        center = centers[g]
        within += float(np.sum((points - center) ** 2))
        between += len(points) * float(np.sum((center - overall) ** 2))
        scatters.append(float(np.linalg.norm(points - center, axis=1).mean()))
    ch = (between / (k - 1)) / (within / (n - k)) if within > 0 else float("inf")

    group_centers = np.asarray([centers[g] for g in groups])
    center_dist = np.linalg.norm(group_centers[:, None, :] - group_centers[None, :, :], axis=2)
    db_values = []
    for i in range(k):
        ratios = []
        for j in range(k):
            if i != j:
                denom = center_dist[i, j]
                ratios.append((scatters[i] + scatters[j]) / denom if denom > 0 else float("inf"))
        db_values.append(max(ratios))
    return float(ch), float(np.mean(db_values))


def make_feature_spaces(df):
    """Tạo các không gian đặc trưng; chuẩn hóa riêng từng không gian."""
    work = df.copy()
    income = work["Annual Income"].replace(0, np.nan)
    work["Spending_to_Income_Ratio"] = (work["Spending Score"] / income).replace([np.inf, -np.inf], np.nan)
    work["Gender_encoded"] = work["Gender"].astype(str).str.strip().str.lower().map(
        {"female": 0.0, "male": 1.0}
    )
    spaces = {
        "2D_Core": ["Annual Income", "Spending Score"],
        "3D_Demographics": ["Age", "Annual Income", "Spending Score"],
        "3D_Ratio": ["Annual Income", "Spending Score", "Spending_to_Income_Ratio"],
        "4D_Core_Ratio": ["Age", "Annual Income", "Spending Score", "Spending_to_Income_Ratio"],
        "5D_With_Gender_Exploratory": ["Age", "Annual Income", "Spending Score", "Spending_to_Income_Ratio", "Gender_encoded"],
    }
    output = {}
    for name, cols in spaces.items():
        valid = work[cols].notna().all(axis=1)
        raw = work.loc[valid, cols].astype(float)
        scaled, means, stds = standardize_features(raw, cols)
        output[name] = {
            "X": scaled.to_numpy(dtype=float),
            "rows": work.index[valid].to_numpy(),
            "columns": cols,
            "means": means,
            "stds": stds,
        }
    return output


def run_space_comparison(spaces):
    rows = []
    for space_name, item in spaces.items():
        X = item["X"]
        for k in range(2, 9):
            model = KMeans(n_clusters=k, init="k-means++", max_iter=120, random_state=42).fit(X)
            ch, db = internal_metrics(X, model.labels_, model.centers_)
            sil = silhouette_score_sample(X, model.labels_, sample_size=1000, random_state=42)
            rows.append({"Feature_Space": space_name, "Features": ", ".join(item["columns"]),
                         "K": k, "Inertia": model.inertia_, "Silhouette": sil,
                         "Calinski_Harabasz": ch, "Davies_Bouldin": db,
                         "Iterations": model.n_iter_})
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "feature_space_comparison.csv", index=False)
    return result


def run_grid_search(spaces):
    # Tập trung tuning trên 3 cấu hình có ý nghĩa; không tối ưu mọi tổ hợp không cần thiết.
    targets = ["2D_Core", "3D_Demographics", "3D_Ratio"]
    rows = []
    for name in targets:
        X = spaces[name]["X"]
        for k in range(2, 9):
            for init in ("random", "k-means++"):
                for max_iter in (60, 120):
                    start = time.perf_counter()
                    model = KMeans(n_clusters=k, init=init, max_iter=max_iter,
                                   tol=1e-4, random_state=42).fit(X)
                    elapsed = time.perf_counter() - start
                    ch, db = internal_metrics(X, model.labels_, model.centers_)
                    sil = silhouette_score_sample(X, model.labels_, sample_size=1000, random_state=42)
                    rows.append({"Feature_Space": name, "K": k, "Init": init,
                                 "Max_Iter": max_iter, "Tolerance": 1e-4,
                                 "Inertia": model.inertia_, "Silhouette": sil,
                                 "Calinski_Harabasz": ch, "Davies_Bouldin": db,
                                 "Iterations": model.n_iter_, "Time_Seconds": elapsed})
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "grid_search_results.csv", index=False)
    return result


def plot_comparison(results):
    IMAGES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for name, part in results.groupby("Feature_Space"):
        axes[0].plot(part["K"], part["Silhouette"], marker="o", label=name)
        axes[1].plot(part["K"], part["Inertia"], marker="o", label=name)
    axes[0].set(title="Silhouette theo K và không gian đặc trưng", xlabel="K", ylabel="Silhouette")
    axes[1].set(title="Elbow theo K và không gian đặc trưng", xlabel="K", ylabel="Inertia")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(IMAGES / "feature_space_comparison.png", dpi=160)
    plt.close(fig)


def main():
    if not DATA.exists():
        raise FileNotFoundError("Chưa có customers_clean.csv. Hãy chạy python -m src.preprocess trước.")
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA)
    spaces = make_feature_spaces(df)
    print("=== SO SÁNH KHÔNG GIAN ĐẶC TRƯNG ===")
    comparison = run_space_comparison(spaces)
    print(comparison.sort_values(["Feature_Space", "K"]).to_string(index=False))
    print("\n=== GRID SEARCH ===")
    tuning = run_grid_search(spaces)
    ranked = tuning.sort_values(["Feature_Space", "Silhouette"], ascending=[True, False])
    best = ranked.groupby("Feature_Space", as_index=False).first()
    print(best[["Feature_Space", "K", "Init", "Max_Iter", "Silhouette", "Calinski_Harabasz", "Davies_Bouldin"]].to_string(index=False))
    plot_comparison(comparison)
    print("\nĐã lưu: feature_space_comparison.csv, grid_search_results.csv và feature_space_comparison.png")
    print("Lưu ý: 5D_With_Gender_Exploratory chỉ là thử nghiệm; không mặc định tốt hơn vì Gender mã hóa nhị phân.")


if __name__ == "__main__":
    main()
