"""Tạo demo lịch sử hội tụ K-means trên mặt phẳng Income-Spending."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.models.kmeans import KMeans

ROOT = Path(__file__).resolve().parents[1]


def main():
    df = pd.read_csv(ROOT / "data/processed/customers_clean.csv")
    cols = ["Annual Income", "Spending Score"]
    X = df[cols].to_numpy(dtype=float)
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    rng = np.random.default_rng(42)
    X_demo = X[rng.choice(len(X), size=min(1200, len(X)), replace=False)]
    model = KMeans(n_clusters=5, init="random", max_iter=100, random_state=42).fit(X_demo)

    out = ROOT / "docs/images"
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot([h["iteration"] for h in model.history_],
                 [h["inertia"] for h in model.history_], marker="o")
    axes[0].set(title="Inertia giảm qua từng vòng lặp", xlabel="Iteration", ylabel="Inertia")
    axes[0].grid(alpha=0.25)
    axes[1].plot([h["iteration"] for h in model.history_],
                 [h["centroid_shift"] for h in model.history_], marker="o")
    axes[1].set(title="Độ dịch chuyển tâm cụm", xlabel="Iteration", ylabel="Centroid shift")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "kmeans_convergence.png", dpi=160)
    plt.close(fig)

    pd.DataFrame([{"iteration": h["iteration"], "inertia": h["inertia"],
                   "centroid_shift": h["centroid_shift"],
                   "cluster_sizes": str(h["cluster_sizes"])} for h in model.history_]
                 ).to_csv(ROOT / "data/processed/convergence_history.csv", index=False)
    print(f"Số vòng lặp: {model.n_iter_}; Inertia: {model.inertia_:.4f}")
    print("Đã lưu convergence_history.csv và kmeans_convergence.png")


if __name__ == "__main__":
    main()
