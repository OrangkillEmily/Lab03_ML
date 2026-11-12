"""Đánh giá cụm và tạo biểu đồ, bảng hồ sơ khách hàng."""
from pathlib import Path
import pickle
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.utils import silhouette_score_sample

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_PATH = ROOT / "models" / "best_kmeans.pkl"
IMAGE_DIR = ROOT / "docs" / "images"


def main():
    """Đọc mô hình tốt nhất, gán nhãn cụm, mô tả cụm và lưu biểu đồ.

    Parameters: Không có.
    Returns: Không trả về giá trị; lưu customer_clusters.csv, cluster_profile.csv và ảnh.
    """
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    customers = pd.read_csv(PROCESSED_DIR / "customers_clean.csv")
    X = pd.read_csv(PROCESSED_DIR / "X_scaled.csv").to_numpy(dtype=float)

    if not MODEL_PATH.exists():
        raise FileNotFoundError("Chưa có mô hình. Hãy chạy python -m src.train trước.")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    customers["Cluster"] = model.predict(X)
    customers.to_csv(PROCESSED_DIR / "customer_clusters.csv", index=False)

    feature_cols = ["Age", "Annual Income", "Spending Score"]
    profile = customers.groupby("Cluster")[feature_cols].agg(["count", "mean", "median"])
    profile.columns = [f"{feature}_{stat}" for feature, stat in profile.columns]
    gender_share = pd.crosstab(
        customers["Cluster"], customers["Gender"], normalize="index"
    ).mul(100).round(2)
    gender_share.columns = [f"Gender_%_{col}" for col in gender_share.columns]
    profile = profile.join(gender_share)
    profile.to_csv(PROCESSED_DIR / "cluster_profile.csv")

    score = silhouette_score_sample(X, customers["Cluster"].to_numpy(),
                                    sample_size=1000, random_state=42)
    print("=" * 70)
    print("ĐÁNH GIÁ KẾT QUẢ PHÂN CỤM")
    print("=" * 70)
    print(f"Số cụm: {customers['Cluster'].nunique()}")
    print(f"Inertia: {model.inertia_:.2f}")
    print(f"Silhouette Score (mẫu tối đa 1.000 điểm): {score:.4f}")
    print("\nSố khách hàng trong từng cụm:")
    print(customers["Cluster"].value_counts().sort_index().to_string())
    print("\nHồ sơ cụm:")
    print(profile.round(2).to_string())

    # Biểu đồ lựa chọn K: Elbow (Inertia) và Silhouette Score.
    results_path = PROCESSED_DIR / "experiment_results.csv"
    if results_path.exists():
        results = pd.read_csv(results_path)
        if {"K", "Inertia"}.issubset(results.columns):
            plt.figure(figsize=(7.5, 4.8))
            plt.plot(results["K"], results["Inertia"], marker="o")
            plt.xticks(results["K"].astype(int))
            plt.xlabel("Số cụm K")
            plt.ylabel("Inertia")
            plt.title("Elbow Method - Inertia theo K")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(IMAGE_DIR / "elbow_method.png", dpi=160)
            plt.close()

        score_col = next((c for c in results.columns if "Silhouette Score" in c), None)
        if score_col is not None:
            plt.figure(figsize=(7.5, 4.8))
            plt.plot(results["K"], results[score_col], marker="o")
            plt.xticks(results["K"].astype(int))
            plt.xlabel("Số cụm K")
            plt.ylabel("Silhouette Score (mẫu)")
            plt.title("So sánh Silhouette Score theo K")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(IMAGE_DIR / "silhouette_by_k.png", dpi=160)
            plt.close()

    # Biểu đồ số lượng khách hàng ở từng cụm.
    cluster_counts = customers["Cluster"].value_counts().sort_index()
    plt.figure(figsize=(7.5, 4.8))
    plt.bar(cluster_counts.index.astype(str), cluster_counts.values)
    plt.xlabel("Cluster")
    plt.ylabel("Số khách hàng")
    plt.title("Số lượng khách hàng theo cụm")
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "cluster_sizes.png", dpi=160)
    plt.close()

    # Tạo bản tóm tắt kết quả có thể đưa trực tiếp vào báo cáo.
    summary_path = ROOT / "docs" / "results_summary.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# Tóm tắt kết quả Lab03\\n\\n")
        f.write("## Kết quả mô hình\\n\\n")
        f.write(f"- Số khách hàng: **{len(customers):,}**\\n")
        f.write(f"- Số cụm được chọn: **{customers['Cluster'].nunique()}**\\n")
        f.write(f"- Inertia: **{model.inertia_:.4f}**\\n")
        f.write(f"- Silhouette Score (mẫu tối đa 1.000 điểm): **{score:.4f}**\\n\\n")
        f.write("## Kích thước các cụm\\n\\n")
        f.write("```text\\n")
        f.write(cluster_counts.rename_axis("Cluster").to_frame("Customer count").to_string())
        f.write("\\n```\\n\\n## Hồ sơ cụm\\n\\n```text\\n")
        f.write(profile.round(2).to_string())
        f.write("\\n```")
        f.write("\\n\\n## Lưu ý diễn giải\\n\\n")
        f.write("Silhouette Score được ước lượng trên mẫu, không phải toàn bộ dữ liệu. ")
        f.write("Cần kết hợp biểu đồ, kích thước cụm và trung bình đặc trưng để diễn giải; ")
        f.write("các số Cluster chỉ là mã nhóm, không phải thứ hạng.\\n")

    # Dùng mẫu cho biểu đồ để ảnh nhẹ hơn; file nhãn cụm vẫn giữ đủ mọi khách hàng.
    plot_customers = customers.sample(n=min(5000, len(customers)), random_state=42)

    # Biểu đồ thu nhập và điểm chi tiêu, tô màu theo cụm.
    plt.figure(figsize=(9, 6))
    for cluster_id in sorted(plot_customers["Cluster"].unique()):
        subset = plot_customers[plot_customers["Cluster"] == cluster_id]
        plt.scatter(subset["Annual Income"], subset["Spending Score"],
                    alpha=0.55, s=18, label=f"Cụm {cluster_id}")
    plt.xlabel("Annual Income")
    plt.ylabel("Spending Score")
    plt.legend()
    plt.title("Phân cụm khách hàng: Thu nhập và điểm chi tiêu")
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "clusters_income_spending.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 6))
    for cluster_id in sorted(plot_customers["Cluster"].unique()):
        subset = plot_customers[plot_customers["Cluster"] == cluster_id]
        plt.scatter(subset["Age"], subset["Spending Score"],
                    alpha=0.55, s=18, label=f"Cụm {cluster_id}")
    plt.xlabel("Age")
    plt.ylabel("Spending Score")
    plt.legend()
    plt.title("Phân cụm khách hàng: Tuổi và điểm chi tiêu")
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "clusters_age_spending.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    mean_scores = customers.groupby("Cluster")["Spending Score"].mean()
    plt.bar(mean_scores.index.astype(str), mean_scores.values)
    plt.xlabel("Cluster")
    plt.ylabel("Spending Score trung bình")
    plt.title("Điểm chi tiêu trung bình theo cụm")
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "cluster_spending_means.png", dpi=160)
    plt.close()

    print(f"\nĐã lưu nhãn và hồ sơ cụm tại: {PROCESSED_DIR}")
    print(f"Đã lưu biểu đồ tại: {IMAGE_DIR}")


if __name__ == "__main__":
    main()
