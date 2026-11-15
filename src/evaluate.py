"""Đánh giá mô hình K-means và phân tích hồ sơ khách hàng."""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
)

from src.utils import silhouette_score_sample


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.yaml"


def load_config():
    """Đọc cấu hình dự án."""

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("config.yaml không hợp lệ.")

    return config


def main():
    """Đánh giá mô hình, lưu kết quả và tạo biểu đồ."""

    print("\n===== K-MEANS EVALUATION =====")

    config = load_config()

    processed_dir = ROOT / config.get(
        "processed_dir", "data/processed"
    )
    model_dir = ROOT / config.get("model_dir", "models")
    image_dir = ROOT / config.get("image_dir", "docs/images")

    model_path = model_dir / "best_kmeans.pkl"

    features = config.get("features", [])
    sample_size = int(
        config.get("silhouette_sample_size", 1000)
    )
    random_state = int(config.get("random_state", 42))

    image_dir.mkdir(parents=True, exist_ok=True)

    # 1. Kiểm tra các file đầu vào.
    customers_path = processed_dir / "customers_clean.csv"
    scaled_path = processed_dir / "X_scaled.csv"

    for path in [customers_path, scaled_path, model_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy file: {path}"
            )

    # 2. Đọc dữ liệu và mô hình.
    customers = pd.read_csv(customers_path)
    X_df = pd.read_csv(scaled_path)
    X = X_df.to_numpy(dtype=float)

    with model_path.open("rb") as file:
        model = pickle.load(file)

    if len(customers) != len(X):
        raise ValueError(
            "Số dòng customers_clean.csv và X_scaled.csv không khớp."
        )

    if X.shape[1] != model.centers_.shape[1]:
        raise ValueError(
            "Số đặc trưng không khớp với mô hình."
        )

    if list(X_df.columns) != features:
        raise ValueError(
            "Các đặc trưng trong X_scaled.csv không khớp config.yaml."
        )

    # 3. Dự đoán nhãn cụm.
    labels = model.predict(X)
    customers["Cluster"] = labels

    cluster_counts = (
        customers["Cluster"]
        .value_counts()
        .sort_index()
    )

    n_clusters = len(np.unique(labels))

    if n_clusters < 2 or n_clusters >= len(X):
        raise ValueError(
            "Số cụm không hợp lệ để tính các chỉ số đánh giá."
        )

    # 4. Tính các chỉ số đánh giá.
    inertia = float(model.inertia_)

    silhouette = silhouette_score_sample(
        X,
        labels,
        sample_size=sample_size,
        random_state=random_state,
    )

    calinski = float(
        calinski_harabasz_score(X, labels)
    )

    davies = float(
        davies_bouldin_score(X, labels)
    )

    metrics = {
        "n_samples": len(X),
        "n_features": X.shape[1],
        "n_clusters": n_clusters,
        "inertia": inertia,
        "silhouette_sample": silhouette,
        "calinski_harabasz": calinski,
        "davies_bouldin": davies,
        "iterations": model.n_iter_,
        "converged": model.converged_,
        "stop_reason": model.stop_reason_,
    }

    # 5. Lưu nhãn cụm.
    customers_path_out = (
        processed_dir / "customer_clusters.csv"
    )

    customers.to_csv(
        customers_path_out,
        index=False,
        encoding="utf-8-sig",
    )

    # 6. Tạo hồ sơ cụm trên dữ liệu gốc.
    profile = (
        customers.groupby("Cluster")[features]
        .agg(["count", "mean", "median"])
    )

    profile.columns = [
        f"{feature}_{stat}"
        for feature, stat in profile.columns
    ]

    # Bổ sung tỷ lệ giới tính nếu có.
    if "Gender" in customers.columns:

        gender_share = pd.crosstab(
            customers["Cluster"],
            customers["Gender"],
            normalize="index",
        ).mul(100).round(2)

        gender_share.columns = [
            f"Gender_%_{col}"
            for col in gender_share.columns
        ]

        profile = profile.join(gender_share)

    profile_path = processed_dir / "cluster_profile.csv"

    profile.to_csv(
        profile_path,
        encoding="utf-8-sig",
    )

    # 7. Lưu bảng chỉ số đánh giá.
    metrics_df = pd.DataFrame([metrics])

    metrics_path = processed_dir / "evaluation_metrics.csv"

    metrics_df.to_csv(
        metrics_path,
        index=False,
        encoding="utf-8-sig",
    )

    metrics_json_path = (
        processed_dir / "evaluation_summary.json"
    )

    with metrics_json_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            ensure_ascii=False,
            indent=4,
        )

    # 8. In kết quả.
    print("\nCÁC CHỈ SỐ ĐÁNH GIÁ")
    print(f"Số khách hàng: {len(X)}")
    print(f"Số đặc trưng: {X.shape[1]}")
    print(f"Số cụm: {n_clusters}")
    print(f"Inertia: {inertia:.4f}")
    print(f"Silhouette Score (sample): {silhouette:.4f}")
    print(f"Calinski-Harabasz: {calinski:.4f}")
    print(f"Davies-Bouldin: {davies:.4f}")
    print(f"Iterations: {model.n_iter_}")
    print(f"Converged: {model.converged_}")
    print(f"Stop reason: {model.stop_reason_}")

    print("\nSỐ LƯỢNG KHÁCH HÀNG TRONG TỪNG CỤM")
    print(cluster_counts.to_string())

    print("\nHỒ SƠ CỤM")
    print(profile.round(2).to_string())

    # 9. Biểu đồ Elbow và Silhouette.
    results_path = processed_dir / "experiment_results.csv"

    if results_path.exists():

        results = pd.read_csv(results_path)

        if {"K", "Inertia"}.issubset(results.columns):

            plt.figure(figsize=(7.5, 4.8))

            plt.plot(
                results["K"],
                results["Inertia"],
                marker="o",
            )

            plt.xticks(results["K"].astype(int))
            plt.xlabel("Number of clusters (K)")
            plt.ylabel("Inertia")
            plt.title("Elbow Method")
            plt.grid(alpha=0.3)
            plt.tight_layout()

            plt.savefig(
                image_dir / "elbow_method.png",
                dpi=160,
            )

            plt.close()

        score_col = next(
            (
                col for col in results.columns
                if "Silhouette Score" in col
            ),
            None,
        )

        if score_col:

            plt.figure(figsize=(7.5, 4.8))

            plt.plot(
                results["K"],
                results[score_col],
                marker="o",
            )

            plt.xticks(results["K"].astype(int))
            plt.xlabel("Number of clusters (K)")
            plt.ylabel("Silhouette Score")
            plt.title("Silhouette Score by K")
            plt.grid(alpha=0.3)
            plt.tight_layout()

            plt.savefig(
                image_dir / "silhouette_by_k.png",
                dpi=160,
            )

            plt.close()

    # 10. Biểu đồ kích thước cụm.
    plt.figure(figsize=(7.5, 4.8))

    plt.bar(
        cluster_counts.index.astype(str),
        cluster_counts.values,
    )

    plt.xlabel("Cluster")
    plt.ylabel("Number of customers")
    plt.title("Customer Count by Cluster")
    plt.tight_layout()

    plt.savefig(
        image_dir / "cluster_sizes.png",
        dpi=160,
    )

    plt.close()

    # 11. Biểu đồ phân cụm theo hai cặp đặc trưng.
    plot_customers = customers.sample(
        n=min(5000, len(customers)),
        random_state=random_state,
    )

    plot_pairs = [
        ("Annual Income", "Spending Score"),
        ("Age", "Spending Score"),
    ]

    for x_col, y_col in plot_pairs:

        if x_col not in customers.columns:
            continue

        if y_col not in customers.columns:
            continue

        plt.figure(figsize=(9, 6))

        for cluster_id in sorted(
            plot_customers["Cluster"].unique()
        ):

            subset = plot_customers[
                plot_customers["Cluster"] == cluster_id
            ]

            plt.scatter(
                subset[x_col],
                subset[y_col],
                alpha=0.55,
                s=18,
                label=f"Cluster {cluster_id}",
            )

        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.legend()
        plt.title(f"Customer Clusters: {x_col} vs {y_col}")
        plt.tight_layout()

        filename = (
            f"clusters_{x_col.lower().replace(' ', '_')}_"
            f"{y_col.lower().replace(' ', '_')}.png"
        )

        plt.savefig(
            image_dir / filename,
            dpi=160,
        )

        plt.close()

    # 12. Biểu đồ Spending Score trung bình.
    if "Spending Score" in customers.columns:

        mean_scores = (
            customers.groupby("Cluster")["Spending Score"]
            .mean()
        )

        plt.figure(figsize=(8, 4.5))

        plt.bar(
            mean_scores.index.astype(str),
            mean_scores.values,
        )

        plt.xlabel("Cluster")
        plt.ylabel("Mean Spending Score")
        plt.title("Average Spending Score by Cluster")
        plt.tight_layout()

        plt.savefig(
            image_dir / "cluster_spending_means.png",
            dpi=160,
        )

        plt.close()

    # 13. Tạo báo cáo Markdown tóm tắt.
    summary_path = ROOT / "docs" / "results_summary.md"

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        file.write("# Lab03 - Evaluation Summary\n\n")

        file.write("## Model Results\n\n")

        for key, value in metrics.items():
            file.write(f"- **{key}:** {value}\n")

        file.write("\n## Cluster Sizes\n\n")
        file.write("```text\n")
        file.write(
            cluster_counts.rename_axis("Cluster")
            .to_frame("Customer count")
            .to_string()
        )
        file.write("\n```\n")

        file.write("\n## Cluster Profiles\n\n")
        file.write("```text\n")
        file.write(profile.round(2).to_string())
        file.write("\n```\n")

        file.write(
            "\n## Interpretation Notes\n\n"
            "- Silhouette càng cao thường thể hiện cụm càng tách biệt.\n"
            "- Davies-Bouldin càng thấp thường càng tốt.\n"
            "- Calinski-Harabasz càng cao thường càng tốt.\n"
            "- Inertia giảm khi số cụm tăng, cần kết hợp Elbow để chọn K.\n"
            "- Cần xem thêm kích thước và hồ sơ từng cụm.\n"
        )

    # 14. Hoàn tất.
    print("\nĐÃ LƯU KẾT QUẢ")
    print(customers_path_out)
    print(profile_path)
    print(metrics_path)
    print(metrics_json_path)
    print(summary_path)
    print(image_dir)

    print("\n[DONE] EVALUATION")


if __name__ == "__main__":
    main()
