"""Khám phá dữ liệu thô và tạo các biểu đồ thống kê cơ bản."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "customer_segmentation.csv"
IMAGE_DIR = ROOT / "docs" / "images"


def main():
    """Đọc dữ liệu, in thông tin tổng quan và lưu biểu đồ khám phá.

    Parameters: Không có.
    Returns: Không trả về giá trị; in thống kê và lưu ảnh vào docs/images/.
    """
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)

    print("=" * 65)
    print("KHÁM PHÁ DỮ LIỆU KHÁCH HÀNG")
    print("=" * 65)
    print(f"Kích thước dữ liệu: {df.shape[0]} dòng, {df.shape[1]} cột")
    print("\nNăm dòng đầu:")
    print(df.head().to_string(index=False))
    print("\nKiểu dữ liệu:")
    print(df.dtypes)
    print("\nGiá trị thiếu theo cột:")
    print(df.isna().sum())
    print(f"\nSố dòng trùng hoàn toàn: {df.duplicated().sum()}")
    numeric_cols = ["Age", "Annual Income", "Spending Score"]
    print("\nThống kê các đặc trưng số:")
    print(df[numeric_cols].describe().round(2))

    # Phát hiện ngoại lệ bằng quy tắc IQR; chỉ thống kê, không tự động xóa.
    print("\nKiểm tra ngoại lệ bằng IQR (chỉ phát hiện, không tự động loại bỏ):")
    outlier_rows = []
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        count = int(((df[col] < lower) | (df[col] > upper)).sum())
        outlier_rows.append({
            "Feature": col,
            "Q1": q1,
            "Q3": q3,
            "IQR": iqr,
            "Lower bound": lower,
            "Upper bound": upper,
            "Outlier count": count,
            "Outlier percent": count / len(df) * 100
        })
    outlier_df = pd.DataFrame(outlier_rows)
    print(outlier_df.round(2).to_string(index=False))
    outlier_df.to_csv(ROOT / "data" / "processed" / "outlier_summary.csv", index=False)

    # Boxplot giúp quan sát phân phối và các điểm có khả năng là ngoại lệ.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, col in zip(axes, numeric_cols):
        ax.boxplot(df[col].dropna(), vert=True, patch_artist=True)
        ax.set_title(f"Boxplot - {col}")
        ax.set_ylabel(col)
        ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "outlier_boxplots.png", dpi=160)
    plt.close()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, numeric_cols):
        ax.hist(df[col], bins=25, edgecolor='black', alpha=0.75)
        ax.set_title(f"Phân phối {col}")
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "feature_distributions.png", dpi=150)
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    pairs = [("Age", "Annual Income"), ("Age", "Spending Score"),
             ("Annual Income", "Spending Score")]
    for ax, (x_col, y_col) in zip(axes, pairs):
        for gender in sorted(df["Gender"].dropna().unique()):
            subset = df[df["Gender"] == gender]
            ax.scatter(subset[x_col], subset[y_col], alpha=0.4, s=12, label=gender)
        ax.set_title(f"{x_col} và {y_col}")
        ax.legend(title="Gender")
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "eda_scatterplots.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    gender_counts = df["Gender"].value_counts()
    plt.bar(gender_counts.index, gender_counts.values)
    plt.title("Số lượng khách hàng theo giới tính")
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "gender_distribution.png", dpi=150)
    plt.close()

    print(f"\nĐã lưu biểu đồ khám phá tại: {IMAGE_DIR}")


if __name__ == "__main__":
    main()
