"""Tiền xử lý dữ liệu Customer Segmentation cho Lab03."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent

RAW_FILE = ROOT / "data" / "raw" / "customer_segmentation.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
IMAGE_DIR = ROOT / "docs" / "images"

FEATURE_COLUMNS = [
    "Age",
    "Annual Income",
    "Spending Score"
]

REQUIRED_COLUMNS = [
    "Customer ID",
    "Age",
    "Gender",
    "Annual Income",
    "Spending Score"
]


def standardize_features(df, columns=None):
    """Chuẩn hóa dữ liệu bằng Z-score."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Dữ liệu đầu vào phải là DataFrame.")

    if columns is None:
        columns = df.select_dtypes(include=np.number).columns.tolist()

    missing = [col for col in columns if col not in df.columns]

    if missing:
        raise ValueError(f"Thiếu các cột: {missing}")

    data = df[columns].astype(float)

    if data.isna().any().any():
        raise ValueError("Dữ liệu còn giá trị thiếu.")

    if not np.isfinite(data.to_numpy()).all():
        raise ValueError("Dữ liệu chứa giá trị vô hạn.")

    means = data.mean()
    stds = data.std(ddof=0)

    safe_stds = stds.replace(0, 1)

    scaled = (data - means) / safe_stds

    return scaled, means, stds


def load_data():
    """Đọc dữ liệu Customer Segmentation."""

    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Không tìm thấy dữ liệu: {RAW_FILE}"
        )

    df = pd.read_csv(RAW_FILE)

    missing = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Dữ liệu thiếu các cột bắt buộc: {missing}"
        )

    return df


def clean_data(df):
    """Làm sạch dữ liệu và giữ lại thông tin khách hàng."""

    data = df.copy()

    rows_before = len(data)

    # Loại bỏ các dòng trùng lặp hoàn toàn.
    data = data.drop_duplicates().copy()

    duplicate_removed = rows_before - len(data)

    # Chuyển các đặc trưng số về kiểu numeric.
    for col in FEATURE_COLUMNS:
        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        )

    # Thay giá trị vô hạn bằng NaN.
    data[FEATURE_COLUMNS] = data[
        FEATURE_COLUMNS
    ].replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Điền giá trị thiếu bằng trung vị.
    for col in FEATURE_COLUMNS:
        median_value = data[col].median()

        if pd.isna(median_value):
            raise ValueError(
                f"Cột {col} không có dữ liệu hợp lệ."
            )

        data[col] = data[col].fillna(median_value)

    # Xử lý Gender nếu có dữ liệu thiếu.
    data["Gender"] = (
        data["Gender"]
        .astype("string")
        .str.strip()
    )

    gender_mode = data["Gender"].mode()

    if not gender_mode.empty:
        data["Gender"] = data["Gender"].fillna(
            gender_mode.iloc[0]
        )

    # Kiểm tra dữ liệu số hợp lệ.
    if (data[FEATURE_COLUMNS] < 0).any().any():
        raise ValueError(
            "Các đặc trưng tuổi, thu nhập và điểm chi tiêu "
            "không được âm."
        )

    data = data.reset_index(drop=True)

    summary = {
        "rows_before": rows_before,
        "rows_after": len(data),
        "duplicates_removed": duplicate_removed,
        "missing_values_after": int(data.isna().sum().sum())
    }

    return data, summary


def save_preprocessing_figures(raw_df, clean_df, scaled_df):
    """Tạo biểu đồ trước và sau chuẩn hóa."""

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    # Biểu đồ so sánh phân phối trước và sau chuẩn hóa.
    fig, axes = plt.subplots(
        len(FEATURE_COLUMNS),
        2,
        figsize=(12, 10)
    )

    for i, col in enumerate(FEATURE_COLUMNS):
        axes[i, 0].hist(
            clean_df[col],
            bins=30,
            edgecolor="black"
        )

        axes[i, 0].set_title(
            f"{col} - Before Scaling"
        )

        axes[i, 1].hist(
            scaled_df[col],
            bins=30,
            edgecolor="black"
        )

        axes[i, 1].set_title(
            f"{col} - After Scaling"
        )

    fig.tight_layout()

    fig.savefig(
        IMAGE_DIR / "before_after_scaling.png",
        dpi=160,
        bbox_inches="tight"
    )

    plt.close(fig)

    # Biểu đồ tổng quan preprocessing.
    fig, ax = plt.subplots(figsize=(9, 5))

    means = scaled_df.mean()
    stds = scaled_df.std(ddof=0)

    x = np.arange(len(FEATURE_COLUMNS))
    width = 0.35

    ax.bar(
        x - width / 2,
        means.values,
        width,
        label="Mean"
    )

    ax.bar(
        x + width / 2,
        stds.values,
        width,
        label="Standard Deviation"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        FEATURE_COLUMNS,
        rotation=15
    )

    ax.set_title("Preprocessing Summary")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        IMAGE_DIR / "preprocessing_summary.png",
        dpi=160,
        bbox_inches="tight"
    )

    plt.close(fig)


def main():
    """Thực hiện toàn bộ giai đoạn Preprocessing."""

    print("\n===== PREPROCESSING =====")

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    IMAGE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # 1. Đọc dữ liệu.
    raw_df = load_data()

    print("Dữ liệu ban đầu:", raw_df.shape)

    # 2. Làm sạch dữ liệu.
    clean_df, summary = clean_data(raw_df)

    print("Dữ liệu sau làm sạch:", clean_df.shape)
    print("Số dòng trùng đã loại:", summary["duplicates_removed"])

    # 3. Chuẩn hóa các đặc trưng dùng cho K-means.
    scaled_df, means, stds = standardize_features(
        clean_df,
        FEATURE_COLUMNS
    )

    # 4. Lưu dữ liệu sạch.
    clean_path = PROCESSED_DIR / "customers_clean.csv"

    clean_df.to_csv(
        clean_path,
        index=False
    )

    # 5. Lưu dữ liệu đã chuẩn hóa.
    scaled_path = PROCESSED_DIR / "X_scaled.csv"

    scaled_df.to_csv(
        scaled_path,
        index=False
    )

    # 6. Lưu thông số chuẩn hóa.
    scaler_info = {
        "method": "Z-score Standardization",
        "features": FEATURE_COLUMNS,
        "means": means.to_dict(),
        "stds": stds.to_dict(),
        "ddof": 0
    }

    scaler_path = PROCESSED_DIR / "scaler.json"

    with open(
        scaler_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            scaler_info,
            file,
            ensure_ascii=False,
            indent=4
        )

    # 7. Lưu thống kê preprocessing.
    summary["features_used"] = FEATURE_COLUMNS
    summary["scaled_shape"] = list(scaled_df.shape)

    summary_path = (
        PROCESSED_DIR / "preprocessing_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=4
        )

    # 8. Tạo biểu đồ.
    save_preprocessing_figures(
        raw_df,
        clean_df,
        scaled_df
    )

    print("\nCác đặc trưng được sử dụng:")
    for col in FEATURE_COLUMNS:
        print(
            f"{col}: mean={means[col]:.4f}, "
            f"std={stds[col]:.4f}"
        )

    print("\nĐã lưu:")
    print(clean_path)
    print(scaled_path)
    print(scaler_path)
    print(summary_path)

    print("\n[DONE] PREPROCESSING")
    

if __name__ == "__main__":
    main()
