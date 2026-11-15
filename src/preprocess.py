"""Tiền xử lý dữ liệu Customer Segmentation."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"


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


def resolve_path(path):
    """Chuyển đường dẫn tương đối thành đường dẫn tuyệt đối."""

    path = Path(path)

    if not path.is_absolute():
        path = ROOT / path

    return path


def standardize_features(df, columns):
    """Chuẩn hóa dữ liệu bằng phương pháp Z-score."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Dữ liệu đầu vào phải là DataFrame.")

    missing = [
        col for col in columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(f"Thiếu các cột: {missing}")

    data = df[columns].astype(float)

    if data.isna().any().any():
        raise ValueError("Dữ liệu còn giá trị thiếu.")

    if not np.isfinite(data.to_numpy()).all():
        raise ValueError("Dữ liệu chứa giá trị vô hạn.")

    means = data.mean()
    stds = data.std(ddof=0)

    # Tránh chia cho 0 đối với đặc trưng có độ lệch chuẩn bằng 0.
    safe_stds = stds.replace(0, 1)

    scaled = (data - means) / safe_stds

    return scaled, means, safe_stds


def load_data(data_path, features):
    """Đọc và kiểm tra dữ liệu ban đầu."""

    if not data_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy dữ liệu: {data_path}"
        )

    df = pd.read_csv(data_path)

    required_columns = ["Customer ID"] + features

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Dữ liệu thiếu các cột bắt buộc: {missing}"
        )

    return df


def clean_data(df, features):
    """Làm sạch dữ liệu và giữ thông tin khách hàng."""

    data = df.copy()

    rows_before = len(data)

    # Loại bỏ các dòng trùng lặp hoàn toàn.
    data = data.drop_duplicates().copy()

    duplicate_removed = rows_before - len(data)

    # Chuyển các đặc trưng về kiểu số.
    for col in features:
        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        )

    # Thay giá trị vô hạn bằng NaN.
    data[features] = data[features].replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Điền giá trị thiếu bằng trung vị.
    for col in features:
        median_value = data[col].median()

        if pd.isna(median_value):
            raise ValueError(
                f"Cột {col} không có dữ liệu hợp lệ."
            )

        data[col] = data[col].fillna(median_value)

    # Kiểm tra dữ liệu âm.
    if (data[features] < 0).any().any():
        raise ValueError(
            "Các đặc trưng không được chứa giá trị âm."
        )

    # Xử lý Gender nếu dataset có cột này.
    if "Gender" in data.columns:

        data["Gender"] = (
            data["Gender"]
            .astype("string")
            .str.strip()
            .replace("", pd.NA)
        )

        gender_mode = data["Gender"].mode()

        if not gender_mode.empty:
            data["Gender"] = data["Gender"].fillna(
                gender_mode.iloc[0]
            )

    data = data.reset_index(drop=True)

    summary = {
        "rows_before": rows_before,
        "rows_after": len(data),
        "duplicates_removed": duplicate_removed,
        "missing_values_after": int(
            data.isna().sum().sum()
        )
    }

    return data, summary


def save_preprocessing_figures(
    clean_df,
    scaled_df,
    features,
    image_dir
):
    """Tạo biểu đồ trước và sau chuẩn hóa."""

    image_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Biểu đồ phân phối trước và sau chuẩn hóa.
    fig, axes = plt.subplots(
        len(features),
        2,
        figsize=(12, 4 * len(features)),
        squeeze=False
    )

    for i, col in enumerate(features):

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
        image_dir / "before_after_scaling.png",
        dpi=160,
        bbox_inches="tight"
    )

    plt.close(fig)

    # Biểu đồ Mean và Standard Deviation.
    fig, ax = plt.subplots(figsize=(9, 5))

    means = scaled_df.mean()
    stds = scaled_df.std(ddof=0)

    x = np.arange(len(features))
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
        features,
        rotation=15
    )

    ax.set_title("Preprocessing Summary")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        image_dir / "preprocessing_summary.png",
        dpi=160,
        bbox_inches="tight"
    )

    plt.close(fig)


def main():
    """Thực hiện toàn bộ giai đoạn Preprocessing."""

    print("\n===== PREPROCESSING =====")

    # 1. Đọc cấu hình.
    config = load_config()

    data_path = resolve_path(
        config.get(
            "data_path",
            "data/raw/customer_segmentation.csv"
        )
    )

    processed_dir = resolve_path(
        config.get("processed_dir", "data/processed")
    )

    image_dir = resolve_path(
        config.get("image_dir", "docs/images")
    )

    features = config.get(
        "features",
        ["Age", "Annual Income", "Spending Score"]
    )

    if not features:
        raise ValueError("Danh sách features không được rỗng.")

    processed_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # 2. Đọc dữ liệu.
    raw_df = load_data(data_path, features)

    print("Dữ liệu ban đầu:", raw_df.shape)

    # 3. Làm sạch dữ liệu.
    clean_df, summary = clean_data(
        raw_df,
        features
    )

    print("Dữ liệu sau làm sạch:", clean_df.shape)
    print(
        "Số dòng trùng đã loại:",
        summary["duplicates_removed"]
    )

    # 4. Chuẩn hóa dữ liệu.
    scaled_df, means, stds = standardize_features(
        clean_df,
        features
    )

    # 5. Lưu dữ liệu sạch.
    clean_path = processed_dir / "customers_clean.csv"

    clean_df.to_csv(
        clean_path,
        index=False
    )

    # 6. Lưu dữ liệu đã chuẩn hóa.
    scaled_path = processed_dir / "X_scaled.csv"

    scaled_df.to_csv(
        scaled_path,
        index=False
    )

    # 7. Lưu thông tin chuẩn hóa.
    scaler_info = {
        "method": "Z-score Standardization",
        "features": features,
        "means": means.to_dict(),
        "stds": stds.to_dict(),
        "ddof": 0
    }

    scaler_path = processed_dir / "scaler.json"

    with scaler_path.open(
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            scaler_info,
            file,
            ensure_ascii=False,
            indent=4
        )

    # 8. Lưu thống kê preprocessing.
    summary["features_used"] = features
    summary["scaled_shape"] = list(scaled_df.shape)

    summary_path = (
        processed_dir / "preprocessing_summary.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=4
        )

    # 9. Tạo biểu đồ.
    save_preprocessing_figures(
        clean_df,
        scaled_df,
        features,
        image_dir
    )

    # 10. Hiển thị kết quả.
    print("\nCác đặc trưng được sử dụng:")

    for col in features:
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
