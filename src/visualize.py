
"""Gom hình ảnh theo từng Run_ID và từng giai đoạn Pipeline như Lab01."""

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

SOURCE_DIR = ROOT / "docs" / "images"
REPORT_FIGURE_DIR = ROOT / "reports" / "figures"

# Các định dạng hình ảnh được hỗ trợ.
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".svg"}

# Phân loại hình ảnh theo đúng thứ tự Pipeline Lab03.
STAGE_BY_FIGURE = {
    # 01 - EDA
    "outlier_boxplots": "01_eda",
    "feature_distributions": "01_eda",
    "eda_scatterplots": "01_eda",
    "gender_distribution": "01_eda",
    "correlation_matrix": "01_eda",

    # 02 - Preprocessing
    "before_after_scaling": "02_preprocessing",
    "preprocessing_summary": "02_preprocessing",

    # 03 - Training
    "elbow_method": "03_training",
    "silhouette_by_k": "03_training",

    # 04 - Evaluation
    "cluster_sizes": "04_evaluation",
    "clusters_income_spending": "04_evaluation",
    "clusters_age_spending": "04_evaluation",
    "cluster_spending_means": "04_evaluation",
    "cluster_profiles": "04_evaluation",

    # 05 - Extended Experiments
    "feature_space_comparison": "05_extended/experiments",
    "kmeans_convergence": "05_extended/convergence",
}


def create_stage_dir(run_id, stage_name):
    """Tạo thư mục hình ảnh cho một giai đoạn của lần chạy."""

    path = REPORT_FIGURE_DIR / run_id / stage_name
    path.mkdir(parents=True, exist_ok=True)

    return path


def collect_run_figures(run_id):
    """
    Sao chép hình ảnh từ docs/images vào:

    reports/figures/<Run_ID>/
        01_eda/
        02_preprocessing/
        03_training/
        04_evaluation/
        05_extended/
            experiments/
            convergence/
        99_other/
    """

    copied = []

    run_dir = REPORT_FIGURE_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    if not SOURCE_DIR.exists():
        return copied

    # Duyệt tất cả hình ảnh trong thư mục nguồn.
    image_paths = sorted(
        path
        for path in SOURCE_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    for image_path in image_paths:

        # Xác định giai đoạn dựa trên tên hình ảnh.
        stage_name = STAGE_BY_FIGURE.get(
            image_path.stem,
            "99_other"
        )

        # Tạo thư mục tương ứng.
        stage_dir = create_stage_dir(
            run_id,
            stage_name
        )

        # Sao chép hình ảnh vào thư mục đích.
        destination = stage_dir / image_path.name

        shutil.copy2(
            image_path,
            destination
        )

        copied.append(
            str(destination.relative_to(ROOT)).replace("\\", "/")
        )

    return copied
