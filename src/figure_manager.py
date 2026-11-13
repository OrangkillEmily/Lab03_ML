"""Quản lý và thu thập hình ảnh theo Pipeline và Run_ID."""

from pathlib import Path
import filecmp
import shutil


# ============================================================
# 1. PATH CONFIGURATION
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

SOURCE_DIR = ROOT_DIR / "docs" / "images"

REPORT_FIGURE_DIR = ROOT_DIR / "reports" / "figures"


# ============================================================
# 2. PIPELINE CONFIGURATION
# ============================================================

PIPELINE_NAMES = {
    "customer_segmentation",
    "additional_image_segmentation",
}

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".svg",
    ".webp",
}


# ============================================================
# 3. RUN ID HELPERS
# ============================================================

def validate_pipeline_name(pipeline_name):
    """Kiểm tra và chuẩn hóa tên pipeline."""

    if pipeline_name is None:
        raise ValueError("pipeline_name không được để trống.")

    pipeline_name = str(pipeline_name).strip().lower()

    if pipeline_name not in PIPELINE_NAMES:
        raise ValueError(
            f"Pipeline không hợp lệ: {pipeline_name}. "
            f"Các pipeline hợp lệ: {sorted(PIPELINE_NAMES)}"
        )

    return pipeline_name


def get_pipeline_name(run_id: str) -> str:
    """
    Lấy tên pipeline từ Run_ID.

    Run_ID mới:
        customer_segmentation--<suffix>
        additional_image_segmentation--<suffix>

    Run_ID cũ không có prefix được xem là Customer.
    """

    if not run_id:
        return "customer_segmentation"

    run_id = str(run_id).strip()

    if "--" in run_id:
        pipeline_name = run_id.split("--", 1)[0]
        return validate_pipeline_name(pipeline_name)

    return "customer_segmentation"


def get_run_suffix(run_id: str) -> str:
    """Lấy phần timestamp và ID phía sau prefix pipeline."""

    if not run_id:
        raise ValueError("run_id không được để trống.")

    run_id = str(run_id).strip()

    suffix = (
        run_id.split("--", 1)[1]
        if "--" in run_id
        else run_id
    )

    if not suffix or suffix in {".", ".."}:
        raise ValueError(f"Run_ID không hợp lệ: {run_id}")

    if "/" in suffix or "\\" in suffix:
        raise ValueError(
            f"Run_ID chứa đường dẫn không hợp lệ: {run_id}"
        )

    return suffix


def validate_run_pipeline(run_id, pipeline_name):
    """Đảm bảo Run_ID thuộc đúng pipeline."""

    pipeline_name = validate_pipeline_name(pipeline_name)
    run_pipeline = get_pipeline_name(run_id)

    if run_pipeline != pipeline_name:
        raise ValueError(
            f"Run_ID thuộc pipeline '{run_pipeline}', "
            f"nhưng pipeline_name được truyền vào là "
            f"'{pipeline_name}'."
        )

    return pipeline_name


# ============================================================
# 4. DIRECTORY HELPERS
# ============================================================

def get_run_source_dir(
    run_id: str,
    pipeline_name: str = None
) -> Path:
    """
    Lấy thư mục ảnh nguồn riêng của một lần chạy.

    Ví dụ:
        docs/images/customer_segmentation/<suffix>/
    """

    if pipeline_name is None:
        pipeline_name = get_pipeline_name(run_id)

    pipeline_name = validate_run_pipeline(
        run_id,
        pipeline_name
    )

    return (
        SOURCE_DIR
        / pipeline_name
        / get_run_suffix(run_id)
    )


def get_source_dirs(
    pipeline_name: str,
    run_id: str = None
) -> list:
    """
    Lấy danh sách thư mục ảnh nguồn.

    Ưu tiên thư mục riêng của Run_ID.

    Customer hỗ trợ thêm ảnh legacy được lưu trực tiếp
    trong thư mục pipeline hoặc docs/images.

    Các thư mục legacy chỉ được quét file trực tiếp,
    không quét đệ quy để tránh lấy ảnh của Run_ID cũ.
    """

    pipeline_name = validate_pipeline_name(pipeline_name)

    source_dirs = []

    if run_id:
        validate_run_pipeline(run_id, pipeline_name)

        run_source = get_run_source_dir(
            run_id,
            pipeline_name
        )

        if run_source.exists():
            source_dirs.append(run_source)

    # Additional chỉ lấy ảnh trong thư mục riêng Run_ID.
    if pipeline_name == "additional_image_segmentation":
        return source_dirs

    # Hỗ trợ ảnh Customer lưu trực tiếp trong thư mục pipeline.
    pipeline_source = SOURCE_DIR / pipeline_name

    if (
        pipeline_source.exists()
        and pipeline_source not in source_dirs
    ):
        source_dirs.append(pipeline_source)

    # Hỗ trợ ảnh Customer legacy lưu trực tiếp trong docs/images.
    if SOURCE_DIR.exists():
        source_dirs.append(SOURCE_DIR)

    return source_dirs


def get_report_figure_dir(
    run_id: str,
    pipeline_name: str = None
) -> Path:
    """
    Lấy thư mục lưu hình ảnh theo Pipeline và Run_ID.

    reports/figures/<pipeline_name>/<run_suffix>/
    """

    if pipeline_name is None:
        pipeline_name = get_pipeline_name(run_id)

    pipeline_name = validate_run_pipeline(
        run_id,
        pipeline_name
    )

    return (
        REPORT_FIGURE_DIR
        / pipeline_name
        / get_run_suffix(run_id)
    )


def get_stage_figure_dir(
    run_id: str,
    stage: str,
    pipeline_name: str = None
) -> Path:
    """
    Lấy và tạo thư mục lưu hình cho một stage.

    Hỗ trợ stage lồng nhau:

        05_extended/convergence
        05_extended/experiments
    """

    if not stage or not str(stage).strip():
        raise ValueError("stage không được để trống.")

    stage_path = Path(str(stage).strip())

    if stage_path.is_absolute():
        raise ValueError(f"Stage không hợp lệ: {stage}")

    if any(part in {"..", "."} for part in stage_path.parts):
        raise ValueError(f"Stage không hợp lệ: {stage}")

    if pipeline_name is None:
        pipeline_name = get_pipeline_name(run_id)

    stage_dir = (
        get_report_figure_dir(run_id, pipeline_name)
        / stage_path
    )

    stage_dir.mkdir(parents=True, exist_ok=True)

    return stage_dir


# ============================================================
# 5. FIGURE STAGE MAPPING
# ============================================================

def detect_customer_stage(image_name: str) -> str:
    """
    Phân loại ảnh Customer Segmentation.

    Output:
        01_eda
        02_preprocessing
        03_training
        04_evaluation
        05_extended/convergence
        05_extended/experiments
    """

    name = Path(image_name).stem.lower()

    # --------------------------------------------------------
    # 05 - EXTENDED: kiểm tra trước các nhóm khác
    # --------------------------------------------------------

    if any(key in name for key in [
        "convergence",
        "kmeans_convergence",
        "iteration",
    ]):
        return "05_extended/convergence"

    if any(key in name for key in [
        "feature_space",
        "feature_spaces",
        "experiments",
        "experiment",
        "grid_search",
        "parameter",
        "tuning",
        "comparison",
    ]):
        return "05_extended/experiments"

    # --------------------------------------------------------
    # 01 - EDA
    # --------------------------------------------------------

    if any(key in name for key in [
        "eda",
        "feature_distribution",
        "scatterplot",
        "scatter",
        "gender_distribution",
        "outlier",
        "correlation",
        "histogram",
        "boxplot",
    ]):
        return "01_eda"

    # --------------------------------------------------------
    # 02 - PREPROCESSING
    # --------------------------------------------------------

    if any(key in name for key in [
        "preprocessing",
        "preprocess",
        "before_after_scaling",
        "scaling",
        "scaled",
        "transform",
    ]):
        return "02_preprocessing"

    # --------------------------------------------------------
    # 03 - TRAINING
    # --------------------------------------------------------

    if any(key in name for key in [
        "elbow",
        "silhouette_by_k",
        "inertia",
        "training",
        "train",
    ]):
        return "03_training"

    # --------------------------------------------------------
    # 04 - EVALUATION
    # --------------------------------------------------------

    if any(key in name for key in [
        "clusters",
        "cluster_sizes",
        "cluster_spending",
        "cluster_income",
        "cluster_age",
        "cluster_profile",
        "evaluation",
        "evaluate",
        "metric",
        "score",
        "result",
    ]):
        return "04_evaluation"

    # Không xác định được thì đưa vào experiments.
    return "05_extended/experiments"


def detect_additional_stage(image_name: str) -> str:
    """
    Phân loại ảnh Additional Image Segmentation.
    """

    name = Path(image_name).stem.lower()

    # 01 - PREPROCESSING
    if any(key in name for key in [
        "preprocess",
        "preprocessing",
        "summary",
        "original_image",
    ]):
        return "01_preprocessing"

    # 02 - ELBOW
    if any(key in name for key in [
        "elbow",
        "inertia",
    ]):
        return "02_elbow"

    # 04 - PALETTE
    if any(key in name for key in [
        "palette",
        "color_palette",
        "colour_palette",
    ]):
        return "04_palette"

    # 03 - SEGMENTATION
    if any(key in name for key in [
        "segmentation",
        "segmented",
        "comparison",
        "result",
    ]):
        return "03_segmentation"

    return "03_segmentation"


def detect_stage(
    image_name: str,
    pipeline_name: str = "customer_segmentation"
) -> str:
    """
    Xác định stage dựa trên tên file và pipeline.
    """

    pipeline_name = validate_pipeline_name(pipeline_name)

    if pipeline_name == "customer_segmentation":
        return detect_customer_stage(image_name)

    return detect_additional_stage(image_name)


# ============================================================
# 6. FILE COLLECTION HELPERS
# ============================================================

def _get_image_files(source_dirs):
    """
    Thu thập file ảnh từ các thư mục nguồn.

    Quy tắc:
    - Thư mục Run_ID: quét đệ quy.
    - Thư mục pipeline dùng chung: chỉ lấy file trực tiếp.
    - docs/images dùng chung: chỉ lấy file trực tiếp.

    Nhờ vậy không lấy nhầm ảnh từ các Run_ID cũ.
    """

    image_files = []
    seen_files = set()

    report_root = REPORT_FIGURE_DIR.resolve()

    legacy_dirs = {
        SOURCE_DIR.resolve(),
        (SOURCE_DIR / "customer_segmentation").resolve(),
    }

    for source_dir in source_dirs:

        if not source_dir.exists():
            continue

        resolved_source = source_dir.resolve()

        # Thư mục legacy chỉ quét cấp trực tiếp.
        if resolved_source in legacy_dirs:
            candidates = source_dir.iterdir()
        else:
            # Thư mục riêng Run_ID được phép quét đệ quy.
            candidates = source_dir.rglob("*")

        for image_file in candidates:

            if not image_file.is_file():
                continue

            if image_file.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            resolved_path = image_file.resolve()

            # Không thu thập ảnh từ thư mục output.
            try:
                resolved_path.relative_to(report_root)
                continue
            except ValueError:
                pass

            if resolved_path in seen_files:
                continue

            seen_files.add(resolved_path)
            image_files.append(image_file)

    return sorted(image_files, key=lambda path: str(path).lower())


def _unique_destination(destination: Path) -> Path:
    """Tạo tên file mới nếu file đích đã tồn tại."""

    if not destination.exists():
        return destination

    counter = 1

    while True:
        new_destination = destination.with_name(
            f"{destination.stem}_{counter}{destination.suffix}"
        )

        if not new_destination.exists():
            return new_destination

        counter += 1


def _copy_image(image_file: Path, destination: Path) -> bool:
    """
    Copy ảnh và tránh tạo bản sao trùng lặp.

    Returns:
        True nếu đã copy.
        False nếu file đích đã có nội dung giống hệt.
    """

    destination.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if destination.exists():

        try:
            if filecmp.cmp(
                image_file,
                destination,
                shallow=False
            ):
                return False

        except OSError:
            pass

        destination = _unique_destination(destination)

    shutil.copy2(image_file, destination)

    return True


# ============================================================
# 7. COLLECT FIGURES
# ============================================================

def collect_run_figures(
    run_id: str,
    pipeline_name: str = None
) -> dict:
    """
    Thu thập hình ảnh của một lần chạy.

    Customer source:
        docs/images/customer_segmentation/<run_suffix>/

    Additional source:
        docs/images/additional_image_segmentation/<run_suffix>/

    Output:
        reports/figures/<pipeline_name>/<run_suffix>/
    """

    if not run_id:
        raise ValueError("run_id không được để trống.")

    if pipeline_name is None:
        pipeline_name = get_pipeline_name(run_id)

    pipeline_name = validate_run_pipeline(
        run_id,
        pipeline_name
    )

    source_dirs = get_source_dirs(
        pipeline_name=pipeline_name,
        run_id=run_id
    )

    output_dir = get_report_figure_dir(
        run_id=run_id,
        pipeline_name=pipeline_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    if not source_dirs:
        print(
            "[Figure Manager] Không tìm thấy thư mục ảnh nguồn "
            f"cho Run_ID: {run_id}"
        )

        return {
            "pipeline": pipeline_name,
            "run_id": run_id,
            "total": 0,
            "stages": {},
            "output_dir": str(output_dir),
        }

    image_files = _get_image_files(source_dirs)

    copied_by_stage = {}

    for image_file in image_files:

        stage = detect_stage(
            image_file.name,
            pipeline_name
        )

        stage_dir = get_stage_figure_dir(
            run_id=run_id,
            stage=stage,
            pipeline_name=pipeline_name
        )

        destination = stage_dir / image_file.name

        copied = _copy_image(
            image_file,
            destination
        )

        if copied:
            copied_by_stage[stage] = (
                copied_by_stage.get(stage, 0) + 1
            )

    total_images = sum(copied_by_stage.values())

    print("\n========== FIGURE COLLECTION ==========")
    print(f"Pipeline : {pipeline_name}")
    print(f"Run ID   : {run_id}")
    print(f"Found    : {len(image_files)}")
    print(f"Copied   : {total_images}")
    print(f"Output   : {output_dir}")

    for stage, count in sorted(copied_by_stage.items()):
        print(f"  - {stage}: {count} image(s)")

    print("=======================================\n")

    return {
        "pipeline": pipeline_name,
        "run_id": run_id,
        "total": total_images,
        "stages": copied_by_stage,
        "output_dir": str(output_dir),
    }


# ============================================================
# 8. STAGE MAPPING PREVIEW
# ============================================================

def print_stage_mapping():
    """Hiển thị quy tắc phân loại ảnh của hai pipeline."""

    customer_examples = [
        "feature_distributions.png",
        "before_after_scaling.png",
        "elbow_method.png",
        "silhouette_by_k.png",
        "clusters_income_spending.png",
        "cluster_sizes.png",
        "feature_space_comparison.png",
        "kmeans_convergence.png",
    ]

    additional_examples = [
        "preprocessing_summary.png",
        "elbow_method.png",
        "segmentation_result.png",
        "color_palette.png",
    ]

    print("Figure Manager")
    print("=" * 60)

    print("\nCustomer Segmentation:")

    for filename in customer_examples:
        stage = detect_stage(
            filename,
            "customer_segmentation"
        )
        print(f"{filename:40} -> {stage}")

    print("\nAdditional Image Segmentation:")

    for filename in additional_examples:
        stage = detect_stage(
            filename,
            "additional_image_segmentation"
        )
        print(f"{filename:40} -> {stage}")

    print()


# ============================================================
# 9. MAIN
# ============================================================

if __name__ == "__main__":
    print_stage_mapping()
