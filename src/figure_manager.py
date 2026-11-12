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

    Run_ID cũ không có prefix được xem là Customer
    để duy trì khả năng tương thích.
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

    # Không cho phép Run_ID tạo đường dẫn ngoài thư mục dự kiến.
    if "/" in suffix or "\\" in suffix:
        raise ValueError(f"Run_ID chứa đường dẫn không hợp lệ: {run_id}")

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
        docs/images/additional_image_segmentation/<suffix>/
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

    Additional:
        Chỉ lấy ảnh trong thư mục riêng của Run_ID.

    Customer:
        Hỗ trợ thư mục riêng của Run_ID, thư mục pipeline
        và thư mục docs/images cũ.
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

    # Additional không quét thư mục dùng chung.
    if pipeline_name == "additional_image_segmentation":
        return source_dirs

    # Tương thích với Customer Segmentation.
    pipeline_source = SOURCE_DIR / pipeline_name

    if (
        pipeline_source.exists()
        and pipeline_source not in source_dirs
    ):
        source_dirs.append(pipeline_source)

    # Hỗ trợ các stage Customer cũ lưu trực tiếp
    # trong docs/images/.
    if SOURCE_DIR.exists():
        source_dirs.append(SOURCE_DIR)

    return source_dirs


def get_report_figure_dir(
    run_id: str,
    pipeline_name: str = None
) -> Path:
    """
    Lấy thư mục lưu hình ảnh kết quả theo Pipeline và Run_ID.

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

    Ví dụ:
        reports/figures/additional_image_segmentation/
            <run_suffix>/segmentation/
    """

    if not stage or not str(stage).strip():
        raise ValueError("stage không được để trống.")

    stage = str(stage).strip().lower()

    if "/" in stage or "\\" in stage:
        raise ValueError(f"Stage không hợp lệ: {stage}")

    stage_dir = (
        get_report_figure_dir(run_id, pipeline_name)
        / stage
    )

    stage_dir.mkdir(parents=True, exist_ok=True)

    return stage_dir


# ============================================================
# 5. FIGURE STAGE MAPPING
# ============================================================

def detect_stage(image_name: str) -> str:
    """Xác định stage dựa trên tên file ảnh."""

    name = Path(image_name).stem.lower()

    additional_mapping = {
        "elbow": [
            "elbow",
            "inertia",
        ],

        "segmentation": [
            "segmentation",
            "segmented",
            "original",
            "comparison",
        ],

        "palette": [
            "palette",
            "color_palette",
            "colour_palette",
        ],
    }

    for stage, keywords in additional_mapping.items():
        if any(keyword in name for keyword in keywords):
            return stage

    common_mapping = {
        "eda": [
            "eda",
            "distribution",
            "correlation",
            "heatmap",
            "outlier",
            "boxplot",
            "histogram",
        ],

        "preprocess": [
            "preprocess",
            "preprocessing",
            "scaled",
            "scaling",
            "transform",
            "feature",
        ],

        "train": [
            "train",
            "training",
            "model",
            "cluster",
            "silhouette",
            "dendrogram",
            "loss",
            "accuracy",
        ],

        "evaluate": [
            "evaluate",
            "evaluation",
            "metric",
            "score",
            "result",
            "confusion",
            "prediction",
        ],

        "experiments": [
            "experiment",
            "experiments",
            "tuning",
            "parameter",
        ],

        "demo_convergence": [
            "convergence",
            "demo",
            "iteration",
        ],
    }

    for stage, keywords in common_mapping.items():
        if any(keyword in name for keyword in keywords):
            return stage

    return "other"


# ============================================================
# 6. FILE COLLECTION HELPERS
# ============================================================

def _get_image_files(source_dirs):
    """Thu thập file ảnh từ các thư mục nguồn."""

    image_files = []
    seen_files = set()

    report_root = REPORT_FIGURE_DIR.resolve()

    for source_dir in source_dirs:

        if not source_dir.exists():
            continue

        for image_file in source_dir.rglob("*"):

            if not image_file.is_file():
                continue

            if image_file.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            resolved_path = image_file.resolve()

            # Không thu thập ảnh nằm trong thư mục output.
            try:
                resolved_path.relative_to(report_root)
                continue
            except ValueError:
                pass

            if resolved_path in seen_files:
                continue

            seen_files.add(resolved_path)
            image_files.append(image_file)

    return image_files


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

    Additional source:
        docs/images/additional_image_segmentation/<run_suffix>/

    Customer source:
        docs/images/customer_segmentation/<run_suffix>/
        docs/images/customer_segmentation/
        docs/images/

    Output:
        reports/figures/<pipeline_name>/<run_suffix>/<stage>/
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

        stage = detect_stage(image_file.name)

        stage_dir = output_dir / stage
        stage_dir.mkdir(parents=True, exist_ok=True)

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
