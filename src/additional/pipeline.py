"""Main pipeline for K-Means image segmentation."""

from pathlib import Path
import traceback

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image

from src.tracking import (
    create_run,
    update_progress,
    write_log,
    finish_run,
)

from src.figure_manager import (
    get_stage_figure_dir,
    get_report_figure_dir,
)

from .image_io import load_rgb_image, image_to_pixels
from .kmeans import KMeansFromScratch, make_sample
from .segmentation import segment_image
from .visualization import (
    plot_elbow,
    plot_segmentation_comparison,
    plot_color_palette,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

RAW_IMAGE_DIR = ROOT_DIR / "data" / "additional"

PIPELINE_NAME = "additional_image_segmentation"

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}

ELBOW_K_VALUES = [2, 3, 4, 5, 6, 8]
SEGMENTATION_K_VALUES = [3, 5, 8]

MAX_FIT_PIXELS = 20_000
MAX_ITER = 100
RANDOM_STATE = 42
PREDICT_BATCH_SIZE = 50_000


# ============================================================
# 2. TRACKING HELPERS
# ============================================================

def log_message(run_id, message, level="INFO"):
    """Write a message to the Run_ID log."""
    print(f"[{level}] {message}")

    write_log(
        run_id,
        "ADDITIONAL_IMAGE_SEGMENTATION",
        level,
        message,
    )


def start_stage(run_id, stage, message):
    """Mark a pipeline stage as running."""
    log_message(run_id, message)
    update_progress(run_id, stage, "RUNNING")


def complete_stage(run_id, stage, message):
    """Mark a pipeline stage as successful."""
    update_progress(run_id, stage, "SUCCESS")
    log_message(run_id, message)


def fail_stage(run_id, stage, error):
    """Log a stage failure without finalizing the entire run."""
    update_progress(run_id, stage, "FAILED")

    log_message(
        run_id,
        f"Giai đoạn {stage} thất bại: {error}",
        level="ERROR",
    )


# ============================================================
# 3. IMAGE DISCOVERY
# ============================================================

def discover_images(input_dir=RAW_IMAGE_DIR):
    """
    Find supported image files in the input directory.

    Images are read recursively, allowing users to organize
    input files into subdirectories.
    """
    input_dir = Path(input_dir)

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục ảnh đầu vào: {input_dir}"
        )

    image_paths = sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not image_paths:
        raise FileNotFoundError(
            f"Không tìm thấy ảnh hợp lệ trong: {input_dir}"
        )

    return image_paths


# ============================================================
# 4. OUTPUT DIRECTORY
# ============================================================

def get_stage_dir(run_id, stage):
    """
    Get the output directory for a pipeline stage.

    All stage directories are managed by the shared Figure Manager.
    """
    output_dir = get_stage_figure_dir(
        run_id,
        stage,
        pipeline_name=PIPELINE_NAME,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return output_dir


def get_run_output_dir(run_id):
    """Get the root output directory for the current run."""
    output_dir = get_report_figure_dir(
        run_id,
        pipeline_name=PIPELINE_NAME,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return output_dir


def save_figure(fig, output_path, dpi=160):
    """Save and close a Matplotlib figure."""
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
    )

    plt.close(fig)

    return output_path


# ============================================================
# 5. ELBOW METHOD
# ============================================================

def run_elbow_experiment(image_name, image_array):
    """
    Evaluate K-Means inertia for different K values.

    K values larger than the available fitting sample count
    are skipped.
    """
    pixels, _, _ = image_to_pixels(image_array)

    fit_pixels = make_sample(
        pixels,
        max_samples=MAX_FIT_PIXELS,
        random_state=RANDOM_STATE,
    )

    if not np.isfinite(fit_pixels).all():
        raise ValueError(
            f"Dữ liệu pixel của {image_name} chứa NaN hoặc Inf."
        )

    tested_k = [
        k for k in ELBOW_K_VALUES
        if k <= len(fit_pixels)
    ]

    if not tested_k:
        raise ValueError(
            f"Không đủ pixel để thực hiện Elbow Method cho {image_name}."
        )

    records = []

    for k in tested_k:
        model = KMeansFromScratch(
            n_clusters=k,
            max_iter=MAX_ITER,
            random_state=RANDOM_STATE,
        )

        model.fit(fit_pixels)

        records.append({
            "image": image_name,
            "K": k,
            "inertia": model.inertia_,
            "n_iter": model.n_iter_,
            "n_samples": len(fit_pixels),
        })

    return records


# ============================================================
# 6. IMAGE SEGMENTATION
# ============================================================

def run_segmentation_experiment(image_path, image_array, run_id):
    """Segment one image with the configured K values."""
    image_name = image_path.name
    image_stem = image_path.stem

    segmented_dir = get_stage_dir(
        run_id,
        "segmentation",
    )

    palette_dir = get_stage_dir(
        run_id,
        "palette",
    )

    segmented_results = {}
    palette_records = []
    output_files = {}

    for k in SEGMENTATION_K_VALUES:
        result = segment_image(
            image_array,
            n_clusters=k,
            max_fit_pixels=MAX_FIT_PIXELS,
            random_state=RANDOM_STATE,
            max_iter=MAX_ITER,
            predict_batch_size=PREDICT_BATCH_SIZE,
        )

        segmented_results[k] = result["segmented"]

        # Save segmented image.
        segmented_path = (
            segmented_dir
            / f"{image_stem}_segmented_k{k}.png"
        )

        Image.fromarray(
            result["segmented"]
        ).save(segmented_path)

        output_files[k] = str(segmented_path)

        # Collect color palette statistics.
        labels = result["labels"].ravel()
        centers = result["model"].cluster_centers_

        counts = np.bincount(
            labels,
            minlength=k,
        )

        total_pixels = max(
            int(counts.sum()),
            1,
        )

        for cluster_id, (center, count) in enumerate(
            zip(centers, counts)
        ):
            palette_records.append({
                "image": image_name,
                "K": k,
                "cluster": cluster_id,
                "red": round(float(center[0]), 2),
                "green": round(float(center[1]), 2),
                "blue": round(float(center[2]), 2),
                "pixel_count": int(count),
                "proportion": float(count / total_pixels),
            })

        # Save palette visualization for each K.
        palette_fig = plot_color_palette(
            labels=labels,
            centers=centers,
            image_name=image_name,
            k=k,
        )

        save_figure(
            palette_fig,
            palette_dir / f"{image_stem}_palette_k{k}.png",
        )

    # Save comparison of original image and segmented results.
    comparison_fig = plot_segmentation_comparison(
        original=image_array,
        segmented_by_k=segmented_results,
        image_name=image_name,
    )

    comparison_path = save_figure(
        comparison_fig,
        segmented_dir / f"{image_stem}_comparison.png",
    )

    return {
        "image_name": image_name,
        "segmented_results": segmented_results,
        "segmented_files": output_files,
        "comparison_file": str(comparison_path),
        "palette_records": palette_records,
    }


# ============================================================
# 7. MAIN PIPELINE
# ============================================================

def main():
    """Run the complete Additional Image Segmentation pipeline."""

    run_id = create_run(
        pipeline_name=PIPELINE_NAME,
    )

    print("\n" + "=" * 65)
    print("ADDITIONAL IMAGE SEGMENTATION")
    print(f"Run_ID: {run_id}")
    print("=" * 65)

    try:
        # ----------------------------------------------------
        # Stage 1: Load images
        # ----------------------------------------------------

        start_stage(
            run_id,
            "IMAGE_LOADING",
            "Bắt đầu tìm và đọc ảnh đầu vào.",
        )

        image_paths = discover_images()

        images = {}

        for image_path in image_paths:
            image_array = load_rgb_image(image_path)

            images[image_path] = image_array

            log_message(
                run_id,
                f"Đã đọc ảnh {image_path.name}: "
                f"{image_array.shape[1]}x"
                f"{image_array.shape[0]} pixels.",
            )

        complete_stage(
            run_id,
            "IMAGE_LOADING",
            f"Đã đọc thành công {len(images)} ảnh.",
        )

        # ----------------------------------------------------
        # Stage 2: Elbow Method
        # ----------------------------------------------------

        start_stage(
            run_id,
            "ELBOW",
            "Bắt đầu thực nghiệm Elbow Method.",
        )

        elbow_records = []

        for image_path, image_array in images.items():
            records = run_elbow_experiment(
                image_path.name,
                image_array,
            )

            elbow_records.extend(records)

        elbow_df = pd.DataFrame(elbow_records)

        elbow_dir = get_stage_dir(
            run_id,
            "elbow",
        )

        elbow_fig = plot_elbow(elbow_df)

        elbow_plot_path = save_figure(
            elbow_fig,
            elbow_dir / "elbow_method.png",
        )

        complete_stage(
            run_id,
            "ELBOW",
            f"Hoàn thành Elbow Method. Biểu đồ: {elbow_plot_path}",
        )

        # ----------------------------------------------------
        # Stage 3: Image segmentation
        # ----------------------------------------------------

        start_stage(
            run_id,
            "SEGMENTATION",
            "Bắt đầu phân đoạn ảnh bằng K-Means.",
        )

        image_results = {}
        palette_records = []

        for image_path, image_array in images.items():
            result = run_segmentation_experiment(
                image_path,
                image_array,
                run_id,
            )

            image_results[image_path.name] = result

            palette_records.extend(
                result["palette_records"]
            )

            log_message(
                run_id,
                f"Đã phân đoạn {image_path.name} "
                f"với K={SEGMENTATION_K_VALUES}.",
            )

        palette_df = pd.DataFrame(palette_records)

        complete_stage(
            run_id,
            "SEGMENTATION",
            f"Đã phân đoạn thành công {len(image_results)} ảnh.",
        )

        # ----------------------------------------------------
        # Stage 4: Save experiment data
        # ----------------------------------------------------

        start_stage(
            run_id,
            "VISUALIZATION",
            "Hoàn tất lưu các biểu đồ và dữ liệu thực nghiệm.",
        )

        output_root = get_run_output_dir(run_id)

        elbow_csv = output_root / "elbow_results.csv"
        palette_csv = output_root / "palette_results.csv"

        elbow_df.to_csv(
            elbow_csv,
            index=False,
            encoding="utf-8-sig",
        )

        palette_df.to_csv(
            palette_csv,
            index=False,
            encoding="utf-8-sig",
        )

        complete_stage(
            run_id,
            "VISUALIZATION",
            "Đã lưu dữ liệu và hình ảnh trực quan hóa.",
        )

        # ----------------------------------------------------
        # Stage 5: Generate report
        # ----------------------------------------------------

        start_stage(
            run_id,
            "REPORT",
            "Bắt đầu tạo báo cáo Markdown, HTML và PDF.",
        )

        from .report import generate_report

        report_paths = generate_report(
            run_id=run_id,
            elbow_df=elbow_df,
            palette_df=palette_df,
            image_results=image_results,
            tested_k=SEGMENTATION_K_VALUES,
            status="SUCCESS",
        )

        complete_stage(
            run_id,
            "REPORT",
            "Đã tạo thành công các định dạng báo cáo.",
        )

        # ----------------------------------------------------
        # Finish pipeline
        # ----------------------------------------------------

        update_progress(
            run_id,
            "PIPELINE",
            "SUCCESS",
        )

        finish_run(
            run_id,
            "SUCCESS",
        )

        print("\n" + "=" * 65)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print(f"Run_ID: {run_id}")
        print(f"Elbow results: {elbow_csv}")
        print(f"Palette results: {palette_csv}")

        for report_type, report_path in report_paths.items():
            print(f"{report_type.upper()}: {report_path}")

        print("=" * 65)

        return {
            "run_id": run_id,
            "elbow_results": elbow_df,
            "palette_results": palette_df,
            "image_results": image_results,
            "report_paths": report_paths,
        }

    except Exception as exc:
        try:
            update_progress(
                run_id,
                "PIPELINE",
                "FAILED",
            )

            log_message(
                run_id,
                f"Pipeline thất bại: {exc}\n"
                f"{traceback.format_exc()}",
                level="ERROR",
            )

            finish_run(
                run_id,
                "FAILED",
            )

        except Exception as tracking_error:
            print(
                f"[ERROR] Không thể cập nhật trạng thái tracking: "
                f"{tracking_error}"
            )

        print("\n[PIPELINE ERROR]")
        traceback.print_exc()

        raise


if __name__ == "__main__":
    main()
