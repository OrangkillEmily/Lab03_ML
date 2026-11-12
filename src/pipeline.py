"""Điều phối Customer Segmentation Pipeline theo từng Run_ID."""

import sys
import time
import argparse
import traceback
from pathlib import Path

# =========================================================
# 1. PROJECT PATH
# =========================================================

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# =========================================================
# 2. TRACKING
# =========================================================

from src.tracking import (
    create_run,
    update_progress,
    write_log,
    finish_run
)

from src.figure_manager import collect_run_figures


# =========================================================
# 3. PIPELINE CONFIGURATION
# =========================================================

PIPELINE_NAME = "customer_segmentation"


# =========================================================
# 4. HELPER FUNCTIONS
# =========================================================

def log_stage_start(run_id, stage_name, input_desc):
    """Ghi nhận thời điểm bắt đầu một giai đoạn."""

    write_log(
        run_id,
        stage_name,
        "START",
        f"Bắt đầu giai đoạn {stage_name}."
    )

    update_progress(
        run_id,
        stage_name,
        "START",
        input_data={
            "description": input_desc
        }
    )


def log_stage_success(
    run_id,
    stage_name,
    input_desc,
    output_data,
    elapsed
):
    """Ghi nhận kết quả thành công của một giai đoạn."""

    output_data["duration_seconds"] = elapsed

    update_progress(
        run_id,
        stage_name,
        "SUCCESS",
        input_data={
            "description": input_desc
        },
        output_data=output_data
    )

    write_log(
        run_id,
        stage_name,
        "SUCCESS",
        f"Hoàn thành trong {elapsed:.4f} giây."
    )


def log_stage_failed(
    run_id,
    stage_name,
    input_desc,
    error,
    elapsed
):
    """Ghi nhận lỗi của một giai đoạn."""

    update_progress(
        run_id,
        stage_name,
        "FAILED",
        input_data={
            "description": input_desc
        },
        output_data={
            "duration_seconds": elapsed
        },
        error=str(error)
    )

    write_log(
        run_id,
        stage_name,
        "FAILED",
        str(error)
    )


def collect_training_results():
    """Đọc kết quả thực nghiệm sau khi Training."""

    import pandas as pd

    result_path = (
        ROOT
        / "data"
        / "processed"
        / "experiment_results.csv"
    )

    if not result_path.exists():
        return {}

    df = pd.read_csv(result_path)

    output = {
        "experiments": df.to_dict(orient="records")
    }

    if (
        not df.empty
        and "Silhouette Score (sample)" in df.columns
        and "K" in df.columns
    ):
        best_index = (
            df["Silhouette Score (sample)"].idxmax()
        )

        output["selected_k"] = int(
            df.loc[best_index, "K"]
        )

        output["best_silhouette_score"] = float(
            df.loc[
                best_index,
                "Silhouette Score (sample)"
            ]
        )

    return output


def collect_evaluation_results():
    """Đọc kết quả phân cụm sau Evaluation."""

    import pandas as pd

    cluster_path = (
        ROOT
        / "data"
        / "processed"
        / "customer_clusters.csv"
    )

    if not cluster_path.exists():
        return {}

    clusters = pd.read_csv(cluster_path)

    output = {
        "customers": len(clusters)
    }

    if "Cluster" in clusters.columns:
        output["cluster_counts"] = {
            str(key): int(value)
            for key, value in (
                clusters["Cluster"]
                .value_counts()
                .sort_index()
                .items()
            )
        }

        output["number_of_clusters"] = int(
            clusters["Cluster"].nunique()
        )

    return output


# =========================================================
# 5. MAIN PIPELINE
# =========================================================

def main():

    from src import (
        eda,
        preprocess,
        train,
        evaluate,
        experiments,
        demo_convergence
    )

    # -----------------------------------------------------
    # ARGUMENTS
    # -----------------------------------------------------

    parser = argparse.ArgumentParser(
        description="Chạy Lab03 Customer Segmentation Pipeline"
    )

    parser.add_argument(
        "--extended",
        action="store_true",
        help=(
            "Chạy thêm feature-space experiments, "
            "grid search và convergence."
        )
    )

    args = parser.parse_args()

    # -----------------------------------------------------
    # INITIALIZE RUN
    # -----------------------------------------------------

    # Chỉ thay đổi ở đây:
    # Tạo Run_ID có prefix riêng cho Customer Segmentation.
    run_id = create_run(
        pipeline_name=PIPELINE_NAME
    )

    pipeline_start = time.perf_counter()

    completed = []
    figures = []

    print("=" * 72)
    print("LAB03 - CUSTOMER SEGMENTATION PIPELINE")
    print(f"Pipeline: {PIPELINE_NAME}")
    print(f"Run_ID: {run_id}")
    print(
        f"Extended experiments: "
        f"{'ON' if args.extended else 'OFF'}"
    )
    print("=" * 72)

    # -----------------------------------------------------
    # MAIN STEPS
    # -----------------------------------------------------

    steps = [
        (
            "EDA",
            eda.main,
            "data/raw/customer_segmentation.csv",
            "EDA statistics; docs/images/customer_segmentation/*"
        ),
        (
            "PREPROCESSING",
            preprocess.main,
            "data/raw/customer_segmentation.csv",
            (
                "customers_clean.csv; "
                "X_scaled.csv; "
                "scaler.json; "
                "preprocessing figures"
            )
        ),
        (
            "TRAINING",
            train.main,
            "X_scaled.csv; config.yaml",
            (
                "best_kmeans.pkl; "
                "experiment_results.csv"
            )
        ),
        (
            "EVALUATION",
            evaluate.main,
            (
                "best_kmeans.pkl; "
                "customers_clean.csv; "
                "X_scaled.csv"
            ),
            (
                "customer_clusters.csv; "
                "cluster_profile.csv; "
                "results_summary.md; "
                "evaluation figures"
            )
        )
    ]

    # -----------------------------------------------------
    # EXTENDED EXPERIMENTS
    # -----------------------------------------------------

    extended_steps = []

    if args.extended:

        extended_steps = [
            (
                "EXPERIMENTS",
                experiments.main,
                (
                    "customers_clean.csv; "
                    "feature-space configuration"
                ),
                (
                    "feature_space_comparison.csv; "
                    "grid_search_results.csv"
                )
            ),
            (
                "CONVERGENCE",
                demo_convergence.main,
                "customers_clean.csv; KMeans",
                (
                    "convergence_history.csv; "
                    "kmeans_convergence.png"
                )
            )
        ]

    # =====================================================
    # 6. EXECUTE PIPELINE STAGES
    # =====================================================

    try:

        all_steps = steps + extended_steps

        for (
            stage_name,
            function,
            input_desc,
            output_desc
        ) in all_steps:

            print(f"\n[RUN] {stage_name}")

            stage_start = time.perf_counter()

            log_stage_start(
                run_id,
                stage_name,
                input_desc
            )

            try:

                # Training nhận Run_ID để theo dõi
                # kết quả thực nghiệm.
                if stage_name == "TRAINING":

                    function(run_id=run_id)

                else:

                    function()

                elapsed = round(
                    time.perf_counter() - stage_start,
                    4
                )

                output = {
                    "description": output_desc
                }

                # -----------------------------------------
                # TRAINING RESULTS
                # -----------------------------------------

                if stage_name == "TRAINING":

                    output.update(
                        collect_training_results()
                    )

                # -----------------------------------------
                # EVALUATION RESULTS
                # -----------------------------------------

                if stage_name == "EVALUATION":

                    output.update(
                        collect_evaluation_results()
                    )

                # -----------------------------------------
                # UPDATE TRACKING
                # -----------------------------------------

                log_stage_success(
                    run_id,
                    stage_name,
                    input_desc,
                    output,
                    elapsed
                )

                completed.append(stage_name)

                print(
                    f"[DONE] {stage_name}: "
                    f"{elapsed:.2f}s"
                )

            except Exception as error:

                elapsed = round(
                    time.perf_counter() - stage_start,
                    4
                )

                log_stage_failed(
                    run_id,
                    stage_name,
                    input_desc,
                    error,
                    elapsed
                )

                raise

        # =================================================
        # 7. VISUALIZATION
        # =================================================

        print("\n[RUN] VISUALIZATION")

        stage_start = time.perf_counter()

        log_stage_start(
            run_id,
            "VISUALIZATION",
            "docs/images/customer_segmentation"
        )

        try:

            # Thu thập hình ảnh riêng cho Customer.
            figures = collect_run_figures(
                run_id,
                pipeline_name=PIPELINE_NAME
            )

            elapsed = round(
                time.perf_counter() - stage_start,
                4
            )

            # figure_manager phiên bản mới trả về dict.
            # Lấy danh sách ảnh theo từng stage.
            figure_stages = figures.get("stages", {})

            output = {
                "figure_count": figures.get("total", 0),
                "figures_by_stage": figure_stages,
                "figure_output_dir": figures.get("output_dir")
            }

            log_stage_success(
                run_id,
                "VISUALIZATION",
                "docs/images/customer_segmentation",
                output,
                elapsed
            )

            completed.append("VISUALIZATION")

            print(
                f"[DONE] VISUALIZATION: "
                f"{elapsed:.2f}s"
            )

            print(
                f"[INFO] Figures collected: "
                f"{figures.get('total', 0)}"
            )

        except Exception as error:

            elapsed = round(
                time.perf_counter() - stage_start,
                4
            )

            log_stage_failed(
                run_id,
                "VISUALIZATION",
                "docs/images/customer_segmentation",
                error,
                elapsed
            )

            raise

        # =================================================
        # 8. PIPELINE SUMMARY
        # =================================================

        total = round(
            time.perf_counter() - pipeline_start,
            4
        )

        figure_count = figures.get("total", 0)

        update_progress(
            run_id,
            "PIPELINE",
            "SUCCESS",
            output_data={
                "pipeline": PIPELINE_NAME,
                "completed_steps": completed,
                "total_steps": len(completed),
                "total_execution_time_seconds": total,
                "figure_count": figure_count,
                "figures": figures
            }
        )

        write_log(
            run_id,
            "PIPELINE",
            "SUCCESS",
            "Customer Segmentation Pipeline hoàn thành."
        )

        # Kết thúc Run.
        finish_run(run_id, "SUCCESS")

        # =================================================
        # 9. FINAL OUTPUT
        # =================================================

        print("\n" + "=" * 72)
        print("PIPELINE COMPLETED")
        print(f"Pipeline: {PIPELINE_NAME}")
        print(f"Run_ID: {run_id}")
        print(f"Completed stages: {len(completed)}")
        print(f"Total time: {total:.2f}s")
        print(f"Figures collected: {figure_count}")
        print("=" * 72)

        print(
            "\n[INFO] Pipeline đã hoàn tất."
            "\n[INFO] Report được tạo riêng bằng report.py."
        )

    # =====================================================
    # 10. ERROR HANDLING
    # =====================================================

    except Exception as error:

        total = round(
            time.perf_counter() - pipeline_start,
            4
        )

        update_progress(
            run_id,
            "PIPELINE",
            "FAILED",
            output_data={
                "pipeline": PIPELINE_NAME,
                "completed_steps": completed,
                "total_execution_time_seconds": total
            },
            error=str(error)
        )

        write_log(
            run_id,
            "PIPELINE",
            "FAILED",
            str(error)
        )

        finish_run(run_id, "FAILED")

        print(
            f"\n[ERROR] Customer Segmentation Pipeline thất bại: "
            f"{error}",
            file=sys.stderr
        )

        traceback.print_exc()

        raise


# =========================================================
# 11. ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()
