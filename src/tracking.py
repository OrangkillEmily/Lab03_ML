"""Theo dõi tiến độ từng lần chạy pipeline Lab03.

Hỗ trợ:
- Customer Segmentation
- Additional Image Segmentation

Mỗi pipeline có thư mục tracking riêng:

    logs/
        customer_segmentation/
        additional_image_segmentation/

    progress_results/
        customer_segmentation/
        additional_image_segmentation/
"""

import json
import uuid
from datetime import datetime
from pathlib import Path


# =========================================================
# 1. PROJECT CONFIGURATION
# =========================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

LOG_ROOT = ROOT_DIR / "logs"
RESULT_ROOT = ROOT_DIR / "progress_results"

DEFAULT_PIPELINE = "customer_segmentation"

SUPPORTED_PIPELINES = {
    "customer_segmentation",
    "additional_image_segmentation",
}

TERMINAL_STATUSES = {"SUCCESS", "FAILED"}

ALLOWED_STATUSES = {
    "START",
    "RUNNING",
    "SUCCESS",
    "FAILED",
}


# =========================================================
# 2. GENERAL HELPER FUNCTIONS
# =========================================================

def _now():
    """Lấy thời gian hiện tại kèm timezone."""
    return datetime.now().astimezone().isoformat(
        timespec="seconds"
    )


def _parse_time(value):
    """Chuyển chuỗi thời gian thành datetime."""
    if not value:
        return None

    try:
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None


def _duration_seconds(start_time, end_time):
    """Tính thời gian thực thi tính bằng giây."""
    start = _parse_time(start_time)
    end = _parse_time(end_time)

    if start is None or end is None:
        return None

    return round(
        (end - start).total_seconds(),
        4
    )


# =========================================================
# 3. RUN_ID AND PIPELINE MANAGEMENT
# =========================================================

def _pipeline_name(run_id):
    """Xác định pipeline từ Run_ID."""

    run_id = str(run_id).strip()

    if not run_id:
        raise ValueError("run_id không được để trống.")

    if "--" in run_id:
        pipeline_name = run_id.split("--", 1)[0]

        return _validate_pipeline_name(pipeline_name)

    # Tương thích với Run_ID cũ.
    return DEFAULT_PIPELINE


def _run_suffix(run_id):
    """Lấy phần ID phía sau prefix pipeline."""

    run_id = str(run_id).strip()

    if not run_id:
        raise ValueError("run_id không được để trống.")

    if "--" in run_id:
        return run_id.split("--", 1)[1]

    return run_id


def _validate_pipeline_name(pipeline_name):
    """Kiểm tra tên pipeline."""

    pipeline_name = str(
        pipeline_name
    ).strip().lower()

    if not pipeline_name:
        raise ValueError(
            "pipeline_name không được để trống."
        )

    if pipeline_name not in SUPPORTED_PIPELINES:
        raise ValueError(
            f"Pipeline không được hỗ trợ: {pipeline_name}. "
            f"Các pipeline hợp lệ: {sorted(SUPPORTED_PIPELINES)}"
        )

    return pipeline_name


# =========================================================
# 4. PIPELINE DIRECTORY MANAGEMENT
# =========================================================

def _log_dir(run_id):
    """Trả về thư mục log của pipeline."""

    return LOG_ROOT / _pipeline_name(run_id)


def _result_dir(run_id):
    """Trả về thư mục progress JSON của pipeline."""

    return RESULT_ROOT / _pipeline_name(run_id)


def _log_path(run_id):
    """Đường dẫn file log theo Run_ID."""

    return (
        _log_dir(run_id)
        / f"pipeline_{_run_suffix(run_id)}.log"
    )


def _progress_path(run_id):
    """Đường dẫn file progress JSON theo Run_ID."""

    return (
        _result_dir(run_id)
        / f"progress_{_run_suffix(run_id)}.json"
    )


def _ensure_directories(run_id):
    """Đảm bảo thư mục của pipeline tồn tại."""

    _log_dir(run_id).mkdir(
        parents=True,
        exist_ok=True
    )

    _result_dir(run_id).mkdir(
        parents=True,
        exist_ok=True
    )


# =========================================================
# 5. PROGRESS FILE MANAGEMENT
# =========================================================

def _read_progress(run_id):
    """Đọc progress của một Run_ID."""

    path = _progress_path(run_id)

    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy Run_ID: {run_id}\n"
            f"Đường dẫn: {path}"
        )

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def _write_progress(data):
    """Ghi progress JSON."""

    run_id = data["run_id"]

    _ensure_directories(run_id)

    path = _progress_path(run_id)

    path.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=4,
            default=str
        ),
        encoding="utf-8"
    )


# =========================================================
# 6. LOG MANAGEMENT
# =========================================================

def write_log(
    run_id,
    step_name,
    status,
    message
):
    """Ghi log theo pipeline và Run_ID."""

    _ensure_directories(run_id)

    path = _log_path(run_id)

    timestamp = _now()

    with path.open(
        "a",
        encoding="utf-8"
    ) as file:

        file.write(
            f"[{timestamp}] "
            f"[{str(status).upper()}] "
            f"[{step_name}] "
            f"{message}\n"
        )


# =========================================================
# 7. CREATE NEW PIPELINE RUN
# =========================================================

def create_run(
    pipeline_name=DEFAULT_PIPELINE
):
    """Khởi tạo một lần chạy pipeline mới."""

    pipeline_name = _validate_pipeline_name(
        pipeline_name
    )

    run_id = (
        pipeline_name
        + "--"
        + datetime.now().astimezone().strftime(
            "%Y%m%d_%H%M%S"
        )
        + "_"
        + uuid.uuid4().hex[:6]
    )

    _ensure_directories(run_id)

    start_time = _now()

    data = {
        "run_id": run_id,
        "project": "Lab03",
        "pipeline": pipeline_name,
        "start_time": start_time,
        "end_time": None,
        "duration_seconds": None,
        "total_execution_time_seconds": None,
        "status": "RUNNING",
        "error": None,
        "steps": [],
    }

    _write_progress(data)

    write_log(
        run_id,
        "PIPELINE",
        "START",
        f"Bắt đầu pipeline {pipeline_name}."
    )

    return run_id


# =========================================================
# 8. UPDATE PIPELINE PROGRESS
# =========================================================

def update_progress(
    run_id,
    step_name,
    status,
    input_data=None,
    output_data=None,
    error=None,
):
    """Ghi tiến độ một Stage và cập nhật thông tin Run.

    Lưu ý:
    - FAILED ở cấp Stage không kết thúc toàn bộ Run.
    - Chỉ finish_run() mới đóng Run.
    """

    status = str(status).upper()

    if status not in ALLOWED_STATUSES:
        raise ValueError(
            f"Status không hợp lệ: {status}. "
            f"Chấp nhận: {sorted(ALLOWED_STATUSES)}"
        )

    data = _read_progress(run_id)

    # Không cho phép cập nhật Run đã kết thúc.
    if data["status"] in TERMINAL_STATUSES:
        raise RuntimeError(
            f"Run {run_id} đã kết thúc với status "
            f"{data['status']}; không thể cập nhật thêm."
        )

    timestamp = _now()

    previous_step = None

    # Tìm Stage đang chạy để cập nhật kết quả.
    if status in TERMINAL_STATUSES:

        for step in reversed(data["steps"]):

            if (
                step["step"] == step_name
                and step["status"] in {
                    "START",
                    "RUNNING"
                }
                and step.get("end_time") is None
            ):
                previous_step = step
                break

    # -----------------------------------------------------
    # UPDATE EXISTING STAGE
    # -----------------------------------------------------

    if previous_step:

        previous_step["status"] = status
        previous_step["end_time"] = timestamp

        previous_step["duration_seconds"] = (
            _duration_seconds(
                previous_step.get("start_time"),
                timestamp
            )
        )

        if input_data is not None:
            previous_step["input"] = input_data

        if output_data is not None:
            previous_step["output"] = output_data

        if error is not None:
            previous_step["error"] = str(error)

    # -----------------------------------------------------
    # CREATE NEW STAGE RECORD
    # -----------------------------------------------------

    else:

        step_record = {
            "step": step_name,
            "timestamp": timestamp,
            "start_time": timestamp,
            "end_time": (
                timestamp
                if status in TERMINAL_STATUSES
                else None
            ),
            "duration_seconds": (
                0
                if status in TERMINAL_STATUSES
                else None
            ),
            "status": status,
            "input": input_data,
            "output": output_data,
            "error": (
                str(error)
                if error is not None
                else None
            ),
        }

        data["steps"].append(step_record)

    # -----------------------------------------------------
    # KHÔNG KẾT THÚC RUN TẠI ĐÂY
    # -----------------------------------------------------
    # Kể cả khi Stage FAILED, Run vẫn giữ RUNNING.
    # pipeline.py sẽ ghi nhận lỗi và gọi finish_run()
    # để đóng Run với trạng thái FAILED.

    _write_progress(data)

    duration = (
        previous_step.get("duration_seconds")
        if previous_step
        else 0
    )

    write_log(
        run_id,
        step_name,
        status,
        (
            f"Stage {step_name}: {status}. "
            f"Duration: {duration} giây."
        )
    )

    return data


# =========================================================
# 9. FINISH PIPELINE RUN
# =========================================================

def finish_run(
    run_id,
    status="SUCCESS"
):
    """Đóng Run và ghi nhận trạng thái cuối."""

    status = str(status).upper()

    if status not in TERMINAL_STATUSES:
        raise ValueError(
            "finish_run() chỉ nhận "
            "status SUCCESS hoặc FAILED."
        )

    data = _read_progress(run_id)

    # Không ghi đè Run đã kết thúc.
    if data["status"] in TERMINAL_STATUSES:

        if data["status"] != status:
            raise RuntimeError(
                f"Run {run_id} đã kết thúc với status "
                f"{data['status']}; không thể đổi thành "
                f"{status}."
            )

        return data

    # Không cho SUCCESS nếu có Stage FAILED.
    failed_steps = [
        step for step in data["steps"]
        if step.get("status") == "FAILED"
    ]

    if status == "SUCCESS" and failed_steps:
        failed_names = [
            step["step"] for step in failed_steps
        ]

        raise RuntimeError(
            "Không thể kết thúc SUCCESS vì có Stage FAILED: "
            + ", ".join(failed_names)
        )

    end_time = _now()

    data["status"] = status
    data["end_time"] = end_time

    data["duration_seconds"] = (
        _duration_seconds(
            data.get("start_time"),
            end_time
        )
    )

    data["total_execution_time_seconds"] = (
        data["duration_seconds"]
    )

    # Đảm bảo Stage đã kết thúc có duration.
    for step in data["steps"]:

        if (
            step.get("end_time")
            and step.get("duration_seconds") is None
        ):
            step["duration_seconds"] = (
                _duration_seconds(
                    step.get("start_time")
                    or step.get("timestamp"),
                    step.get("end_time")
                )
            )

    if status == "FAILED":

        if not data.get("error"):

            if failed_steps:
                data["error"] = (
                    failed_steps[-1].get("error")
                    or (
                        f"Stage {failed_steps[-1]['step']} "
                        "thất bại."
                    )
                )
            else:
                data["error"] = (
                    "Pipeline kết thúc thất bại."
                )

    _write_progress(data)

    write_log(
        run_id,
        "PIPELINE",
        status,
        (
            "Kết thúc pipeline. "
            f"Tổng thời gian: "
            f"{data['total_execution_time_seconds']} giây."
        )
    )

    return data
