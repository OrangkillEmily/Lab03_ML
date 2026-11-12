"""
Tạo báo cáo Lab03 theo từng Run_ID.

Đầu ra:
- reports/report_<Run_ID>.md
- reports/report_<Run_ID>.html
- reports/report_<Run_ID>.pdf

Dữ liệu đầu vào:
- progress_results/progress_<Run_ID>.json
- logs/pipeline_<Run_ID>.log
- data/processed/experiment_results.csv
- data/processed/customers_clean.csv
- data/processed/customer_clusters.csv
- data/processed/cluster_profile.csv
- reports/figures/<Run_ID>/

Có thể chạy độc lập:
    python -m src.report --run-id <Run_ID>

Hoặc gọi từ pipeline:
    generate_report(run_id)
"""

import argparse
import base64
import csv
import html
import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd


# ============================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

RESULT_DIR = ROOT_DIR / "progress_results"
LOG_DIR = ROOT_DIR / "logs"

DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"

REPORT_DIR = ROOT_DIR / "reports" / "customer_segmentation"
FIGURE_DIR = ROOT_DIR / "reports" / "figures"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# Các file dữ liệu đầu vào của Lab03
PROGRESS_PATTERN = "progress_{run_id}.json"
LOG_PATTERN = "pipeline_{run_id}.log"

CLEAN_DATA_FILE = PROCESSED_DIR / "customers_clean.csv"
SCALED_DATA_FILE = PROCESSED_DIR / "X_scaled.csv"
EXPERIMENT_FILE = PROCESSED_DIR / "experiment_results.csv"
CLUSTER_FILE = PROCESSED_DIR / "customer_clusters.csv"
PROFILE_FILE = PROCESSED_DIR / "cluster_profile.csv"

# Tên cột đặc trưng dùng cho K-means
FEATURES = [
    "Age",
    "Annual Income",
    "Spending Score",
]

def get_latest_run_id():
    """
    Tìm Run_ID mới nhất trong progress_results.

    Run_ID được đọc từ file:
        progress_<Run_ID>.json
    """

    if not RESULT_DIR.exists():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục progress: {RESULT_DIR}"
        )

    progress_files = list(
        RESULT_DIR.glob("progress_*.json")
    )

    if not progress_files:
        raise FileNotFoundError(
            f"Không tìm thấy file progress trong {RESULT_DIR}"
        )

    # Lấy file progress được cập nhật gần nhất
    latest_file = max(
        progress_files,
        key=lambda file: file.stat().st_mtime,
    )

    with latest_file.open("r", encoding="utf-8") as file:
        progress_data = json.load(file)

    run_id = progress_data.get("run_id")

    if not run_id:
        raise ValueError(
            f"File progress không có Run_ID: {latest_file}"
        )

    print(f"[REPORT] Run_ID được chọn: {run_id}")

    return run_id

# ============================================================
# 2. HÀM ĐỌC DỮ LIỆU
# ============================================================

def read_json(path):
    """Đọc JSON Tracking; báo lỗi rõ ràng nếu file thiếu hoặc sai định dạng."""
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file JSON: {path}")

    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON không hợp lệ: {path} ({exc})") from exc
    except OSError as exc:
        raise OSError(f"Không thể đọc file JSON: {path} ({exc})") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Nội dung Tracking phải là object JSON: {path}")

    return data


def read_csv(path):
    """Đọc CSV an toàn."""
    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def read_log(path):
    """Đọc nội dung log của lần chạy."""
    if not path.exists():
        return ""

    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def format_number(value, digits=4):
    """Định dạng số để đưa vào báo cáo."""
    try:
        number = float(value)

        if number.is_integer():
            return f"{int(number):,}"

        return f"{number:,.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def format_timestamp(value):
    """Rút gọn timestamp ISO để bảng Tracking không bị tràn cột."""
    if value is None or str(value).strip() == "":
        return ""
    text_value = str(value).strip()
    try:
        parsed = datetime.fromisoformat(text_value.replace("Z", "+00:00"))
        return parsed.strftime("%d-%m-%Y %H:%M:%S")
    except (TypeError, ValueError):
        return text_value


def format_duration(value):
    """Định dạng thời gian chạy."""
    try:
        seconds = float(value)

        if seconds < 1:
            return f"{seconds * 1000:.2f} ms"

        if seconds < 60:
            return f"{seconds:.2f} giây"

        minutes = int(seconds // 60)
        remaining = seconds % 60

        return f"{minutes} phút {remaining:.2f} giây"
    except (TypeError, ValueError):
        return "Không có dữ liệu"


def dataframe_to_markdown(df, max_rows=100):
    """Chuyển DataFrame thành bảng Markdown."""
    if df.empty:
        return "_Không có dữ liệu._"

    display_df = df.head(max_rows).copy()
    display_df = display_df.fillna("")

    headers = [str(column) for column in display_df.columns]

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for _, row in display_df.iterrows():
        values = []

        for value in row.tolist():
            if isinstance(value, float):
                value = format_number(value)

            value = str(value).replace("|", "\\|").replace("\n", " ")
            values.append(value)

        lines.append("| " + " | ".join(values) + " |")

    if len(df) > max_rows:
        lines.append("")
        lines.append(
            f"_Bảng hiển thị {max_rows}/{len(df)} dòng đầu tiên._"
        )

    return "\n".join(lines)


def dataframe_to_html(df, max_rows=100):
    """Chuyển DataFrame thành bảng HTML."""
    if df.empty:
        return '<p class="empty">Không có dữ liệu.</p>'

    display_df = df.head(max_rows).copy()

    table = ['<div class="table-wrapper"><table>']
    table.append("<thead><tr>")

    for column in display_df.columns:
        table.append(f"<th>{html.escape(str(column))}</th>")

    table.append("</tr></thead><tbody>")

    for _, row in display_df.iterrows():
        table.append("<tr>")

        for value in row.tolist():
            if pd.isna(value):
                value = ""
            elif isinstance(value, float):
                value = format_number(value)

            table.append(f"<td>{html.escape(str(value))}</td>")

        table.append("</tr>")

    table.append("</tbody></table></div>")

    if len(df) > max_rows:
        table.append(
            f'<p class="note">Hiển thị {max_rows}/{len(df)} dòng đầu tiên.</p>'
        )

    return "".join(table)


# ============================================================
# 3. XỬ LÝ PROGRESS / TRACKING
# ============================================================

def flatten_progress(data, parent_key=""):
    """
    Chuyển cấu trúc JSON lồng nhau thành danh sách thông tin.

    Hàm này không phụ thuộc cứng vào một cấu trúc JSON cụ thể.
    """
    rows = []

    if isinstance(data, dict):
        for key, value in data.items():
            current_key = f"{parent_key}.{key}" if parent_key else str(key)

            if isinstance(value, dict):
                rows.extend(flatten_progress(value, current_key))

            elif isinstance(value, list):
                for index, item in enumerate(value):
                    if isinstance(item, dict):
                        rows.extend(
                            flatten_progress(
                                item,
                                f"{current_key}[{index}]"
                            )
                        )

            else:
                rows.append({
                    "Field": current_key,
                    "Value": value,
                })

    elif isinstance(data, list):
        for index, item in enumerate(data):
            if isinstance(item, dict):
                rows.extend(
                    flatten_progress(item, f"{parent_key}[{index}]")
                )

    return rows


def extract_stage_records(data):
    """
    Cố gắng tìm danh sách các giai đoạn trong progress JSON.

    Hỗ trợ một số dạng phổ biến:
    - steps: [...]
    - stages: [...]
    - progress: [...]
    - pipeline: {steps: [...]}
    """
    candidates = []

    def walk(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                key_lower = str(key).lower()

                if (
                    key_lower in {"steps", "stages", "progress"}
                    and isinstance(value, list)
                ):
                    candidates.append(value)

                if isinstance(value, (dict, list)):
                    walk(value)

        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(data)

    for candidate in candidates:
        records = [
            item for item in candidate
            if isinstance(item, dict)
        ]

        if records:
            return records

    return []


def build_stage_dataframe(progress):
    """Tạo bảng stage theo đúng cấu trúc tracking.py."""
    records = extract_stage_records(progress)
    rows = []

    for item in records:
        output_data = item.get("output")
        if not isinstance(output_data, dict):
            output_data = {}

        start_time = (
            item.get("start_time")
            or item.get("timestamp")
            or item.get("started_at")
            or ""
        )
        end_time = item.get("end_time") or item.get("finished_at") or ""

        duration = item.get("duration_seconds")
        if duration is None:
            duration = output_data.get("duration_seconds")
        duration_text = (
            format_duration(duration)
            if duration is not None
            else "Chưa có dữ liệu"
        )

        error_text = item.get("error")
        info = (
            error_text
            or output_data.get("description")
            or item.get("message")
            or item.get("description")
            or ""
        )

        rows.append({
            "Giai đoạn": (
                item.get("step")
                or item.get("stage")
                or item.get("name")
                or item.get("task")
                or "Không xác định"
            ),
            "Trạng thái": item.get("status") or item.get("state") or "Không có dữ liệu",
            "Thời gian bắt đầu": format_timestamp(start_time),
            "Thời gian kết thúc": format_timestamp(end_time),
            "Thời gian chạy": duration_text,
            "Thông tin": info,
        })

    if rows:
        return pd.DataFrame(rows)

    # Hỗ trợ JSON cũ không có steps.
    flattened = flatten_progress(progress)
    return pd.DataFrame(flattened) if flattened else pd.DataFrame()


def get_progress_value(progress, *keys):
    """Tìm giá trị theo tên key ở nhiều cấp JSON."""
    if not isinstance(progress, dict):
        return None

    for key in keys:
        if key in progress and not isinstance(progress[key], (dict, list)):
            return progress[key]

    for value in progress.values():
        if isinstance(value, dict):
            found = get_progress_value(value, *keys)

            if found is not None:
                return found

    return None


# ============================================================
# 4. ĐỌC VÀ PHÂN TÍCH KẾT QUẢ
# ============================================================

def get_cluster_column(df):
    """Tìm tên cột chứa nhãn cụm."""
    if df.empty:
        return None

    candidates = [
        "Cluster",
        "cluster",
        "Cluster_Label",
        "cluster_label",
        "KMeans_Cluster",
        "label",
    ]

    for column in candidates:
        if column in df.columns:
            return column

    for column in df.columns:
        if "cluster" in str(column).lower():
            return column

    return None


def build_cluster_counts(cluster_df):
    """Tính số lượng và tỷ lệ khách hàng từng cụm."""
    if cluster_df.empty:
        return pd.DataFrame()

    cluster_col = get_cluster_column(cluster_df)

    if cluster_col is None:
        return pd.DataFrame()

    counts = (
        cluster_df[cluster_col]
        .value_counts(dropna=False)
        .sort_index()
        .reset_index()
    )

    counts.columns = ["Cluster", "Customer_Count"]

    total = counts["Customer_Count"].sum()

    counts["Percentage"] = (
        counts["Customer_Count"] / total * 100
    ).round(2)

    return counts


def find_metric_column(df, keywords):
    """Tìm cột metric theo từ khóa."""
    if df.empty:
        return None

    for column in df.columns:
        name = str(column).lower().replace(" ", "_")

        if any(keyword in name for keyword in keywords):
            return column

    return None


def get_best_experiment(experiment_df):
    """Lấy dòng có Silhouette Score cao nhất."""
    if experiment_df.empty:
        return None

    silhouette_col = find_metric_column(
        experiment_df,
        ["silhouette"]
    )

    if silhouette_col is None:
        return None

    values = pd.to_numeric(
        experiment_df[silhouette_col],
        errors="coerce"
    )

    if values.notna().sum() == 0:
        return None

    return experiment_df.loc[values.idxmax()]


def build_experiment_table(experiment_df, run_id):
    """
    Lọc kết quả thực nghiệm theo Run_ID nếu file có cột Run_ID.
    Nếu không có cột Run_ID thì sử dụng dữ liệu hiện có.
    """
    if experiment_df.empty:
        return pd.DataFrame()

    df = experiment_df.copy()

    run_column = next(
        (
            col for col in df.columns
            if str(col).lower() == "run_id"
        ),
        None
    )

    if run_column:
        filtered = df[
            df[run_column].astype(str) == str(run_id)
        ]

        if not filtered.empty:
            df = filtered

    return df


def build_cluster_profile(profile_df):
    """Chuẩn hóa bảng hồ sơ cụm để hiển thị."""
    if profile_df.empty:
        return pd.DataFrame()

    return profile_df.copy()


def describe_cluster_profiles(profile_df):
    """Tạo nhận xét sơ bộ từ bảng hồ sơ cụm."""
    if profile_df.empty:
        return (
            "Chưa có dữ liệu hồ sơ cụm để phân tích. "
            "Cần kiểm tra file cluster_profile.csv."
        )

    lines = []

    for _, row in profile_df.iterrows():
        values = []

        for column, value in row.items():
            if str(column).lower() in {
                "cluster", "cluster_id", "cluster_label"
            }:
                continue

            if pd.notna(value):
                values.append(f"{column}: {format_number(value)}")

        if values:
            # Không dùng toán tử `or` vì nhãn cụm 0 là giá trị hợp lệ.
            cluster_value = None
            for column in ("Cluster", "cluster", "Cluster_Label", "cluster_label"):
                if column in row.index and pd.notna(row[column]):
                    cluster_value = row[column]
                    break

            if cluster_value is None:
                cluster_value = "Không xác định"

            lines.append(
                f"- **Cụm {format_number(cluster_value)}:** " + "; ".join(values)
            )

    return "\n".join(lines) if lines else "Không có thông tin hồ sơ cụm."


# ============================================================
# 5. THU THẬP HÌNH ẢNH THEO GIAI ĐOẠN
# ============================================================

STAGE_TITLES = {
    "01_eda": "EDA – Khám phá dữ liệu",
    "02_preprocessing": "Preprocessing – Tiền xử lý dữ liệu",
    "03_training": "Training – Huấn luyện K-means",
    "04_evaluation": "Evaluation – Đánh giá phân cụm",
    "05_extended": "Thực nghiệm mở rộng",
    "experiments": "Thực nghiệm mở rộng",
    "convergence": "Phân tích hội tụ",
    "99_other": "Biểu đồ khác",
}


FIGURE_DESCRIPTIONS = {
    "outlier_boxplots": (
        "Biểu đồ boxplot hỗ trợ quan sát phân phối dữ liệu "
        "và phát hiện các giá trị có khả năng là ngoại lệ."
    ),
    "feature_distributions": (
        "Biểu đồ phân phối giúp quan sát hình dạng phân phối "
        "của từng đặc trưng."
    ),
    "eda_scatterplots": (
        "Biểu đồ phân tán thể hiện mối quan hệ giữa các đặc trưng "
        "và hỗ trợ quan sát cấu trúc dữ liệu."
    ),
    "gender_distribution": (
        "Biểu đồ thể hiện phân bố giới tính trong tập dữ liệu."
    ),
    "correlation_matrix": (
        "Ma trận tương quan thể hiện mức độ tương quan tuyến tính "
        "giữa các đặc trưng số."
    ),
    "before_after_scaling": (
        "So sánh phân phối đặc trưng trước và sau chuẩn hóa Z-score."
    ),
    "preprocessing_summary": (
        "Biểu đồ so sánh giá trị trung bình (Mean) và độ lệch chuẩn "
        "(Standard Deviation) của các đặc trưng trong dữ liệu."
    ),
    "elbow_method": (
        "Phương pháp Elbow thể hiện sự thay đổi Inertia theo số cụm K. "
        "Điểm khuỷu tay có thể hỗ trợ lựa chọn K."
    ),
    "silhouette_by_k": (
        "Silhouette Score dùng để đánh giá mức độ gắn kết trong cụm "
        "và tách biệt giữa các cụm."
    ),
    "cluster_sizes": (
        "Biểu đồ thể hiện số lượng khách hàng thuộc từng cụm."
    ),
    "clusters_income_spending": (
        "Biểu đồ phân bố khách hàng theo Annual Income và Spending Score, "
        "được tô màu theo nhãn cụm."
    ),
    "clusters_age_spending": (
        "Biểu đồ phân bố khách hàng theo Age và Spending Score, "
        "được tô màu theo nhãn cụm."
    ),
    "cluster_spending_means": (
        "So sánh giá trị Spending Score trung bình giữa các cụm."
    ),
    "cluster_profiles": (
        "Biểu đồ hoặc bảng trực quan hóa đặc điểm trung bình của từng cụm."
    ),
    "feature_space_comparison": (
        "So sánh kết quả phân cụm trên các không gian đặc trưng khác nhau."
    ),
    "kmeans_convergence": (
        "Minh họa quá trình tâm cụm thay đổi qua các vòng lặp của K-means."
    ),
}


def collect_figures(run_id):
    """Lấy danh sách ảnh trong thư mục figures của Run_ID."""
    run_figure_dir = FIGURE_DIR / str(run_id)

    if not run_figure_dir.exists():
        return {}

    grouped = {}

    for image_path in sorted(run_figure_dir.rglob("*")):
        if not image_path.is_file():
            continue

        if image_path.suffix.lower() not in {
            ".png", ".jpg", ".jpeg", ".svg"
        }:
            continue

        relative = image_path.relative_to(run_figure_dir)
        stage = relative.parts[0] if len(relative.parts) > 1 else "99_other"

        grouped.setdefault(stage, []).append(image_path)

    return grouped


def describe_figure(image_path):
    """Lấy mô tả ảnh theo tên file."""
    stem = image_path.stem.lower()

    for key, description in FIGURE_DESCRIPTIONS.items():
        if key in stem:
            return description

    readable_name = stem.replace("_", " ").replace("-", " ").capitalize()

    return (
        f"Biểu đồ {readable_name}, được tạo trong quá trình thực hiện Lab03."
    )


# ============================================================
# 6. TẠO NỘI DUNG MARKDOWN
# ============================================================

def build_report_data(run_id):
    """Đọc và tổng hợp toàn bộ dữ liệu của một lần chạy."""
    progress_path = RESULT_DIR / PROGRESS_PATTERN.format(run_id=run_id)
    log_path = LOG_DIR / LOG_PATTERN.format(run_id=run_id)

    if not progress_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy Tracking của Run_ID {run_id}: {progress_path}"
        )

    progress = read_json(progress_path)
    if not progress or progress.get("run_id") != str(run_id):
        raise ValueError(
            f"Tracking JSON không hợp lệ hoặc Run_ID không khớp: {progress_path}"
        )

    log_text = read_log(log_path)

    clean_df = read_csv(CLEAN_DATA_FILE)
    scaled_df = read_csv(SCALED_DATA_FILE)
    experiment_df = build_experiment_table(
        read_csv(EXPERIMENT_FILE),
        run_id
    )
    cluster_df = read_csv(CLUSTER_FILE)
    profile_df = build_cluster_profile(
        read_csv(PROFILE_FILE)
    )

    cluster_counts = build_cluster_counts(cluster_df)
    best_experiment = get_best_experiment(experiment_df)

    return {
        "run_id": run_id,
        "progress": progress,
        "log_text": log_text,
        "clean_df": clean_df,
        "scaled_df": scaled_df,
        "experiment_df": experiment_df,
        "cluster_df": cluster_df,
        "profile_df": profile_df,
        "cluster_counts": cluster_counts,
        "best_experiment": best_experiment,
        "figures": collect_figures(run_id),
        "progress_path": progress_path,
        "log_path": log_path,
    }


def build_markdown(data):
    """Tạo nội dung báo cáo Markdown đầy đủ."""
    run_id = data["run_id"]
    progress = data["progress"]

    clean_df = data["clean_df"]
    scaled_df = data["scaled_df"]
    experiment_df = data["experiment_df"]
    cluster_df = data["cluster_df"]
    profile_df = data["profile_df"]
    cluster_counts = data["cluster_counts"]

    status = (
    data.get("report_status")
    or get_progress_value(progress, "status", "pipeline_status")
    or "Không xác định"
)

    start_time = (
        get_progress_value(progress, "start_time", "started_at")
        or "Không có dữ liệu"
    )

    end_time = (
        get_progress_value(progress, "end_time", "finished_at")
        or "Không có dữ liệu"
    )

    duration = get_progress_value(
        progress,
        "total_execution_time_seconds",
        "duration_seconds",
        "total_duration_seconds",
        "total_time",
    )

    best = data["best_experiment"]

    lines = []

    def add(text=""):
        lines.append(str(text))

    # --------------------------------------------------------
    # Tiêu đề và thông tin lần chạy
    # --------------------------------------------------------

    add("# BÁO CÁO THỰC NGHIỆM LAB03 – CUSTOMER SEGMENTATION")
    add()
    add("> Báo cáo tổng hợp quá trình khám phá dữ liệu, tiền xử lý, "
        "huấn luyện và đánh giá mô hình K-means.")
    add()

    add("## 1. Thông tin lần chạy")
    add()
    add("| Thông tin | Giá trị |")
    add("|---|---|")
    add(f"| Run_ID | `{run_id}` |")
    add(f"| Trạng thái | **{status}** |")
    add(f"| Bắt đầu | {start_time} |")
    add(f"| Kết thúc | {end_time} |")
    add(f"| Tổng thời gian | {format_duration(duration)} |")
    add(f"| Progress file | `{data['progress_path'].relative_to(ROOT_DIR)}` |")
    add(f"| Log file | `{data['log_path'].relative_to(ROOT_DIR)}` |")
    add()

    # --------------------------------------------------------
    # Tổng quan pipeline
    # --------------------------------------------------------

    add("## 2. Tổng quan Pipeline")
    add()
    add("Pipeline Lab03 được tổ chức theo năm giai đoạn chính:")
    add()
    add("```text")
    add("Dữ liệu đầu vào")
    add("      |")
    add("      v")
    add("1. EDA – Khám phá dữ liệu")
    add("      |")
    add("      v")
    add("2. PREPROCESSING – Làm sạch và chuẩn hóa")
    add("      |")
    add("      v")
    add("3. TRAINING – Huấn luyện K-means")
    add("      |")
    add("      v")
    add("4. EVALUATION – Đánh giá kết quả phân cụm")
    add("      |")
    add("      v")
    add("Visualization -> Tracking -> Report")
    add("```")
    add()
    add("Mỗi giai đoạn nhận đầu vào từ giai đoạn trước, thực hiện "
        "nhiệm vụ riêng và tạo ra đầu ra phục vụ bước tiếp theo.")
    add()

    # --------------------------------------------------------
    # Tiến trình tracking
    # --------------------------------------------------------

    add("## 3. Tiến trình thực hiện và Tracking")
    add()
    stage_df = build_stage_dataframe(progress)

    if not stage_df.empty:
        add(dataframe_to_markdown(stage_df))
    else:
        add("_Không tìm thấy danh sách các giai đoạn trong progress JSON._")

    add()
    add("File `progress_results/progress_<Run_ID>.json` lưu trạng thái "
        "và thông tin tiến trình của lần chạy. File log lưu các sự kiện "
        "bắt đầu, hoàn thành hoặc lỗi trong quá trình thực thi.")
    add()

    # --------------------------------------------------------
    # Giải thích các stage
    # --------------------------------------------------------

    add("## 4. Giải thích các giai đoạn trong Pipeline")
    add()

    add("### 4.1. EDA – Exploratory Data Analysis")
    add()
    add("**Mục đích:** Khảo sát cấu trúc dữ liệu, kiểu dữ liệu, phân phối "
        "các thuộc tính và mối quan hệ giữa các biến.")
    add()
    add("**Input:** `data/raw/customer_segmentation.csv`.")
    add()
    add("**Các công việc chính:**")
    add("- Đọc và kiểm tra dữ liệu đầu vào.")
    add("- Kiểm tra số lượng bản ghi, thuộc tính và kiểu dữ liệu.")
    add("- Khảo sát dữ liệu thiếu, dữ liệu trùng lặp và ngoại lệ.")
    add("- Quan sát phân phối Age, Annual Income và Spending Score.")
    add("- Phân tích mối quan hệ giữa các đặc trưng.")
    add()
    add("**Output:** Các thống kê mô tả, bảng phân tích và biểu đồ EDA "
        "được lưu trong thư mục hình ảnh của lần chạy.")
    add()

    add("### 4.2. Preprocessing – Tiền xử lý dữ liệu")
    add()
    add("**Mục đích:** Chuẩn bị dữ liệu hợp lệ và đưa các đặc trưng về "
        "cùng thang đo trước khi huấn luyện K-means.")
    add()
    add("**Input:** Dữ liệu gốc từ `data/raw/customer_segmentation.csv`.")
    add()
    add("**Các công việc chính:**")
    add("- Loại bỏ các bản ghi trùng lặp.")
    add("- Chuyển các thuộc tính số về kiểu dữ liệu phù hợp.")
    add("- Loại bỏ bản ghi thiếu hoặc chứa giá trị không hợp lệ.")
    add("- Chọn ba đặc trưng: Age, Annual Income, Spending Score.")
    add("- Không sử dụng Customer ID làm đặc trưng phân cụm.")
    add("- Giữ Gender để hỗ trợ phân tích hồ sơ cụm nếu cần.")
    add()
    add("**Chuẩn hóa Z-score:**")
    add()
    add("**z = (x − μ) / σ**")
    add()
    add("Trong đó:")
    add("- `x`: giá trị ban đầu.")
    add("- `μ`: giá trị trung bình của đặc trưng.")
    add("- `σ`: độ lệch chuẩn của đặc trưng.")
    add("- `z`: giá trị sau chuẩn hóa.")
    add()
    add("**Output:**")
    add("- `data/processed/customers_clean.csv`.")
    add("- `data/processed/X_scaled.csv`.")
    add("- `data/processed/scaler.json`.")
    add()

    add("#### Kết quả tiền xử lý")
    add()
    add(f"- Số dòng sau làm sạch: **{len(clean_df):,}**.")
    add(f"- Số dòng dữ liệu sau chuẩn hóa: **{len(scaled_df):,}**.")
    add(f"- Số đặc trưng được sử dụng: **{len(FEATURES)}**.")
    add(f"- Các đặc trưng: `{', '.join(FEATURES)}`.")
    add()

    if not clean_df.empty:
        add("Một số dòng dữ liệu sau làm sạch:")
        add()
        add(dataframe_to_markdown(clean_df, max_rows=5))
        add()

    if not scaled_df.empty:
        add("Một số dòng dữ liệu sau chuẩn hóa:")
        add()
        add(dataframe_to_markdown(scaled_df, max_rows=5))
        add()

    add("### 4.3. Training – Huấn luyện mô hình K-means")
    add()
    add("**Mục đích:** Tìm cách chia tập khách hàng thành các cụm sao cho "
        "các điểm dữ liệu trong cùng cụm gần nhau theo khoảng cách trong "
        "không gian đặc trưng.")
    add()
    add("**Input:** `X_scaled.csv` và cấu hình mô hình.")
    add()
    add("**Quy trình:**")
    add("1. Khởi tạo các tâm cụm.")
    add("2. Gán mỗi khách hàng vào tâm cụm gần nhất.")
    add("3. Cập nhật tâm cụm bằng trung bình các điểm thuộc cụm.")
    add("4. Lặp lại quá trình gán cụm và cập nhật tâm cho đến khi hội tụ "
        "hoặc đạt giới hạn số vòng lặp.")
    add("5. Thử nghiệm các giá trị K và so sánh chỉ số đánh giá.")
    add()
    add("**Các chỉ số sử dụng:**")
    add()
    add("- **Inertia:** Tổng bình phương khoảng cách từ các điểm dữ liệu "
        "đến tâm cụm tương ứng. Inertia càng thấp thì các điểm trong cụm "
        "càng gần tâm cụm, nhưng chỉ số này thường giảm khi tăng K.")
    add("- **Silhouette Score:** Đánh giá mức độ gắn kết trong cụm và "
        "mức độ tách biệt với cụm gần nhất. Giá trị nằm trong khoảng "
        "[-1, 1], càng cao thường thể hiện cấu trúc cụm càng rõ.")
    add()
    add("**Output:** Mô hình K-means tốt nhất và bảng kết quả thử nghiệm K.")
    add()

    add("#### Bảng kết quả thử nghiệm K")
    add()

    if not experiment_df.empty:
        add(dataframe_to_markdown(experiment_df))
    else:
        add("_Không tìm thấy dữ liệu experiment_results.csv._")

    add()

    if best is not None:
        add("#### Cấu hình có Silhouette Score cao nhất")
        add()

        for column, value in best.items():
            add(f"- **{column}:** {format_number(value)}")

        add()
    else:
        add("Chưa xác định được cấu hình tốt nhất từ bảng thực nghiệm. "
            "Cần kiểm tra cột Silhouette trong `experiment_results.csv`.")
        add()

    add("### 4.4. Evaluation – Đánh giá kết quả phân cụm")
    add()
    add("**Mục đích:** Phân tích kết quả phân cụm sau khi mô hình K-means "
        "đã gán nhãn cho từng khách hàng.")
    add()
    add("**Input:** Mô hình K-means tốt nhất, dữ liệu đã làm sạch và "
        "dữ liệu chuẩn hóa.")
    add()
    add("**Các công việc chính:**")
    add("- Gán nhãn cụm cho từng khách hàng.")
    add("- Tính số lượng khách hàng trong mỗi cụm.")
    add("- Tính tỷ lệ khách hàng của từng cụm.")
    add("- Tính các giá trị trung bình để xây dựng hồ sơ cụm.")
    add("- Trực quan hóa sự phân bố giữa các cụm.")
    add()
    add("**Output:**")
    add("- `customer_clusters.csv`.")
    add("- `cluster_profile.csv`.")
    add("- Các biểu đồ đánh giá phân cụm.")
    add()

    # --------------------------------------------------------
    # Kết quả evaluation
    # --------------------------------------------------------

    add("## 5. Kết quả phân cụm")
    add()

    add("### 5.1. Số lượng khách hàng trong từng cụm")
    add()

    if not cluster_counts.empty:
        add(dataframe_to_markdown(cluster_counts))
    else:
        add("_Không thể xác định cột nhãn cụm trong customer_clusters.csv._")

    add()

    add("### 5.2. Hồ sơ và đặc điểm các cụm")
    add()

    if not profile_df.empty:
        # Chia hồ sơ cụm thành hai bảng ngắn để tránh tiêu đề bị ép hẹp.
        cluster_col = next(
            (c for c in profile_df.columns
             if str(c).strip().lower() in {"cluster", "cluster_id", "cluster_label"}),
            profile_df.columns[0],
        )
        profile_groups = [
            (
                "Độ tuổi và thu nhập",
                [cluster_col, "Age_count", "Age_mean", "Age_median",
                 "Annual Income_mean", "Annual Income_median"],
                {
                    cluster_col: "Cụm", "Age_count": "Số KH",
                    "Age_mean": "Tuổi TB", "Age_median": "Tuổi trung vị",
                    "Annual Income_mean": "Thu nhập TB",
                    "Annual Income_median": "Thu nhập trung vị",
                },
            ),
            (
                "Chi tiêu và giới tính",
                [cluster_col, "Spending Score_count", "Spending Score_mean",
                 "Spending Score_median", "Gender_%_Female", "Gender_%_Male"],
                {
                    cluster_col: "Cụm", "Spending Score_count": "Số KH",
                    "Spending Score_mean": "Chi tiêu TB",
                    "Spending Score_median": "Chi tiêu trung vị",
                    "Gender_%_Female": "Nữ (%)", "Gender_%_Male": "Nam (%)",
                },
            ),
        ]
        for group_title, columns, headers in profile_groups:
            available = [column for column in columns if column in profile_df.columns]
            if len(available) > 1:
                add(f"**{group_title}:**")
                add()
                add(dataframe_to_markdown(profile_df[available].rename(columns=headers)))
                add()
        add("#### Nhận xét theo hồ sơ cụm")
        add()
        add(describe_cluster_profiles(profile_df))
    else:
        add("_Chưa có dữ liệu cluster_profile.csv để phân tích._")

    add()

    add("### 5.3. Dữ liệu khách hàng sau khi gán cụm")
    add()

    if not cluster_df.empty:
        add(f"- Tổng số bản ghi: **{len(cluster_df):,}**.")
        add(f"- Số thuộc tính: **{len(cluster_df.columns)}**.")
        add()
        add("Một số bản ghi:")
        add()
        add(dataframe_to_markdown(cluster_df, max_rows=5))
    else:
        add("_Không tìm thấy customer_clusters.csv._")

    add()

    # --------------------------------------------------------
    # Biểu đồ
    # --------------------------------------------------------

    add("## 6. Biểu đồ và trực quan hóa")
    add()
    add("Các biểu đồ được nhóm theo từng giai đoạn của Pipeline. "
        "Mỗi hình ảnh thể hiện một khía cạnh của dữ liệu hoặc kết quả "
        "phân cụm.")
    add()

    figure_groups = data["figures"]

    if not figure_groups:
        add("_Không tìm thấy hình ảnh trong thư mục figures của Run_ID._")
    else:
        for stage, image_paths in figure_groups.items():
            title = STAGE_TITLES.get(
                stage,
                STAGE_TITLES.get(
                    Path(stage).name,
                    stage.replace("_", " ").title()
                )
            )

            add(f"### 6.{list(figure_groups).index(stage) + 1}. {title}")
            add()

            for image_path in image_paths:
                relative_path = image_path.relative_to(REPORT_DIR)
                image_name = image_path.stem.replace("_", " ").title()

                add(f"#### {image_name}")
                add()
                add(f"![{image_name}]({relative_path.as_posix()})")
                add()
                add(describe_figure(image_path))
                add()

    # --------------------------------------------------------
    # Thực nghiệm mở rộng
    # --------------------------------------------------------

    add("## 7. Thực nghiệm mở rộng")
    add()

    extended_stages = [
        stage for stage in figure_groups
        if stage in {"05_extended", "experiments", "convergence"}
    ]

    has_extended = bool(extended_stages)

    if has_extended:
        add("Lần chạy này có dữ liệu hoặc hình ảnh thuộc nhóm thực nghiệm mở rộng.")
        add()
        add("Các hình ảnh của nhóm thực nghiệm mở rộng đã được trình bày "
            "trong mục 6 theo đúng nhóm thư mục. Mục này tóm tắt các nhóm "
            "nội dung, không chèn lại hình ảnh để tránh trùng lặp.")
        add()
        for stage in extended_stages:
            stage_title = STAGE_TITLES.get(
                stage,
                stage.replace("_", " ").title()
            )
            add(f"- **{stage_title}:** {len(figure_groups[stage])} hình ảnh.")
    else:
        add("Không phát hiện dữ liệu hình ảnh thuộc nhóm thực nghiệm mở rộng "
            "trong lần chạy này. Các nội dung mở rộng không được đưa vào "
            "kết quả chính để tránh nhầm lẫn với Pipeline cơ bản.")
    add()

    # --------------------------------------------------------
    # Logs
    # --------------------------------------------------------

    add("## 8. Nhật ký thực thi")
    add()

    if data["log_text"]:
        add("Nội dung log được lấy từ file:")
        add()
        add(f"`{data['log_path'].relative_to(ROOT_DIR)}`")
        add()
        add("```text")
        add(data["log_text"].rstrip())
        add("```")
    else:
        add("_Không tìm thấy file log cho Run_ID này._")

    add()

    # --------------------------------------------------------
    # Cấu trúc output
    # --------------------------------------------------------

    add("## 9. Các tệp đầu ra")
    add()
    add("| Tệp | Vai trò |")
    add("|---|---|")
    add("| `customers_clean.csv` | Dữ liệu sau làm sạch |")
    add("| `X_scaled.csv` | Dữ liệu đã chuẩn hóa |")
    add("| `scaler.json` | Thông tin tham số chuẩn hóa |")
    add("| `experiment_results.csv` | Kết quả thử nghiệm các giá trị K |")
    add("| `best_kmeans.pkl` | Mô hình K-means đã huấn luyện |")
    add("| `customer_clusters.csv` | Dữ liệu khách hàng kèm nhãn cụm |")
    add("| `cluster_profile.csv` | Thống kê đặc trưng của từng cụm |")
    add("| `progress_results/` | Trạng thái Pipeline theo Run_ID |")
    add("| `logs/` | Nhật ký thực thi theo Run_ID |")
    add("| `reports/figures/` | Biểu đồ theo Run_ID và giai đoạn |")
    add()

    # --------------------------------------------------------
    # Kết luận
    # --------------------------------------------------------

    add("## 10. Kết luận và nhận xét")
    add()

    if best is not None:
        silhouette_col = find_metric_column(
            experiment_df,
            ["silhouette"]
        )
        inertia_col = find_metric_column(
            experiment_df,
            ["inertia"]
        )

        k_col = find_metric_column(
            experiment_df,
            ["best_k", "n_clusters", "clusters", "k"]
        )

        if k_col:
            add(
                f"- Giá trị K trong cấu hình tốt nhất: "
                f"**{format_number(best[k_col])}**."
            )

        if silhouette_col:
            add(
                f"- Silhouette Score: "
                f"**{format_number(best[silhouette_col])}**."
            )

        if inertia_col:
            add(
                f"- Inertia: **{format_number(best[inertia_col])}**."
            )

    if not cluster_counts.empty:
        add(
            f"- Số cụm được tạo trong kết quả: "
            f"**{len(cluster_counts)}**."
        )

        add(
            f"- Tổng số khách hàng được phân cụm: "
            f"**{int(cluster_counts['Customer_Count'].sum()):,}**."
        )

    add()
    add("**Ưu điểm:**")
    add("- K-means có quy trình đơn giản, dễ triển khai và dễ giải thích.")
    add("- Chuẩn hóa dữ liệu giúp hạn chế ảnh hưởng của sự khác biệt "
        "thang đo giữa các đặc trưng.")
    add("- Có thể sử dụng hồ sơ cụm để phân tích hành vi và đặc điểm "
        "khách hàng.")
    add()
    add("**Hạn chế:**")
    add("- K-means cần xác định số cụm K trước khi huấn luyện.")
    add("- Kết quả có thể phụ thuộc vào cách khởi tạo tâm cụm.")
    add("- Silhouette Score và Inertia cần được xem xét kết hợp, "
        "không nên sử dụng riêng một chỉ số để kết luận.")
    add("- Silhouette Score thấp có thể phản ánh các cụm chưa tách biệt "
        "rõ ràng trong không gian đặc trưng đã chọn.")
    add()
    add("**Kết luận chung:** Kết quả phân cụm cần được đánh giá đồng thời "
        "qua chỉ số định lượng, kích thước cụm, hồ sơ đặc trưng và biểu đồ "
        "trực quan. Việc lựa chọn K cần cân bằng giữa độ gắn kết trong cụm, "
        "độ tách biệt giữa các cụm và khả năng diễn giải trong thực tế.")
    add()

    add("---")
    add()
    add(f"_Báo cáo được tạo tự động cho Run_ID: `{run_id}`._")

    return "\n".join(lines)


# ============================================================
# 7. CHUYỂN MARKDOWN SANG HTML
# ============================================================

def markdown_to_html(markdown_text, report_dir):
    """
    Chuyển Markdown sang HTML.
    Xử lý heading, paragraph, list, code block, bảng và ảnh.
    """
    lines = markdown_text.splitlines()
    output = []

    in_code = False
    in_list = False
    in_table = False
    paragraph = []

    def close_paragraph():
        nonlocal paragraph

        if paragraph:
            content = " ".join(paragraph)
            output.append(f"<p>{inline_markdown(content, report_dir)}</p>")
            paragraph = []

    def close_list():
        nonlocal in_list

        if in_list:
            output.append("</ul>")
            in_list = False

    def close_table():
        nonlocal in_table

        if in_table:
            output.append("</tbody></table></div>")
            in_table = False

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("```"):
            close_paragraph()
            close_list()
            close_table()

            if not in_code:
                output.append("<pre><code>")
                in_code = True
            else:
                output.append("</code></pre>")
                in_code = False

            continue

        if in_code:
            output.append(html.escape(line))
            continue

        if not stripped:
            close_paragraph()
            close_list()
            close_table()
            continue

        # Bảng Markdown
        if stripped.startswith("|") and stripped.endswith("|"):
            close_paragraph()
            close_list()

            cells = [
                cell.strip()
                for cell in stripped.strip("|").split("|")
            ]

            # Bỏ dòng phân cách bảng
            if all(
                re.fullmatch(r":?-{3,}:?", cell.replace(" ", ""))
                for cell in cells
            ):
                continue

            if not in_table:
                output.append(
                    '<div class="table-wrapper"><table><thead><tr>'
                )

                for cell in cells:
                    output.append(
                        f"<th>{inline_markdown(cell, report_dir)}</th>"
                    )

                output.append("</tr></thead><tbody>")
                in_table = True
            else:
                output.append("<tr>")

                for cell in cells:
                    output.append(
                        f"<td>{inline_markdown(cell, report_dir)}</td>"
                    )

                output.append("</tr>")

            continue

        close_table()

        # Heading
        heading = re.match(r"^(#{1,6})\s+(.*)$", stripped)

        if heading:
            close_paragraph()
            close_list()

            level = len(heading.group(1))
            content = inline_markdown(heading.group(2), report_dir)

            output.append(f"<h{level}>{content}</h{level}>")
            continue

        # Horizontal line
        if stripped == "---":
            close_paragraph()
            close_list()
            output.append("<hr>")
            continue

        # Blockquote
        if stripped.startswith(">"):
            close_paragraph()
            close_list()

            content = inline_markdown(
                stripped.lstrip("> ").strip(),
                report_dir
            )
            output.append(f"<blockquote>{content}</blockquote>")
            continue

        # Bullet list
        if stripped.startswith("- "):
            close_paragraph()

            if not in_list:
                output.append("<ul>")
                in_list = True

            content = inline_markdown(stripped[2:], report_dir)
            output.append(f"<li>{content}</li>")
            continue

        close_list()

        paragraph.append(stripped)

    close_paragraph()
    close_list()
    close_table()

    if in_code:
        output.append("</code></pre>")

    return "\n".join(output)


def inline_markdown(text, report_dir):
    """Xử lý định dạng inline cơ bản."""
    text = html.escape(str(text))

    # Ảnh Markdown
    image_pattern = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")

    def replace_image(match):
        alt = match.group(1)
        raw_path = match.group(2)

        image_path = (report_dir / raw_path).resolve()

        try:
            image_path.relative_to(report_dir.resolve())
        except ValueError:
            return ""

        if not image_path.exists():
            return (
                f'<p class="missing-image">Không tìm thấy ảnh: '
                f'{html.escape(alt)}</p>'
            )

        if image_path.suffix.lower() == ".svg":
            encoded = base64.b64encode(
                image_path.read_bytes()
            ).decode("ascii")

            src = f"data:image/svg+xml;base64,{encoded}"
        else:
            mime = (
                "image/jpeg"
                if image_path.suffix.lower() in {".jpg", ".jpeg"}
                else "image/png"
            )

            encoded = base64.b64encode(
                image_path.read_bytes()
            ).decode("ascii")

            src = f"data:{mime};base64,{encoded}"

        return (
            f'<img src="{src}" alt="{html.escape(alt)}" '
            f'class="report-image">'
        )

    text = image_pattern.sub(replace_image, text)

    # Bold
    text = re.sub(
        r"\*\*(.+?)\*\*",
        r"<strong>\1</strong>",
        text
    )

    # Inline code
    text = re.sub(
        r"`(.+?)`",
        r"<code>\1</code>",
        text
    )

    # Italic
    text = re.sub(
        r"(?<!\*)\*([^*]+)\*(?!\*)",
        r"<em>\1</em>",
        text
    )

    return text


def build_html(markdown_text, run_id):
    """Tạo HTML hoàn chỉnh từ nội dung Markdown."""
    body = markdown_to_html(markdown_text, REPORT_DIR)

    return f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Lab03 Report - {html.escape(str(run_id))}</title>

<style>
* {{
    box-sizing: border-box;
}}

body {{
    margin: 0;
    padding: 32px 16px;
    background: #f3f5f9;
    color: #263238;
    font-family: Arial, Helvetica, sans-serif;
    line-height: 1.7;
}}

.container {{
    max-width: 1100px;
    margin: auto;
    padding: 42px;
    background: #ffffff;
    border-radius: 12px;
    box-shadow: 0 5px 24px rgba(0,0,0,0.07);
}}

h1 {{
    color: #17365d;
    text-align: center;
    font-size: 27px;
    padding-bottom: 20px;
    border-bottom: 3px solid #2f75b5;
}}

h2 {{
    color: #1f4e79;
    margin-top: 38px;
    padding-bottom: 8px;
    border-bottom: 1px solid #d8e1eb;
}}

h3 {{
    color: #2f75b5;
    margin-top: 26px;
}}

h4 {{
    color: #40566f;
    margin-top: 20px;
}}

p {{
    text-align: justify;
}}

table {{
    width: 100%;
    border-collapse: collapse;
    margin: 16px 0 24px;
    font-size: 14px;
}}

th {{
    background: #1f4e79;
    color: white;
    font-weight: bold;
    text-align: left;
}}

th, td {{
    border: 1px solid #d9e1ea;
    padding: 10px 12px;
    vertical-align: top;
}}

tr:nth-child(even) {{
    background: #f5f8fc;
}}

.table-wrapper {{
    width: 100%;
    overflow-x: auto;
}}

code {{
    background: #eef2f7;
    padding: 2px 5px;
    border-radius: 4px;
    color: #b03a2e;
}}

pre {{
    background: #17212f;
    color: #e8edf4;
    padding: 18px;
    border-radius: 8px;
    overflow-x: auto;
    line-height: 1.5;
}}

pre code {{
    color: inherit;
    background: transparent;
}}

blockquote {{
    margin: 20px 0;
    padding: 12px 18px;
    border-left: 4px solid #2f75b5;
    background: #eef5fc;
}}

.report-image {{
    display: block;
    max-width: 100%;
    height: auto;
    margin: 18px auto;
    border: 1px solid #e0e5eb;
    border-radius: 8px;
}}

li {{
    margin: 5px 0;
}}

hr {{
    margin: 35px 0;
    border: none;
    border-top: 1px solid #d9e1ea;
}}

.empty, .note, .missing-image {{
    color: #77818c;
    font-style: italic;
}}

@media print {{
    body {{
        padding: 0;
        background: white;
    }}

    .container {{
        box-shadow: none;
        max-width: none;
        padding: 15px;
    }}

    table, img, pre {{
        page-break-inside: avoid;
    }}

    h2, h3, h4 {{
        page-break-after: avoid;
    }}
}}
</style>
</head>

<body>
<div class="container">
{body}
</div>
</body>
</html>
"""


# ============================================================
# 8. TẠO PDF
# ============================================================

def create_pdf(markdown_text, output_path, report_dir):
    """
    Tạo PDF bằng ReportLab.

    PDF xử lý:
    - Heading
    - Paragraph
    - Danh sách
    - Bảng Markdown
    - Code block
    - Hình ảnh
    """

    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import (
            getSampleStyleSheet,
            ParagraphStyle,
        )
        from reportlab.lib.units import mm
        from reportlab.platypus import (
            SimpleDocTemplate,
            Paragraph,
            Spacer,
            Table,
            TableStyle,
            Image,
            PageBreak,
            Preformatted,
        )
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont

    except ImportError as exc:
        raise RuntimeError(
            "Chưa cài ReportLab. Hãy chạy: pip install reportlab"
        ) from exc

    # Tìm font Unicode có hỗ trợ tiếng Việt.
    # Ưu tiên font có sẵn trên Windows, sau đó thử Linux/macOS.
    font_candidates = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/calibri.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
        Path("/Library/Fonts/Arial.ttf"),
    ]

    bold_candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf"),
        Path("C:/Windows/Fonts/segoeuib.ttf"),
        Path("C:/Windows/Fonts/calibrib.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"),
        Path("/Library/Fonts/Arial Bold.ttf"),
    ]

    normal_path = next((path for path in font_candidates if path.exists()), None)
    bold_path = next((path for path in bold_candidates if path.exists()), None)

    if normal_path is None:
        raise RuntimeError(
            "Không tìm thấy font Unicode để xuất PDF tiếng Việt. "
            "Trên Windows, hãy kiểm tra C:/Windows/Fonts/arial.ttf "
            "hoặc cài font DejaVu Sans."
        )

    pdfmetrics.registerFont(TTFont("VietnameseFont", str(normal_path)))
    normal_font = "VietnameseFont"

    # Nếu không có font đậm riêng, dùng font thường để tránh lỗi glyph.
    if bold_path:
        pdfmetrics.registerFont(TTFont("VietnameseFontBold", str(bold_path)))
        bold_font = "VietnameseFontBold"
    else:
        bold_font = normal_font

    pdfmetrics.registerFontFamily(
        "VietnameseFont",
        normal="VietnameseFont",
        bold=bold_font,
        italic="VietnameseFont",
        boldItalic=bold_font,
    )

    styles = getSampleStyleSheet()

    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            fontName=bold_font,
            fontSize=17,
            leading=23,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#17365D"),
            spaceAfter=18,
        )
    )

    styles.add(
        ParagraphStyle(
            name="ReportH2",
            parent=styles["Heading2"],
            fontName=bold_font,
            fontSize=13,
            leading=17,
            textColor=colors.HexColor("#1F4E79"),
            spaceBefore=15,
            spaceAfter=8,
            keepWithNext=True,
        )
    )

    styles.add(
        ParagraphStyle(
            name="ReportH3",
            parent=styles["Heading3"],
            fontName=bold_font,
            fontSize=11,
            leading=15,
            textColor=colors.HexColor("#2F75B5"),
            spaceBefore=10,
            spaceAfter=6,
            keepWithNext=True,
        )
    )

    styles.add(
        ParagraphStyle(
            name="ReportBody",
            parent=styles["BodyText"],
            fontName=normal_font,
            fontSize=9,
            leading=14,
            alignment=TA_LEFT,
            spaceAfter=7,
        )
    )

    styles.add(
        ParagraphStyle(
            name="ReportSmall",
            parent=styles["BodyText"],
            fontName=normal_font,
            fontSize=6.8,
            leading=8.2,
            wordWrap="CJK",
        )
    )

    story = []

    def clean_inline(value):
        value = html.escape(str(value))

        value = re.sub(
            r"\*\*(.+?)\*\*",
            r"<b>\1</b>",
            value
        )

        value = re.sub(
            r"`(.+?)`",
            r"<font color='#B03A2E'>\1</font>",
            value
        )

        value = re.sub(
            r"(?<!\*)\*([^*]+)\*(?!\*)",
            r"<i>\1</i>",
            value
        )

        return value

    def add_paragraph(text):
        text = clean_inline(text)

        if text.strip():
            story.append(
                Paragraph(text, styles["ReportBody"])
            )

    def add_table(table_lines):
        parsed = []

        for line in table_lines:
            cells = [
                cell.strip()
                for cell in line.strip().strip("|").split("|")
            ]

            if all(
                re.fullmatch(r":?-{3,}:?", cell.replace(" ", ""))
                for cell in cells
            ):
                continue

            parsed.append(cells)

        if not parsed:
            return

        column_count = max(len(row) for row in parsed)

        table_data = []
        dense_table = column_count >= 9
        table_font_size = 5.6 if dense_table else (6.4 if column_count >= 6 else 7.2)
        table_leading = table_font_size + 1.2
        cell_style = ParagraphStyle(
            name=f"ReportTableCell{column_count}",
            parent=styles["ReportSmall"],
            fontName=normal_font,
            fontSize=table_font_size,
            leading=table_leading,
            wordWrap="CJK",
        )

        for row in parsed:
            row = row + [""] * (column_count - len(row))

            table_data.append([
                Paragraph(
                    clean_inline(value),
                    cell_style
                )
                for value in row
            ])

        available_width = A4[0] - 36 * mm

        # Bảng Tracking có 6 cột: dành thêm không gian cho timestamp và thông tin.
        if column_count == 6 and parsed and "Thời gian bắt đầu" in parsed[0]:
            ratios = [0.105, 0.085, 0.19, 0.19, 0.105, 0.325]
            column_widths = [available_width * ratio for ratio in ratios]
        else:
            column_widths = [available_width / column_count] * column_count

        table = Table(
            table_data,
            colWidths=column_widths,
            repeatRows=1,
            hAlign="LEFT",
        )

        table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0),
                 colors.HexColor("#1F4E79")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.4,
                 colors.HexColor("#C9D3DF")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 2 if dense_table else 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2 if dense_table else 3),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [
                    colors.white,
                    colors.HexColor("#F3F6FA"),
                ]),
            ])
        )

        story.append(table)
        story.append(Spacer(1, 8))

    lines = markdown_text.splitlines()

    index = 0
    in_code = False
    code_lines = []
    table_lines = []

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()

        if stripped.startswith("```"):
            if not in_code:
                in_code = True
                code_lines = []
            else:
                in_code = False

                story.append(
                    Preformatted(
                        "\n".join(code_lines),
                        ParagraphStyle(
                            name="CodeBlock",
                            fontName=normal_font,
                            fontSize=7,
                            leading=10,
                            wordWrap="CJK",
                            backColor=colors.HexColor("#F0F2F5"),
                            borderPadding=7,
                        )
                    )
                )

                story.append(Spacer(1, 8))

            index += 1
            continue

        if in_code:
            code_lines.append(line)
            index += 1
            continue

        # Gom bảng liên tiếp
        if stripped.startswith("|") and stripped.endswith("|"):
            table_lines.append(stripped)
            index += 1
            continue

        if table_lines:
            add_table(table_lines)
            table_lines = []

        # Tiêu đề
        heading = re.match(r"^(#{1,6})\s+(.*)$", stripped)

        if heading:
            level = len(heading.group(1))
            title = heading.group(2)

            if level == 1:
                story.append(
                    Paragraph(
                        clean_inline(title),
                        styles["ReportTitle"]
                    )
                )
            elif level == 2:
                story.append(
                    Paragraph(
                        clean_inline(title),
                        styles["ReportH2"]
                    )
                )
            else:
                story.append(
                    Paragraph(
                        clean_inline(title),
                        styles["ReportH3"]
                    )
                )

            index += 1
            continue

        # Ảnh Markdown
        image_match = re.match(
            r"!\[([^\]]*)\]\(([^)]+)\)",
            stripped
        )

        if image_match:
            relative_path = image_match.group(2)
            image_path = (report_dir / relative_path).resolve()

            try:
                image_path.relative_to(report_dir.resolve())
            except ValueError:
                image_path = None

            if image_path and image_path.exists():
                try:
                    image = Image(str(image_path))
                    image._restrictSize(165 * mm, 105 * mm)
                    story.append(image)
                    story.append(Spacer(1, 6))
                except Exception:
                    add_paragraph(
                        f"Không thể hiển thị hình ảnh: {image_match.group(1)}"
                    )

            index += 1
            continue

        # Danh sách
        if stripped.startswith("- "):
            add_paragraph("• " + stripped[2:])
            index += 1
            continue

        numbered = re.match(r"^\d+\.\s+(.*)$", stripped)

        if numbered:
            add_paragraph("• " + numbered.group(1))
            index += 1
            continue

        # Blockquote
        if stripped.startswith(">"):
            add_paragraph(stripped.lstrip("> ").strip())
            index += 1
            continue

        # Ngăn cách
        if stripped == "---":
            story.append(Spacer(1, 10))
            index += 1
            continue

        if stripped:
            add_paragraph(stripped)

        index += 1

    if table_lines:
        add_table(table_lines)

    def add_page_number(canvas, document):
        canvas.saveState()
        canvas.setFont(normal_font, 8)
        canvas.setFillColor(colors.HexColor("#6B7280"))

        page_number = canvas.getPageNumber()

        canvas.drawRightString(
            A4[0] - 18 * mm,
            10 * mm,
            f"Lab03 Report | Trang {page_number}"
        )

        canvas.restoreState()

    document = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=17 * mm,
        bottomMargin=18 * mm,
        title="Lab03 Customer Segmentation Report",
        author="Lab03 Pipeline",
    )

    document.build(
        story,
        onFirstPage=add_page_number,
        onLaterPages=add_page_number,
    )


# ============================================================
# 9. HÀM TẠO BÁO CÁO CHÍNH
# ============================================================

def generate_report(run_id, status=None):
    """
    Tạo báo cáo Markdown, HTML và PDF cho một Run_ID.

    Chạy độc lập:
        python -m src.report --run-id <Run_ID>

    Có thể truyền trạng thái nếu cần:
        generate_report(run_id, status="SUCCESS")
    """
    run_id = str(run_id)

    print(
        f"\n[REPORT] Đang tạo báo cáo cho Run_ID: {run_id}"
    )

    # Đọc dữ liệu tracking và các kết quả của Run_ID.
    data = build_report_data(run_id)

    # Chỉ ghi đè status nếu caller chủ động truyền vào.
    # Khi chạy report độc lập, status sẽ được đọc từ tracking JSON.
    if status is not None:
        data["report_status"] = str(status).upper()

    markdown_text = build_markdown(data)

    md_path = REPORT_DIR / f"report_{run_id}.md"
    html_path = REPORT_DIR / f"report_{run_id}.html"
    pdf_path = REPORT_DIR / f"report_{run_id}.pdf"

    # --------------------------------------------------------
    # 1. Markdown
    # --------------------------------------------------------

    md_path.write_text(
        markdown_text,
        encoding="utf-8"
    )

    print(f"[REPORT] Markdown: {md_path}")

    # --------------------------------------------------------
    # 2. HTML
    # --------------------------------------------------------

    html_text = build_html(
        markdown_text,
        run_id
    )

    html_path.write_text(
        html_text,
        encoding="utf-8"
    )

    print(f"[REPORT] HTML: {html_path}")

    # --------------------------------------------------------
    # 3. PDF
    # --------------------------------------------------------

    create_pdf(
        markdown_text,
        pdf_path,
        REPORT_DIR
    )

    print(f"[REPORT] PDF: {pdf_path}")

    print(
        "[REPORT] Hoàn thành tạo báo cáo."
    )

    return {
        "markdown": str(md_path),
        "html": str(html_path),
        "pdf": str(pdf_path),
    }


# ============================================================
# 10. CHẠY REPORT
# ============================================================

def main():
    """Điểm chạy độc lập; hỗ trợ chọn Run_ID hoặc lấy lần chạy mới nhất."""
    parser = argparse.ArgumentParser(
        description="Tạo báo cáo Lab03 ở định dạng Markdown, HTML và PDF."
    )
    parser.add_argument(
        "--run-id",
        dest="run_id",
        default=None,
        help="Run_ID cần xuất báo cáo. Nếu bỏ trống sẽ chọn Run_ID mới nhất.",
    )
    args = parser.parse_args()

    run_id = args.run_id or get_latest_run_id()
    generate_report(run_id=run_id)


# ============================================================
# 11. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()