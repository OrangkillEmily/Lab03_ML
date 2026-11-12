"""Generate detailed Markdown, HTML and PDF reports for Image Segmentation."""

from pathlib import Path
from datetime import datetime
import base64
import html
import re
import os
import sys
from PIL import Image
import numpy as np
import pandas as pd


# ============================================================
# 1. PATH CONFIGURATION
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]
PACKAGE_DIR = Path(__file__).resolve().parent

REPORT_DIR = ROOT_DIR / "reports" / "additional_image_segmentation"

REPORT_MD = REPORT_DIR / "image_segmentation_report.md"
REPORT_HTML = REPORT_DIR / "image_segmentation_report.html"
REPORT_PDF = REPORT_DIR / "image_segmentation_report.pdf"

FIGURE_ROOT = (
    ROOT_DIR
    / "reports"
    / "figures"
    / "additional_image_segmentation"
)


# ============================================================
# 2. GENERAL HELPERS
# ============================================================

def get_run_suffix(run_id):
    """Extract the suffix from a pipeline Run_ID."""
    prefix = "additional_image_segmentation--"
    run_id = str(run_id)

    if run_id.startswith(prefix):
        suffix = run_id[len(prefix):]
    else:
        suffix = run_id

    if not suffix or Path(suffix).name != suffix:
        raise ValueError(f"Run_ID không hợp lệ: {run_id}")

    return suffix


def get_run_figure_dir(run_id):
    """Get the figure directory of the current run."""
    return FIGURE_ROOT / get_run_suffix(run_id)


def safe_relative_path(path, base_dir=PACKAGE_DIR):
    """Return a safe relative path for report images."""
    path = Path(path).resolve()

    try:
        path.relative_to(ROOT_DIR.resolve())
    except ValueError as exc:
        raise ValueError(
            f"Đường dẫn nằm ngoài thư mục dự án: {path}"
        ) from exc

    return Path(os.path.relpath(path, base_dir)).as_posix()


def format_number(value, decimals=3):
    """Format numeric values for report display."""
    try:
        value = float(value)

        if not np.isfinite(value):
            return "N/A"

        return f"{value:,.{decimals}f}"

    except (TypeError, ValueError):
        return str(value)


def dataframe_to_markdown(df, max_rows=None):
    """Convert DataFrame to Markdown without requiring tabulate."""
    if df is None or df.empty:
        return "_Không có dữ liệu._"

    display_df = df.copy()

    if max_rows is not None:
        display_df = display_df.head(max_rows)

    headers = [str(column) for column in display_df.columns]

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for _, row in display_df.iterrows():
        values = []

        for value in row.tolist():
            if pd.isna(value):
                value = ""

            value = str(value).replace("|", "\\|")
            value = value.replace("\n", " ")

            values.append(value)

        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)



def collect_run_figures(run_id):
    """Collect all figures belonging to the current run."""
    run_dir = get_run_figure_dir(run_id)

    stages = {
        "elbow": "Elbow Method",
        "segmentation": "Image Segmentation",
        "palette": "Color Palette",
    }

    figures = {}

    for stage, title in stages.items():
        stage_dir = run_dir / stage
        files = []

        if stage_dir.is_dir():
            files = sorted(
                path
                for path in stage_dir.iterdir()
                if path.is_file()
                and path.suffix.lower() in {
                    ".png", ".jpg", ".jpeg", ".svg"
                }
            )

        figures[stage] = {
            "title": title,
            "files": files,
        }

    return figures



def get_image_statistics(image_results):
    """Build basic image statistics from available experiment results."""
    records = []

    for image_name, result in image_results.items():
        segmented_results = result.get("segmented_results", {})

        if segmented_results:
            sample_image = next(iter(segmented_results.values()))
            height, width = sample_image.shape[:2]
            n_pixels = height * width
        else:
            height = width = n_pixels = 0

        records.append({
            "Image": image_name,
            "Height": height,
            "Width": width,
            "Total pixels": n_pixels,
            "Tested K": ", ".join(
                map(str, sorted(segmented_results.keys()))
            ),
        })

    return pd.DataFrame(records)


# ============================================================
# 3. REPORT DATA
# ============================================================

def prepare_report_data(
    run_id,
    elbow_df,
    palette_df,
    image_results,
    tested_k,
    status="SUCCESS",
):
    """Prepare all data used by Markdown, HTML and PDF."""
    image_results = image_results or {}

    return {
        "run_id": str(run_id),
        "status": str(status).upper(),
        "created_at": datetime.now().astimezone().strftime(
            "%Y-%m-%d %H:%M:%S %Z"
        ),
        "elbow_df": (
            elbow_df.copy()
            if elbow_df is not None
            else pd.DataFrame()
        ),
        "palette_df": (
            palette_df.copy()
            if palette_df is not None
            else pd.DataFrame()
        ),
        "image_results": image_results,
        "tested_k": list(tested_k or []),
        "image_stats": get_image_statistics(image_results),
        "figures": collect_run_figures(run_id),
        "run_figure_dir": get_run_figure_dir(run_id),
    }


# ============================================================
# 4. ANALYSIS HELPERS
# ============================================================

def analyze_elbow_results(elbow_df):
    """Generate observations from Elbow Method results."""
    if elbow_df is None or elbow_df.empty:
        return [
            "Không có dữ liệu Elbow Method để phân tích."
        ]

    observations = []

    for image_name, group in elbow_df.groupby("image"):
        group = group.sort_values("K")

        if len(group) < 2:
            observations.append(
                f"- Ảnh **{image_name}** không có đủ giá trị K "
                "để so sánh mức thay đổi Inertia."
            )
            continue

        first = float(group.iloc[0]["inertia"])
        last = float(group.iloc[-1]["inertia"])

        reduction = (
            (first - last) / first * 100
            if first > 0 else 0
        )

        observations.append(
            f"- Với ảnh **{image_name}**, Inertia giảm khoảng "
            f"**{reduction:.2f}%** trong phạm vi K được thử nghiệm."
        )

        values = group["inertia"].to_numpy(dtype=float)
        ks = group["K"].to_numpy(dtype=float)

        if len(values) >= 3:
            improvements = -np.diff(values)
            ratios = []

            for index in range(len(improvements) - 1):
                current = improvements[index]
                following = improvements[index + 1]

                if current > 0:
                    ratios.append(
                        (index, following / current)
                    )

            if ratios:
                best_index = min(
                    ratios,
                    key=lambda item: item[1],
                )[0]

                candidate_k = int(ks[best_index + 1])

                observations.append(
                    f"- Mức giảm Inertia có xu hướng chậm lại "
                    f"sau vùng K xấp xỉ **{candidate_k}**. "
                    "Đây chỉ là gợi ý tham khảo, không phải "
                    "K tối ưu được xác định chắc chắn."
                )

    return observations


def analyze_palette_results(palette_df):
    """Generate observations about cluster proportions."""
    if palette_df is None or palette_df.empty:
        return [
            "Không có dữ liệu bảng màu để phân tích."
        ]

    observations = []

    required = {
        "image",
        "K",
        "cluster",
        "proportion",
    }

    if not required.issubset(palette_df.columns):
        return observations

    for (image_name, k), group in palette_df.groupby(
        ["image", "K"]
    ):
        group = group.sort_values(
            "proportion",
            ascending=False,
        )

        largest = group.iloc[0]
        smallest = group.iloc[-1]

        observations.append(
            f"- Ảnh **{image_name}**, K={k}: cụm chiếm tỷ lệ lớn "
            f"nhất là cluster {int(largest['cluster'])}, "
            f"chiếm **{float(largest['proportion']) * 100:.2f}%** "
            "tổng số pixel."
        )

        if len(group) > 1:
            observations.append(
                f"  Cụm có tỷ lệ nhỏ nhất là cluster "
                f"{int(smallest['cluster'])}, chiếm "
                f"**{float(smallest['proportion']) * 100:.2f}%**."
            )

    return observations


# ============================================================
# 5. BUILD MARKDOWN REPORT
# ============================================================

def build_markdown_report(data):
    """Build a detailed Markdown report."""
    lines = []

    def add(text=""):
        lines.append(str(text))

    run_id = data["run_id"]
    elbow_df = data["elbow_df"]
    palette_df = data["palette_df"]
    image_results = data["image_results"]
    tested_k = data["tested_k"]
    image_stats = data["image_stats"]
    figures = data["figures"]

    # --------------------------------------------------------
    # Cover and metadata
    # --------------------------------------------------------

    add("# BÁO CÁO ĐỒ ÁN MACHINE LEARNING")
    add()
    add("# ADDITIONAL: IMAGE SEGMENTATION USING K-MEANS")
    add()

    add("## THÔNG TIN THỰC NGHIỆM")
    add()
    add(f"- **Run_ID:** `{run_id}`")
    add(f"- **Trạng thái:** {data['status']}")
    add(f"- **Thời gian tạo:** {data['created_at']}")
    add("- **Phương pháp:** K-Means với K-Means++ initialization")
    add("- **Đặc trưng:** RGB color space")
    add("- **Đơn vị dữ liệu:** Pixel")
    add()

    # --------------------------------------------------------
    # 1. Introduction
    # --------------------------------------------------------

    add("## 1. GIỚI THIỆU")
    add()

    add("### 1.1. Bối cảnh bài toán")
    add()
    add(
        "Image Segmentation là quá trình phân chia ảnh thành các "
        "nhóm pixel có đặc điểm tương đồng. Trong bài thực nghiệm "
        "này, phân cụm được thực hiện dựa trên màu sắc RGB của pixel."
    )
    add()
    add(
        "K-Means là thuật toán học không giám sát, không yêu cầu "
        "nhãn phân đoạn có sẵn. Thuật toán tìm các tâm cụm sao cho "
        "tổng bình phương khoảng cách giữa pixel và tâm cụm tương ứng "
        "được tối thiểu hóa."
    )
    add()

    add("### 1.2. Mục tiêu")
    add()
    add("- Xây dựng thuật toán K-Means từ đầu.")
    add("- Sử dụng K-Means++ để khởi tạo tâm cụm.")
    add("- Khảo sát ảnh hưởng của số cụm K.")
    add("- Trực quan hóa kết quả phân đoạn.")
    add("- Phân tích màu đại diện và tỷ lệ pixel của từng cụm.")
    add()

    add("### 1.3. Phạm vi thực nghiệm")
    add()
    add(
        "Thực nghiệm sử dụng ảnh RGB. Mỗi pixel được biểu diễn bằng "
        "vector ba chiều gồm giá trị Red, Green và Blue. Thuật toán "
        "không sử dụng vị trí không gian hoặc đặc trưng ngữ nghĩa."
    )
    add()

    # --------------------------------------------------------
    # 2. Data
    # --------------------------------------------------------

    add("## 2. DỮ LIỆU ĐẦU VÀO")
    add()

    add("### 2.1. Danh sách ảnh")
    add()

    if not image_stats.empty:
        add(dataframe_to_markdown(image_stats))
    else:
        add("_Không có ảnh đầu vào._")

    add()

    add("### 2.2. Cấu trúc dữ liệu ảnh")
    add()
    add(
        "Ảnh được đọc và chuyển sang RGB để đảm bảo mỗi pixel có "
        "ba kênh màu. Ảnh có kích thước H × W × 3 được chuyển thành "
        "ma trận N × 3, với N = H × W."
    )
    add()
    add(
        "Mỗi hàng của ma trận tương ứng với một pixel. Ba cột lần lượt "
        "là giá trị Red, Green và Blue."
    )
    add()

    add("### 2.3. Đặc điểm dữ liệu")
    add()
    add(
        "- Số lượng mẫu phụ thuộc vào độ phân giải của ảnh."
    )
    add(
        "- Các giá trị RGB nằm trong khoảng 0 đến 255."
    )
    add(
        "- Pixel trong cùng một vùng ảnh có thể có màu gần nhau."
    )
    add(
        "- Các vùng vật thể khác nhau có thể có màu tương tự."
    )
    add()

    # --------------------------------------------------------
    # 3. Preprocessing
    # --------------------------------------------------------

    add("## 3. TIỀN XỬ LÝ DỮ LIỆU")
    add()

    add("### 3.1. Đọc và chuẩn hóa định dạng ảnh")
    add()
    add(
        "Ảnh đầu vào được đọc bằng Pillow và chuyển về RGB. "
        "Bước này giúp thống nhất số kênh màu, tránh trường hợp "
        "ảnh đầu vào sử dụng grayscale, RGBA hoặc các định dạng "
        "màu khác."
    )
    add()

    add("### 3.2. Chuyển ảnh thành ma trận pixel")
    add()
    add(
        "Ma trận ảnh ba chiều được reshape thành ma trận hai chiều. "
        "Dữ liệu được chuyển sang float32 để thực hiện phép tính "
        "khoảng cách và cập nhật tâm cụm."
    )
    add()

    add("### 3.3. Lấy mẫu pixel")
    add()
    add(
        "Với ảnh có độ phân giải lớn, việc huấn luyện trên toàn bộ "
        "pixel có thể làm tăng chi phí tính toán. Pipeline sử dụng "
        "lấy mẫu ngẫu nhiên không hoàn lại để tạo tập pixel huấn luyện."
    )
    add()
    add(
        "Sau khi huấn luyện, mô hình dự đoán nhãn cho toàn bộ pixel "
        "theo từng batch để hạn chế sử dụng bộ nhớ."
    )
    add()

    add("### 3.4. Tái tạo ảnh")
    add()
    add(
        "Sau khi xác định nhãn cụm của từng pixel, mỗi pixel được "
        "thay bằng màu RGB của tâm cụm tương ứng. Ma trận pixel "
        "được reshape về kích thước ảnh ban đầu."
    )
    add()

    # --------------------------------------------------------
    # 4. Methodology
    # --------------------------------------------------------

    add("## 4. PHƯƠNG PHÁP")
    add()

    add("### 4.1. K-Means")
    add()
    add(
        "K-Means tối ưu tổng bình phương khoảng cách trong mỗi cụm:"
    )
    add()
    add(
        "$$J = \\\\sum_{i=1}^{N} "
        "\\\\left\\\\|x_i - \\\\mu_{c_i}\\\\right\\\\|^2$$"
    )
    add()
    add("Trong đó:")
    add("- $x_i$: vector RGB của pixel thứ i.")
    add("- $\\\\mu_{c_i}$: tâm cụm được gán cho pixel.")
    add("- $N$: tổng số pixel.")
    add("- $c_i$: nhãn cụm của pixel.")
    add()

    add("### 4.2. Khởi tạo K-Means++")
    add()
    add(
        "K-Means++ lựa chọn tâm đầu tiên ngẫu nhiên. Các tâm tiếp "
        "theo được chọn dựa trên xác suất tỷ lệ với bình phương "
        "khoảng cách đến tâm gần nhất đã chọn."
    )
    add()
    add(
        "Cách khởi tạo này giúp các tâm ban đầu phân bố hợp lý hơn "
        "so với việc chọn hoàn toàn ngẫu nhiên."
    )
    add()

    add("### 4.3. Gán cụm")
    add()
    add(
        "Mỗi pixel được gán vào tâm cụm có khoảng cách Euclidean "
        "nhỏ nhất trong không gian RGB."
    )
    add()
    add(
        "$$c_i = \\\\arg\\\\min_k "
        "\\\\left\\\\|x_i - \\\\mu_k\\\\right\\\\|^2$$"
    )
    add()

    add("### 4.4. Cập nhật tâm cụm")
    add()
    add(
        "Tâm cụm được cập nhật bằng trung bình cộng của tất cả "
        "pixel thuộc cụm đó."
    )
    add()
    add(
        "$$\\\\mu_k = \\\\frac{1}{|C_k|}"
        "\\\\sum_{x_i \\\\in C_k} x_i$$"
    )
    add()

    add("### 4.5. Điều kiện dừng")
    add()
    add(
        "Thuật toán dừng khi độ dịch chuyển tâm cụm nhỏ hơn ngưỡng "
        "tolerance hoặc đạt số vòng lặp tối đa."
    )
    add()

    # --------------------------------------------------------
    # 5. Experiment setup
    # --------------------------------------------------------

    add("## 5. THIẾT LẬP THỰC NGHIỆM")
    add()

    add("### 5.1. Cấu hình")
    add()
    add(f"- **Các giá trị K khảo sát Elbow:** `{sorted(elbow_df['K'].unique().tolist()) if not elbow_df.empty else []}`")
    add(f"- **Các giá trị K phân đoạn:** `{tested_k}`")
    add("- **Khởi tạo:** K-Means++")
    add("- **Random state:** 42")
    add("- **Số vòng lặp tối đa:** 100")
    add("- **Số pixel huấn luyện tối đa:** 20.000")
    add("- **Kênh màu:** RGB")
    add()

    add("### 5.2. Quy trình thực nghiệm")
    add()
    add("1. Đọc ảnh và chuyển sang RGB.")
    add("2. Chuyển ảnh thành ma trận pixel.")
    add("3. Lấy mẫu pixel để huấn luyện.")
    add("4. Chạy Elbow Method trên nhiều giá trị K.")
    add("5. Huấn luyện mô hình với các giá trị K phân đoạn.")
    add("6. Dự đoán nhãn cho toàn bộ pixel.")
    add("7. Tái tạo ảnh và lưu kết quả.")
    add("8. Thống kê màu đại diện và trực quan hóa.")
    add()

    # --------------------------------------------------------
    # 6. Elbow
    # --------------------------------------------------------

    add("## 6. KẾT QUẢ ELBOW METHOD")
    add()

    add("### 6.1. Bảng kết quả")
    add()

    if not elbow_df.empty:
        display_df = elbow_df.copy()

        if "inertia" in display_df.columns:
            display_df["inertia"] = display_df["inertia"].round(3)

        add(dataframe_to_markdown(display_df))
    else:
        add("_Không có dữ liệu Elbow Method._")

    add()

    add("### 6.2. Biểu đồ Elbow")
    add()

    elbow_files = figures.get("elbow", [])

    if elbow_files:
        for path in elbow_files:
            relative = safe_relative_path(path)
            add(f"![Elbow Method]({relative})")
            add()
    else:
        add("_Không tìm thấy biểu đồ Elbow._")
        add()

    add("### 6.3. Nhận xét")
    add()

    for observation in analyze_elbow_results(elbow_df):
        add(observation)

    add()
    add(
        "Elbow Method cung cấp thông tin về mức giảm Inertia khi "
        "tăng K. Tuy nhiên, điểm gấp khúc có thể không rõ ràng "
        "đối với một số ảnh. Do đó, cần kết hợp quan sát trực quan "
        "để lựa chọn số cụm phù hợp."
    )
    add()

    # --------------------------------------------------------
    # 7. Segmentation
    # --------------------------------------------------------

    add("## 7. KẾT QUẢ IMAGE SEGMENTATION")
    add()

    if image_results:
        for index, (image_name, result) in enumerate(
            image_results.items(),
            start=1,
        ):
            add(f"### 7.{index}. Ảnh: {image_name}")
            add()

            comparison_file = result.get("comparison_file")

            if comparison_file and Path(comparison_file).exists():
                relative = safe_relative_path(comparison_file)

                add("#### So sánh ảnh gốc và ảnh phân đoạn")
                add()
                add(f"![Comparison - {image_name}]({relative})")
                add()

            segmented_files = result.get("segmented_files", {})

            add("#### Kết quả theo từng K")
            add()

            for k, file_path in sorted(
                segmented_files.items(),
                key=lambda item: int(item[0]),
            ):
                if not Path(file_path).exists():
                    continue

                relative = safe_relative_path(file_path)

                add(f"**K = {k}**")
                add()
                add(f"![{image_name} K={k}]({relative})")
                add()

            add("#### Nhận xét")
            add()
            add(
                "K nhỏ thường làm ảnh có ít màu đại diện hơn, "
                "giúp đơn giản hóa màu sắc nhưng có thể làm mất "
                "các chi tiết màu nhỏ."
            )
            add()
            add(
                "K lớn giữ được nhiều sắc thái màu hơn. Tuy nhiên, "
                "việc tăng K không đồng nghĩa với việc ranh giới "
                "đối tượng trong ảnh được nhận diện chính xác hơn."
            )
            add()

    else:
        add("_Không có kết quả phân đoạn._")
        add()

    # --------------------------------------------------------
    # 8. Color palette
    # --------------------------------------------------------

    add("## 8. PHÂN TÍCH BẢNG MÀU")
    add()

    add("### 8.1. Bảng màu RGB")
    add()

    if not palette_df.empty:
        display_df = palette_df.copy()

        if "proportion" in display_df.columns:
            display_df["proportion_percent"] = (
                display_df["proportion"] * 100
            ).round(2)

            display_df = display_df.drop(
                columns=["proportion"]
            )

        numeric_columns = display_df.select_dtypes(
            include=["number"]
        ).columns

        display_df[numeric_columns] = (
            display_df[numeric_columns].round(2)
        )

        add(dataframe_to_markdown(display_df))
    else:
        add("_Không có dữ liệu bảng màu._")

    add()

    add("### 8.2. Biểu đồ tỷ lệ màu")
    add()

    palette_files = figures.get("palette", {}).get("files", [])

    if palette_files:
        for path in palette_files:
            relative = safe_relative_path(path)
            add(f"![Color palette]({relative})")
            add()
    else:
        add("_Không tìm thấy biểu đồ bảng màu._")
        add()

    add("### 8.3. Nhận xét về phân bố màu")
    add()

    for observation in analyze_palette_results(palette_df):
        add(observation)

    add()

    # --------------------------------------------------------
    # 9. Evaluation
    # --------------------------------------------------------

    add("## 9. ĐÁNH GIÁ KẾT QUẢ")
    add()

    add("### 9.1. Khả năng giảm số lượng màu")
    add()
    add(
        "K-Means thay thế các màu RGB ban đầu bằng một tập hợp "
        "màu đại diện có số lượng giới hạn. Khi K nhỏ, mức độ "
        "lượng tử hóa màu mạnh hơn. Khi K lớn, ảnh giữ lại nhiều "
        "sắc thái màu hơn."
    )
    add()

    add("### 9.2. Khả năng bảo toàn chi tiết")
    add()
    add(
        "Các vùng có màu sắc tương đồng có thể được biểu diễn "
        "bằng cùng một tâm cụm. Điều này có thể làm mất những "
        "khác biệt màu sắc nhỏ giữa các vùng ảnh."
    )
    add()

    add("### 9.3. Đánh giá bằng Inertia")
    add()
    add(
        "Inertia phản ánh mức độ gần nhau của pixel và tâm cụm "
        "trong không gian RGB. Inertia thấp hơn cho thấy các pixel "
        "gần tâm cụm hơn trên tập dữ liệu được đánh giá."
    )
    add()
    add(
        "Do Inertia thường giảm khi tăng K, chỉ số này cần được "
        "xem xét cùng số lượng cụm và kết quả trực quan."
    )
    add()

    add("### 9.4. Đánh giá trực quan")
    add()
    add(
        "Ảnh so sánh và biểu đồ bảng màu giúp đánh giá mức độ "
        "thay đổi màu sắc, sự xuất hiện của các cụm lớn và nhỏ, "
        "cũng như mức độ giữ lại chi tiết khi thay đổi K."
    )
    add()

    # --------------------------------------------------------
    # 10. Limitations
    # --------------------------------------------------------

    add("## 10. HẠN CHẾ")
    add()

    add("### 10.1. Chỉ sử dụng thông tin màu RGB")
    add()
    add(
        "Thuật toán không sử dụng tọa độ pixel, kết cấu hoặc "
        "đặc trưng hình dạng. Vì vậy, các vùng ảnh không liên "
        "quan nhưng có màu gần nhau có thể bị gom vào cùng cụm."
    )
    add()

    add("### 10.2. Không hiểu ngữ nghĩa đối tượng")
    add()
    add(
        "K-Means không biết đâu là người, cây, bầu trời, tòa nhà "
        "hay vật thể. Kết quả là phân cụm màu chứ không phải "
        "phân đoạn ngữ nghĩa."
    )
    add()

    add("### 10.3. Phụ thuộc vào số cụm K")
    add()
    add(
        "K cần được lựa chọn trước. Một giá trị K không phù hợp "
        "có thể làm ảnh quá đơn giản hoặc vẫn giữ quá nhiều màu."
    )
    add()

    add("### 10.4. Ảnh hưởng của lấy mẫu")
    add()
    add(
        "Mô hình được huấn luyện trên tập pixel lấy mẫu. Nếu "
        "mẫu không đại diện tốt cho phân bố màu của ảnh, tâm "
        "cụm có thể không phản ánh đầy đủ màu sắc trong ảnh."
    )
    add()

    # --------------------------------------------------------
    # 11. Future improvements
    # --------------------------------------------------------

    add("## 11. HƯỚNG PHÁT TRIỂN")
    add()
    add(
        "- Thử nghiệm không gian màu Lab để biểu diễn khác biệt "
        "màu sắc gần với cảm nhận thị giác hơn."
    )
    add(
        "- Kết hợp tọa độ không gian của pixel với đặc trưng màu."
    )
    add(
        "- Thử nghiệm các thuật toán phân đoạn khác như Mean Shift, "
        "DBSCAN hoặc Gaussian Mixture Model."
    )
    add(
        "- Sử dụng phương pháp đánh giá bổ sung và so sánh trực quan "
        "giữa các cấu hình."
    )
    add(
        "- Tối ưu hóa việc lấy mẫu và xử lý batch cho ảnh độ phân giải lớn."
    )
    add()

    # --------------------------------------------------------
    # 12. Conclusion
    # --------------------------------------------------------

    add("## 12. KẾT LUẬN")
    add()
    add(
        "Pipeline đã xây dựng quy trình phân đoạn ảnh bằng K-Means "
        "từ bước đọc ảnh, chuyển đổi dữ liệu pixel, lấy mẫu, "
        "huấn luyện, dự đoán đến tái tạo và trực quan hóa kết quả."
    )
    add()
    add(
        "Elbow Method hỗ trợ khảo sát mối quan hệ giữa số cụm K "
        "và Inertia. Bảng màu giúp giải thích màu đại diện và "
        "tỷ lệ pixel của từng cụm. Ảnh so sánh giúp quan sát "
        "mức độ thay đổi khi điều chỉnh K."
    )
    add()
    add(
        "K-Means phù hợp với mục tiêu lượng tử hóa màu và phân "
        "cụm pixel theo màu sắc. Tuy nhiên, phương pháp không "
        "thể thay thế các mô hình phân đoạn có khả năng nhận "
        "diện cấu trúc và ngữ nghĩa đối tượng."
    )
    add()

    # --------------------------------------------------------
    # 13. Output files
    # --------------------------------------------------------

    add("## 13. CẤU TRÚC KẾT QUẢ")
    add()
    add("### 13.1. Hình ảnh và dữ liệu")
    add()
    add(
        f"`reports/figures/additional_image_segmentation/"
        f"{get_run_suffix(run_id)}/`"
    )
    add()
    add("### 13.2. Báo cáo")
    add()
    add("- `image_segmentation_report.md`")
    add("- `image_segmentation_report.html`")
    add("- `image_segmentation_report.pdf`")
    add()

    add("---")
    add()
    add(
        f"_Báo cáo được tạo tự động bởi Additional Image Segmentation "
        f"Pipeline. Run_ID: `{run_id}`._"
    )

    return "\n".join(lines)


# ============================================================
# 6. MARKDOWN TO HTML
# ============================================================

def inline_markdown(text, report_dir):
    """Convert inline Markdown and embed local images as Base64."""
    text = html.escape(str(text))

    image_pattern = re.compile(
        r"!\[([^\]]*)\]\(([^)]+)\)"
    )

    def replace_image(match):
        alt = match.group(1)
        raw_path = match.group(2)

        image_path = (
            report_dir / raw_path
        ).resolve()

        try:
            image_path.relative_to(ROOT_DIR.resolve())
        except ValueError:
            return ""

        if not image_path.is_file():
            return (
                f'<p class="missing-image">'
                f'Không tìm thấy ảnh: {html.escape(alt)}'
                f'</p>'
            )

        mime_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
        }

        mime = mime_types.get(image_path.suffix.lower())

        if mime is None:
            return ""

        encoded = base64.b64encode(
            image_path.read_bytes()
        ).decode("ascii")

        return (
            f'<img src="data:{mime};base64,{encoded}" '
            f'alt="{html.escape(alt)}" '
            f'class="report-image">'
        )

    text = image_pattern.sub(replace_image, text)

    text = re.sub(
        r"\*\*(.+?)\*\*",
        r"<strong>\1</strong>",
        text,
    )

    text = re.sub(
        r"`(.+?)`",
        r"<code>\1</code>",
        text,
    )

    text = re.sub(
        r"(?<!\*)\*([^*]+)\*(?!\*)",
        r"<em>\1</em>",
        text,
    )

    return text


def markdown_to_html(markdown_text, report_dir):
    """Convert Markdown into HTML with tables, images and lists."""
    lines = markdown_text.splitlines()
    output = []

    in_code = False
    code_lines = []
    in_list = False
    in_table = False
    paragraph = []

    def close_paragraph():
        nonlocal paragraph

        if paragraph:
            output.append(
                "<p>"
                + inline_markdown(
                    " ".join(paragraph),
                    report_dir,
                )
                + "</p>"
            )
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
                code_lines = []
                in_code = True
            else:
                output.append(
                    html.escape("\n".join(code_lines))
                )
                output.append("</code></pre>")
                in_code = False

            continue

        if in_code:
            code_lines.append(line)
            continue

        if not stripped:
            close_paragraph()
            close_list()
            close_table()
            continue

        if stripped.startswith("|") and stripped.endswith("|"):
            close_paragraph()
            close_list()

            cells = [
                cell.strip()
                for cell in stripped.strip("|").split("|")
            ]

            if all(
                re.fullmatch(
                    r":?-{3,}:?",
                    cell.replace(" ", ""),
                )
                for cell in cells
            ):
                continue

            if not in_table:
                output.append(
                    '<div class="table-wrapper">'
                    '<table><thead><tr>'
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

        heading = re.match(
            r"^(#{1,6})\s+(.*)$",
            stripped,
        )

        if heading:
            close_paragraph()
            close_list()

            level = len(heading.group(1))
            content = inline_markdown(
                heading.group(2),
                report_dir,
            )

            output.append(
                f"<h{level}>{content}</h{level}>"
            )
            continue

        if stripped == "---":
            close_paragraph()
            close_list()
            output.append("<hr>")
            continue

        if stripped.startswith("- "):
            close_paragraph()

            if not in_list:
                output.append("<ul>")
                in_list = True

            output.append(
                "<li>"
                + inline_markdown(
                    stripped[2:],
                    report_dir,
                )
                + "</li>"
            )
            continue

        close_list()

        numbered = re.match(
            r"^\d+\.\s+(.*)$",
            stripped,
        )

        if numbered:
            paragraph.append(numbered.group(1))
            continue

        if stripped.startswith(">"):
            close_paragraph()
            output.append(
                "<blockquote>"
                + inline_markdown(
                    stripped.lstrip("> ").strip(),
                    report_dir,
                )
                + "</blockquote>"
            )
            continue

        paragraph.append(stripped)

    close_paragraph()
    close_list()
    close_table()

    if in_code:
        output.append(
            html.escape("\n".join(code_lines))
        )
        output.append("</code></pre>")

    return "\n".join(output)


def build_html(markdown_text, report_dir, run_id):
    """Build a standalone HTML document."""
    body = markdown_to_html(
        markdown_text,
        report_dir,
    )

    return f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">

<title>Image Segmentation Report</title>

<style>
* {{
    box-sizing: border-box;
}}

body {{
    margin: 0;
    padding: 35px 15px;
    background: #eef1f5;
    color: #263238;
    font-family: Arial, Helvetica, sans-serif;
    line-height: 1.75;
}}

.container {{
    max-width: 1150px;
    margin: auto;
    padding: 45px 55px;
    background: #ffffff;
    border-radius: 10px;
    box-shadow: 0 4px 22px rgba(0,0,0,0.08);
}}

h1 {{
    text-align: center;
    color: #17365d;
    font-size: 27px;
    line-height: 1.5;
    margin-top: 22px;
    margin-bottom: 25px;
}}

h2 {{
    color: #1f4e79;
    font-size: 21px;
    margin-top: 42px;
    padding-bottom: 10px;
    border-bottom: 2px solid #d9e4ef;
}}

h3 {{
    color: #2f75b5;
    font-size: 17px;
    margin-top: 28px;
}}

h4 {{
    color: #40566f;
    font-size: 15px;
    margin-top: 20px;
}}

p {{
    text-align: justify;
    margin: 10px 0;
}}

ul {{
    padding-left: 26px;
}}

li {{
    margin: 6px 0;
}}

.table-wrapper {{
    width: 100%;
    overflow-x: auto;
    margin: 18px 0 25px;
}}

table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
}}

th {{
    background: #1f4e79;
    color: white;
    font-weight: 600;
    text-align: left;
}}

td, th {{
    border: 1px solid #d5dee8;
    padding: 10px 12px;
    vertical-align: top;
}}

tr:nth-child(even) {{
    background: #f4f7fa;
}}

.report-image {{
    display: block;
    max-width: 100%;
    height: auto;
    margin: 22px auto;
    border: 1px solid #e1e6ed;
    border-radius: 7px;
}}

code {{
    background: #edf1f5;
    padding: 3px 6px;
    border-radius: 4px;
    color: #a93226;
}}

pre {{
    background: #17212f;
    color: #e7edf5;
    padding: 18px;
    border-radius: 7px;
    overflow-x: auto;
}}

blockquote {{
    margin: 15px 0;
    padding: 12px 20px;
    border-left: 4px solid #2f75b5;
    background: #f3f7fb;
}}

hr {{
    border: none;
    border-top: 1px solid #d7dee7;
    margin: 38px 0;
}}

.missing-image {{
    color: #a93226;
    font-style: italic;
}}

@media print {{
    body {{
        padding: 0;
        background: white;
    }}

    .container {{
        max-width: none;
        box-shadow: none;
        padding: 15px;
    }}

    img, table, pre {{
        page-break-inside: avoid;
    }}

    h2, h3, h4 {{
        page-break-after: avoid;
    }}
}}

@media (max-width: 700px) {{
    .container {{
        padding: 25px 18px;
    }}
}}
</style>
</head>

<body>
<div class="container">
{body}
</div>
</body>
</html>"""


# ============================================================
# 7. PDF REPORT
# ============================================================

def find_unicode_fonts():
    """Find fonts that support Vietnamese characters."""
    normal_candidates = [
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

    normal = next(
        (path for path in normal_candidates if path.exists()),
        None,
    )

    bold = next(
        (path for path in bold_candidates if path.exists()),
        None,
    )

    if normal is None:
        raise RuntimeError(
            "Không tìm thấy font Unicode để tạo PDF tiếng Việt. "
            "Hãy cài Arial hoặc DejaVu Sans."
        )

    return normal, bold or normal


def create_pdf(markdown_text, output_path, report_dir):
    """Render Markdown content into a formatted PDF."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import (
            getSampleStyleSheet,
            ParagraphStyle,
        )
        from reportlab.lib.units import mm
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.platypus import (
            SimpleDocTemplate,
            Paragraph,
            Spacer,
            Table,
            TableStyle,
            Image,
            PageBreak,
            Preformatted,
            KeepTogether,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Thiếu ReportLab. Cài bằng: pip install reportlab"
        ) from exc

    normal_path, bold_path = find_unicode_fonts()

    normal_font = "SegmentationReportFont"
    bold_font = "SegmentationReportFontBold"

    pdfmetrics.registerFont(
        TTFont(normal_font, str(normal_path))
    )

    pdfmetrics.registerFont(
        TTFont(bold_font, str(bold_path))
    )

    pdfmetrics.registerFontFamily(
        normal_font,
        normal=normal_font,
        bold=bold_font,
        italic=normal_font,
        boldItalic=bold_font,
    )

    styles = getSampleStyleSheet()

    styles.add(
        ParagraphStyle(
            name="ReportCover",
            fontName=bold_font,
            fontSize=19,
            leading=27,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#17365D"),
            spaceAfter=18,
        )
    )

    styles.add(
        ParagraphStyle(
            name="ReportH2",
            fontName=bold_font,
            fontSize=13,
            leading=18,
            textColor=colors.HexColor("#1F4E79"),
            spaceBefore=16,
            spaceAfter=8,
            keepWithNext=True,
        )
    )

    styles.add(
        ParagraphStyle(
            name="ReportH3",
            fontName=bold_font,
            fontSize=11,
            leading=15,
            textColor=colors.HexColor("#2F75B5"),
            spaceBefore=12,
            spaceAfter=6,
            keepWithNext=True,
        )
    )

    styles.add(
        ParagraphStyle(
            name="ReportH4",
            fontName=bold_font,
            fontSize=9.5,
            leading=13,
            textColor=colors.HexColor("#40566F"),
            spaceBefore=9,
            spaceAfter=5,
            keepWithNext=True,
        )
    )

    styles.add(
        ParagraphStyle(
            name="ReportBody",
            fontName=normal_font,
            fontSize=9,
            leading=14,
            alignment=TA_LEFT,
            spaceAfter=7,
            wordWrap="CJK",
        )
    )

    styles.add(
        ParagraphStyle(
            name="ReportTable",
            fontName=normal_font,
            fontSize=7,
            leading=9,
            wordWrap="CJK",
        )
    )

    styles.add(
        ParagraphStyle(
            name="ReportTableHeader",
            fontName=bold_font,
            fontSize=7,
            leading=9,
            textColor=colors.white,
            wordWrap="CJK",
        )
    )

    styles.add(
        ParagraphStyle(
            name="ReportCode",
            fontName=normal_font,
            fontSize=7,
            leading=10,
            backColor=colors.HexColor("#F0F2F5"),
            borderPadding=7,
        )
    )

    story = []

    def convert_inline(text):
        """Convert a limited set of Markdown inline syntax to ReportLab XML."""
        text = html.escape(str(text))

        text = re.sub(
            r"\*\*(.+?)\*\*",
            r"<b>\1</b>",
            text,
        )

        text = re.sub(
            r"`(.+?)`",
            r'<font color="#A93226">\1</font>',
            text,
        )

        text = re.sub(
            r"(?<!\*)\*([^*]+)\*(?!\*)",
            r"<i>\1</i>",
            text,
        )

        return text

    def add_table(table_lines):
        """Create a paginated PDF table."""
        rows = []

        for line in table_lines:
            cells = [
                cell.strip()
                for cell in line.strip().strip("|").split("|")
            ]

            if all(
                re.fullmatch(
                    r":?-{3,}:?",
                    cell.replace(" ", ""),
                )
                for cell in cells
            ):
                continue

            rows.append(cells)

        if not rows:
            return

        column_count = max(len(row) for row in rows)

        usable_width = A4[0] - 36 * mm
        col_width = usable_width / column_count

        table_data = []

        for row_index, row in enumerate(rows):
            row = row + [""] * (column_count - len(row))

            style = (
                styles["ReportTableHeader"]
                if row_index == 0
                else styles["ReportTable"]
            )

            table_data.append([
                Paragraph(
                    convert_inline(value),
                    style,
                )
                for value in row
            ])

        table = Table(
            table_data,
            colWidths=[col_width] * column_count,
            repeatRows=1,
            hAlign="LEFT",
        )

        table.setStyle(
            TableStyle([
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor("#1F4E79"),
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.4,
                    colors.HexColor("#C8D2DE"),
                ),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [
                        colors.white,
                        colors.HexColor("#F2F5F9"),
                    ],
                ),
            ])
        )

        story.append(table)
        story.append(Spacer(1, 9))

    def add_image(raw_path, alt=""):
        """Insert an image if it belongs to the project."""
        image_path = (
            report_dir / raw_path
        ).resolve()

        try:
            image_path.relative_to(ROOT_DIR.resolve())
        except ValueError:
            return

        if not image_path.is_file():
            story.append(
                Paragraph(
                    f"Không tìm thấy hình: {html.escape(alt)}",
                    styles["ReportBody"],
                )
            )
            return

        try:
            image = Image(str(image_path))
            image._restrictSize(
                165 * mm,
                105 * mm,
            )

            story.append(
                KeepTogether([
                    image,
                    Spacer(1, 8),
                ])
            )
        except Exception:
            story.append(
                Paragraph(
                    f"Không thể hiển thị hình: {html.escape(alt)}",
                    styles["ReportBody"],
                )
            )

    # --------------------------------------------------------
    # Parse Markdown
    # --------------------------------------------------------

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
                        styles["ReportCode"],
                    )
                )

                story.append(Spacer(1, 8))

            index += 1
            continue

        if in_code:
            code_lines.append(line)
            index += 1
            continue

        if stripped.startswith("|") and stripped.endswith("|"):
            table_lines.append(stripped)
            index += 1
            continue

        if table_lines:
            add_table(table_lines)
            table_lines = []

        if not stripped:
            index += 1
            continue

        heading = re.match(
            r"^(#{1,6})\s+(.*)$",
            stripped,
        )

        if heading:
            level = len(heading.group(1))
            title = heading.group(2)
            content = convert_inline(title)

            if level == 1:
                story.append(
                    Paragraph(
                        content,
                        styles["ReportCover"],
                    )
                )
            elif level == 2:
                story.append(
                    Paragraph(
                        content,
                        styles["ReportH2"],
                    )
                )
            elif level == 3:
                story.append(
                    Paragraph(
                        content,
                        styles["ReportH3"],
                    )
                )
            else:
                story.append(
                    Paragraph(
                        content,
                        styles["ReportH4"],
                    )
                )

            index += 1
            continue

        image_match = re.match(
            r"!\[([^\]]*)\]\(([^)]+)\)",
            stripped,
        )

        if image_match:
            add_image(
                image_match.group(2),
                image_match.group(1),
            )

            index += 1
            continue

        if stripped.startswith("- "):
            story.append(
                Paragraph(
                    "• " + convert_inline(stripped[2:]),
                    styles["ReportBody"],
                )
            )

            index += 1
            continue

        numbered = re.match(
            r"^\d+\.\s+(.*)$",
            stripped,
        )

        if numbered:
            story.append(
                Paragraph(
                    "• " + convert_inline(numbered.group(1)),
                    styles["ReportBody"],
                )
            )

            index += 1
            continue

        if stripped.startswith(">"):
            story.append(
                Paragraph(
                    convert_inline(stripped.lstrip("> ").strip()),
                    styles["ReportBody"],
                )
            )

            index += 1
            continue

        if stripped == "---":
            story.append(Spacer(1, 12))
            index += 1
            continue

        story.append(
            Paragraph(
                convert_inline(stripped),
                styles["ReportBody"],
            )
        )

        index += 1

    if table_lines:
        add_table(table_lines)

    if in_code:
        story.append(
            Preformatted(
                "\n".join(code_lines),
                styles["ReportCode"],
            )
        )

    # --------------------------------------------------------
    # Header and footer
    # --------------------------------------------------------

    def draw_page(canvas, document):
        canvas.saveState()

        page_width, page_height = A4

        canvas.setFont(normal_font, 7)
        canvas.setFillColor(
            colors.HexColor("#687789")
        )

        canvas.drawString(
            18 * mm,
            page_height - 10 * mm,
            "LAB03 - ADDITIONAL IMAGE SEGMENTATION",
        )

        canvas.setStrokeColor(
            colors.HexColor("#D5DEE8")
        )

        canvas.line(
            18 * mm,
            page_height - 13 * mm,
            page_width - 18 * mm,
            page_height - 13 * mm,
        )

        canvas.line(
            18 * mm,
            13 * mm,
            page_width - 18 * mm,
            13 * mm,
        )

        canvas.drawString(
            18 * mm,
            8 * mm,
            "K-Means Image Segmentation Report",
        )

        canvas.drawRightString(
            page_width - 18 * mm,
            8 * mm,
            f"Page {document.page}",
        )

        canvas.restoreState()

    document = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
        title="Additional Image Segmentation Report",
        author="Lab03 Machine Learning",
    )

    document.build(
        story,
        onFirstPage=draw_page,
        onLaterPages=draw_page,
    )


# ============================================================
# 8. MAIN REPORT GENERATOR
# ============================================================

def generate_report(
    run_id,
    elbow_df,
    palette_df,
    image_results,
    tested_k,
    status="SUCCESS",
):
    """
    Generate Markdown, HTML and PDF reports.

    All reports are saved directly inside this package directory.
    Existing reports are overwritten on each run.
    """
    print(
        f"\n[REPORT] Generating reports for Run_ID: {run_id}"
    )

    REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
    )

    data = prepare_report_data(
        run_id=run_id,
        elbow_df=elbow_df,
        palette_df=palette_df,
        image_results=image_results,
        tested_k=tested_k,
        status=status,
    )

    markdown_text = build_markdown_report(data)

    # Markdown
    REPORT_MD.write_text(
        markdown_text,
        encoding="utf-8",
    )

    print(f"[REPORT] Markdown: {REPORT_MD}")

    # HTML
    html_text = build_html(
    markdown_text,
    REPORT_DIR,
    run_id,
    )

    REPORT_HTML.write_text(
        html_text,
        encoding="utf-8",
    )

    print(f"[REPORT] HTML: {REPORT_HTML}")

    # PDF
    create_pdf(
    markdown_text,
    REPORT_PDF,
    REPORT_DIR,
    )

    print(f"[REPORT] PDF: {REPORT_PDF}")

    print("[REPORT] All reports generated successfully.")

    return {
        "markdown": str(REPORT_MD),
        "html": str(REPORT_HTML),
        "pdf": str(REPORT_PDF),
    }


# ============================================================
# 9. SCRIPT ENTRY POINT
# ============================================================


def load_saved_report_data(run_id):
    """Load saved experiment data without rerunning K-Means."""
    run_dir = get_run_figure_dir(run_id)

    elbow_csv = run_dir / "elbow_results.csv"
    palette_csv = run_dir / "palette_results.csv"

    if not elbow_csv.is_file():
        raise FileNotFoundError(f"Missing file: {elbow_csv}")

    if not palette_csv.is_file():
        raise FileNotFoundError(f"Missing file: {palette_csv}")

    elbow_df = pd.read_csv(elbow_csv)
    palette_df = pd.read_csv(palette_csv)

    if "K" in palette_df.columns:
        tested_k = sorted(
            int(k) for k in palette_df["K"].dropna().unique()
        )
    else:
        tested_k = []

    image_dir = ROOT_DIR / "data" / "additional"
    segmentation_dir = run_dir / "segmentation"

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    if not image_dir.is_dir():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")

    if not segmentation_dir.is_dir():
        raise FileNotFoundError(
            f"Missing segmentation directory: {segmentation_dir}"
        )

    saved_images = [
        p for p in segmentation_dir.iterdir()
        if p.is_file() and p.suffix.lower() in image_extensions
    ]

    image_results = {}

    for image_path in sorted(image_dir.iterdir()):
        if not image_path.is_file():
            continue
        if image_path.suffix.lower() not in image_extensions:
            continue

        image_stem = image_path.stem.lower()

        with Image.open(image_path) as img:
            width, height = img.size

        related_files = [
            p for p in saved_images
            if image_stem in p.stem.lower()
        ]

        comparison_candidates = [
            p for p in related_files
            if "comparison" in p.stem.lower()
        ]

        comparison_file = (
            comparison_candidates[0]
            if comparison_candidates else None
        )

        segmented_files = {}

        for k in tested_k:
            pattern = re.compile(
                rf"(?<![a-z0-9])k[_-]?{k}(?!\d)",
                re.IGNORECASE,
            )

            candidates = [
                p for p in related_files
                if p != comparison_file
                and pattern.search(p.stem)
            ]

            if candidates:
                segmented_files[k] = candidates[0]

        image_results[image_path.name] = {
            "height": height,
            "width": width,
            "comparison_file": comparison_file,
            "segmented_files": segmented_files,
        }

    return elbow_df, palette_df, image_results, tested_k


def main():
    """Generate reports from a previously completed run."""

    if len(sys.argv) > 1:
        run_id = sys.argv[1]
    else:
        if not FIGURE_ROOT.is_dir():
            raise FileNotFoundError(
                f"Không tìm thấy thư mục kết quả: {FIGURE_ROOT}"
            )

        run_dirs = [
            p for p in FIGURE_ROOT.iterdir()
            if p.is_dir()
            and (p / "elbow_results.csv").is_file()
            and (p / "palette_results.csv").is_file()
        ]

        if not run_dirs:
            raise FileNotFoundError(
                "Không tìm thấy Run_ID có đủ hai file CSV."
            )

        latest = max(
            run_dirs,
            key=lambda p: p.stat().st_mtime,
        )

        run_id = "additional_image_segmentation--" + latest.name

    print(f"\n[REPORT] Loading Run_ID: {run_id}")

    elbow_df, palette_df, image_results, tested_k = (
        load_saved_report_data(run_id)
    )

    generate_report(
        run_id=run_id,
        elbow_df=elbow_df,
        palette_df=palette_df,
        image_results=image_results,
        tested_k=tested_k,
        status="SUCCESS",
    )


if __name__ == "__main__":
    main()

