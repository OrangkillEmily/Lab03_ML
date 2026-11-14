# Tracking and Reports — Theo dõi và tổng hợp báo cáo

## 1. Purpose — Mục đích

Tracking and Reports là hệ thống theo dõi, quản lý và tổng hợp kết quả trong quá trình thực thi dự án Customer Segmentation.

Mục tiêu của hệ thống là ghi nhận toàn bộ thông tin của từng lần chạy Pipeline, bao gồm trạng thái thực thi, thời gian xử lý, dữ liệu đầu vào, kết quả đầu ra và các lỗi phát sinh.

Ngoài ra, hệ thống quản lý hình ảnh và tổng hợp kết quả thực nghiệm thành các báo cáo có cấu trúc để phục vụ quá trình phân tích, đánh giá và trình bày kết quả.

Hệ thống gồm bốn thành phần chính:

- **Tracking System:** Quản lý Run_ID và tiến trình thực thi.
- **Logging System:** Ghi nhận nhật ký hoạt động và lỗi.
- **Figure Management:** Thu thập và tổ chức hình ảnh theo từng giai đoạn.
- **Report Generation:** Tổng hợp dữ liệu và xuất báo cáo MD, HTML, PDF.

Ngoài ra, hệ thống còn lưu lịch sử Inference để theo dõi các lần dự đoán cụm khách hàng mới.

Tracking and Reports không trực tiếp huấn luyện mô hình K-means mà hỗ trợ quản lý và lưu trữ kết quả từ các giai đoạn khác nhau.

## 2. Input — Output

### Input

| Input | Description |
|---|---|
| `Run_ID` | Mã định danh của lần chạy Pipeline |
| `progress_results/` | Dữ liệu tiến trình được lưu dưới dạng JSON |
| `logs/` | Nhật ký thực thi của từng Pipeline |
| `experiment_results.csv` | Kết quả thử nghiệm các giá trị K |
| `customer_clusters.csv` | Dữ liệu khách hàng kèm nhãn Cluster |
| `cluster_profile.csv` | Thống kê đặc điểm của từng cụm |
| `docs/images/` | Các hình ảnh được tạo từ những giai đoạn khác nhau |
| `inference_history.csv` | Lịch sử phân cụm khách hàng mới |

### Output

| Output | Description |
|---|---|
| `progress_<Run_ID>.json` | Thông tin tiến trình và trạng thái thực thi |
| `pipeline_<Run_ID>.log` | Nhật ký hoạt động của Pipeline |
| `reports/figures/` | Hình ảnh được tổ chức theo Pipeline, Run_ID và Stage |
| `report_<Run_ID>.md` | Báo cáo tổng hợp dạng Markdown |
| `report_<Run_ID>.html` | Báo cáo tổng hợp dạng HTML |
| `report_<Run_ID>.pdf` | Báo cáo tổng hợp dạng PDF |
| `inference_history.csv` | Lịch sử dự đoán cụm khách hàng mới |

Các kết quả được lưu trữ theo từng lần chạy nhằm hỗ trợ kiểm tra, so sánh và tái sử dụng dữ liệu thực nghiệm.

## 3. Tracking System — Hệ thống theo dõi tiến trình

Tracking System được hiện thực trong `src/tracking.py`.

### 3.1. Run_ID — Mã định danh lần chạy

Mỗi lần thực thi Pipeline, hệ thống tạo một Run_ID duy nhất thông qua hàm:

```python
run_id = create_run(
    pipeline_name="customer_segmentation"
)
```

Run_ID có cấu trúc:

```text
customer_segmentation--YYYYMMDD_HHMMSS_xxxxxx
```

Trong đó:

- `customer_segmentation`: Tên Pipeline.
- `YYYYMMDD_HHMMSS`: Thời điểm bắt đầu chạy.
- `xxxxxx`: Sáu ký tự ngẫu nhiên được tạo từ UUID.

Ví dụ:

```text
customer_segmentation--20261111_194455_f6e9ea
```

Run_ID được sử dụng để liên kết thông tin của một lần thực thi, bao gồm:

- Trạng thái Pipeline.
- Thời gian bắt đầu và kết thúc.
- Kết quả của từng Stage.
- Nhật ký thực thi.
- Thông tin lỗi.
- Hình ảnh và báo cáo tương ứng.

Nhờ Run_ID, hệ thống có thể phân biệt các lần chạy khác nhau và tránh nhầm lẫn giữa các kết quả thực nghiệm.

### 3.2. Progress Tracking — Theo dõi tiến trình

Hệ thống sử dụng hàm `create_run()` để khởi tạo một lần chạy mới.

```python
def create_run(
    pipeline_name=DEFAULT_PIPELINE
):
    run_id = (
        pipeline_name
        + "--"
        + datetime.now().astimezone().strftime(
            "%Y%m%d_%H%M%S"
        )
        + "_"
        + uuid.uuid4().hex[:6]
    )
```

Sau khi tạo Run_ID, hệ thống khởi tạo dữ liệu Tracking:

```python
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
```

Thông tin được lưu vào file JSON trong thư mục:

`progress_results/customer_segmentation/`

Ví dụ:

`progress_results/customer_segmentation/progress_20261111_194455_f6e9ea.json`

**Các trường thông tin chính:**

| Field | Description |
|---|---|
| `run_id` | Mã định danh lần chạy |
| `project` | Tên dự án |
| `pipeline` | Tên Pipeline |
| `start_time` | Thời điểm bắt đầu |
| `end_time` | Thời điểm kết thúc |
| `duration_seconds` | Tổng thời gian thực thi |
| `total_execution_time_seconds` | Tổng thời gian chạy Pipeline |
| `status` | Trạng thái thực thi |
| `error` | Thông tin lỗi nếu có |
| `steps` | Danh sách các giai đoạn đã thực hiện |

### 3.3. Stage Tracking — Theo dõi từng giai đoạn

Trong Pipeline, mỗi giai đoạn được theo dõi bằng ba hàm chính:

**1. log_stage_start()**

Ghi nhận thời điểm bắt đầu một giai đoạn.

```python
log_stage_start(
    run_id,
    stage_name,
    input_desc
)
```

Thông tin được ghi nhận:

- Run_ID.
- Tên Stage.
- Trạng thái START.
- Mô tả dữ liệu đầu vào.

**2. log_stage_success()**

Ghi nhận kết quả khi một giai đoạn hoàn thành thành công.

```python
log_stage_success(
    run_id,
    stage_name,
    input_desc,
    output_data,
    elapsed
)
```

Thông tin được lưu:

- Trạng thái SUCCESS.
- Dữ liệu đầu vào.
- Dữ liệu đầu ra.
- Thời gian thực thi.

**3. log_stage_failed()**

Ghi nhận khi một giai đoạn xảy ra lỗi.

```python
log_stage_failed(
    run_id,
    stage_name,
    input_desc,
    error,
    elapsed
)
```

Thông tin được lưu:

- Trạng thái FAILED.
- Mô tả lỗi.
- Thời gian thực thi.
- Tên giai đoạn xảy ra lỗi.

**Các trạng thái sử dụng:**

| Status | Description |
|---|---|
| START | Giai đoạn bắt đầu thực hiện |
| RUNNING | Trạng thái Run đang được thực thi |
| SUCCESS | Giai đoạn hoặc Pipeline hoàn thành thành công |
| FAILED | Giai đoạn hoặc Pipeline xảy ra lỗi |

Lưu ý: START được sử dụng cho trạng thái bắt đầu của Stage, trong khi Run tổng thể được khởi tạo với trạng thái RUNNING.

### 3.4. Finish Run — Kết thúc lần chạy

Sau khi hoàn thành tất cả các giai đoạn, Pipeline gọi:

```python
finish_run(
    run_id,
    status="SUCCESS"
)
```

Hàm này thực hiện:

- Ghi nhận thời điểm kết thúc.
- Tính tổng thời gian thực thi.
- Cập nhật trạng thái cuối cùng.
- Hoàn thiện thông tin các Stage.
- Ghi nhận lỗi nếu Pipeline thất bại.

Nếu có Stage FAILED, hệ thống không cho phép kết thúc Run với trạng thái SUCCESS.

## 4. Logging System — Hệ thống ghi nhật ký

Logging System được hiện thực trong `src/tracking.py`.

Mục đích là ghi lại các sự kiện xảy ra trong quá trình thực thi Pipeline để hỗ trợ kiểm tra và phát hiện lỗi.

### 4.1. Write Log

Hàm `write_log()` chịu trách nhiệm ghi nhật ký:

```python
write_log(
    run_id,
    step_name,
    status,
    message
)
```

Mỗi dòng Log có cấu trúc:

```text
[Timestamp] [Status] [Stage] Message
```

Ví dụ:

```text
[2026-11-11T19:44:55+07:00]
[START]
[EDA]
Bắt đầu giai đoạn EDA.
```

Các thông tin được ghi nhận:

- Timestamp: Thời điểm xảy ra sự kiện.
- Status: Trạng thái thực thi.
- Stage: Tên giai đoạn.
- Message: Nội dung thông báo.

### 4.2. Log File Organization

Log được lưu riêng theo từng Pipeline:

```text
logs/
└── customer_segmentation/
    ├── pipeline_20261111_194455_f6e9ea.log
    ├── pipeline_20261111_194850_5d9753.log
    └── pipeline_20261111_195039_45c8f0.log
```

Mỗi file Log tương ứng với một Run_ID.

Hệ thống sử dụng chế độ append (`"a"`) để bổ sung nội dung mới vào file thay vì ghi đè nhật ký đã tồn tại.

### 4.3. Error Handling — Xử lý lỗi

Khi một Stage xảy ra lỗi, Pipeline sử dụng cơ chế `try-except` để bắt ngoại lệ.

Quy trình xử lý:

1. Bắt lỗi phát sinh trong Stage.
2. Ghi nhận trạng thái FAILED.
3. Lưu thông tin lỗi vào Progress JSON.
4. Ghi thông báo lỗi vào Log File.
5. Phát sinh lại ngoại lệ để dừng các Stage tiếp theo.
6. Kết thúc Run với trạng thái FAILED.

Cơ chế này giúp xác định chính xác giai đoạn xảy ra lỗi và hỗ trợ quá trình kiểm tra, sửa chữa chương trình.

## 5. Visualization Management — Quản lý hình ảnh

Chức năng quản lý hình ảnh được hiện thực trong `src/figure_manager.py`.

Mục đích là thu thập các hình ảnh được tạo từ những giai đoạn khác nhau và tổ chức chúng theo từng Run_ID để phục vụ báo cáo.

### 5.1. Collect Figures

Hàm chính:

```python
figures = collect_run_figures(
    run_id,
    pipeline_name="customer_segmentation"
)
```

Hàm thực hiện:

- Xác định thư mục hình ảnh nguồn.
- Tìm kiếm các file hình ảnh.
- Nhận diện Stage dựa trên tên file.
- Phân loại hình ảnh.
- Sao chép hình ảnh vào thư mục tương ứng.
- Tránh tạo bản sao trùng lặp.

Các định dạng được hỗ trợ:

- PNG.
- JPG.
- JPEG.
- SVG.
- WEBP.

### 5.2. Detect Stage

Hệ thống sử dụng hàm `detect_stage()` để xác định hình ảnh thuộc giai đoạn nào dựa trên tên file.

Ví dụ:

| Image Name | Stage |
|---|---|
| `eda_scatterplots.png` | EDA |
| `preprocessing_summary.png` | Preprocessing |
| `silhouette_by_k.png` | Training |
| `cluster_sizes.png` | Training |
| `feature_space_comparison.png` | Experiments |
| `kmeans_convergence.png` | Convergence |

Nếu không xác định được giai đoạn, hình ảnh được đưa vào nhóm `other`.

### 5.3. Figure Directory

Hình ảnh được tổ chức theo cấu trúc:

```text
reports/
└── figures/
    └── customer_segmentation/
        └── <Run_Suffix>/
            ├── eda/
            ├── preprocess/
            ├── train/
            ├── evaluate/
            ├── experiments/
            ├── demo_convergence/
            └── other/
```

Trong đó:

- `customer_segmentation`: Tên Pipeline.
- `Run_Suffix`: Phần timestamp và mã định danh của Run_ID.
- Các thư mục con: Những giai đoạn tạo ra hình ảnh.

Kết quả thu thập trả về:

```python
{
    "pipeline": pipeline_name,
    "run_id": run_id,
    "total": total_images,
    "stages": copied_by_stage,
    "output_dir": str(output_dir)
}
```

Nhờ đó, Pipeline có thể ghi nhận tổng số hình ảnh và số lượng hình ảnh theo từng giai đoạn.

## 6. Report Generation — Tạo báo cáo

Chức năng tạo báo cáo được hiện thực trong `src/report.py`.

Mục đích là tổng hợp kết quả thực nghiệm, thông tin Tracking, Log và hình ảnh để tạo báo cáo hoàn chỉnh cho từng Run_ID.

### 6.1. Collect Report Data

Trước khi tạo báo cáo, hệ thống thu thập dữ liệu từ nhiều nguồn:

| Source | Information |
|---|---|
| Progress JSON | Run_ID, trạng thái, thời gian và các Stage |
| Pipeline Log | Nhật ký thực thi |
| `customers_clean.csv` | Dữ liệu khách hàng sau làm sạch |
| `X_scaled.csv` | Dữ liệu đã chuẩn hóa |
| `experiment_results.csv` | Kết quả thực nghiệm K-means |
| `customer_clusters.csv` | Kết quả phân cụm khách hàng |
| `cluster_profile.csv` | Đặc điểm của từng cụm |
| Figures Directory | Các biểu đồ theo từng giai đoạn |

Hàm `build_report_data()` chịu trách nhiệm đọc và tổng hợp các dữ liệu trên thành một cấu trúc thống nhất.

### 6.2. Build Report Content

Sau khi thu thập dữ liệu, hệ thống sử dụng:

```python
markdown_text = build_markdown(data)
```

Hàm `build_markdown()` tạo nội dung báo cáo dựa trên thông tin thực nghiệm và Tracking.

Nội dung báo cáo bao gồm:

- Thông tin chung của lần chạy.
- Run_ID.
- Trạng thái thực thi.
- Thời gian chạy.
- Kết quả từng giai đoạn.
- Kết quả huấn luyện K-means.
- Chỉ số đánh giá phân cụm.
- Thống kê các cụm khách hàng.
- Các biểu đồ kết quả.
- Nhật ký thực thi.

### 6.3. Generate Markdown Report

Báo cáo Markdown được tạo bằng:

```python
md_path.write_text(
    markdown_text,
    encoding="utf-8"
)
```

File đầu ra:

`reports/customer_segmentation/report_<Run_ID>.md`

Markdown phù hợp để đọc, chỉnh sửa và quản lý nội dung báo cáo.

### 6.4. Generate HTML Report

Hệ thống chuyển đổi nội dung Markdown thành HTML:

```python
html_text = build_html(
    markdown_text,
    run_id
)
```

Sau đó ghi ra file:

`reports/customer_segmentation/report_<Run_ID>.html`

HTML giúp hiển thị báo cáo với bố cục trực quan, bảng biểu và hình ảnh.

### 6.5. Generate PDF Report

Hệ thống sử dụng thư viện ReportLab để tạo báo cáo PDF:

```python
create_pdf(
    markdown_text,
    pdf_path,
    REPORT_DIR
)
```

File đầu ra:

`reports/customer_segmentation/report_<Run_ID>.pdf`

PDF được sử dụng để lưu trữ, chia sẻ và trình bày báo cáo hoàn chỉnh.

### 6.6. Run Report Generator

Có thể tạo báo cáo riêng sau khi Pipeline hoàn thành:

```bash
python -m src.report --run-id <RUN_ID>
```

Nếu không truyền Run_ID, chương trình sẽ tìm lần chạy mới nhất theo file Progress mà nó tìm thấy.

Hàm chính:

```python
generate_report(run_id)
```

Hàm này thực hiện lần lượt:

1. Đọc dữ liệu Tracking.
2. Thu thập kết quả thực nghiệm.
3. Đọc Log.
4. Thu thập hình ảnh.
5. Xây dựng nội dung Markdown.
6. Xuất HTML.
7. Xuất PDF.

**Lưu ý:** Report Generation là chức năng riêng, không phải một giai đoạn huấn luyện K-means. Pipeline và Report Generator có trách nhiệm khác nhau.

## 7. Inference History — Lịch sử dự đoán

Ngoài việc theo dõi Pipeline, hệ thống còn lưu lịch sử dự đoán khách hàng mới trong `src/predict.py`.

Hàm thực hiện:

```python
save_inference_history(result)
```

Mỗi lần dự đoán, hệ thống ghi thêm một dòng vào:

`reports/customer_segmentation/inference_history.csv`

Các thông tin được lưu:

| Field | Description |
|---|---|
| `timestamp` | Thời điểm thực hiện dự đoán |
| `Age` | Độ tuổi khách hàng |
| `Annual Income` | Thu nhập hằng năm |
| `Spending Score` | Điểm chi tiêu |
| `Cluster` | Cụm được dự đoán |
| `prediction_time_seconds` | Thời gian xử lý dự đoán |

Hệ thống sử dụng CSV Writer ở chế độ append để bổ sung kết quả mới mà không ghi đè lịch sử cũ.

Inference History giúp theo dõi các lần sử dụng mô hình sau khi huấn luyện.

**Lưu ý:** Inference History là lịch sử dự đoán độc lập, không phải lịch sử thực thi Pipeline và không thay thế Progress JSON.

## 8. Output Directory Structure — Cấu trúc thư mục đầu ra

Các kết quả Tracking, Logging, Visualization và Reports được tổ chức như sau:

```text
Lab03_ML/
│
├── progress_results/
│   └── customer_segmentation/
│       ├── progress_<Run_Suffix>.json
│       └── ...
│
├── logs/
│   └── customer_segmentation/
│       ├── pipeline_<Run_Suffix>.log
│       └── ...
│
├── reports/
│   │
│   ├── customer_segmentation/
│   │   ├── report_<Run_ID>.md
│   │   ├── report_<Run_ID>.html
│   │   ├── report_<Run_ID>.pdf
│   │   ├── metrics_summary.csv
│   │   └── inference_history.csv
│   │
│   └── figures/
│       └── customer_segmentation/
│           └── <Run_Suffix>/
│               ├── eda/
│               ├── preprocess/
│               ├── train/
│               ├── evaluate/
│               ├── experiments/
│               └── demo_convergence/
│
├── docs/
│   └── experiment_logs.csv
│
└── src/
    ├── tracking.py
    ├── figure_manager.py
    ├── report.py
    └── predict.py
```

**Ý nghĩa các thư mục:**

| Directory | Responsibility |
|---|---|
| `progress_results/` | Lưu trạng thái và tiến trình từng Run |
| `logs/` | Lưu nhật ký thực thi |
| `reports/figures/` | Lưu hình ảnh theo Run_ID và Stage |
| `reports/customer_segmentation/` | Lưu báo cáo và lịch sử inference |
| `docs/` | Lưu tài liệu và lịch sử thực nghiệm |

## Code References

| File | Responsibility |
|---|---|
| `src/tracking.py` | Quản lý Run_ID, Progress JSON và Logging |
| `src/tracking.py` – `create_run()` | Tạo một lần chạy mới |
| `src/tracking.py` – `update_progress()` | Cập nhật trạng thái và thông tin từng Stage |
| `src/tracking.py` – `write_log()` | Ghi nhật ký thực thi |
| `src/tracking.py` – `finish_run()` | Kết thúc Run và cập nhật trạng thái cuối |
| `src/pipeline.py` | Tích hợp Tracking vào toàn bộ quy trình |
| `src/figure_manager.py` | Thu thập và tổ chức hình ảnh |
| `src/figure_manager.py` – `detect_stage()` | Phân loại hình ảnh theo Stage |
| `src/figure_manager.py` – `collect_run_figures()` | Thu thập hình ảnh theo Run_ID |
| `src/report.py` | Tổng hợp và xuất báo cáo |
| `src/report.py` – `build_report_data()` | Thu thập dữ liệu từ Tracking, Log và các kết quả |
| `src/report.py` – `build_markdown()` | Xây dựng nội dung báo cáo |
| `src/report.py` – `build_html()` | Chuyển nội dung thành HTML |
| `src/report.py` – `create_pdf()` | Xuất báo cáo PDF |
| `src/report.py` – `generate_report()` | Điều phối quá trình tạo báo cáo |
| `src/predict.py` – `save_inference_history()` | Lưu lịch sử dự đoán khách hàng mới |
| `docs/experiment_logs.csv` | Lưu lịch sử thực nghiệm |
| `reports/customer_segmentation/` | Lưu báo cáo và lịch sử inference |
| `progress_results/` | Lưu kết quả Tracking |
| `logs/` | Lưu nhật ký Pipeline |
| `reports/figures/` | Lưu hình ảnh phục vụ báo cáo |