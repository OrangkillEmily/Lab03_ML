# Pipeline Phase — Điều phối toàn bộ quy trình

## 1. Purpose — Mục đích

Pipeline Phase là giai đoạn điều phối toàn bộ quy trình Machine Learning trong dự án Customer Segmentation, từ khám phá dữ liệu, tiền xử lý, huấn luyện mô hình đến đánh giá và trực quan hóa kết quả.

Mục tiêu của Pipeline là kết nối các module độc lập thành một quy trình thực thi thống nhất, đảm bảo các giai đoạn được thực hiện theo đúng thứ tự và sử dụng kết quả đầu ra của giai đoạn trước làm đầu vào cho giai đoạn tiếp theo.

Pipeline không trực tiếp thực hiện thuật toán K-means mà đóng vai trò điều phối các module:

- EDA: Khám phá dữ liệu.
- Preprocessing: Làm sạch và chuẩn hóa dữ liệu.
- Training: Huấn luyện và lựa chọn mô hình K-means.
- Evaluation: Đánh giá kết quả phân cụm.
- Experiments: Thực nghiệm mở rộng (tùy chọn).
- Convergence: Theo dõi quá trình hội tụ (tùy chọn).
- Visualization: Thu thập và quản lý các biểu đồ kết quả.

Ngoài ra, Pipeline tích hợp cơ chế Tracking để theo dõi trạng thái thực thi, thời gian, dữ liệu đầu vào, đầu ra và lỗi của từng giai đoạn.

Mỗi lần chạy Pipeline được gán một `Run_ID` riêng biệt nhằm phân biệt và quản lý kết quả của từng lần thực nghiệm.

## 2. Input — Output

### Input

| Input | Description |
|---|---|
| `customer_segmentation.csv` | Bộ dữ liệu khách hàng ban đầu |
| `config.yaml` | File cấu hình toàn bộ tham số của dự án |
| `src/eda.py` | Module khám phá dữ liệu |
| `src/preprocess.py` | Module tiền xử lý dữ liệu |
| `src/train.py` | Module huấn luyện K-means |
| `src/evaluate.py` | Module đánh giá kết quả phân cụm |
| `src/experiments.py` | Module thực nghiệm mở rộng |
| `src/demo_convergence.py` | Module theo dõi hội tụ |
| `src/tracking.py` | Module quản lý Run_ID và tiến trình thực thi |

Pipeline sử dụng dữ liệu ban đầu từ:

`data/raw/customer_segmentation.csv`

Các tham số được đọc từ `config.yaml`, bao gồm:

- Danh sách đặc trưng sử dụng.
- Các giá trị K cần thử nghiệm.
- Số vòng lặp tối đa.
- Random State.
- Kích thước mẫu tính Silhouette Score.
- Các cấu hình thực nghiệm mở rộng.

### Output

| Output | Description |
|---|---|
| `Run_ID` | Mã định danh duy nhất cho mỗi lần chạy |
| `progress_<Run_ID>.json` | Lưu tiến trình, trạng thái, input, output và thời gian |
| `pipeline_<Run_ID>.log` | Nhật ký thực thi của toàn bộ Pipeline |
| `best_kmeans.pkl` | Mô hình K-means tốt nhất |
| `customer_clusters.csv` | Dữ liệu khách hàng kèm nhãn Cluster |
| `experiment_results.csv` | Kết quả thử nghiệm các giá trị K |
| `cluster_profile.csv` | Thống kê đặc điểm của từng cụm |
| `reports/figures/<Run_ID>/` | Hình ảnh được thu thập và phân loại theo từng giai đoạn |
| `report_<Run_ID>.md` | Báo cáo tổng hợp dạng Markdown |
| `report_<Run_ID>.html` | Báo cáo tổng hợp dạng HTML |
| `report_<Run_ID>.pdf` | Báo cáo tổng hợp dạng PDF |

Các file báo cáo MD, HTML và PDF được tạo riêng bằng `src/report.py` sau khi Pipeline hoàn thành.

## 3. How to do — Quy trình thực hiện

### Pipeline Workflow

**Basic Pipeline — Quy trình mặc định**

```text
             Raw Dataset
                  |
                  v
           Create Run_ID
                  |
                  v
              EDA Phase
                  |
                  v
         Preprocessing Phase
                  |
                  v
            Training Phase
                  |
                  v
           Evaluation Phase
                  |
                  v
          Visualization Phase
                  |
                  v
          Collect Run Figures
                  |
                  v
          Update Pipeline Status
                  |
                  v
           Finish Run SUCCESS
                  |
                  v
             Pipeline End
```

**Extended Pipeline — Thực nghiệm mở rộng**

Khi chạy với tham số `--extended`, hệ thống bổ sung hai giai đoạn:

```text
           Evaluation Phase
                  |
                  v
           Experiments Phase
                  |
                  v
          Convergence Phase
                  |
                  v
         Visualization Phase
                  |
                  v
             Pipeline End
```

### Các bước thực hiện

**Step 1 — Initialize Pipeline**

Khởi tạo quá trình thực thi trong `src/pipeline.py`.

Đầu tiên, hệ thống xác định thư mục gốc của dự án và import các module cần thiết:

```python
from src.tracking import (
    create_run,
    update_progress,
    write_log,
    finish_run
)

from src.figure_manager import collect_run_figures
```

Sau đó, hệ thống tạo một Run_ID mới:

```python
run_id = create_run(
    pipeline_name=PIPELINE_NAME
)
```

Run_ID có dạng:

```text
customer_segmentation--YYYYMMDD_HHMMSS_xxxxxx
```

Trong đó:

- `customer_segmentation`: Tên Pipeline.
- `YYYYMMDD_HHMMSS`: Thời điểm bắt đầu.
- `xxxxxx`: Chuỗi ký tự ngẫu nhiên giúp phân biệt các lần chạy.

Run_ID được sử dụng xuyên suốt các giai đoạn để liên kết thông tin thực thi và kết quả tracking.

**Step 2 — Execute EDA Phase**

Gọi module:

```python
eda.main()
```

EDA thực hiện khám phá dữ liệu ban đầu.

Các công việc chính:

- Kiểm tra kích thước dữ liệu.
- Kiểm tra dữ liệu thiếu và dữ liệu trùng lặp.
- Thống kê các đặc trưng.
- Phân tích phân phối Age, Annual Income và Spending Score.
- Trực quan hóa dữ liệu.

Kết quả được sử dụng để hiểu cấu trúc và đặc điểm của tập dữ liệu trước khi tiền xử lý.

**Step 3 — Execute Preprocessing Phase**

Gọi module:

```python
preprocess.main()
```

Preprocessing thực hiện:

- Đọc dữ liệu khách hàng.
- Làm sạch dữ liệu.
- Loại bỏ hoặc xử lý dữ liệu không hợp lệ theo quy tắc tiền xử lý.
- Lựa chọn ba đặc trưng sử dụng trong mô hình.
- Chuẩn hóa dữ liệu bằng Z-score.
- Lưu các thông số chuẩn hóa.

Các kết quả quan trọng:

- `customers_clean.csv`
- `X_scaled.csv`
- `scaler.json`
- `preprocessing_summary.json`

Trong đó, `X_scaled.csv` được sử dụng trực tiếp trong Training Phase.

**Step 4 — Execute Training Phase**

Gọi module:

```python
train.main(run_id=run_id)
```

Khác với các giai đoạn khác, Training nhận thêm Run_ID để ghi nhận kết quả thực nghiệm theo từng lần chạy.

Các công việc chính:

- Đọc dữ liệu đã chuẩn hóa.
- Đọc cấu hình từ `config.yaml`.
- Thử nghiệm các giá trị K.
- Huấn luyện từng mô hình K-means.
- Tính Inertia và Silhouette Score.
- Lựa chọn mô hình có Silhouette Score cao nhất.
- Lưu mô hình tốt nhất.

Kết quả quan trọng:

- `models/best_kmeans.pkl`
- `data/processed/experiment_results.csv`
- `docs/experiment_logs.csv`

Sau khi huấn luyện hoàn tất, Pipeline thu thập kết quả thông qua `collect_training_results()`.

Thông tin được ghi nhận bao gồm:

- Các giá trị K đã thử nghiệm.
- Silhouette Score.
- Giá trị K được lựa chọn.
- Silhouette Score tốt nhất.

**Step 5 — Execute Evaluation Phase**

Gọi module:

```python
evaluate.main()
```

Evaluation sử dụng mô hình tốt nhất cùng dữ liệu khách hàng để đánh giá chất lượng phân cụm.

Các công việc chính:

- Gán nhãn Cluster cho khách hàng.
- Thống kê số lượng khách hàng trong từng cụm.
- Xây dựng hồ sơ đặc điểm của từng cụm.
- Tính các chỉ số đánh giá phân cụm.
- Trực quan hóa kết quả.

Các kết quả quan trọng:

- `customer_clusters.csv`
- `cluster_profile.csv`
- `results_summary.md`
- Các biểu đồ đánh giá.

Sau đó, Pipeline gọi:

```python
collect_evaluation_results()
```

Để thu thập:

- Tổng số khách hàng.
- Số lượng cụm.
- Số khách hàng thuộc từng cụm.

**Step 6 — Execute Extended Experiments (Optional)**

Nếu người dùng chạy Pipeline với tham số `--extended`, hệ thống bổ sung các thực nghiệm mở rộng.

Có hai giai đoạn:

**6.1. Experiments Phase**

Gọi:

```python
experiments.main()
```

Thực hiện các thử nghiệm nhằm khảo sát ảnh hưởng của cách lựa chọn đặc trưng và cấu hình K-means đến chất lượng phân cụm.

Các nội dung chính:

- So sánh các Feature Spaces.
- Thử nghiệm các phương pháp khởi tạo tâm cụm.
- Thử nghiệm các giá trị Max Iterations.
- Grid Search các cấu hình.
- So sánh nhiều chỉ số đánh giá.

Kết quả:

- `feature_space_comparison.csv`
- `grid_search_results.csv`

**6.2. Convergence Phase**

Gọi:

```python
demo_convergence.main()
```

Mục đích là theo dõi sự thay đổi của các tâm cụm và Inertia trong quá trình thuật toán K-means hội tụ.

Kết quả:

- `convergence_history.csv`
- `kmeans_convergence.png`

Hai giai đoạn này chỉ được thực hiện khi bật chế độ Extended.

**Step 7 — Execute Visualization Phase**

Sau khi các giai đoạn thực thi hoàn tất, Pipeline thu thập các biểu đồ thông qua:

```python
figures = collect_run_figures(
    run_id,
    pipeline_name=PIPELINE_NAME
)
```

Module `src/figure_manager.py` chịu trách nhiệm quản lý và thu thập hình ảnh theo từng giai đoạn.

Các nhóm hình ảnh bao gồm:

- EDA.
- Preprocessing.
- Training.
- Evaluation.
- Experiments (nếu chạy Extended).
- Convergence (nếu chạy Extended).

Các hình ảnh được tổ chức theo Run_ID trong:

`reports/figures/<Run_ID>/`

Pipeline ghi nhận số lượng hình ảnh và thông tin thư mục đầu ra để phục vụ quá trình tổng hợp báo cáo.

**Step 8 — Tracking and Logging**

Trong suốt quá trình thực thi, Pipeline sử dụng `src/tracking.py` để theo dõi trạng thái của từng giai đoạn.

Có ba trạng thái chính:

| Status | Description |
|---|---|
| START | Giai đoạn bắt đầu thực hiện |
| SUCCESS | Giai đoạn hoàn thành thành công |
| FAILED | Giai đoạn xảy ra lỗi |

Hai hàm quan trọng trong Pipeline:

```python
log_stage_start(
    run_id,
    stage_name,
    input_desc
)
```

Ghi nhận thời điểm bắt đầu, tên giai đoạn và thông tin đầu vào.

```python
log_stage_success(
    run_id,
    stage_name,
    input_desc,
    output_data,
    elapsed
)
```

Ghi nhận kết quả đầu ra và thời gian thực hiện khi giai đoạn hoàn thành.

Nếu xảy ra lỗi, Pipeline gọi:

```python
log_stage_failed(
    run_id,
    stage_name,
    input_desc,
    error,
    elapsed
)
```

Thông tin được lưu tại hai vị trí:

**1. Progress Results**

`progress_results/customer_segmentation/progress_<Run_ID>.json`

Lưu thông tin có cấu trúc:

- Run_ID.
- Pipeline Name.
- Start Time.
- End Time.
- Duration.
- Status.
- Input Data.
- Output Data.
- Error.
- Danh sách các Steps.

**2. Logs**

`logs/customer_segmentation/pipeline_<Run_ID>.log`

Lưu nhật ký thực thi theo thời gian, bao gồm:

- Thời điểm bắt đầu.
- Tên giai đoạn.
- Trạng thái.
- Thông báo thực thi.
- Thông tin lỗi nếu có.

**Step 9 — Finish Pipeline and Generate Reports**

Sau khi tất cả các giai đoạn hoàn thành, Pipeline tổng hợp kết quả:

- Các giai đoạn đã hoàn thành.
- Tổng số giai đoạn.
- Tổng thời gian thực thi.
- Số lượng hình ảnh được thu thập.

Sau đó cập nhật trạng thái Pipeline:

```python
finish_run(run_id, "SUCCESS")
```

Nếu tất cả giai đoạn hoàn thành thành công, Run được đánh dấu `SUCCESS`.

Nếu xảy ra lỗi, Pipeline chuyển sang cơ chế xử lý ngoại lệ:

```python
except Exception as error:

    update_progress(
        run_id,
        "PIPELINE",
        "FAILED",
        error=str(error)
    )

    finish_run(run_id, "FAILED")

    raise
```

Nhờ đó, hệ thống ghi nhận lỗi, cập nhật trạng thái thất bại và kết thúc Run.

**Tạo báo cáo:**

Sau khi Pipeline hoàn thành, báo cáo được tạo riêng bằng `src/report.py`.

Chạy:

```bash
python -m src.report --run-id <RUN_ID>
```

Hệ thống tổng hợp thông tin thực nghiệm, kết quả đánh giá, tracking và hình ảnh để tạo:

- `reports/report_<Run_ID>.md`
- `reports/report_<Run_ID>.html`
- `reports/report_<Run_ID>.pdf`

Như vậy, Pipeline chịu trách nhiệm thực thi và thu thập kết quả, còn Report chịu trách nhiệm tổng hợp và xuất báo cáo.

## 4. Execution Management — Quản lý quá trình thực thi

Pipeline sử dụng Run_ID để quản lý từng lần chạy độc lập.

| Component | Responsibility |
|---|---|
| Run_ID | Định danh từng lần thực thi |
| `create_run()` | Tạo Run mới |
| `update_progress()` | Cập nhật trạng thái và kết quả từng giai đoạn |
| `write_log()` | Ghi nhật ký thực thi |
| `finish_run()` | Kết thúc Run với SUCCESS hoặc FAILED |
| `time.perf_counter()` | Đo thời gian thực thi |
| `try-except` | Bắt và ghi nhận lỗi |

**Cơ chế xử lý lỗi:**

Khi một giai đoạn xảy ra lỗi:

1. Ghi nhận trạng thái FAILED cho giai đoạn đó.
2. Lưu thông tin lỗi và thời gian thực thi.
3. Phát sinh ngoại lệ để dừng các giai đoạn tiếp theo.
4. Cập nhật trạng thái toàn bộ Pipeline thành FAILED.
5. Kết thúc Run và in thông tin lỗi ra màn hình.

Pipeline không tiếp tục chạy các giai đoạn phía sau khi một giai đoạn bắt buộc thất bại.

## 5. Used Data — Dữ liệu sử dụng

Pipeline sử dụng dữ liệu và kết quả được tạo xuyên suốt các giai đoạn.

| Stage | Input | Output |
|---|---|---|
| EDA | `customer_segmentation.csv` | Thống kê và biểu đồ EDA |
| Preprocessing | Dữ liệu gốc | `customers_clean.csv`, `X_scaled.csv`, `scaler.json` |
| Training | `X_scaled.csv`, `config.yaml` | `best_kmeans.pkl`, `experiment_results.csv` |
| Evaluation | Model, dữ liệu khách hàng, dữ liệu chuẩn hóa | `customer_clusters.csv`, `cluster_profile.csv` |
| Experiments | Dữ liệu khách hàng, cấu hình thực nghiệm | `feature_space_comparison.csv`, `grid_search_results.csv` |
| Convergence | Dữ liệu khách hàng, K-means | `convergence_history.csv` |
| Visualization | Các biểu đồ từ những giai đoạn trước | Hình ảnh được tổ chức theo Run_ID |

**Lưu ý:** Experiments và Convergence là hai giai đoạn tùy chọn, chỉ được thực hiện khi bật `--extended`.

Inference không nằm trong luồng Pipeline huấn luyện mặc định. Đây là quy trình riêng sử dụng mô hình đã lưu để phân cụm khách hàng mới.

## 6. How to Run — Cách thực thi Pipeline

Đứng tại thư mục gốc `Lab03_ML`.

**1. Chạy Pipeline mặc định**

```bash
python -m src.pipeline
```

Thực hiện các giai đoạn:

EDA → Preprocessing → Training → Evaluation → Visualization.

**2. Chạy Pipeline mở rộng**

```bash
python -m src.pipeline --extended
```

Bổ sung:

- Feature-space experiments.
- Grid Search.
- Convergence demonstration.

**3. Tạo báo cáo cho một Run_ID**

```bash
python -m src.report --run-id <RUN_ID>
```

Xuất báo cáo MD, HTML và PDF cho lần chạy tương ứng.

**4. Chạy Inference riêng**

```bash
python -m src.predict
```

Hoặc mở giao diện:

```bash
python -m streamlit run app.py
```

## Code References

| File | Responsibility |
|---|---|
| `src/pipeline.py` | Điều phối toàn bộ quy trình Lab03 |
| `src/pipeline.py` – `main()` | Khởi tạo Run, thực thi các giai đoạn và tổng hợp kết quả |
| `src/pipeline.py` – `log_stage_start()` | Ghi nhận trạng thái bắt đầu |
| `src/pipeline.py` – `log_stage_success()` | Ghi nhận trạng thái hoàn thành |
| `src/pipeline.py` – `log_stage_failed()` | Ghi nhận lỗi của từng giai đoạn |
| `src/pipeline.py` – `collect_training_results()` | Thu thập kết quả Training |
| `src/pipeline.py` – `collect_evaluation_results()` | Thu thập kết quả Evaluation |
| `src/tracking.py` | Quản lý Run_ID, tiến trình và nhật ký |
| `src/figure_manager.py` | Thu thập và tổ chức hình ảnh theo Run_ID |
| `src/eda.py` | Khám phá dữ liệu |
| `src/preprocess.py` | Tiền xử lý dữ liệu |
| `src/train.py` | Huấn luyện và lựa chọn K-means |
| `src/evaluate.py` | Đánh giá kết quả phân cụm |
| `src/experiments.py` | Thực nghiệm mở rộng |
| `src/demo_convergence.py` | Theo dõi quá trình hội tụ |
| `src/report.py` | Tổng hợp và xuất báo cáo MD, HTML, PDF |
| `src/predict.py` | Dự đoán cụm cho khách hàng mới |
| `config.yaml` | Cấu hình tham số và thực nghiệm |
| `progress_results/` | Lưu tiến trình từng Run |
| `logs/` | Lưu nhật ký thực thi |
| `reports/` | Lưu báo cáo và hình ảnh kết quả |