# LAB03 – CUSTOMER SEGMENTATION

## 1. Mục tiêu
Phân cụm khách hàng bằng thuật toán K-means tự cài đặt với NumPy. Ba đặc trưng cơ sở là `Age`, `Annual Income`, `Spending Score`; `Customer ID` không được dùng để tính khoảng cách. `Gender` chỉ dùng để mô tả cụm.

## 2. Cấu trúc dự án

```text
Lab03_ML/
├── agents/                 # Mô tả bài toán, quy tắc và hướng dẫn
├── app.py                  # Streamlit inference cho khách hàng mới
├── config.yaml             # Cấu hình dữ liệu, K và tham số K-means
├── data/
│   ├── raw/                # customer_segmentation.csv
│   └── processed/          # dữ liệu sạch, scaler, nhãn cụm, kết quả
├── docs/
│   ├── images/             # Biểu đồ EDA, evaluation, experiments
│   ├── experiment_logs.csv # Lịch sử thử nghiệm theo Run_ID
│   ├── results_summary.md  # Tóm tắt kết quả đánh giá
│   └── ...                 # Tài liệu các giai đoạn
├── logs/                   # pipeline_<Run_ID>.log
├── models/                 # best_kmeans.pkl
├── notebooks/              # Notebook thực nghiệm
├── progress_results/       # progress_<Run_ID>.json
├── reports/
│   ├── figures/<Run_ID>/{eda,training,evaluation,experiments,convergence}/ # Ảnh theo giai đoạn
│   ├── archive/            # Báo cáo cũ nếu cần lưu trữ
│   ├── inference_history.csv
│   ├── metrics_summary.csv
│   └── report_<Run_ID>.{md,html,pdf}
├── src/
│   ├── models/kmeans.py    # K-means tự cài đặt
│   ├── eda.py              # Khám phá dữ liệu
│   ├── preprocess.py       # Làm sạch và chuẩn hóa
│   ├── train.py            # Thử K=2..8, chọn mô hình
│   ├── evaluate.py         # Đánh giá và mô tả cụm
│   ├── experiments.py      # So sánh feature space và Grid Search
│   ├── demo_convergence.py # Lịch sử hội tụ
│   ├── predict.py          # Inference và lưu lịch sử
│   ├── tracking.py         # Run_ID, progress JSON, log
│   ├── visualize.py        # Gom hình theo Run_ID
│   ├── report.py           # Sinh MD, HTML, PDF
│   └── pipeline.py         # Điều phối toàn bộ quy trình
└── tests/                  # Unit tests cho K-means
```

## 3. Cài đặt

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 4. Chạy toàn bộ Pipeline

Đứng tại thư mục gốc `Lab03_ML`:

```powershell
python -m src.pipeline
```

Mỗi lần chạy tạo một `Run_ID` duy nhất. Pipeline cơ bản gồm EDA, Preprocessing, Training, Evaluation, Visualization và Report. Để chạy thêm feature-space experiments, Grid Search và convergence (tốn thời gian hơn), dùng:

```powershell
python -m src.pipeline --extended
```

## 5. Chạy riêng từng giai đoạn

```powershell
python -m src.eda
python -m src.preprocess
python -m src.train
python -m src.evaluate
python -m src.experiments
python -m src.demo_convergence
```

## 6. Tracking và báo cáo

- `progress_results/progress_<Run_ID>.json`: trạng thái, input, output, thời gian và lỗi của từng giai đoạn.
- `logs/pipeline_<Run_ID>.log`: nhật ký chạy theo thời gian.
- `docs/experiment_logs.csv`: lịch sử các lần thử K, có Run_ID và Timestamp.
- `reports/metrics_summary.csv`: tóm tắt metric theo Run_ID, không ghi đè lần chạy trước.
- `reports/figures/<Run_ID>/eda/`, `training/`, `evaluation/`, `experiments/`, `convergence/`: ảnh được gom theo từng giai đoạn và từng Run_ID.
- `reports/report_<Run_ID>.md`, `.html`, `.pdf`: ba định dạng báo cáo được tạo tự động.

Có thể tạo lại báo cáo cho một lần chạy đã có:

```powershell
python -m src.report --run-id <RUN_ID>
```

## 7. Inference – Dự đoán khách hàng mới

Sau khi đã có model và scaler:

```powershell
python -m src.predict
```

Hoặc chạy giao diện:

```powershell
python -m streamlit run app.py
```

Inference nhận Age, Annual Income, Spending Score; chuẩn hóa bằng đúng `scaler.json` của tập huấn luyện; gọi `model.predict()`; trả về Cluster và lưu lịch sử tại `reports/inference_history.csv`.

## 8. Kiểm thử

```powershell
python -m unittest discover -s tests -v
```

## 9. Lưu ý phương pháp
- K-means là học không giám sát, không có nhãn đúng/sai để tính Accuracy.
- Inertia càng thấp thường biểu thị các điểm gần tâm cụm hơn, nhưng giảm khi tăng K.
- Silhouette càng cao thường biểu thị cụm tách biệt tốt hơn; cần kết hợp hồ sơ cụm và kích thước cụm.
- Silhouette được tính trên mẫu tối đa 1.000 điểm để giảm thời gian.
- `Gender` không đưa vào mô hình cơ sở; các feature space mở rộng trong `experiments.py` chỉ phục vụ khảo sát.
- Cluster là nhãn định danh do thuật toán sinh ra, không phải mức độ tốt/xấu của khách hàng.
