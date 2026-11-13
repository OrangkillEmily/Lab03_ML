# LAB03 – CUSTOMER SEGMENTATION

## 1. Giới thiệu

Lab03 thực hiện bài toán **Customer Segmentation (Phân khúc khách hàng)** bằng thuật toán K-means tự cài đặt với NumPy.

Mục tiêu là phân nhóm khách hàng dựa trên các đặc trưng về độ tuổi, thu nhập và mức độ chi tiêu. Qua đó, có thể phân tích đặc điểm và hành vi của từng nhóm khách hàng.

### Đặc trưng sử dụng

| Đặc trưng | Vai trò |
|---|---|
| `Age` | Độ tuổi khách hàng |
| `Annual Income` | Thu nhập hằng năm |
| `Spending Score` | Điểm chi tiêu |
| `Customer ID` | Chỉ dùng để định danh, không đưa vào mô hình |
| `Gender` | Dùng để mô tả và phân tích đặc điểm cụm |

Mô hình cơ sở sử dụng ba đặc trưng `Age`, `Annual Income` và `Spending Score`. `Gender` không tham gia tính khoảng cách trong mô hình cơ sở.

## 2. Cấu trúc dự án

```text
Lab03_ML/
├── agents/                         # Mô tả bài toán, quy tắc và hướng dẫn
├── app.py                          # Streamlit inference cho khách hàng mới
├── config.yaml                     # Cấu hình dữ liệu và tham số K-means
├── data/
│   ├── raw/                        # Dữ liệu gốc
│   └── processed/                  # Dữ liệu đã xử lý và kết quả
├── docs/
│   ├── images/                     # Hình ảnh trực quan hóa
│   ├── experiment_logs.csv         # Lịch sử thực nghiệm theo Run_ID
│   └── results_summary.md          # Tóm tắt kết quả
├── logs/                           # Nhật ký thực thi Pipeline
├── models/                         # Mô hình K-means đã huấn luyện
├── notebooks/                      # Notebook thực nghiệm
├── progress_results/               # Kết quả tracking theo Run_ID
├── reports/
│   ├── figures/
│   │   ├── customer_segmentation/
│   │   │   └── <Run_ID>/
│   │   │       ├── 01_eda/
│   │   │       ├── 02_preprocessing/
│   │   │       ├── 03_training/
│   │   │       ├── 04_evaluation/
│   │   │       └── 05_extended/
│   │   │           ├── convergence/
│   │   │           └── experiments/
│   │   └── additional_image_segmentation/
│   │       └── <Run_ID>/
│   │           ├── 01_preprocessing/
│   │           ├── 02_elbow/
│   │           ├── 03_segmentation/
│   │           └── 04_palette/
│   ├── archive/                    # Báo cáo cũ
│   ├── inference_history.csv       # Lịch sử dự đoán
│   ├── metrics_summary.csv         # Tổng hợp chỉ số đánh giá
│   └── report_<Run_ID>.{md,html,pdf}
├── src/
│   ├── models/
│   │   └── kmeans.py               # K-means tự cài đặt bằng NumPy
│   ├── eda.py                      # Khám phá dữ liệu
│   ├── preprocess.py               # Làm sạch và chuẩn hóa
│   ├── train.py                    # Huấn luyện và lựa chọn K
│   ├── evaluate.py                 # Đánh giá và phân tích cụm
│   ├── experiments.py              # So sánh feature space và Grid Search
│   ├── demo_convergence.py         # Minh họa quá trình hội tụ
│   ├── predict.py                  # Dự đoán khách hàng mới
│   ├── tracking.py                 # Quản lý Run_ID và tracking
│   ├── figure_manager.py           # Thu thập và phân loại hình ảnh
│   ├── report.py                   # Sinh báo cáo MD, HTML, PDF
│   └── pipeline.py                 # Điều phối toàn bộ quy trình
├── tests/                          # Unit tests cho K-means
├── requirements.txt
└── README.md
```

## 3. Cài đặt môi trường

Yêu cầu Python và các thư viện được khai báo trong `requirements.txt`.

Mở PowerShell tại thư mục gốc `Lab03_ML` và thực hiện:

```powershell
# Tạo môi trường ảo
python -m venv .venv

# Kích hoạt môi trường ảo
.\.venv\Scripts\Activate.ps1

# Cài đặt thư viện
pip install -r requirements.txt
```

## 4. Chạy toàn bộ Pipeline

Đứng tại thư mục gốc của dự án.

### 4.1. Pipeline cơ bản

```powershell
python -m src.pipeline
```

Pipeline cơ bản thực hiện các giai đoạn:

1. **EDA:** Khám phá dữ liệu, phân tích phân phối và mối quan hệ giữa các đặc trưng.
2. **Preprocessing:** Kiểm tra dữ liệu, làm sạch và chuẩn hóa đặc trưng.
3. **Training:** Thử nghiệm các giá trị K và lựa chọn mô hình K-means.
4. **Evaluation:** Đánh giá chất lượng phân cụm và phân tích đặc điểm từng cụm.
5. **Visualization:** Thu thập, phân loại và lưu hình ảnh theo Run_ID.
6. **Report:** Tổng hợp kết quả thành báo cáo Markdown, HTML và PDF.

Mỗi lần chạy tạo một `Run_ID` riêng để phân biệt và theo dõi kết quả thực nghiệm.

### 4.2. Pipeline mở rộng

```powershell
python -m src.pipeline --extended
```

Ngoài các giai đoạn cơ bản, chế độ mở rộng thực hiện thêm:

- So sánh các không gian đặc trưng (Feature Space Comparison).
- Grid Search để khảo sát các cấu hình K-means.
- Minh họa lịch sử hội tụ của thuật toán (K-means Convergence).

Chế độ này phục vụ việc khảo sát sâu hơn ảnh hưởng của đặc trưng và tham số đến kết quả phân cụm.

## 5. Chạy riêng từng giai đoạn

Có thể chạy từng module độc lập từ thư mục gốc:

```powershell
# Khám phá dữ liệu
python -m src.eda

# Tiền xử lý
python -m src.preprocess

# Huấn luyện
python -m src.train

# Đánh giá
python -m src.evaluate

# Thực nghiệm mở rộng
python -m src.experiments

# Minh họa hội tụ
python -m src.demo_convergence
```

Lưu ý: Khi chạy từng module riêng lẻ, cần đảm bảo dữ liệu đầu vào và các kết quả của giai đoạn trước đã tồn tại. Việc chạy riêng không nhất thiết tạo ra đầy đủ tracking và báo cáo như chạy toàn bộ Pipeline.

## 6. Tracking và quản lý kết quả

Dự án sử dụng `Run_ID` để theo dõi từng lần thực nghiệm. Các kết quả được lưu riêng nhằm hỗ trợ kiểm tra, so sánh và tái lập quá trình chạy.

| Thành phần | Đường dẫn | Chức năng |
|---|---|---|
| Progress | `progress_results/progress_<Run_ID>.json` | Trạng thái, đầu vào, đầu ra, thời gian và lỗi của từng giai đoạn |
| Log | `logs/pipeline_<Run_ID>.log` | Nhật ký thực thi Pipeline theo thời gian |
| Experiment Logs | `docs/experiment_logs.csv` | Lịch sử thử nghiệm K và thông tin Run_ID |
| Metrics Summary | `reports/metrics_summary.csv` | Tổng hợp các chỉ số đánh giá theo Run_ID |
| Figures | `reports/figures/` | Hình ảnh được phân loại theo Pipeline và Run_ID |
| Reports | `reports/report_<Run_ID>` | Báo cáo của từng lần chạy |

### 6.1. Tổ chức hình ảnh

Hình ảnh được thu thập và phân loại tự động bằng `src/figure_manager.py`.

Đối với Customer Segmentation:

```text
customer_segmentation/
└── <Run_ID>/
    ├── 01_eda/
    ├── 02_preprocessing/
    ├── 03_training/
    ├── 04_evaluation/
    └── 05_extended/
        ├── convergence/
        └── experiments/
```

Ý nghĩa các thư mục:

| Thư mục | Nội dung |
|---|---|
| `01_eda` | Biểu đồ khám phá và phân tích dữ liệu |
| `02_preprocessing` | Hình ảnh trước và sau chuẩn hóa |
| `03_training` | Elbow Method, Silhouette theo K |
| `04_evaluation` | Biểu đồ kết quả và phân tích cụm |
| `05_extended/convergence` | Quá trình hội tụ của K-means |
| `05_extended/experiments` | So sánh các không gian đặc trưng và thực nghiệm mở rộng |

Việc phân loại theo từng Run_ID giúp hạn chế nhầm lẫn hình ảnh giữa các lần chạy.

### 6.2. Tạo lại báo cáo

Có thể tạo lại báo cáo cho một Run_ID đã tồn tại:

```powershell
python -m src.report --run-id <RUN_ID>
```

Thay `<RUN_ID>` bằng mã thực nghiệm cần tạo báo cáo.

Các định dạng báo cáo:

- `.md`: Báo cáo Markdown.
- `.html`: Báo cáo HTML.
- `.pdf`: Báo cáo PDF.

## 7. Inference – Dự đoán khách hàng mới

Sau khi hoàn tất huấn luyện, có thể sử dụng mô hình đã lưu để dự đoán nhóm khách hàng mới.

### 7.1. Chạy inference bằng dòng lệnh

```powershell
python -m src.predict
```

### 7.2. Chạy giao diện Streamlit

```powershell
python -m streamlit run app.py
```

### 7.3. Quy trình dự đoán

Inference nhận ba thông tin đầu vào:

- `Age`
- `Annual Income`
- `Spending Score`

Quy trình xử lý:

1. Nhận thông tin khách hàng mới.
2. Đọc scaler đã được lưu từ giai đoạn Preprocessing.
3. Chuẩn hóa dữ liệu bằng đúng tham số đã học từ tập huấn luyện.
4. Đưa dữ liệu đã chuẩn hóa vào `model.predict()`.
5. Xác định cụm khách hàng tương ứng.
6. Hiển thị kết quả và lưu lịch sử dự đoán.

Lịch sử inference được lưu tại:

```text
reports/inference_history.csv
```

**Lưu ý:** Không được fit lại scaler trên từng khách hàng mới. Phải sử dụng các tham số chuẩn hóa được lưu từ tập huấn luyện để đảm bảo dữ liệu inference nằm trong cùng không gian đặc trưng với dữ liệu huấn luyện.

## 8. Kiểm thử

Chạy unit tests bằng lệnh:

```powershell
python -m unittest discover -s tests -v
```

Các kiểm thử nhằm kiểm tra hoạt động của thuật toán K-means và những chức năng được triển khai trong bộ kiểm thử.

## 9. Phương pháp và lưu ý đánh giá

### 9.1. Thuật toán K-means

K-means là thuật toán học không giám sát (Unsupervised Learning). Thuật toán phân chia dữ liệu thành K cụm dựa trên khoảng cách giữa các điểm dữ liệu và tâm cụm (Centroid).

Quy trình cơ bản:

1. Khởi tạo K tâm cụm.
2. Tính khoảng cách từ từng điểm dữ liệu đến các tâm cụm.
3. Gán mỗi điểm vào cụm có tâm gần nhất.
4. Cập nhật tâm cụm bằng trung bình các điểm thuộc cụm.
5. Lặp lại quá trình gán cụm và cập nhật tâm cho đến khi hội tụ hoặc đạt số vòng lặp tối đa.

Trong dự án, K-means được tự cài đặt bằng NumPy thay vì chỉ gọi trực tiếp mô hình K-means có sẵn từ thư viện.

### 9.2. Các chỉ số đánh giá

| Chỉ số | Ý nghĩa | Hướng đánh giá |
|---|---|---|
| Inertia | Tổng bình phương khoảng cách từ điểm dữ liệu đến tâm cụm tương ứng | Thấp hơn thường tốt hơn |
| Silhouette Score | Đánh giá mức độ gắn kết trong cụm và tách biệt giữa các cụm | Cao hơn thường tốt hơn |
| Davies-Bouldin Index | Đánh giá độ tương đồng giữa các cụm | Thấp hơn thường tốt hơn |
| Calinski-Harabasz Index | So sánh độ phân tán giữa cụm và trong cụm | Cao hơn thường tốt hơn |

Không nên lựa chọn K chỉ dựa trên một chỉ số. Cần kết hợp các chỉ số đánh giá với kích thước cụm và đặc điểm thực tế của từng nhóm khách hàng.

### 9.3. Những điểm cần lưu ý

- K-means là thuật toán học không giám sát nên không có nhãn đúng/sai để tính Accuracy như bài toán phân loại.
- Inertia thường giảm khi tăng K, vì vậy không thể chỉ chọn K có Inertia thấp nhất.
- Silhouette Score càng cao thường cho thấy các cụm càng gắn kết và tách biệt tốt.
- Silhouette được tính trên mẫu tối đa 1.000 điểm nhằm giảm thời gian tính toán.
- `Customer ID` không được đưa vào mô hình vì chỉ có vai trò định danh.
- `Gender` chỉ được sử dụng để mô tả cụm trong mô hình cơ sở.
- Các feature space mở rộng có sử dụng `Gender` chỉ mang tính khảo sát, vì mã hóa nhị phân có thể tạo ra giả định khoảng cách nhân tạo.
- Nhãn Cluster chỉ là nhãn định danh do thuật toán tạo ra, không đại diện cho mức độ tốt hoặc xấu của khách hàng.

## 10. Kết luận

Lab03 xây dựng một quy trình phân cụm khách hàng tương đối hoàn chỉnh, từ khám phá dữ liệu, tiền xử lý, xây dựng thuật toán K-means, huấn luyện, đánh giá đến trực quan hóa và inference.

Bên cạnh mô hình cơ sở, dự án còn có các thực nghiệm mở rộng để khảo sát ảnh hưởng của không gian đặc trưng và quá trình hội tụ của thuật toán.

Việc tích hợp tracking theo `Run_ID`, quản lý hình ảnh và tự động tạo báo cáo giúp kết quả thực nghiệm được lưu trữ có hệ thống, thuận tiện cho việc kiểm tra, so sánh và tái lập.