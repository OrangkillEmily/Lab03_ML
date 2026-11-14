# Inference Phase — Dự đoán cụm khách hàng mới

## 1. Purpose — Mục đích

Inference Phase là giai đoạn sử dụng mô hình K-means đã được huấn luyện trong Training Phase để phân cụm những khách hàng mới chưa xuất hiện trong tập dữ liệu huấn luyện.

Mục tiêu là xác định khách hàng mới thuộc cụm (Cluster) nào dựa trên các đặc trưng:

- **Age:** Độ tuổi khách hàng.
- **Annual Income:** Thu nhập hằng năm.
- **Spending Score:** Điểm chi tiêu của khách hàng.

Khác với Training Phase, Inference Phase không thực hiện quá trình huấn luyện lại mô hình hoặc tìm kiếm các tâm cụm mới.

Hệ thống sử dụng mô hình K-means đã lưu cùng với các thông số chuẩn hóa từ Preprocessing Phase để chuyển đổi dữ liệu đầu vào và dự đoán cụm tương ứng.

**Lưu ý:** Cluster chỉ là nhãn nhóm do thuật toán tạo ra, không thể hiện thứ hạng tốt hay xấu của khách hàng.

## 2. Input — Output

### Input

| Input | Description |
|---|---|
| Age | Độ tuổi khách hàng mới |
| Annual Income | Thu nhập hằng năm của khách hàng |
| Spending Score | Điểm chi tiêu của khách hàng (0–100) |
| `best_kmeans.pkl` | Mô hình K-means đã được huấn luyện và lưu từ Training Phase |
| `scaler.json` | Các thông số chuẩn hóa Z-score được lưu từ Preprocessing Phase |

### Output

| Output | Description |
|---|---|
| Predicted Cluster | Cụm mà khách hàng mới được phân vào |
| Scaled Features | Các đặc trưng sau khi chuẩn hóa Z-score |
| Prediction Time | Thời gian thực hiện dự đoán |
| Timestamp | Thời điểm thực hiện inference |
| Inference History | Lịch sử các lần dự đoán |

Kết quả được hiển thị cho người dùng thông qua chương trình hoặc giao diện Streamlit và được lưu vào:

`reports/customer_segmentation/inference_history.csv`

## 3. How to do — Quy trình thực hiện

### Inference Workflow

```text
          New Customer Data
                    |
                    v
             Receive Input
                    |
                    v
              Validate Input
                    |
                    v
          Load Trained Artifacts
                    |
             +------+------+
             |             |
             v             v
       best_kmeans.pkl   scaler.json
             |             |
             +------+------+
                    |
                    v
          Extract Scaling Parameters
             Mean and Standard Deviation
                    |
                    v
          Z-score Standardization
                    |
                    v
             Scaled Features
                    |
                    v
          K-means predict()
                    |
                    v
           Calculate Distances
           to Cluster Centers
                    |
                    v
          Assign Nearest Cluster
                    |
                    v
           Predicted Cluster
                    |
             +------+------+
             |             |
             v             v
       Display Result   Calculate Time
             |             |
             +------+------+
                    |
                    v
          Save Inference History
```

### Các bước thực hiện

**Step 1 — Receive Input**

Nhận thông tin của khách hàng mới từ người dùng thông qua chương trình hoặc giao diện Streamlit.

Ba đặc trưng đầu vào bao gồm:

- Age.
- Annual Income.
- Spending Score.

Ví dụ:

```python
result = predict_customer(
    age=30,
    annual_income=75000,
    spending_score=65
)
```

Đây là dữ liệu mới cần được phân cụm, không bắt buộc phải có nhãn Cluster thực tế.

**Step 2 — Load Trained Artifacts**

Sử dụng các file đã được tạo từ những giai đoạn trước:

- `models/best_kmeans.pkl`
- `data/processed/scaler.json`

Trong `src/predict.py`, hàm `load_model()` thực hiện:

```python
with MODEL_PATH.open("rb") as f:
    model = pickle.load(f)

scaler = json.loads(
    SCALER_PATH.read_text(encoding="utf-8")
)
```

Trong đó:

- `best_kmeans.pkl`: Chứa mô hình K-means tốt nhất đã được lựa chọn trong Training Phase.
- `scaler.json`: Chứa danh sách đặc trưng, giá trị trung bình (Mean) và độ lệch chuẩn (Standard Deviation) của dữ liệu huấn luyện.

Mô hình và scaler được tải lại từ các artifact đã lưu, không cần huấn luyện lại.

**Step 3 — Validate Input**

Kiểm tra tính hợp lệ của dữ liệu đầu vào trước khi thực hiện dự đoán.

Trong `predict_customer()`, các giá trị được chuyển về kiểu `float`:

```python
values = {
    "Age": float(age),
    "Annual Income": float(annual_income),
    "Spending Score": float(spending_score)
}
```

Các điều kiện kiểm tra:

- Age phải lớn hơn 0.
- Annual Income không được âm.
- Spending Score phải nằm trong khoảng từ 0 đến 100.

Nếu dữ liệu không hợp lệ, chương trình sẽ phát sinh `ValueError` để tránh đưa dữ liệu không phù hợp vào mô hình.

**Step 4 — Z-score Standardization**

Sau khi kiểm tra dữ liệu, hệ thống chuẩn hóa các đặc trưng bằng phương pháp Z-score.

Công thức:

\[
Z = \frac{X-\mu}{\sigma}
\]

Trong đó:

- \(X\): Giá trị đặc trưng đầu vào.
- \(\mu\): Giá trị trung bình của đặc trưng trong tập huấn luyện.
- \(\sigma\): Độ lệch chuẩn của đặc trưng trong tập huấn luyện.
- \(Z\): Giá trị sau chuẩn hóa.

Code thực hiện:

```python
features = scaler["features"]

x_scaled = [[
    (values[col] - scaler["means"][col])
    / scaler["stds"][col]
    for col in features
]]
```

**Điểm quan trọng:** Inference sử dụng Mean và Standard Deviation đã lưu từ Preprocessing Phase, không tính lại các thông số chuẩn hóa từ khách hàng mới.

Điều này đảm bảo dữ liệu mới được chuyển đổi về cùng không gian đặc trưng với dữ liệu mà mô hình đã học.

**Step 5 — K-means Prediction**

Sau khi chuẩn hóa, vector đặc trưng được đưa vào mô hình K-means thông qua hàm:

```python
cluster = int(model.predict(x_scaled)[0])
```

K-means thực hiện quá trình gán cụm dựa trên khoảng cách từ điểm dữ liệu mới đến các tâm cụm đã học trong Training Phase.

Công thức khoảng cách Euclidean:

\[
d(x,\mu_k)=
\sqrt{\sum_{j=1}^{n}(x_j-\mu_{kj})^2}
\]

Trong đó:

- \(x\): Vector đặc trưng của khách hàng mới sau chuẩn hóa.
- \(\mu_k\): Tâm của cụm thứ \(k\).
- \(n\): Số lượng đặc trưng.

Khách hàng được gán vào cụm có tâm gần nhất:

\[
Cluster(x)=\arg\min_k d(x,\mu_k)
\]

Ở bước này, mô hình chỉ sử dụng các tâm cụm đã được học trước đó, không cập nhật lại tâm cụm.

**Step 6 — Generate Prediction Result**

Sau khi dự đoán, hệ thống tổng hợp kết quả:

```python
result = {
    "timestamp": datetime.now().astimezone().isoformat(
        timespec="seconds"
    ),
    "input": values,
    "output": {
        "cluster": cluster,
        "scaled_features": x_scaled[0],
        "prediction_time_seconds": round(elapsed, 6)
    }
}
```

Kết quả bao gồm:

- Timestamp: Thời điểm thực hiện dự đoán.
- Input: Thông tin khách hàng ban đầu.
- Cluster: Cụm được dự đoán.
- Scaled Features: Các giá trị sau chuẩn hóa.
- Prediction Time: Thời gian xử lý.

**Step 7 — Display and Save Results**

Hiển thị kết quả dự đoán cho người dùng.

Trong `app.py`, giao diện Streamlit hiển thị:

- Cluster mà khách hàng được phân vào.
- Thời gian dự đoán.
- Đặc điểm trung bình của cụm tương ứng nếu có `cluster_profile.csv`.
- Chi tiết dữ liệu đầu vào và các giá trị sau chuẩn hóa.

Sau đó, hàm `save_inference_history()` ghi kết quả vào:

`reports/customer_segmentation/inference_history.csv`

Các thông tin được lưu bao gồm:

- Timestamp.
- Age.
- Annual Income.
- Spending Score.
- Cluster.
- Prediction Time.

File CSV được ghi theo chế độ append, nghĩa là mỗi lần dự đoán mới sẽ được bổ sung vào lịch sử thay vì ghi đè các kết quả trước đó.

## 4. Evaluation Metrics — Các chỉ số theo dõi

Trong Inference Phase, hệ thống tập trung theo dõi thời gian xử lý, dữ liệu đầu vào và kết quả phân cụm.

| Metric | Description |
|---|---|
| Cleaning / Validation | Kiểm tra tính hợp lệ của dữ liệu đầu vào |
| Standardization | Chuyển đổi đặc trưng bằng Z-score |
| Prediction Time | Thời gian thực hiện dự đoán |
| Total Processing Time | Thời gian xử lý trong hàm `predict_customer()` |
| Feature Shape | Kích thước vector đặc trưng đầu vào |
| Predicted Cluster | Cụm được dự đoán cho khách hàng mới |

**Lưu ý:** Các chỉ số như Silhouette Score, Inertia, Calinski-Harabasz và Davies-Bouldin không được tính trực tiếp cho một khách hàng mới trong quy trình inference hiện tại.

Những chỉ số này được sử dụng trong Training và Evaluation Phase để đánh giá chất lượng phân cụm trên tập dữ liệu.

Ngoài ra, Inference Phase không có nhãn thực tế để so sánh nên không thể đánh giá độ chính xác phân loại theo cách thông thường như Accuracy, Precision, Recall hoặc F1-score.

## 5. Used Data — Dữ liệu sử dụng

Inference Phase sử dụng hai nhóm dữ liệu chính:

**1. New Customer Data**

Thông tin của khách hàng mới do người dùng cung cấp.

Bao gồm:

- Age.
- Annual Income.
- Spending Score.

Đây là dữ liệu đầu vào cần được phân cụm và không bắt buộc phải có nhãn.

**2. Trained Artifacts**

Các artifact được tạo từ Preprocessing và Training Phase:

- `models/best_kmeans.pkl`
- `data/processed/scaler.json`

Trong đó:

- `best_kmeans.pkl` chứa các tâm cụm và những thông tin cần thiết của mô hình K-means đã huấn luyện.
- `scaler.json` chứa Mean, Standard Deviation và danh sách đặc trưng được sử dụng khi chuẩn hóa dữ liệu.

Các artifact này đảm bảo khách hàng mới được xử lý theo đúng không gian đặc trưng mà mô hình đã học.

Inference Phase không sử dụng dữ liệu mới để huấn luyện lại mô hình hoặc cập nhật các tâm cụm.

## 6. Output Files — Các file đầu ra

| File | Description |
|---|---|
| `reports/customer_segmentation/inference_history.csv` | Lưu lịch sử dự đoán của khách hàng mới |
| `models/best_kmeans.pkl` | Mô hình K-means được sử dụng để dự đoán |
| `data/processed/scaler.json` | Thông số chuẩn hóa được sử dụng trong inference |

## Code References

| File | Responsibility |
|---|---|
| `src/predict.py` | Điều phối toàn bộ quá trình inference |
| `src/predict.py` – `load_model()` | Tải mô hình K-means và scaler |
| `src/predict.py` – `predict_customer()` | Kiểm tra dữ liệu, chuẩn hóa và dự đoán Cluster |
| `src/predict.py` – `save_inference_history()` | Lưu lịch sử dự đoán vào CSV |
| `src/models/kmeans.py` | Hiện thực thuật toán K-means và phương thức `predict()` |
| `src/preprocess.py` | Làm sạch dữ liệu và tạo thông số chuẩn hóa |
| `src/train.py` | Huấn luyện, lựa chọn và lưu mô hình K-means tốt nhất |
| `app.py` | Giao diện Streamlit cho phép người dùng nhập dữ liệu và xem kết quả |
| `models/` | Lưu mô hình K-means đã huấn luyện |
| `data/processed/scaler.json` | Lưu các thông số chuẩn hóa |
| `reports/customer_segmentation/inference_history.csv` | Lưu lịch sử inference |