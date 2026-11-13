# Training Phase — Huấn luyện mô hình

## 1. Purpose — Mục đích

Training Phase là giai đoạn huấn luyện thuật toán Machine Learning nhằm xây dựng mô hình có khả năng phân cụm khách hàng dựa trên các đặc trưng về độ tuổi, thu nhập và hành vi chi tiêu.

Mục tiêu của giai đoạn này là sử dụng dữ liệu đã được tiền xử lý để thuật toán K-means học cách xác định các nhóm khách hàng có đặc điểm tương đồng.

Khác với bài toán phân loại có nhãn (Classification), Customer Segmentation là bài toán học không giám sát (Unsupervised Learning). Dữ liệu huấn luyện không có nhãn Cluster được cung cấp trước. Thuật toán tự tìm kiếm cấu trúc phân nhóm dựa trên khoảng cách giữa các điểm dữ liệu.

Dự án sử dụng thuật toán:

- **K-means:** Phân chia dữ liệu thành K cụm dựa trên khoảng cách Euclidean đến các tâm cụm (Centroids).

Trong quá trình huấn luyện, hệ thống thử nghiệm nhiều giá trị K khác nhau để tìm số cụm phù hợp.

Sau khi hoàn thành, mô hình có kết quả phân cụm tốt nhất được lưu thành file `best_kmeans.pkl` để tái sử dụng trong Inference Phase mà không cần huấn luyện lại.

## 2. Input — Output

### Input

| Input | Description |
|---|---|
| `X_scaled.csv` | Ma trận đặc trưng khách hàng đã được chuẩn hóa bằng Z-score |
| `config.yaml` | File cấu hình các tham số huấn luyện K-means |
| `src/models/kmeans.py` | Cài đặt thuật toán K-means sử dụng NumPy |

Dữ liệu đầu vào được tạo từ Preprocessing Phase, bao gồm:

- Làm sạch dữ liệu khách hàng.
- Lựa chọn ba đặc trưng Age, Annual Income và Spending Score.
- Chuẩn hóa dữ liệu bằng phương pháp Z-score.

**Các đặc trưng sử dụng trong huấn luyện:**

| Feature | Description |
|---|---|
| Age | Độ tuổi khách hàng |
| Annual Income | Thu nhập hằng năm |
| Spending Score | Điểm chi tiêu của khách hàng (0–100) |

### Output

| Output | Description |
|---|---|
| `best_kmeans.pkl` | Mô hình K-means tốt nhất đã huấn luyện |
| `experiment_results.csv` | Kết quả thử nghiệm các giá trị K |
| `experiment_logs.csv` | Lịch sử thực nghiệm qua các lần chạy |
| `centers_` | Tọa độ các tâm cụm sau huấn luyện |
| `labels_` | Nhãn cụm của từng mẫu dữ liệu |
| `inertia_` | Tổng bình phương khoảng cách từ các điểm đến tâm cụm tương ứng |
| `history_` | Lịch sử hội tụ qua từng vòng lặp |

Mô hình sau khi huấn luyện được lưu trong thư mục `models/` và được sử dụng lại trong Inference Phase.

Kết quả thực nghiệm được lưu trong `data/processed/` và lịch sử thực nghiệm được lưu tại `docs/experiment_logs.csv`.

## 3. How to do — Quy trình thực hiện

### Training Workflow

```text
             X_scaled.csv
                   |
                   v
           Load Training Data
                   |
                   v
              Load Config
                   |
                   v
        Read K Values [2,3,4,5,6,7,8]
                   |
                   v
           Initialize K-means
                   |
                   v
          Loop Through Each K
                   |
                   v
          Initialize Centroids
                   |
                   v
             K-means fit()
                   |
          +--------+--------+
          |                 |
          v                 v
     Assign Labels     Update Centroids
          |                 |
          +--------+--------+
                   |
                   v
           Check Convergence
                   |
             +-----+-----+
             |           |
          Not Stop     Converged
             |           |
             +-----<-----+
                   |
                   v
            Calculate Inertia
                   |
                   v
         Calculate Silhouette Score
                   |
                   v
          Save Experiment Results
                   |
                   v
         Compare Silhouette Scores
                   |
                   v
          Select Best K and Model
                   |
                   v
          Save best_kmeans.pkl
                   |
                   v
           Save Experiment Logs
                   |
                   v
             Evaluation Phase
```

### Các bước thực hiện

**Step 1 — Load Training Data**

Đọc dữ liệu đặc trưng đã chuẩn hóa từ file `X_scaled.csv`.

Code thực hiện trong `src/train.py`:

```python
X = pd.read_csv(x_path).to_numpy(dtype=float)
```

Trong đó:

- `X`: Ma trận đặc trưng đầu vào.
- Mỗi hàng đại diện cho một khách hàng.
- Mỗi cột đại diện cho một đặc trưng.

Kích thước dữ liệu hiện tại:

```text
X.shape = (15079, 3)
```

Có nghĩa là dữ liệu huấn luyện gồm 15.079 khách hàng và 3 đặc trưng.

**Step 2 — Load Configuration**

Đọc các tham số được khai báo trong `config.yaml`.

Các tham số chính:

| Parameter | Value | Description |
|---|---:|---|
| `k_values` | [2, 3, 4, 5, 6, 7, 8] | Các giá trị K cần thử nghiệm |
| `max_iter` | 60 | Số vòng lặp tối đa |
| `random_state` | 42 | Seed để tái lập kết quả |
| `silhouette_sample_size` | 1000 | Số mẫu tối đa dùng tính Silhouette |
| `tolerance` | 0.0001 | Ngưỡng dừng dựa trên độ dịch chuyển tâm cụm |

Trong `train.py`, các tham số được đọc như sau:

```python
k_values = config.get(
    "k_values", [2, 3, 4, 5, 6, 7, 8]
)

max_iter = int(config.get("max_iter", 60))

random_state = int(
    config.get("random_state", 42)
)

sample_size = int(
    config.get("silhouette_sample_size", 1000)
)

tolerance = float(
    config.get("tolerance", 1e-4)
)
```

Các tham số này giúp kiểm soát quá trình huấn luyện và đảm bảo thực nghiệm có thể tái lập.

**Step 3 — Initialize K-means Models**

Hệ thống lần lượt thử nghiệm các giá trị K được khai báo trong `config.yaml`.

Ví dụ:

```python
k_values = [2, 3, 4, 5, 6, 7, 8]
```

Với mỗi giá trị K, hệ thống khởi tạo một mô hình K-means mới:

```python
model = KMeans(
    n_clusters=k,
    init="random",
    max_iter=max_iter,
    tol=tolerance,
    random_state=random_state
)
```

Trong đó:

- `n_clusters`: Số cụm cần phân chia.
- `init="random"`: Khởi tạo tâm cụm bằng cách chọn ngẫu nhiên K điểm dữ liệu khác nhau.
- `max_iter`: Số vòng lặp tối đa.
- `tol`: Ngưỡng hội tụ.
- `random_state`: Đảm bảo kết quả khởi tạo có thể tái lập.

**Lưu ý:** Trong Training Phase chính, chương trình sử dụng phương pháp khởi tạo `random`. Mặc dù lớp K-means hỗ trợ thêm `k-means++`, phương pháp này không được sử dụng trong vòng huấn luyện chính của `train.py`.

**Step 4 — Train K-means Models**

Gọi phương thức:

```python
model.fit(X)
```

Đây là bước thực hiện thuật toán K-means được cài đặt trong `src/models/kmeans.py`.

Quá trình huấn luyện gồm các bước chính:

**4.1. Initialize Centroids — Khởi tạo tâm cụm**

Chọn K điểm dữ liệu làm tâm cụm ban đầu.

```python
centers = self._initialize_centers(X, rng)
```

Các tâm cụm ban đầu được lấy từ những điểm dữ liệu khác nhau trong tập huấn luyện.

**4.2. Assign Labels — Gán điểm dữ liệu vào cụm**

Tính khoảng cách giữa từng điểm dữ liệu và các tâm cụm.

Công thức khoảng cách Euclidean:

\[
d(x,\mu_k)=
\sqrt{\sum_{j=1}^{n}(x_j-\mu_{kj})^2}
\]

Trong đó:

- \(x\): Điểm dữ liệu khách hàng.
- \(\mu_k\): Tâm cụm thứ k.
- \(n\): Số lượng đặc trưng.

Sau đó, mỗi khách hàng được gán vào cụm có tâm gần nhất.

```python
labels = np.argmin(distances, axis=1)
```

**4.3. Update Centroids — Cập nhật tâm cụm**

Sau khi gán các điểm dữ liệu vào cụm, hệ thống tính lại tâm cụm bằng trung bình các điểm thuộc cụm đó.

Công thức:

\[
\mu_k =
\frac{1}{N_k}
\sum_{x_i \in C_k}x_i
\]

Trong đó:

- \(C_k\): Tập hợp các điểm thuộc cụm k.
- \(N_k\): Số lượng điểm dữ liệu trong cụm.
- \(\mu_k\): Tâm cụm mới.

Code:

```python
members = X[labels == cluster_id]

if len(members) > 0:
    new_centers[cluster_id] = members.mean(axis=0)
```

Nếu một cụm không có điểm dữ liệu nào (Empty Cluster), thuật toán sẽ chọn điểm đang nằm xa tâm cụm gần nhất để khởi tạo lại tâm cụm rỗng.

**4.4. Check Convergence — Kiểm tra hội tụ**

Sau khi cập nhật tâm cụm, hệ thống tính độ dịch chuyển của các tâm:

\[
Shift =
\left\|
Centers_{new}-Centers_{old}
\right\|_2
\]

Code:

```python
centroid_shift = float(
    np.linalg.norm(new_centers - centers)
)
```

Nếu:

\[
centroid\_shift \leq tolerance
\]

Thuật toán dừng vì các tâm cụm đã gần như không thay đổi.

Nếu chưa đạt điều kiện hội tụ, thuật toán tiếp tục lặp lại hai bước Assign Labels và Update Centroids cho đến khi hội tụ hoặc đạt `max_iter`.

**4.5. Calculate Inertia — Tính tổng sai số trong cụm**

Sau khi cập nhật tâm cụm, hệ thống tính Inertia.

Công thức:

\[
Inertia =
\sum_{k=1}^{K}
\sum_{x_i \in C_k}
\|x_i-\mu_k\|^2
\]

Inertia biểu diễn tổng bình phương khoảng cách từ mỗi điểm dữ liệu đến tâm cụm tương ứng.

- Inertia càng nhỏ: Các điểm trong cụm càng gần tâm cụm.
- Inertia càng lớn: Các điểm dữ liệu phân tán xa tâm cụm hơn.

Tuy nhiên, Inertia thường giảm khi tăng K nên không thể sử dụng riêng chỉ số này để quyết định số cụm tốt nhất.

**4.6. Save Convergence History — Lưu lịch sử hội tụ**

Sau mỗi vòng lặp, thuật toán ghi nhận:

- Iteration: Số thứ tự vòng lặp.
- Inertia: Giá trị Inertia tại vòng lặp.
- Centroid Shift: Độ dịch chuyển tâm cụm.
- Cluster Sizes: Số lượng điểm trong từng cụm.
- Centers: Tọa độ các tâm cụm.

Các thông tin này được lưu trong thuộc tính `history_` của mô hình.

Sau khi huấn luyện hoàn tất, mô hình lưu lại:

- `centers_`: Các tâm cụm cuối cùng.
- `labels_`: Nhãn cụm của từng điểm dữ liệu.
- `inertia_`: Inertia cuối cùng.
- `n_iter_`: Số vòng lặp thực tế.

**Step 5 — Evaluate and Select Best K**

Sau khi huấn luyện từng mô hình, hệ thống tính Silhouette Score để đánh giá chất lượng phân cụm.

Code:

```python
score = silhouette_score_sample(
    X,
    model.labels_,
    sample_size=sample_size,
    random_state=random_state
)
```

Silhouette Score đánh giá mức độ phù hợp của một điểm dữ liệu với cụm hiện tại so với các cụm khác.

Công thức:

\[
S(i)=\frac{b(i)-a(i)}
{\max(a(i),b(i))}
\]

Trong đó:

- \(a(i)\): Khoảng cách trung bình từ điểm i đến các điểm khác trong cùng cụm.
- \(b(i)\): Khoảng cách trung bình nhỏ nhất từ điểm i đến một cụm khác.

Giá trị Silhouette Score nằm trong khoảng [-1, 1]:

- Gần 1: Điểm dữ liệu được phân vào cụm phù hợp.
- Gần 0: Điểm dữ liệu nằm gần ranh giới giữa các cụm.
- Gần -1: Điểm dữ liệu có khả năng được phân vào cụm chưa phù hợp.

Trong dự án, Silhouette Score được tính trên tối đa 1.000 mẫu để giảm thời gian và chi phí tính toán.

Sau đó, hệ thống so sánh các kết quả và lựa chọn mô hình có Silhouette Score cao nhất:

```python
if pd.notna(score) and score > best_score:
    best_score = score
    best_model = model
    best_k = k
```

Như vậy, tiêu chí lựa chọn mô hình tốt nhất trong Training Phase là Silhouette Score cao nhất, không phải Inertia thấp nhất.

**Kết quả thực nghiệm hiện tại:**

| K | Inertia | Silhouette Score |
|---:|---:|---:|
| 2 | 33927.6926 | 0.2401 |
| 3 | 26997.0444 | 0.2320 |
| 4 | 21502.2601 | 0.2550 |
| 5 | 18088.5254 | 0.2672 |
| **6** | **14905.6510** | **0.2859** |
| 7 | 13170.8885 | 0.2721 |
| 8 | 11315.0007 | 0.2724 |

<caption> Kết quả từ experiment_results.csv trong lần chạy hiện tại.</caption>

Mô hình K-means với K = 6 đạt Silhouette Score cao nhất (0.2859), do đó được lựa chọn làm mô hình tốt nhất.

**Step 6 — Save Trained Model**

Sau khi xác định mô hình tốt nhất, hệ thống lưu mô hình bằng thư viện `pickle`.

Code:

```python
with (MODEL_DIR / "best_kmeans.pkl").open("wb") as file:
    pickle.dump(best_model, file)
```

File được lưu tại:

`models/best_kmeans.pkl`

Mô hình đã lưu chứa các thông tin quan trọng như:

- Số lượng cụm K.
- Các tâm cụm đã học.
- Nhãn cụm của dữ liệu huấn luyện.
- Inertia.
- Số vòng lặp.
- Lịch sử hội tụ.

Nhờ đó, mô hình có thể được tải lại và sử dụng để dự đoán cụm cho khách hàng mới trong Inference Phase.

**Step 7 — Save Experiment Results and Tracking**

Sau mỗi lần thử nghiệm, hệ thống lưu bảng kết quả vào:

`data/processed/experiment_results.csv`

Các thông tin bao gồm:

- K.
- Inertia.
- Silhouette Score.
- Iterations.
- Time (seconds).

Ngoài ra, lịch sử thực nghiệm được ghi vào:

`docs/experiment_logs.csv`

Các thông tin được lưu gồm:

- Run_ID.
- Timestamp.
- K.
- Inertia.
- Silhouette Score.
- Iterations.
- Time (seconds).

File lịch sử được ghi theo chế độ append, giúp lưu lại nhiều lần thực nghiệm mà không ghi đè các kết quả trước đó.

Sau khi hoàn thành Training Phase, hệ thống chuyển sang Evaluation Phase để đánh giá chất lượng phân cụm chi tiết hơn.

## 4. Evaluation Metrics — Các chỉ số đánh giá

Trong Training Phase, hệ thống sử dụng các chỉ số để theo dõi quá trình hội tụ và lựa chọn số lượng cụm phù hợp.

| Metric | Description |
|---|---|
| Inertia | Tổng bình phương khoảng cách từ các điểm dữ liệu đến tâm cụm tương ứng |
| Silhouette Score | Đánh giá mức độ phù hợp và tách biệt giữa các cụm |
| Iterations | Số vòng lặp thực tế của thuật toán K-means |
| Centroid Shift | Độ dịch chuyển của các tâm cụm giữa hai vòng lặp |
| Cluster Sizes | Số lượng điểm dữ liệu được phân vào từng cụm |
| Training Time | Thời gian thực hiện huấn luyện và đánh giá một giá trị K |

**Lưu ý:**

- Inertia được sử dụng để theo dõi mức độ tập trung của các điểm dữ liệu trong cụm.
- Silhouette Score được sử dụng làm tiêu chí lựa chọn mô hình tốt nhất.
- Centroid Shift được sử dụng để xác định điều kiện hội tụ.
- Training Time giúp theo dõi hiệu suất thực thi.

Các chỉ số Calinski-Harabasz và Davies-Bouldin được sử dụng trong Evaluation Phase và các thực nghiệm mở rộng, không phải tiêu chí lựa chọn mô hình chính trong `src/train.py`.

Do đây là bài toán Unsupervised Learning nên dự án không sử dụng Accuracy, Precision, Recall hoặc F1-score để đánh giá mô hình K-means.

## 5. Used Data — Dữ liệu sử dụng

Dữ liệu được sử dụng trong Training Phase có nguồn gốc từ bộ dữ liệu Customer Segmentation.

Sau Preprocessing Phase, dữ liệu được làm sạch và lựa chọn ba đặc trưng chính:

- Age.
- Annual Income.
- Spending Score.

**Dữ liệu huấn luyện:**

| Dataset | Number of Samples | Number of Features | Purpose |
|---|---:|---:|---|
| Training Data | 15,079 | 3 | Dùng để huấn luyện K-means |

Các file dữ liệu:

- `data/processed/X_scaled.csv`: Ma trận đặc trưng đã chuẩn hóa.
- `data/processed/customers_clean.csv`: Dữ liệu khách hàng sau làm sạch.
- `data/processed/scaler.json`: Các thông số Mean và Standard Deviation được sử dụng trong chuẩn hóa.

**Lưu ý:** Khác với bài toán phân loại trong Lab01, K-means là thuật toán học không giám sát nên không cần `y_train` hoặc nhãn thực tế để huấn luyện.

Toàn bộ 15.079 khách hàng được sử dụng làm dữ liệu huấn luyện. Sau khi huấn luyện, thuật toán tự tạo nhãn Cluster cho từng khách hàng.

## Code References

| File | Responsibility |
|---|---|
| `src/train.py` | Điều phối quá trình thử nghiệm nhiều giá trị K và lựa chọn mô hình tốt nhất |
| `src/models/kmeans.py` | Hiện thực thuật toán K-means bằng NumPy |
| `src/models/kmeans.py` – `__init__()` | Khởi tạo các tham số và thuộc tính của mô hình |
| `src/models/kmeans.py` – `_initialize_centers()` | Khởi tạo các tâm cụm ban đầu |
| `src/models/kmeans.py` – `_squared_distances()` | Tính bình phương khoảng cách Euclidean |
| `src/models/kmeans.py` – `_assign_labels()` | Gán các điểm dữ liệu vào cụm gần nhất |
| `src/models/kmeans.py` – `_update_centers()` | Cập nhật tâm cụm và xử lý cụm rỗng |
| `src/models/kmeans.py` – `fit()` | Thực hiện toàn bộ quá trình huấn luyện K-means |
| `src/models/kmeans.py` – `predict()` | Dự đoán cụm cho dữ liệu mới |
| `src/utils.py` | Cài đặt hàm tính Silhouette Score trên mẫu dữ liệu |
| `src/preprocess.py` | Làm sạch và chuẩn hóa dữ liệu đầu vào |
| `src/evaluate.py` | Đánh giá chất lượng phân cụm |
| `src/experiments.py` | Thực hiện các thực nghiệm mở rộng |
| `src/tracking.py` | Theo dõi trạng thái và thông tin các lần chạy pipeline |
| `config.yaml` | Cấu hình các tham số huấn luyện và thực nghiệm |
| `models/best_kmeans.pkl` | Lưu mô hình K-means tốt nhất |
| `data/processed/experiment_results.csv` | Lưu kết quả thử nghiệm các giá trị K |
| `docs/experiment_logs.csv` | Lưu lịch sử thực nghiệm qua các lần chạy |