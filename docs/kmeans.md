# K-Means Model

## 1. Purpose

K-Means là thuật toán học máy không giám sát (Unsupervised Learning), được sử dụng để phân nhóm dữ liệu khách hàng thành K cụm dựa trên mức độ tương đồng về đặc trưng.

Trong Lab03, mô hình được xây dựng bằng NumPy, không sử dụng `sklearn.cluster.KMeans`.

**Mục tiêu:**
- Phân chia khách hàng thành các nhóm có đặc điểm tương đồng.
- Tìm tâm cụm đại diện cho từng nhóm khách hàng.
- Tối thiểu hóa tổng bình phương khoảng cách giữa các điểm dữ liệu và tâm cụm tương ứng (Inertia).
- Hỗ trợ phân cụm dữ liệu mới thông qua hàm `predict()`.

## 2. Input - Output

### Input

Mô hình nhận các tham số:

| Tham số | Ý nghĩa | Giá trị mặc định |
|---|---|---|
| `X` | Ma trận dữ liệu đầu vào | Bắt buộc |
| `n_clusters` | Số lượng cụm K | 3 |
| `max_iter` | Số vòng lặp tối đa | 100 |
| `init` | Phương pháp khởi tạo tâm cụm | k-means++ |
| `tol` | Ngưỡng hội tụ | 1e-4 |
| `random_state` | Hạt giống ngẫu nhiên | 42 |

Dữ liệu `X` có dạng ma trận 2 chiều `(n_samples, n_features)`, trong đó:
- `n_samples`: Số lượng mẫu dữ liệu.
- `n_features`: Số lượng đặc trưng của mỗi mẫu.

### Output

Sau khi huấn luyện, mô hình trả về:

| Thuộc tính | Ý nghĩa |
|---|---|
| `centers_` | Tọa độ tâm của các cụm |
| `labels_` | Nhãn cụm của từng điểm dữ liệu |
| `inertia_` | Tổng bình phương khoảng cách trong cụm |
| `n_iter_` | Số vòng lặp thực tế |
| `history_` | Lịch sử huấn luyện qua từng vòng lặp |
| `converged_` | Trạng thái hội tụ |
| `stop_reason_` | Lý do dừng huấn luyện |

Ngoài ra:
- `predict(X)`: Dự đoán cụm cho dữ liệu mới.
- `fit_predict(X)`: Huấn luyện mô hình và trả về nhãn cụm.

## 3. How to Do

Mô hình K-Means được triển khai qua các bước chính:

### Step 1: Initialize Model

Khởi tạo mô hình thông qua `__init__()` với các tham số như số cụm K, số vòng lặp tối đa, phương pháp khởi tạo tâm và ngưỡng hội tụ.

### Step 2: Validate Input

Trong `fit()`, kiểm tra:
- Số cụm và số vòng lặp phải lớn hơn 0.
- Ngưỡng hội tụ không được âm.
- Dữ liệu phải là ma trận 2 chiều.
- Dữ liệu không được rỗng hoặc chứa NaN, Inf.
- Số lượng mẫu phải lớn hơn hoặc bằng số cụm.

### Step 3: Initialize Centroids

Sử dụng `_initialize_centers()` để chọn K tâm cụm ban đầu.

Hỗ trợ hai phương pháp:

- **Random:** Chọn ngẫu nhiên K điểm dữ liệu khác nhau làm tâm cụm ban đầu.
- **K-means++:** Chọn tâm đầu tiên ngẫu nhiên, sau đó chọn các tâm tiếp theo dựa trên xác suất tỷ lệ với bình phương khoảng cách đến tâm gần nhất. Phương pháp này giúp các tâm ban đầu phân bố hợp lý hơn.

### Step 4: Assign Labels

Sử dụng `_assign_labels()` để tính khoảng cách từ mỗi điểm dữ liệu đến các tâm cụm.

Khoảng cách Euclidean bình phương:

\[
d(x_i,\mu_j)^2 = \sum_{k=1}^{n}(x_{ik}-\mu_{jk})^2
\]

Mỗi điểm được gán vào cụm có tâm gần nhất.

### Step 5: Update Centroids

Sử dụng `_update_centers()` để cập nhật tâm cụm bằng trung bình cộng của các điểm thuộc cụm đó.

\[
\mu_j = \frac{1}{|C_j|}\sum_{x_i \in C_j}x_i
\]

Nếu xuất hiện cụm rỗng, mô hình chọn điểm dữ liệu xa tâm cụm gần nhất để khởi tạo lại tâm của cụm đó.

### Step 6: Check Convergence

Tính độ dịch chuyển của các tâm cụm giữa hai lần cập nhật:

\[
centroid\_shift = ||\mu_{new}-\mu_{old}||_2
\]

Nếu:

\[
centroid\_shift \leq tol
\]

Mô hình được xem là hội tụ và dừng huấn luyện.

Nếu chưa hội tụ, tiếp tục cho đến khi đạt `max_iter`.

### Step 7: Save Training Results

Mỗi vòng lặp lưu lại:
- Inertia.
- Centroid shift.
- Số lượng điểm trong từng cụm.
- Tọa độ các tâm cụm.

Kết thúc huấn luyện, mô hình lưu tâm cụm, nhãn cụm, Inertia và số vòng lặp thực tế.

### Step 8: Prediction

Sử dụng `predict()` để gán dữ liệu mới vào cụm có tâm gần nhất dựa trên các tâm cụm đã học.

## 4. Evaluate Metrics

Các chỉ số được sử dụng để theo dõi và đánh giá chất lượng phân cụm:

| Metric | Ý nghĩa | Đánh giá |
|---|---|---|
| Inertia | Tổng bình phương khoảng cách từ các điểm đến tâm cụm tương ứng | Càng thấp càng tốt |
| Silhouette Score | Đánh giá mức độ liên kết trong cụm và sự tách biệt giữa các cụm | Càng gần 1 càng tốt |
| Centroid Shift | Độ dịch chuyển của tâm cụm qua mỗi vòng lặp | Càng nhỏ càng thể hiện sự ổn định |
| Cluster Sizes | Số lượng điểm dữ liệu thuộc từng cụm | Dùng để đánh giá phân bố dữ liệu |
| Convergence | Trạng thái hội tụ của mô hình | Kiểm tra mô hình đã hội tụ hay chưa |

**Lưu ý:**
- Inertia được tính trực tiếp trong `kmeans.py`.
- Silhouette Score được sử dụng ở bước đánh giá bên ngoài mô hình, không được tính trực tiếp trong class `KMeans`.
- Inertia thường giảm khi tăng K, vì vậy cần kết hợp Elbow Method và Silhouette Score để lựa chọn số cụm phù hợp.

## 5. Used Data

### Dataset: Customer Segmentation

Sử dụng bộ dữ liệu khách hàng trong bài toán Customer Segmentation của Lab03.

**Thông tin dữ liệu:**
- Số lượng khách hàng: 15,079.
- Loại bài toán: Unsupervised Learning.
- Không sử dụng nhãn mục tiêu (Target Label) để huấn luyện.

**Các đặc trưng khách hàng được sử dụng trong phân tích:**
- Age: Độ tuổi khách hàng.
- Annual Income: Thu nhập hằng năm.
- Spending Score: Điểm chi tiêu của khách hàng.
- Gender: Giới tính khách hàng (nếu được giữ lại trong ma trận đặc trưng sau tiền xử lý).

**Quy trình sử dụng dữ liệu:**

1. Đọc dữ liệu khách hàng.
2. Tiền xử lý và lựa chọn các đặc trưng phù hợp.
3. Chuẩn bị ma trận dữ liệu số `X`.
4. Đưa dữ liệu vào mô hình K-Means.
5. Phân nhóm khách hàng theo số cụm K đã lựa chọn.
6. Sử dụng nhãn cụm để phân tích và diễn giải đặc điểm từng nhóm khách hàng.

**Kết quả:** Mô hình tạo ra các nhóm khách hàng có đặc điểm tương đồng, hỗ trợ phân tích hành vi và xây dựng chiến lược chăm sóc khách hàng phù hợp.