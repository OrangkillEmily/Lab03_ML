
# Tóm tắt kết quả Lab03

## 1. Kết quả mô hình

- **Tổng số khách hàng:** 15,079
- **Số cụm được chọn (K):** 6
- **Inertia:** 14,905.6510
- **Silhouette Score:** 0.2859 (ước lượng trên mẫu tối đa 1,000 điểm)

## 2. Kích thước các cụm

| Cluster | Số lượng khách hàng |
|:---:|---:|
| 0 | 2,629 |
| 1 | 2,486 |
| 2 | 2,461 |
| 3 | 2,352 |
| 4 | 2,628 |
| 5 | 2,523 |
| **Tổng** | **15,079** |

## 3. Hồ sơ đặc trưng của từng cụm

| Cluster | Age Mean | Age Median | Income Mean | Income Median | Spending Mean | Spending Median | Female (%) | Male (%) |
|:---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 34.12 | 33 | 59,143.55 | 56,232 | 41.74 | 42 | 48.88 | 51.12 |
| 1 | 35.84 | 35 | 141,740.80 | 143,512.5 | 78.38 | 79 | 49.44 | 50.56 |
| 2 | 72.61 | 73 | 78,488.11 | 78,191 | 22.68 | 22 | 49.78 | 50.22 |
| 3 | 40.67 | 41 | 155,059.18 | 157,199 | 22.82 | 22 | 51.40 | 48.60 |
| 4 | 74.59 | 75 | 160,891.41 | 162,428 | 57.68 | 58 | 48.33 | 51.67 |
| 5 | 66.58 | 67 | 65,904.02 | 64,788 | 78.16 | 79 | 50.18 | 49.82 |

*Lưu ý: Age, Annual Income và Spending Score được trình bày theo giá trị trung bình (Mean) và trung vị (Median).*

## 4. Nhận xét và diễn giải các cụm

- **Cluster 0:** Khách hàng trẻ (khoảng 34 tuổi), thu nhập trung bình thấp và mức chi tiêu trung bình.
- **Cluster 1:** Khách hàng trẻ (khoảng 36 tuổi), thu nhập cao và mức chi tiêu cao.
- **Cluster 2:** Khách hàng lớn tuổi (khoảng 73 tuổi), thu nhập trung bình và mức chi tiêu thấp.
- **Cluster 3:** Khách hàng khoảng 41 tuổi, thu nhập cao và mức chi tiêu thấp.
- **Cluster 4:** Khách hàng lớn tuổi (khoảng 75 tuổi), thu nhập cao và mức chi tiêu tương đối cao.
- **Cluster 5:** Khách hàng lớn tuổi (khoảng 67 tuổi), thu nhập thấp hơn các nhóm còn lại và mức chi tiêu cao.

## 5. Lưu ý khi đánh giá kết quả

- **Inertia:** Đo tổng bình phương khoảng cách từ các điểm dữ liệu đến tâm cụm tương ứng. Giá trị càng thấp thường cho thấy các điểm trong cụm càng gần tâm cụm, nhưng cần xét cùng số lượng cụm K.
- **Silhouette Score:** Đạt 0.2859, được ước lượng trên mẫu tối đa 1,000 điểm thay vì toàn bộ dữ liệu. Kết quả cho thấy mức độ phân tách giữa các cụm chưa thực sự rõ ràng.
- **Kích thước cụm:** Sáu cụm có số lượng khách hàng tương đối đồng đều, dao động từ 2,352 đến 2,629 khách hàng.
- **Diễn giải cụm:** Cần kết hợp biểu đồ phân cụm, kích thước cụm và các đặc trưng thống kê để đánh giá. Các giá trị Cluster (0–5) chỉ là mã định danh nhóm, không biểu thị thứ hạng hay mức độ quan trọng.
