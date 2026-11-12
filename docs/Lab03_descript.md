# BÁO CÁO LAB 03: PHÂN CỤM KHÁCH HÀNG

## 1. Mục tiêu
Phân nhóm khách hàng bằng học máy không giám sát dựa trên tuổi, thu nhập hằng năm và điểm chi tiêu. Mục đích là nhận diện các nhóm có đặc điểm khác nhau để hỗ trợ phân tích khách hàng.

## 2. Mô tả dữ liệu
Bộ dữ liệu có 15.079 dòng và 5 cột: `Customer ID`, `Age`, `Gender`, `Annual Income`, `Spending Score`.
- `Customer ID`: mã định danh duy nhất, không dùng làm đặc trưng.
- `Age`: tuổi khách hàng.
- `Gender`: giới tính, giữ lại để phân tích đặc điểm cụm.
- `Annual Income`: thu nhập hằng năm.
- `Spending Score`: điểm chi tiêu.

Kiểm tra dữ liệu gồm giá trị thiếu, dòng trùng, thống kê mô tả, biểu đồ phân phối và biểu đồ phân tán. Ngoại lệ được thống kê theo quy tắc IQR và trực quan hóa bằng boxplot; ngoại lệ không tự động bị xóa vì giá trị cực đoan có thể là dữ liệu hợp lệ. Ba đặc trưng số được dùng để phân cụm.

## 3. Tiền xử lý
Dữ liệu được chuyển về kiểu số cho các đặc trưng cần thiết và loại các dòng thiếu hoặc không hữu hạn. Sau đó chuẩn hóa theo z-score:

`z = (x - mean) / standard_deviation`

Chuẩn hóa giúp tuổi, thu nhập và điểm chi tiêu có thang đo tương đương hơn khi tính khoảng cách. `Customer ID` và `Gender` không đưa vào tính khoảng cách K-means.

## 4. Phương pháp
K-means được tự cài đặt bằng NumPy. Thuật toán khởi tạo tâm bằng K-means++, lặp giữa hai thao tác:
1. Gán mỗi điểm vào tâm gần nhất.
2. Cập nhật mỗi tâm bằng trung bình các điểm trong cụm.

Thử các giá trị K từ 2 đến 8. Inertia được dùng để quan sát mức độ thu gọn cụm và trực quan hóa bằng Elbow Method; Silhouette Score được dùng để so sánh độ tách biệt và được vẽ theo K. Do tính khoảng cách cặp đôi tốn chi phí, Silhouette Score được ước lượng trên mẫu tối đa 1.000 khách hàng.

## 5. Kết quả thực nghiệm
Trong lần chạy hiện tại, K=6 có Silhouette Score mẫu cao nhất trong khoảng đã thử:
- K được chọn: 6
- Inertia: khoảng 14.886,31
- Silhouette Score trên mẫu: khoảng 0,2847
- Số khách hàng trong 6 cụm lần lượt: 2.482; 2.387; 2.491; 2.595; 2.608; 2.516.

Silhouette Score khoảng 0,28 cho thấy các cụm có một mức phân biệt nhất định nhưng chưa tách biệt mạnh. Vì vậy cần kết hợp biểu đồ và hồ sơ cụm để diễn giải; không nên chỉ dựa vào điểm số để kết luận K=6 là lựa chọn tốt nhất tuyệt đối.

## 6. Diễn giải sơ bộ các cụm
Dựa trên trung bình đặc trưng trong lần chạy hiện tại:
- Cụm 0: tuổi trung bình khoảng 63,8; thu nhập khoảng 60.475; điểm chi tiêu khoảng 77,8.
- Cụm 1: tuổi trung bình khoảng 43,3; thu nhập khoảng 159.932; điểm chi tiêu khoảng 23,3.
- Cụm 2: tuổi trung bình khoảng 34,0; thu nhập khoảng 136.339; điểm chi tiêu khoảng 77,9.
- Cụm 3: tuổi trung bình khoảng 35,0; thu nhập khoảng 61.668; điểm chi tiêu khoảng 35,4.
- Cụm 4: tuổi trung bình khoảng 73,8; thu nhập khoảng 157.879; điểm chi tiêu khoảng 64,6.
- Cụm 5: tuổi trung bình khoảng 74,5; thu nhập khoảng 84.085; điểm chi tiêu khoảng 23,8.

Các mô tả này là giá trị trung bình của từng cụm trong lần chạy hiện tại. Tên cụm nên được đặt dựa trên đặc trưng nổi bật, không phải do thuật toán tự tạo ra tên.

## 7. Kết luận
K-means đã phân nhóm được khách hàng dựa trên ba đặc trưng số. Trong các giá trị K thử nghiệm từ 2 đến 8, K=6 đạt Silhouette Score mẫu cao nhất. Kết quả có thể hỗ trợ phân tích các nhóm khách hàng khác nhau, tuy nhiên điểm Silhouette chưa cao nên cần thận trọng khi diễn giải và có thể thử thêm giá trị K hoặc thuật toán khác nếu yêu cầu bài học đòi hỏi so sánh.

## 8. Tệp kết quả
- `data/processed/experiment_results.csv`: kết quả thử nghiệm K.
- `data/processed/customer_clusters.csv`: dữ liệu khách hàng kèm nhãn cụm.
- `data/processed/cluster_profile.csv`: thống kê mô tả từng cụm.
- `data/processed/outlier_summary.csv`: thống kê ngoại lệ theo IQR.
- `docs/results_summary.md`: tóm tắt kết quả mô hình sau khi chạy.
- `docs/images/`: biểu đồ EDA, boxplot ngoại lệ, Elbow, Silhouette, kích thước cụm và phân cụm.


## 9. Thực nghiệm mở rộng và lựa chọn đặc trưng

Ngoài không gian 3D cơ sở (Age, Annual Income, Spending Score), dự án khảo sát:

| Không gian | Đặc trưng | Mục đích |
|---|---|---|
| 2D Core | Annual Income, Spending Score | Dễ trực quan hóa và đối chiếu phân khúc kinh doanh |
| 3D Demographics | Age, Annual Income, Spending Score | Không gian cơ sở của báo cáo |
| 3D Ratio | Annual Income, Spending Score, Spending/Income | Kiểm tra giá trị của đặc trưng tỷ lệ |
| 4D Core+Ratio | Age, Annual Income, Spending Score, Spending/Income | Kiểm tra thêm đặc trưng dẫn xuất |
| 5D Exploratory | 4 đặc trưng trên + Gender mã hóa | Chỉ dùng để khảo sát ảnh hưởng của biến phân loại |

Mỗi không gian được chuẩn hóa độc lập trước khi tính khoảng cách. `Customer ID` luôn bị loại khỏi đầu vào. Không gian có Gender chỉ là thử nghiệm vì mã hóa Male/Female thành 0/1 không đảm bảo khoảng cách số học mang ý nghĩa tự nhiên. Tương tự, Spending/Income có thể làm tăng ảnh hưởng của các biến đã có; cần căn cứ kết quả thực nghiệm để quyết định giữ hay bỏ.

## 10. Đánh giá và tuning

Ngoài Inertia và Silhouette Score, dự án bổ sung:
- **Calinski-Harabasz:** tỷ lệ độ phân tán giữa cụm so với trong cụm; thường càng cao càng tốt.
- **Davies-Bouldin:** mức tương đồng trung bình giữa cụm với cụm gần nhất; thường càng thấp càng tốt.

Grid Search khảo sát K=2..8, `init` thuộc `random`/`k-means++`, `max_iter` thuộc 60/120 và `tol=1e-4` trên ba không gian cốt lõi. Silhouette được ước lượng trên mẫu tối đa 1.000 điểm để kiểm soát thời gian; CH và DB tính trên toàn bộ dữ liệu. Kết quả nằm trong `data/processed/grid_search_results.csv`.

Không sử dụng Manhattan + median trong thuật toán K-means chuẩn. Tổ hợp đó là K-medians, có hàm mục tiêu và quy tắc cập nhật tâm khác. Dự án giữ đúng K-means Euclidean để nhất quán với mục tiêu tối thiểu hóa tổng bình phương khoảng cách.

## 11. Hội tụ

Mô hình lưu `history_` gồm Inertia, độ dịch chuyển tâm và kích thước từng cụm qua mỗi vòng lặp. Chạy `python -m src.demo_convergence` để tạo biểu đồ `docs/images/kmeans_convergence.png` và bảng `data/processed/convergence_history.csv`.
