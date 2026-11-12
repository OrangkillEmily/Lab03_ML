# Phân tích thực nghiệm mở rộng Lab03

Kết quả được tạo từ `python -m src.experiments` trên 15.079 khách hàng. Silhouette được ước lượng trên mẫu 1.000 điểm; CH và DB tính trên toàn bộ dữ liệu.

## Cấu hình tốt nhất theo Silhouette trong từng không gian

| Không gian | K | Silhouette | Calinski-Harabasz | Davies-Bouldin | Inertia |
|---|---:|---:|---:|---:|---:|
| 2D_Core | 4 | 0.3955 | 15050.62 | 0.7692 | 7548.65 |
| 3D_Demographics | 6 | 0.2843 | 6146.45 | 0.9562 | 14886.01 |
| 3D_Ratio | 3 | 0.3973 | 11218.41 | 0.9282 | 18180.26 |
| 4D_Core_Ratio | 2 | 0.3027 | 6091.30 | 1.3381 | 42959.72 |
| 5D_With_Gender_Exploratory | 3 | 0.2522 | 4718.49 | 1.4934 | 46369.53 |

## Kết luận thận trọng

- Không gian 2D Core đạt Silhouette cao nhất tại K=4 (0,3955), đồng thời Davies-Bouldin thấp (0,7692) và CH cao trong các thí nghiệm. Đây là lựa chọn dễ trực quan hóa và dễ giải thích.
- Không gian 3D Ratio đạt Silhouette 0,3973 tại K=3, nhỉnh hơn 2D rất ít. Tuy nhiên, đặc trưng tỷ lệ là biến dẫn xuất từ Income và Spending, cần cân nhắc nguy cơ làm tăng trọng số của hai biến này.
- Không gian 3D Demographics tại K=6 đạt Silhouette khoảng 0,2843; đây là cấu hình cơ sở trước đó của project. Nó có thể vẫn hữu ích nếu mục tiêu nghiệp vụ cần phân nhóm theo tuổi.
- Không gian 4D/5D không cho thấy lợi ích rõ ràng trong lần chạy này. Gender mã hóa 0/1 không nên được đưa vào mô hình chính chỉ vì muốn tăng số chiều.
- Các điểm Silhouette giữa những không gian có số chiều/đặc trưng khác nhau không phải phép so sánh tuyệt đối; cần xem thêm biểu đồ, ý nghĩa biến và độ ổn định qua nhiều seed.

## Grid Search

Grid Search khảo sát K=2..8, init=random/k-means++, max_iter=60/120 và tol=1e-4. Silhouette là tiêu chí xếp hạng chính; các metric còn lại được dùng làm bằng chứng đối chiếu. Kết quả đầy đủ nằm trong `data/processed/grid_search_results.csv`.

## Khuyến nghị báo cáo

Nên trình bày 3D Demographics K=6 là mô hình cơ sở của notebook chính, sau đó dùng thí nghiệm mở rộng để chứng minh rằng không gian 2D/3D Ratio có Silhouette tốt hơn trong bộ dữ liệu này. Nếu ưu tiên khả năng trình bày, chọn 2D Core K=4; nếu ưu tiên giữ thông tin tuổi, tiếp tục phân tích 3D Demographics K=6. Không nên đổi mô hình chính chỉ dựa trên một lần lấy mẫu Silhouette.
