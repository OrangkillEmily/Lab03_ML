# Lab03 - Evaluation Summary

## Model Results

- **n_samples:** 15079
- **n_features:** 3
- **n_clusters:** 6
- **inertia:** 14903.291482110133
- **silhouette_sample:** 0.2850621421447773
- **calinski_harabasz:** 6135.825619984679
- **davies_bouldin:** 0.9565635548402845
- **iterations:** 88
- **converged:** True
- **stop_reason:** centroid_shift <= tol

## Cluster Sizes

```text
         Customer count
Cluster                
0                  2617
1                  2485
2                  2471
3                  2367
4                  2644
5                  2495
```

## Cluster Profiles

```text
         Age_count  Age_mean  Age_median  Annual Income_count  Annual Income_mean  Annual Income_median  Spending Score_count  Spending Score_mean  Spending Score_median  Gender_%_Female  Gender_%_Male
Cluster                                                                                                                                                                                                  
0             2617     34.05        33.0                 2617            59563.63               56878.0                  2617                40.78                   41.0            49.33          50.67
1             2485     35.65        35.0                 2485           141295.63              142719.0                  2485                78.30                   79.0            49.26          50.74
2             2471     72.86        74.0                 2471            78976.24               78309.0                  2471                22.91                   22.0            49.78          50.22
3             2367     41.27        41.0                 2367           156024.63              158396.0                  2367                22.83                   22.0            51.16          48.84
4             2644     74.57        75.0                 2644           159837.30              161532.0                  2644                58.99                   59.0            48.71          51.29
5             2495     65.96        66.0                 2495            64426.78               62751.0                  2495                78.14                   79.0            49.70          50.30
```

## Interpretation Notes

- Silhouette càng cao thường thể hiện cụm càng tách biệt.
- Davies-Bouldin càng thấp thường càng tốt.
- Calinski-Harabasz càng cao thường càng tốt.
- Inertia giảm khi số cụm tăng, cần kết hợp Elbow để chọn K.
- Cần xem thêm kích thước và hồ sơ từng cụm.


## Model Selection Rationale

Mô hình chính sử dụng K = 6 với ba đặc trưng Age, Annual Income và Spending Score.

Trong phạm vi thử nghiệm K = 2..8, K = 6 đạt Silhouette Score cao nhất là 0.2843. Mô hình tạo ra 6 cụm có kích thước tương đối cân bằng, hỗ trợ phân tích khách hàng theo cả độ tuổi, thu nhập và hành vi chi tiêu.

Trong các thí nghiệm mở rộng:
- 2D Core (K = 4) đạt Silhouette 0.3955.
- 3D Ratio (K = 3) đạt Silhouette 0.3973.

Mặc dù hai cấu hình trên có Silhouette cao hơn, chúng sử dụng không gian đặc trưng khác với mô hình chính. Vì vậy, không thể chỉ dựa vào Silhouette để kết luận mô hình nào tốt hơn trong mọi mục tiêu phân tích.

K = 6 được giữ làm mô hình chính nhằm bảo toàn thông tin về tuổi khách hàng. Các cấu hình 2D Core và 3D Ratio được xem là những phương án thay thế trong phân tích mở rộng.
