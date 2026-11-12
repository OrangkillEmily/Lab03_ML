# Tóm tắt kết quả Lab03\n\n## Kết quả mô hình\n\n- Số khách hàng: **15,079**\n- Số cụm được chọn: **6**\n- Inertia: **14905.6510**\n- Silhouette Score (mẫu tối đa 1.000 điểm): **0.2859**\n\n## Kích thước các cụm\n\n```text\n         Customer count
Cluster                
0                  2629
1                  2486
2                  2461
3                  2352
4                  2628
5                  2523\n```\n\n## Hồ sơ cụm\n\n```text\n         Age_count  Age_mean  Age_median  Annual Income_count  Annual Income_mean  Annual Income_median  Spending Score_count  Spending Score_mean  Spending Score_median  Gender_%_Female  Gender_%_Male
Cluster                                                                                                                                                                                                  
0             2629     34.12        33.0                 2629            59143.55               56232.0                  2629                41.74                   42.0            48.88          51.12
1             2486     35.84        35.0                 2486           141740.80              143512.5                  2486                78.38                   79.0            49.44          50.56
2             2461     72.61        73.0                 2461            78488.11               78191.0                  2461                22.68                   22.0            49.78          50.22
3             2352     40.67        41.0                 2352           155059.18              157199.0                  2352                22.82                   22.0            51.40          48.60
4             2628     74.59        75.0                 2628           160891.41              162428.0                  2628                57.68                   58.0            48.33          51.67
5             2523     66.58        67.0                 2523            65904.02               64788.0                  2523                78.16                   79.0            50.18          49.82\n```\n\n## Lưu ý diễn giải\n\nSilhouette Score được ước lượng trên mẫu, không phải toàn bộ dữ liệu. Cần kết hợp biểu đồ, kích thước cụm và trung bình đặc trưng để diễn giải; các số Cluster chỉ là mã nhóm, không phải thứ hạng.\n