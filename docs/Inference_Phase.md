# Inference Phase – Lab03

Inference dùng `models/best_kmeans.pkl` và `data/processed/scaler.json`. Dữ liệu khách hàng mới gồm Age, Annual Income, Spending Score. Các giá trị được chuẩn hóa bằng mean/std của tập huấn luyện, không tính scaler mới từ một khách hàng đơn lẻ.

Sau đó mô hình K-means gọi `predict()` để gán Cluster gần nhất. Kết quả và thời gian dự đoán được lưu nối tiếp vào `reports/inference_history.csv`. Có thể chạy qua `python -m src.predict` hoặc giao diện `python -m streamlit run app.py`.
