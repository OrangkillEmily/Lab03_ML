# Tracking, Visualization và Reports

- `src/tracking.py`: tạo Run_ID, cập nhật progress JSON, ghi log và kết thúc run.
- `src/visualize.py`: gom các ảnh trong `docs/images/` vào `reports/figures/<Run_ID>/`.
- `src/report.py`: đọc kết quả, progress và ảnh để tạo báo cáo `.md`, `.html`, `.pdf`.
- `reports/metrics_summary.csv`: lưu một dòng tóm tắt cho mỗi Run_ID.
- `reports/inference_history.csv`: lịch sử phân cụm khách hàng mới.

Báo cáo được tạo tự động ở cuối `python -m src.pipeline`. Có thể tạo lại bằng `python -m src.report --run-id <RUN_ID>`.
