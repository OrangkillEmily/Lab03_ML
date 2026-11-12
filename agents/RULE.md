# Project Rules

- Chạy lệnh từ thư mục gốc dự án bằng `python -m src.<module>`.
- Không sửa dữ liệu gốc trong `data/raw/`.
- Dùng cùng scaler đã fit từ tập huấn luyện khi inference.
- Mỗi lần chạy pipeline phải có Run_ID riêng.
- Không ghi đè lịch sử experiment, progress, logs, inference.
- Không dùng Customer ID làm đặc trưng.
- Khi kết luận, kết hợp nhiều metric và diễn giải nghiệp vụ.
