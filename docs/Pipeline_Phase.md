# Pipeline Phase – Lab03

Pipeline điều phối EDA → Preprocessing → Training → Evaluation → Experiments → Convergence → Visualization → Report.

Mỗi lần chạy được gán Run_ID. Với mỗi giai đoạn, hệ thống ghi trạng thái START/SUCCESS/FAILED, dữ liệu đầu vào, đầu ra, thời gian và lỗi (nếu có) vào `progress_results/progress_<Run_ID>.json`; nhật ký dạng dòng được ghi vào `logs/pipeline_<Run_ID>.log`.

Chạy bằng `python -m src.pipeline` tại thư mục gốc dự án.
