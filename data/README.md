# Dữ liệu dùng chung v1 - đã chuẩn bị ngày 29/10/2026
Raw AEP_hourly.csv giữ nguyên byte nguồn; raw/manifest.json ghi URL và SHA256.
Chuẩn bị dữ liệu do assistant hoàn thành, không giao lại cho TV02.
- interim/AEP_hourly_clean.csv: Datetime,y,observed; trục giờ tăng dần và duy nhất.
- interim/data_quality.json: số liệu chất lượng, giả định và ranh giới train/test.
- interim/duplicate_timestamps.csv: các dòng nguồn có timestamp trùng để đối chiếu.
- interim/missing_hours.csv: danh sách giờ danh nghĩa thiếu quan sát.
- interim/manifest.json: hash của các file v1; loader kiểm tra trước khi đọc.
Missing y vẫn trống và observed=False, không dùng làm ground truth.
History chỉ ffill bên trong feature builder; không bfill/nội suy hai chiều.
Xem docs/DATA_V1.md để đọc nhanh; tái tạo bằng python scripts/prepare_data.py.
Từ 30/10 TV02 làm EDA, TV03 nhận trực tiếp v1 để làm features/split.
