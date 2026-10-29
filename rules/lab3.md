# Rule Lab 3 | cập nhật 29/10/2026
1. Chỉ dùng AEP_hourly.csv; Lab 2 là tham khảo cấu trúc, không phải dữ liệu Lab 3.
2. Không shuffle; giữ test_start trong configs/experiment.json cho mọi mô hình.
3. Lag_i(t) = y(t-i); rolling phải shift(1) trước khi rolling.
4. Không bfill, không nội suy hai chiều; không điền target thiếu để chấm điểm.
5. Chọn p, feature, siêu tham số chỉ trong train bằng TimeSeriesSplit.
6. Phân biệt one-step cập nhật quan sát với multi-step tại một origin cố định.
7. Ghi timestamp, protocol, seed, feature set, tham số và thời gian fit khi xuất kết quả.
8. Mỗi TV dùng nhánh và notebook riêng; mọi PR phải có code tái tạo và nhận xét.
9. Kiểm thử scaffold không đồng nghĩa bài phân tích/model cuối đã hoàn thành.
10. Lịch làm việc chỉ dùng ngày và sản phẩm bàn giao; không đặt khung online hay định mức giờ làm. Cập nhật handoff bất đồng bộ mỗi ngày làm việc; giữ nguyên các đơn vị thời gian của dữ liệu.
11. Dữ liệu dùng chung v1 do assistant chuẩn bị ngày 29/10; TV02 chỉ EDA, TV03 dùng ngay dữ liệu này để làm features. Thành viên bắt đầu từ 30/10, không có yêu cầu bàn giao ngày 29/10.
12. Repo chung là https://github.com/thvhieu/Time_series_lab3; chỉ đưa file dự án cần thiết lên GitHub. Không commit môi trường, work/outputs tạm, cache, log, backup hoặc thông tin xác thực.
