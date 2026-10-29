# Làm việc nhóm - kế thừa cách làm notebook riêng của Lab 2
## Owner, nhánh và review
- TV01: `integration/report`; TV02: `feature/data-eda`; TV03: `feature/features-split`.
- TV04: `feature/xgboost`; TV05: `feature/baselines-tuning`; TV06: `feature/evaluation-multistep`.
- Repo chung: https://github.com/thvhieu/Time_series_lab3. Mỗi thành viên tạo nhánh riêng, mở PR vào main để leader review.
- Một người sửa một notebook. Đổi contract/config phải được TV01 và owner liên quan review.
- TV04 sửa modeling.py; TV05 gọi factory từ module đó và viết tuning trong notebook 04.

```powershell
git switch main
git pull --ff-only
git switch -c feature/data-eda
# Sửa file đúng owner và ghi handoff
.\.venv\Scripts\python.exe scripts/check_project.py
git add notebooks/01_data_eda.ipynb src/lab3/data.py docs/handoffs/TV02.md
git commit -m "feat: add data audit and EDA"
git push -u origin feature/data-eda
```
Sau khi clone repo, thực hiện pull/push trên nhánh cá nhân của Lab3. Không dùng remote của Lab1/Lab2.

## PR/handoff phải có
Input + data hash; lệnh tái tạo; file output; config/seed; số liệu thực; nhận xét;
TODO/blocker còn lại; người review và ngày review. Không chỉ gửi ảnh màn hình.
Không commit .venv, cache, model nặng hoặc dữ liệu trung gian. Bảng/hình nhỏ trong reports
được track khi có kết quả thật. Không đổi raw CSV để cho test qua.

## Tích hợp
Merge file notebook không tự ghép cell. TV01 tích hợp theo thứ tự đề vào notebook 00,
gom import, gọi src thay cho copy-paste helper, giữ tên config/index thống nhất.
TV06 Restart Kernel -> Run All bản tổng hợp; checklist nội dung do cả nhóm rà.
Nếu lưu output, chỉ giữ kết quả đúng đã chạy; bỏ traceback, biến thử và output quá lớn.

## Online bất đồng bộ
Theo dõi đầu ra theo ngày, không chia ca/định mức làm việc. Mỗi người cập nhật handoff trong ngày theo TEAM_PLAN; ghi DONE khi có sản phẩm và review. Nếu BLOCKED, nêu đầu vào còn thiếu và báo người phụ trách/leader.

## Nền dữ liệu có sẵn
Assistant đã chuẩn bị Data v1; đọc docs/DATA_V1.md. TV02 chỉ làm EDA; TV03 làm features từ 30/10. Mọi thành viên bắt đầu từ 30/10, không yêu cầu bàn giao ngày 29/10. Các file v1 đã whitelist để đi kèm repo; không tự sửa CSV, tái tạo bằng scripts/prepare_data.py.
