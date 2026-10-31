# Lab 3 | Dự đoán chuỗi thời gian bằng Machine Learning

**Trạng thái: khung đã dựng; phần phân tích, huấn luyện trên AEP và kết luận còn TODO.**
Nhóm có 6 người tính cả leader. Hạn theo yêu cầu: **02/11/2026**.
Phân công theo ngày, làm online bất đồng bộ; mỗi người tự chọn lúc làm và cập nhật sản phẩm trong ngày.

## Đọc trước
- `docs/TEAM_PLAN.md`: phân công 6 người, ngày bàn giao, phụ thuộc và lịch từng ngày.
- `docs/REQUIREMENTS_MAP.md`: đối chiếu đủ 7 bước, 4 nâng cao, 3 câu hỏi của đề.
- `docs/DATA_CONTRACT.md`: quy ước dữ liệu, split, feature và đánh giá chung.
- `CONTRIBUTING.md`: cách dùng nhánh, review và tích hợp.

## Dữ liệu đã sẵn
**Dữ liệu dùng chung v1 đã chuẩn bị và kiểm tra ngày 29/10.** Xem `docs/DATA_V1.md`.
TV02 chỉ làm EDA; TV03 làm features từ 30/10, không phải chờ EDA. Không có yêu cầu bàn giao của thành viên ngày 29/10.

## Lấy bài từ GitHub
```bash
git clone https://github.com/thvhieu/Time_series_lab3.git
cd Time_series_lab3
```
Dữ liệu v1 đã kèm sẵn. Repo chỉ gồm bài làm, dữ liệu, tài liệu và công cụ kiểm tra; không kèm môi trường Python/cache/file tạm.

## Bắt đầu trên Windows / PowerShell
Chạy tại thư mục gốc Lab3, dùng Python 3.12 (môi trường kiểm thử của khung).
```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/check_project.py --execute-notebooks
.\.venv\Scripts\python.exe -m jupyter lab
```
Chọn kernel của `.venv`. `requirements.txt` ghim các thư viện trực tiếp đã kiểm thử;
`requirements.lock.txt` lưu toàn bộ môi trường kiểm thử Python 3.12/Windows.

## Cấu trúc
```text
Lab3/
├── configs/experiment.json       # một cấu hình chung cho cả nhóm
├── data/
│   ├── raw/                     # CSV gốc + manifest SHA256
│   ├── interim/                 # dữ liệu kiểm tra/làm sạch, sinh lại bằng code
│   ├── processed/               # feature matrix, không chỉnh tay
│   └── external/                # nguồn bổ sung nếu được thống nhất
├── notebooks/                   # 00 tổng hợp + 01..05 cho 5 thành viên
├── src/lab3/                    # data, features, evaluation, modeling
├── scripts/                     # prepare_data.py, check_project.py
├── tests/                       # test thời gian, leakage, metrics, model smoke
├── docs/                        # lịch, phân công, hợp đồng và handoff
├── reports/{figures,tables}/     # kết quả thật + report.md để viết kết luận
├── models/                      # mô hình đã fit, sinh local
├── references/                  # đề bài PDF gốc
├── rules/                       # quy ước riêng của Lab 3
└── .github/workflows/ci.yml      # kiểm tra khung khi dùng GitHub
```
Dữ liệu đã tải sẵn đúng URL trong đề. CSV có 121.273 dòng, 4 dòng timestamp trùng
và 27 giờ danh nghĩa bị thiếu; thứ tự gốc chưa tăng dần. Xem hợp đồng trước khi EDA.

## Notebook của từng người
| Mã | Notebook | Vai trò |
|---|---|---|
| TV01 | `notebooks/00_lab3_report.ipynb` | Bạn - leader, tích hợp và báo cáo |
| TV02 | `notebooks/01_data_eda.ipynb` | Chỉ EDA trên dữ liệu v1 có sẵn |
| TV03 | `notebooks/02_features_split.ipynb` | Lag, rolling, split, thí nghiệm p |
| TV04 | `notebooks/03_xgboost_importance.ipynb` | XGBoost + feature importance |
| TV05 | `notebooks/04_baselines_tuning.ipynb` | Linear/RF + TimeSeriesSplit tuning |
| TV06 | `notebooks/05_evaluation_multistep.ipynb` | Đánh giá + multi-step |

Mỗi notebook có sẵn import/đọc dữ liệu/chia thời gian và các ô TODO theo đúng phần.
`src/` là tiện ích khởi đầu có test; chưa có kết quả thực nghiệm được điền sẵn.
Không chép kết quả ARIMA/SARIMA của Lab 2 sang dữ liệu AEP.

## Tái tạo dữ liệu v1 (đã kèm sẵn, chỉ chạy khi cần tái tạo)
```powershell
.\.venv\Scripts\python.exe scripts/prepare_data.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```
Sau khi nhóm hoàn thành, leader Restart Kernel -> Run All notebook 00 và điền checklist
nghiệm thu trong `docs/TEAM_PLAN.md`. Check scaffold không tự đánh giá chất lượng kết luận.
