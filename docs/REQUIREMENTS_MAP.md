# Đối chiếu đề bài Lab 3
Nguồn: `references/excercise_ML_models.pdf`, trang duy nhất. Các hàng dưới là yêu cầu
bài tập được chuyển thành checklist nhóm; không phải kết quả bài làm đã hoàn thành.

| Yêu cầu trong đề | Owner | Vị trí | Bằng chứng nghiệm thu |
|---|---|---|---|
| B1 Chuẩn bị dữ liệu | Đã chuẩn bị bởi assistant; TV02 chỉ EDA | docs/DATA_V1.md + notebook 01 | V1 có audit/hash/loader; TV02 bổ sung biểu đồ và nhận xét |
| B2 Tạo lag features | TV03 | notebook 02 | X(t) chỉ chứa quá khứ, bảng minh họa 5 dòng |
| B3 Chia X-y | TV03 | notebook 02 | Không chứa target trong X, shape/index khớp |
| B4 Train/test theo thời gian | TV03 | notebook 02 + config | Một ranh giới chung, không shuffle |
| B5 Huấn luyện XGBoost | TV04 | notebook 03 | Fit trên train, seed/params/runtime |
| B6 Dự đoán/đánh giá | TV06 | notebook 05 | MAE/RMSE/MAPE, cùng timestamp và MW |
| B7 Feature importance | TV04 | notebook 03 | Gain importance, top lags + nhận xét giới hạn |
| Nâng cao 1: rolling mean/std | TV03 | notebook 02 | shift(1), so sánh ablation bằng CV |
| Nâng cao 2: multi-step | TV06 | notebook 05 | Recursive h=1/6/24, bảng lỗi theo horizon |
| Nâng cao 3: TimeSeriesSplit tuning | TV05 | notebook 04 | 3 folds trong train, CV table và best_params |
| Nâng cao 4: Random Forest | TV05 | notebook 04 | Cùng features/holdout và ngân sách ghi rõ |
| Câu 1: XGB so với Linear Regression | TV05 + TV06 | notebook 04/05, report | So sánh số liệu thật, phi tuyến, overfit/runtime |
| Câu 2: Tăng số lag p | TV03 | notebook 02, report | p=6/12/24/48/168 trên cùng folds, số chiều/thời gian/lỗi |
| Câu 3: Khi nào XGB hơn ARIMA | TV01 + TV04 | notebook 00, report | Lập luận phù hợp đặc trưng; không kết luận thắng ARIMA khi chưa chạy trên AEP |
| Dataset AEP_hourly.csv | Đã chuẩn bị bởi assistant | data/raw + data/interim | Hash/columns đúng, v1 được kiểm tra; không giao lại cho TV02 |

Nhóm bổ sung naive/seasonal naive để có mốc so sánh. Đề không buộc huấn luyện ARIMA
trong Lab 3; câu 3 là thảo luận. Nếu có so sánh số liệu ARIMA, phải chạy lại trên AEP
với cùng protocol; không dùng số liệu a10/AirPassengers của Lab 2.
