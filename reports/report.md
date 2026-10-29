# Báo cáo Lab 3 - Dự đoán AEP bằng Machine Learning
**Bản khung - kết quả bên dưới do nhóm điền sau khi chạy thực nghiệm.**
## Thành viên và đóng góp
TODO: tên thật 6 người, mã SV, phần đảm nhiệm và nguồn dữ liệu.
## 1. Dữ liệu v1 đã chuẩn bị + EDA (TV02)
Nền dữ liệu đã được chuẩn bị, xem docs/DATA_V1.md và data/interim/data_quality.json.
TODO TV02: chỉ bổ sung EDA, biểu đồ và nhận xét; mô tả ngắn các giả định đã có, không làm sạch lại.
## 2. Lag features và rolling (TV03)
TODO: công thức, bảng X-y minh họa, shift trước rolling, warmup, feature ablation.
## 3. Tách X-y và train/test (TV03)
TODO: ngày split, số mẫu, protocol one-step, CV và chứng cứ không leakage.
## 4. XGBoost cơ sở (TV04)
TODO: params/seed/runtime, cách fit, fit train không dùng test chọn params.
## 5. Baseline / Linear / Random Forest và tuning (TV05)
TODO: bảng CV, best params, cấu hình chốt và ngân sách so sánh.
## 6. Dự đoán và đánh giá (TV06)
TODO: bảng MAE/RMSE/MAPE, dự báo-vs-thực, phần dư và nhận xét cùng holdout.
## 7. Feature importance (TV04)
TODO: top lag, gain, tính tương quan giữa lag và giới hạn diễn giải nhân quả.
## 8. Multi-step (TV06)
TODO: recursive, origin/horizon, không actual tương lai, bảng lỗi h=1/6/24.
## 9. Ba câu hỏi (TV01 tích hợp)
### 9.1 XGBoost so với Linear Regression (TV05 + TV06)
TODO: số liệu thực + năng lực phi tuyến, overfit, tốc độ, giới hạn kết luận.
### 9.2 Điều gì xảy ra khi tăng p? (TV03)
TODO: bảng CV cùng folds cho 6/12/24/48/168, độ phức tạp, lỗi và thời gian.
### 9.3 Khi nào XGBoost phù hợp hơn ARIMA? (TV01 + TV04)
TODO: lập luận có điều kiện; không suy diễn chiến thắng từ Lab 2 khác dataset.
## 10. Kết luận, giới hạn và tái tạo
TODO: kết luận bám số liệu, giới hạn timezone/DST/holdout, cấu hình và lệnh tái tạo.
## Tài liệu tham khảo
- Đề bài: references/excercise_ML_models.pdf.
- Dataset: https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption?select=AEP_hourly.csv
- TimeSeriesSplit: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
- XGBoost: https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBRegressor
