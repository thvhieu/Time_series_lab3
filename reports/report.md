# BÁO CÁO LAB 3: DỰ BÁO CHUỖI THỜI GIAN PHỤ TẢI ĐIỆN (`AEP_hourly`) BẰNG MACHINE LEARNING

## 1. Chuẩn bị Dữ liệu & Phân tích Khám phá (EDA)
- **Nguồn dữ liệu:** `data/raw/AEP_hourly.csv` (`121.273` dòng gốc, từ `2004-10-01 01:00:00` đến `2018-08-03 00:00:00`, đơn vị **MW**).
- **Làm sạch nhân quả:** Sắp xếp thời gian tăng dần, gộp 4 mốc trùng lặp bằng trung bình (`mean`), chuẩn hóa lên lưới giờ liên tục `121.296` giờ (ghi nhận `27` giờ khuyết với `observed=False`). Chỉ sử dụng `ffill()` một chiều từ quá khứ khi tạo biến trễ; tuyệt đối không dùng `bfill`, không nội suy hai chiều và không chấm điểm trên các giờ khuyết.
- **Đặc trưng thống kê & Chu kỳ:**
  - Toàn chuỗi quan sát (`121.269` giờ) có `mean = 15.499,65 MW`, `std = 2.591,32 MW`, `min = 9.581 MW`, `max = 25.695 MW`. Tập `Train` (`mean = 15.695,56 MW`) cao hơn tập `Test` (`mean = 14.716,17 MW`) khoảng `6,2%`.
  - Chuỗi thể hiện rõ **3 chu kỳ lồng nhau**: chu kỳ ngày (`24h`, đáy `04:00` sáng, đỉnh `19:00–20:00` tối), chu kỳ tuần (`168h`, cuối tuần giảm `~9,8%`) và chu kỳ năm (2 đỉnh vào mùa đông tháng 1–2 và mùa hè tháng 7–8).
  - Hệ số tự tương quan (ACF) đạt cực đại tại `lag_1` (`0,9751`), `lag_24` (`0,8700`) và `lag_168` (`0,7679`).

## 2. Thiết kế Lag Features, Rolling Statistics & Chia Tập Dữ liệu
- **Công thức đặc trưng nhân quả:**
  - Biến trễ: $\text{lag}_i(t) = y_{t-i}$ với $i = 1, \dots, p$ (mặc định $p=24$).
  - Thống kê trượt (`roll_mean_24`, `roll_std_24`, `roll_mean_168`, `roll_std_168`): Tính trên chuỗi `y.ffill().shift(1)` với `ddof=1`, đảm bảo tại thời điểm $t$ chỉ dùng thông tin đến $t-1$.
- **Ablation Study (Rolling Features):** Trên 3 folds `TimeSeriesSplit(n_splits=3, test_size=672)`, việc bổ sung 4 biến rolling vào 24 biến lag giúp giảm `CV RMSE` từ `251,14 MW` xuống `250,54 MW` và `CV MAE` từ `194,38 MW` xuống `194,14 MW`.
- **Chia `Train / Test` theo thời gian:** Cố định `warmup = 168` giờ và mốc cắt `test_start = 2015-10-27 05:00:00` (80% đầu làm `Train`: `96.844` mẫu; 20% cuối làm `Test`: `24.257` mẫu quan sát thực tế).

## 3. Mô hình XGBoost Cơ sở, Kiểm soát Overfit & Feature Importance
- **XGBoost cơ sở (`n_estimators=200, max_depth=4, learning_rate=0.05, seed=42`):**
  - Kiểm tra trên 3 folds `TimeSeriesSplit` trong tập `Train`: `train_RMSE = 261,15 MW` so với `val_RMSE = 250,20 MW` (`gap = -10,95 MW`), xác nhận mô hình không bị quá khớp (*overfit*).
  - Đánh giá trên tập `Test` (`24.257` mẫu): đạt `MAE = 185,42 MW`, `RMSE = 245,13 MW`, `MAPE = 1,28%` (`fit_seconds = 5,70s`).
- **Feature Importance (`Gain`):**
  - `lag_1` chiếm **73,19%** tổng Gain do quán tính phụ tải ngắn hạn rất lớn; tiếp theo là các biến trễ chu kỳ ngày (`lag_22` 5,24%, `lag_23` 4,92%, `lag_3` 3,64%, `lag_21` 2,62%).
  - Các biến trễ liền kề có tương quan mạnh nên cây quyết định ưu tiên chọn một biến đại diện ở nút gốc; điểm `Gain` phản ánh mức giảm hàm mất mát thống kê, không suy ra quan hệ nhân quả vật lý.

## 4. Các Mô hình Baselines, Random Forest & Tuning XGBoost
- **Tuning XGBoost trên `TimeSeriesSplit(n_splits=3, test_size=672)`:**
  - Khảo sát 5 cấu hình trong tập `Train`, cấu hình tối ưu là **`max_depth = 6, learning_rate = 0.05, n_estimators = 200`** đạt `mean_RMSE = 203,19 MW` (`std_RMSE = 9,99 MW`), giảm `47,01 MW RMSE` so với cấu hình cơ sở.

## 5. Kết quả Đánh giá Holdout One-step & Multi-step Đệ quy

### 5.1. Bảng kết quả One-step (`h = 1`, `24.257` mẫu Test)
| Mô hình | Giao thức | Horizon | Số mẫu | MAE (MW) | RMSE (MW) | MAPE (%) | Thời gian fit (s) |
|---|---|---:|---:|---:|---:|---:|---:|
| `linear_regression` | `rolling_origin_one_step_observed_history` | 1 | 24.257 | **147,48** | **203,06** | **1,01** | 0,14 |
| `xgboost_tuned` | `rolling_origin_one_step_observed_history` | 1 | 24.257 | 152,94 | 205,96 | 1,06 | 2,60 |
| `random_forest` | `rolling_origin_one_step_observed_history` | 1 | 24.257 | 152,05 | 207,66 | 1,04 | 181,72 |
| `xgboost_base` | `rolling_origin_one_step_observed_history` | 1 | 24.257 | 185,42 | 245,13 | 1,28 | 5,70 |
| `naive_lag1` | `rolling_origin_one_step_observed_history` | 1 | 24.257 | 407,68 | 525,64 | 2,81 | 0,00 |
| `seasonal_naive_24` | `rolling_origin_one_step_observed_history` | 1 | 24.257 | 921,07 | 1.220,87 | 6,24 | 0,00 |
| `seasonal_naive_168` | `rolling_origin_one_step_observed_history` | 1 | 24.257 | 1.432,69 | 1.906,76 | 9,62 | 0,00 |

### 5.2. Bảng kết quả Multi-step Đệ quy (`h = 1, 6, 24` trên `1.007` origins cố định)
| Mô hình | Horizon $h=1$ RMSE (MAPE) | Horizon $h=6$ RMSE (MAPE) | Horizon $h=24$ RMSE (MAPE) |
|---|---:|---:|---:|
| `xgboost_tuned` | 162,72 MW (1,02%) | **742,80 MW (3,56%)** | 1.038,91 MW (5,93%) |
| `linear_regression` | **140,09 MW (0,89%)** | 825,85 MW (4,17%) | **969,57 MW (5,94%)** |
| `xgboost_base` | 181,87 MW (1,16%) | 905,89 MW (4,36%) | 1.029,71 MW (6,30%) |
| `seasonal_naive_24` | 1.043,17 MW (6,07%) | 1.337,80 MW (6,65%) | 991,76 MW (5,77%) |

## 6. Trả lời 3 Câu hỏi Thảo luận
1. **Câu 1 (XGBoost so với Linear Regression):** Ở dự báo 1 bước ($h=1$) chỉ dùng biến trễ, `Linear Regression` (`RMSE = 203,06 MW`) nhỉnh hơn nhẹ so với `XGBoost Tuned` (`RMSE = 205,96 MW`) vì biến thiên 1 giờ mang tính tự hồi quy tuyến tính liên tục. Tuy nhiên, khi dự báo đệ quy nhiều bước ($h=6$), **`XGBoost Tuned` (`RMSE = 742,80 MW`) vượt qua `Linear Regression` (`RMSE = 825,85 MW`) tới `83,05 MW RMSE`** nhờ khả năng học ngưỡng phi tuyến và kiềm chế độ trôi ngoại suy đệ quy.
2. **Câu 2 (Ảnh hưởng khi tăng số bước trễ $p$):** Khi tăng $p$ từ $6 \to 12 \to 24 \to 48$, `CV RMSE` giảm mạnh từ `282,17 MW` xuống `213,65 MW` (-24,3%) nhờ bao phủ đủ chu kỳ ngày 24h–48h. Tuy nhiên từ $p=48 \to 168$, `CV RMSE` chỉ giảm thêm `0,90 MW` (`212,75 MW`) trong khi số chiều tăng từ 52 lên 172 cột và thời gian huấn luyện tăng gấp ~3,5 lần.
3. **Câu 3 (Khi nào XGBoost phù hợp hơn ARIMA?):** `XGBoost` phù hợp hơn khi chuỗi có quan hệ phi tuyến, đa chu kỳ với độ trễ dài ($24\text{h}, 168\text{h}, 8.760\text{h}$), có nhiều biến ngoại sinh (thời tiết, lịch, ngày lễ) và quy mô mẫu lớn (>100.000 dòng). Ngược lại, `ARIMA` có ưu thế khi chuỗi ngắn, thuần tuyến tính hoặc cần ngoại suy xu hướng vượt ngoài miền giá trị lịch sử $[\min, \max]$ của tập huấn luyện.

## 7. Tài liệu Tham khảo
- Đề bài: `references/excercise_ML_models.pdf`
- Dataset: [Hourly Energy Consumption (`AEP_hourly.csv`)](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption?select=AEP_hourly.csv)
- Scikit-learn: [`TimeSeriesSplit`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
- XGBoost: [`XGBRegressor` Documentation](https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBRegressor)
