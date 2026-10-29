# Hợp đồng dữ liệu & đánh giá | v1 - 29/10/2026

## Dữ liệu nguồn và chất lượng đã quan sát
- AEP_hourly.csv, `Datetime` và `AEP_MW`; đích là mức công suất tiêu thụ theo giờ (MW), không phải MWh.
- 121.273 dòng gốc; 2004-10-01 01:00:00 đến 2018-08-03 00:00:00.
- 4 dòng timestamp trùng; 27 giờ danh nghĩa bị thiếu; file gốc không sắp tăng dần.
- `data/raw/manifest.json` ghi hash và URL; không sửa byte CSV gốc.
- Timestamp không có timezone. Có thể liên quan DST nhưng không suy ra nguyên nhân chỉ từ CSV.
  Khung dùng **trục giờ địa phương danh nghĩa**, không tự gán UTC/timezone hoặc khẳng định đã phục hồi DST.

## Quy tắc làm sạch dùng chung
1. Parse datetime; kiểm tra đúng đầu giờ, numeric dương/hữu hạn; sắp thời gian tăng dần.
2. Gộp timestamp trùng bằng mean tại cùng nhãn giờ; công khai đây là lựa chọn mất phân biệt hai quan sát.
3. Reindex mỗi giờ. Giữ `y=NaN`, `observed=False` ở giờ thiếu.
4. Chỉ history tạo feature dùng `ffill()` từ quá khứ; không bfill, không nội suy hai chiều.
5. Giờ thiếu target không dùng để fit/chấm; không coi giá trị được điền là ground truth.
   Audit và quy tắc đã có trong docs/DATA_V1.md; TV02 chỉ dùng thông tin này để mô tả giới hạn khi làm EDA.

## Định nghĩa supervised và split cố định
- Dự báo y(t) tại cuối t-1: `lag_i(t)=y(t-i)`, tương đương dự báo y(t+1) từ quá khứ khi đổi chỉ số.
- p mặc định 24; thử p=6/12/24/48/168 (lag liên tiếp 1..p).
- Rolling 24/168 giờ tính trên `y.ffill().shift(1)`; std dùng ddof=1.
- Warmup cố định 168 giờ cho **mọi p**, giữ tập timestamp đánh giá như nhau.
- Ranh giới lấy 80% **toàn trục giờ sạch trước khi bỏ NaN/warmup**.
- `test_start = 2015-10-27 05:00:00`; train < mốc, test >= mốc. Không tính lại 80% theo từng feature table.
- Tất cả X/y dùng cùng DatetimeIndex và target MW gốc. Không chứa cột y/observed trong X.

## One-step, CV và holdout
- One-step rolling-origin: model fit trên train; dự báo từng giờ test được dùng actual đã quan sát
  ở các giờ test trước đó làm lag. Đây không phải forecast toàn test tại một origin cố định.
- TimeSeriesSplit chỉ nhận phần train; 3 folds, mỗi validation 672 giờ nếu đủ dữ liệu.
  Mọi p dùng cùng folds/timestamp. Chọn bằng RMSE CV; test giữ lại tới khi freeze cấu hình.
- Causal precomputed lag/rolling phù hợp one-step; không fit scaler/imputer bằng cả train+test.
  Nếu thêm scaler, đưa vào Pipeline và fit riêng trong từng fold. Cây không cần scale.
- Multi-step direct có label tương lai sẽ cần purge/gap theo horizon; không dùng split one-step
  một cách máy móc. Thiết kế khung hiện tại là **recursive**, fit model một bước rồi lặp.

## Multi-step (TV06)
- Horizon 1, 6, 24; dùng origin cố định, cập nhật lag/rolling bằng prediction trong từng rollout.
- Không đưa actual của bất kỳ bước tương lai nào trong rollout vào feature.
- Nếu đánh giá nhiều origin, lịch sử quan sát chỉ được cập nhật tới từng origin mới.
- Các model dùng cùng origin/horizon; chấm ở target gốc có quan sát, ghi số mẫu theo horizon.
- Bảng multi-step riêng với one-step; ghi rõ không trộn hai protocol trong một hạng mục so sánh.

## Output contract
- Prediction: `Datetime,y_true,y_pred,model,protocol,horizon`; tăng dần, không trùng nhãn.
- Metrics: `model,protocol,horizon,p,n_test,MAE,RMSE,MAPE,fit_seconds,seed`.
- MAE/RMSE đơn vị MW; MAPE đơn vị %. Hàm chung bỏ actual=0 khi tính MAPE và báo số điểm hợp lệ.
  Loader AEP hiện từ chối giá trị không dương nên MAPE AEP dùng được mọi actual còn lại.
- Lưu config, model params, feature list, p, rolling, data hash, library versions cạnh mỗi run.
- Naive lag_1, seasonal-naive lag_24/lag_168, Linear, RF và XGB dùng cùng index để so sánh.

## Tài liệu phương pháp
- [Nguồn dữ liệu theo đề](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption?select=AEP_hourly.csv)
- [TimeSeriesSplit chính thức](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
- [XGBoost sklearn API](https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBRegressor)
