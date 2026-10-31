# Dữ liệu dùng chung v1 - đã chuẩn bị ngày 29/10/2026

**Assistant đã hoàn thành phần chuẩn bị dữ liệu. TV02 chỉ làm EDA; TV03 nhận v1 để làm features từ 30/10. Không có yêu cầu thành viên làm hoặc bàn giao trong ngày 29/10.**

## Đã có những gì?

| File trong dự án | Nội dung |
|---|---|
| `data/raw/AEP_hourly.csv` | Toàn bộ CSV nguồn, giữ nguyên byte và SHA256 |
| `data/interim/AEP_hourly_clean.csv` | Dữ liệu dùng chung, sắp thời gian và gộp timestamp trùng |
| `data/interim/data_quality.json` | Số liệu chất lượng, quy tắc xử lý và ranh giới train/test |
| `data/interim/duplicate_timestamps.csv` | 8 dòng gốc thuộc 4 timestamp trùng để đối chiếu |
| `data/interim/missing_hours.csv` | 27 giờ danh nghĩa thiếu quan sát |
| `data/interim/manifest.json` | Hash của 4 file v1; loader kiểm tra trước khi đọc |

## Kết quả chuẩn bị trên toàn bộ dữ liệu thật

- Nguồn: [AEP Hourly Energy Consumption theo đề](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption?select=AEP_hourly.csv).
- Đích: AEP_MW, đơn vị **MW**; đây không phải dữ liệu mẫu hay một phần cắt nhỏ.
- 121.273 dòng gốc; phạm vi 01/10/2004 đến 03/08/2018.
- 4 dòng timestamp bị lặp so với các nhãn đã có; audit chứa đủ 8 dòng của các cặp trùng.
- Sau gộp mean tại cùng nhãn giờ: **121.269 target có quan sát**.
- Sau bổ sung trục giờ danh nghĩa đầy đủ: **121.296 dòng**, gồm 27 giờ thiếu target.
- Raw SHA256: `109d122f7b485555c609eecdec2cd5a03172e0a08acd358d863acc93eb452585`.

SHA256 chuẩn xác được loader đối chiếu trực tiếp với `data/raw/manifest.json`; manifest là nguồn máy đọc có thẩm quyền cho hash.

## Schema dùng chung

| Cột | Ý nghĩa |
|---|---|
| `Datetime` | Giờ địa phương danh nghĩa, tăng dần, duy nhất; CSV không cung cấp timezone |
| `y` | AEP_MW sau khi gộp trùng bằng mean; để trống ở giờ không có quan sát |
| `observed` | True khi có target gốc; False ở 27 giờ thiếu |

**Không điền target thiếu để làm đáp án.** Hàm tạo features hiện có chỉ ffill history từ quá khứ, sau đó shift để tạo lag/rolling. Không bfill, không nội suy hai chiều.

Timestamp không có timezone; quy tắc gộp mean làm mất phân biệt hai quan sát cùng nhãn giờ. V1 không khẳng định đã khôi phục DST hoặc đổi sang UTC. Đây là giả định đã ghi sẵn để nhóm nêu trong báo cáo, không phải việc làm sạch lại giao cho TV02.

## Train/test chung - không đổi theo từng người

- Ranh giới cố định: **2015-10-27 05:00:00**.
- Train: 97.036 giờ danh nghĩa, trong đó 97.012 target có quan sát.
- Test: 24.260 giờ danh nghĩa, trong đó 24.257 target có quan sát.
- Các số trên là trước warmup và lọc features; TV03 ghi thêm số mẫu thực tế sau tạo X-y.
- Giữ warmup 168 giờ cho mọi p, không shuffle, không chọn p/rolling/tham số bằng test.

## Cách dùng

Notebook đã được chuyển sang loader v1. Phần import/setup có sẵn; TV02 không cần viết lại pipeline.

```python
from lab3.data import load_prepared

frame, quality = load_prepared(
    ROOT / "data/interim",
    raw_path=ROOT / "data/raw/AEP_hourly.csv",
)
# frame: DatetimeIndex, cột y và observed
```

Tái tạo khi cần, từ thư mục gốc Lab3:

```powershell
python scripts/prepare_data.py
python scripts/check_project.py --execute-notebooks
```

## Phân công sau khi v1 đã sẵn

- **TV02, ngày 30/10:** chỉ EDA - mô tả dữ liệu, hình xu hướng/chu kỳ/phân phối và nhận xét. Không phải tải, làm sạch hay bàn giao Data v1.
- **TV03, ngày 30/10:** nhận v1 ngay để làm lag/rolling/X-y/split, khảo sát p và câu 2. Không cần chờ EDA của TV02.
- **TV04/TV05:** chuẩn bị mô hình song song, dùng Feature v1 của TV03 cho kết quả chính thức.
- **Ngày 29/10:** phần nền tảng đã được assistant làm; không đặt deadline thành viên.

Chưa có kết quả EDA/model cuối được điền thay nhóm. Đây là nền dữ liệu dùng chung đã chuẩn bị để nhóm bắt đầu thuận lợi.
