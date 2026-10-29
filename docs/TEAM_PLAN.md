# PHÂN CÔNG & TIẾN ĐỘ LAB 3 - THEO NGÀY

**6 thành viên gồm leader | Thành viên làm từ 30/10–02/11/2026 | Online bất đồng bộ**

TV01 là bạn (leader); TV02–TV06 sẽ được điền tên thật trong docs/team.json.
**Dữ liệu dùng chung v1 đã được assistant chuẩn bị ngày 29/10. Ngày này không có yêu cầu bàn giao của thành viên.**
Mỗi người tự chọn lúc làm. Nhóm quản lý bằng sản phẩm bàn giao theo ngày, không chia ca,
không chấm công và không yêu cầu cả nhóm online cùng lúc.
Mục tiêu hoàn thiện bản tích hợp ngày **01/11**, dành **02/11** cho kiểm tra và nộp bài.
Leader xác nhận hạn trên LMS; ngày 02/11 thực hiện theo hạn chính thức của học phần.

## 1. Phần việc của từng người

| Người | Nhiệm vụ chính | Sản phẩm bàn giao | Reviewer |
|---|---|---|---|
| TV01 - bạn, leader | Chốt quy ước, tích hợp notebook 00, viết kết luận và câu 3, rà đề, nộp | Notebook tổng hợp chạy lại được + báo cáo đủ mục + xác nhận nộp | TV06 |
| TV02 | Chỉ EDA trên dữ liệu v1 có sẵn: mô tả, biểu đồ và nhận xét; đọc các giả định đã ghi sẵn | Notebook EDA có hình/nhận xét ngày 30/10; không phải chuẩn bị/làm sạch lại dữ liệu | TV03 |
| TV03 | Lag, rolling, X-y, split chung, khảo sát p và câu 2 | Feature v1 + tests + lựa chọn p/rolling dựa trên CV ngày 30/10 | TV04 |
| TV04 | XGBoost cơ sở, kiểm tra overfit, feature importance, hỗ trợ câu 3 | Code/model tái tạo được + importance + nhận xét ngày 31/10 | TV05 |
| TV05 | Naive/seasonal naive, Linear, RF, tuning XGB bằng TimeSeriesSplit | Bảng CV + best_params + cấu hình/model chốt ngày 31/10 | TV06 |
| TV06 | Metrics, biểu đồ, multi-step, so sánh và QA tích hợp | Khung đánh giá ngày 30/10; multi-step bản thử ngày 31/10; kết quả cuối ngày 01/11 | TV02 |

Mỗi người giữ notebook riêng và tự viết nhận xét phần mình. TV04 sở hữu modeling.py;
TV05 gọi factory đó và viết tuning trong notebook 04. Leader tích hợp, không làm lại phần của cả nhóm.

## 2. Lịch theo ngày

| Ngày | Việc cần hoàn thành | Đầu ra để chuyển sang ngày kế tiếp |
|---|---|---|
| **29/10** | Assistant chuẩn bị dữ liệu v1, kiểm tra chất lượng, loader và tài liệu dùng chung. Thành viên chưa có yêu cầu làm/bàn giao | Dữ liệu sẵn để TV02 và TV03 bắt đầu độc lập từ 30/10 |
| **30/10** | TV02 bắt đầu và hoàn thiện EDA từ dữ liệu v1. TV03 nhận ngay dữ liệu v1 để chốt features/split, khảo sát p và rolling trên CV bằng XGB cơ sở có sẵn. TV04/TV05 chuẩn bị pipeline, chạy thử khi nhận đầu vào. TV06 chuẩn bị metrics/plot/multi-step. TV01 tích hợp phần dữ liệu và phương pháp | Feature v1 + p/rolling + folds CV được chốt; EDA hoàn chỉnh; code model và đánh giá sẵn sàng |
| **31/10** | TV04 chạy XGB/importance; TV05 chạy Linear/RF và tuning trên train; TV03 hỗ trợ kiểm tra features; TV06 thử recursive multi-step bằng model cơ sở; TV02 review hình/nhận xét EDA; TV01 ghép các phần đã bàn giao | Cấu hình/model cuối được chốt từ CV; bàn giao code refit/predict, đúng thứ tự features; bản thử multi-step và nhận xét từng phần |
| **01/11** | TV06 chạy đánh giá cuối/multi-step với cấu hình đã chốt; TV04/TV05 hỗ trợ refit. TV03 rà index/leakage; TV02 rà phần EDA. TV01 hoàn thiện notebook tổng hợp, 3 câu hỏi và báo cáo; cả nhóm review | Notebook tổng hợp chạy sạch, bảng/hình/kết luận đầy đủ, đủ 7 bước + 4 nâng cao + 3 câu hỏi; khóa bản nộp |
| **02/11** | TV01 + TV06 kiểm tra bản nộp; cả nhóm xác nhận đóng góp và sửa lỗi phát hiện; TV01 nộp theo hạn LMS, lưu bằng chứng | Bản đã nộp mở được và có xác nhận; không dành ngày này cho thêm thí nghiệm mới |

Đây là hạn bàn giao theo ngày, không phải lịch có mặt. Phần nào ảnh hưởng người khác
thì chia sẻ ngay khi có bản dùng được, không giữ lại đợi tới lúc tổng hợp trạng thái.

## 3. Bảng công việc hằng ngày theo người

| Người | 29/10 | 30/10 | 31/10 | 01/11 | 02/11 |
|---|---|---|---|---|---|
| TV01 | Không yêu cầu bàn giao | Ghép dữ liệu + phương pháp | Tích hợp dần model và nhận xét | Hoàn thiện báo cáo, rà đủ đề | Kiểm tra và nộp |
| TV02 | Không yêu cầu bàn giao | Hoàn thiện EDA | Review hình/nhận xét EDA | Rà phần EDA trong báo cáo | Xác nhận phần mình |
| TV03 | Không yêu cầu bàn giao | Bàn giao features/p/rolling/CV | Hỗ trợ model, kiểm tra leakage | Rà split và câu 2 | Xác nhận phần mình |
| TV04 | Không yêu cầu bàn giao | Chuẩn bị fit/CV, chạy thử | Hoàn thiện XGB + importance | Hỗ trợ refit và câu 3 | Sửa lỗi nếu có |
| TV05 | Không yêu cầu bàn giao | Chuẩn bị Linear/RF/tuning | Hoàn thiện CV + chốt params | Hỗ trợ refit và câu 1 | Sửa lỗi nếu có |
| TV06 | Không yêu cầu bàn giao | Chuẩn bị plot và rollout | Multi-step bản thử | Đánh giá cuối + QA tích hợp | Kiểm tra gói nộp |

## 4. Phụ thuộc - tránh để người sau ngồi chờ

- **Data v1 (assistant đã chuẩn bị, 29/10) -> features/split (TV03, 30/10) -> model (TV04/TV05, 31/10) -> đánh giá cuối (TV06, 01/11).**
- TV02 và TV03 cùng bắt đầu từ 30/10, dùng dữ liệu v1 đã có. TV03 không phải chờ TV02 làm xong EDA.
- Phần chuẩn bị dữ liệu đã xong: xem docs/DATA_V1.md. TV02 không nhận việc tải/làm sạch/audit lại dữ liệu.
- TV03 dùng factory XGB cơ sở có sẵn cho khảo sát p trên cùng folds; không chờ model tune xong mới chọn features.
- TV04/TV05 chuẩn bị code song song; kết quả chính thức dùng Feature v1 đã chốt.
- TV06 có thể viết metrics/rollout trước; bản thử multi-step ngày 31/10 không phải kết quả cuối.
  Ngày 01/11 chạy lại với cấu hình chốt, không dùng actual tương lai trong một rollout.
- TV01 tích hợp dần từ 30/10; không đợi tất cả xong mới bắt đầu.
- Nếu đầu vào chưa bàn giao, cập nhật trạng thái **BLOCKED**, ghi rõ cần file/hàm nào;
  tiếp tục phần độc lập và báo leader trong lần cập nhật của ngày đó.

## 5. Cách phối hợp online - một lần cập nhật mỗi ngày

Từ ngày 30/10, mỗi thành viên tự cập nhật docs/handoffs/TVxx.md trong ngày làm việc, không có khung cập nhật cố định.
Đăng ký nhiệm vụ bằng Discussion/comment trên GitHub từ 30/10; leader chốt Assignee của từng Issue, không ép chốt vote ngày 29/10.
Leader đọc các handoff để biết tiến độ, không cần giám sát lúc online.

```text
Ngày:
Trạng thái: TODO / DOING / REVIEW / DONE / BLOCKED
Đã xong:
Link/file bàn giao + lệnh tái tạo:
Đang vướng / cần ai hỗ trợ:
Việc tiếp theo:
```

- DONE chỉ khi có sản phẩm mở/chạy lại được, có nhận xét và reviewer xác nhận; không chỉ báo phần trăm.
- Nếu trễ, báo phần đã có + phần còn lại + ngày dự kiến bàn giao. Leader điều phối theo đầu ra.
- Mỗi người sửa notebook riêng. Review bằng comment/tin nhắn/PR, không bắt buộc họp trực tiếp.
- TV02 hỗ trợ TV06 sau EDA; TV04 hỗ trợ TV05 sau phần XGB; giữ ngày 02/11 làm dự phòng nộp.

## 6. Hợp đồng bàn giao và kiểm tra cuối

- Notebook chạy độc lập từ kernel mới; raw CSV giữ nguyên; cấu hình chung trong configs/experiment.json.
- Chọn features/p/siêu tham số chỉ bằng CV trên train; holdout dùng cho đánh giá cuối sau khi cấu hình chốt.
- Prediction: Datetime,y_true,y_pred,model,protocol,horizon.
- Metrics: model,protocol,horizon,p,n_test,MAE,RMSE,MAPE,fit_seconds,seed.
  fit_seconds chỉ ghi thời gian chạy model cho báo cáo, không phải định mức làm việc của thành viên.
- Model one-step cùng test index; bảng multi-step riêng theo horizon/origin; kết quả ghi đúng MW/%.
- Ghi input/hash/config/seed, lệnh chạy, output và nhận xét thật; chưa chạy thì giữ TODO.
- [ ] Điền tên thật đủ 6 người, xác nhận hạn LMS.
- [ ] Đủ 7 bước cơ bản, 4 nâng cao, 3 câu hỏi.
- [ ] Không shuffle/leakage/tune trên test; ghi rõ giả định timestamp/DST.
- [ ] Notebook 00 Restart Kernel -> Run All thành công; hình và bảng hiển thị đầy đủ.
- [ ] Đối chiếu nguồn, ghi phiên bản thư viện, lưu gói nộp và xác nhận nộp.
