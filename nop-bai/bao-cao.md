# Báo Cáo Lab Day 21 - CI/CD cho AI Systems

| | |
|---|---|
| Họ và tên | Trần Thu Phương |
| MSSV | 2A202602734 |
| Lớp / Khóa | K4 |
| Repo GitHub | https://github.com/TranThuPhuong1111/K4-L3-DAY21-TranThuPhuong-2A202602734-CI-CD-for-AI-Systems |
| Ngày nộp | 07/10/2026 |

---

## 1. Bộ Siêu Tham Số Đã Chọn và Lý Do

| Lần chạy | n_estimators | learning_rate | max_depth | f1_score | accuracy |
|---|---|---|---|---|---|
| 1 | 100 | 0.1 | 3 | 0.7109 | 0.878 |
| 2 | 50 | 0.05 | 2 | 0.6051 | 0.846 |
| 3 | 200 | 0.1 | 5 | 0.7149 | 0.874 |
| 4 | 200 | 0.05 | 4 | **0.7222** | **0.880** |
| 5 | 300 | 0.2 | 4 | 0.7085 | 0.870 |

**Bộ siêu tham số đã chọn:** `n_estimators=200`, `learning_rate=0.05`, `max_depth=4`.

**Lý do:** Lần 4 có F1 cao nhất (0.7222). Thứ hạng accuracy không khớp với F1: lần 1 có accuracy cao hơn lần 3 nhưng F1 thấp hơn. Accuracy chỉ dao động 0.846 - 0.880, còn F1 dao động 0.605 - 0.722, nên F1 phân biệt mô hình rõ hơn. Learning rate nhỏ cần nhiều cây để bù lại: lần 2 vừa ít cây vừa nông nên học thiếu (F1 dưới ngưỡng), còn lần 5 với learning rate 0.2 và 300 cây không tốt hơn, có thể do bước học quá lớn khiến mô hình bắt đầu quá khớp.

---

## 2. Vì Sao Ngưỡng Chất Lượng Đặt Trên F1 Chứ Không Phải Accuracy

Chỉ 24,8% mẫu thuộc lớp thu nhập trên 50K, nên mô hình luôn đoán "thu nhập thấp" vẫn đạt accuracy 0.752 dù không tìm được người thu nhập cao nào. F1 của lớp dương kết hợp precision và recall trên chính lớp thiểu số, nên mô hình vô dụng đó có F1 bằng 0. Tôi không dùng `average="weighted"` hay `"macro"`, vì chúng cộng thêm điểm của lớp đa số vốn dễ đạt, làm ngưỡng 0.65 mất ý nghĩa.

---

## 3. Khó Khăn Gặp Phải và Cách Giải Quyết

Không có khó khăn nào ảnh hưởng đến kết quả cuối cùng. Các lỗi nhỏ về môi trường và cấu hình phát sinh trong quá trình làm đều đã được xử lý.

---

## 4. So Sánh Bước 2 và Bước 3 (bắt buộc, 2 - 3 câu)

| | f1_score | accuracy |
|---|---|---|
| Bước 2 (chỉ `train_batch1`) | 0.7222 | 0.880 |
| Bước 3 (thêm `train_batch2`) | 0.7339 | 0.884 |

**Nhận xét:** Gấp đôi dữ liệu chỉ làm F1 tăng 0.0117, vì `train_batch2` có cùng phân phối với dữ liệu cũ (tỷ lệ lớp dương 24,80% so với 24,77%) nên mang thêm ít thông tin. Theo confusion matrix, mô hình mới chỉ nhận ra thêm đúng 2 trong 124 người thu nhập cao của holdout (78 lên 80), nên mức tăng nằm trong biên dao động và không chứng minh thêm dữ liệu luôn tốt hơn. Điều được kiểm chứng là commit dữ liệu đã tự kích hoạt cả bốn job và triển khai model mới lên VM.

---

## 5. Phần Bonus Đã Thực Hiện (nếu có)

- [x] Bonus 1 - Tracking MLflow từ xa với DagsHub: repo DagsHub mirror từ GitHub; job Train ghi run lên `<repo>.mlflow` qua 3 secret `MLFLOW_TRACKING_*`, gắn tag commit (ảnh `06a`, `06b`).
- [x] Bonus 2 - Điều chỉnh ngưỡng quyết định: ngưỡng 0.30 cho F1 0.7341 so với 0.7222 ở ngưỡng 0.5 (Bước 2).
- [x] Bonus 3 - Báo cáo precision / recall tự động: `detail.txt` chứa confusion matrix và precision/recall; recall lớp thu nhập cao chỉ 0.63 so với precision 0.85, nên lỗi bỏ sót người thu nhập cao là lỗi phổ biến và tốn kém hơn.
- [x] Bonus 4 - Hoàn trả về phiên bản trước: chặn Release nếu F1 mới thấp hơn `artifacts/current/report.json`.
- [x] Bonus 5 - Cảnh báo lệch lạc dữ liệu: cảnh báo khi tỷ lệ lớp dương lệch quá 5 điểm so với 24.8%.
