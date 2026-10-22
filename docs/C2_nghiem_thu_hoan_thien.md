# Báo cáo hoàn thiện C2

Ngày nghiệm thu kỹ thuật: 2026-10-22. Đã khắc phục lỗi đồng bộ metadata, tái tạo B03/B04/B05/B07, xác suất và artifacts, kiểm tra giao diện tiêu thụ C3 trên dữ liệu thật. Lần chạy cuối đạt **75/75 test, không lỗi hoặc skip**, bao gồm đủ 55 trường hợp trước đợt hoàn thiện. Chất lượng thống kê của mô hình vẫn có cảnh báo độ nhạy seed.

## Các vấn đề đã sửa

Lỗi xác nhận được ở cả BTC và ETH: `.joblib` trước đây được ghi trước khi có `test_evaluated`, `n_by_period`, số segment/pair cuối cùng; khóa mapping cũng khác kiểu giữa Python và JSON. `run.py` hiện ghi khóa lựa chọn vào file `model_*_selection.json` trước khi xem test, rồi chỉ lưu `.joblib` sau khi metadata cuối cùng hoàn chỉnh. Khóa mapping được chuẩn hóa thành chuỗi. Mỗi lần lưu có tải lại artifact để so metadata và score; ghi model qua file tạm rồi thay thế file đích. JSON cuối, hai model HMM/GMM và manifest được kiểm tra bằng đối chiếu toàn bộ metadata, cùng kiểm mapping/K/model ID/số dòng trong CSV.

Reader C3 bổ sung kiểm schema, UTC, ngày duy nhất, số cột theo K, nhãn/mapping, state ID, high-risk, tổng xác suất và số dòng từng split. Dữ liệu smoothed/hindsight bị từ chối. Với mô hình unstable, `require_stable=True` tiếp tục từ chối; nghiên cứu phải truyền rõ `acknowledge_unstable=True`, nhận `UnstableModelWarning` và lưu quyết định vào thuộc tính dữ liệu. Không có đường đọc nghiên cứu mặc định bỏ qua cảnh báo.

Không sửa source C1/C3. Giữ thuật toán fit, scaler C1, mốc chia mẫu và tiêu chí chọn mô hình. BTC vẫn K=3 seed=1; ETH vẫn K=4 seed=1. Cấu hình chỉ tăng phiên bản lên 1.2.2 để ghi nhận thay đổi giao diện và quy trình artifacts.

## Độ ổn định và nguyên nhân

| Chỉ số validation tại K đang chọn | BTC | ETH |
|---|---:|---:|
| Seed hợp lệ / số cặp | 5 / 10 | 5 / 10 |
| ARI trung vị | -0.035005 | 0.026412 |
| ARI nhỏ nhất | -0.035005 | -0.077506 |
| Khoảng likelihood train giữa seed | 249.226809 | 309.317707 |
| Bất đồng high-risk trên mọi ngày | 1.2381% | 0.7619% |
| Bất đồng trên ngày ít nhất một seed báo high-risk | 39.0000% | 74.7354% |
| Jaccard sự kiện high-risk trung bình | 0.610000 | 0.252646 |
| Kết luận | unstable | unstable |

Tất cả 30 fit đạt tiêu chí hội tụ EM; BTC K=4 bị loại cả 5 seed vì state train nhỏ nhất chỉ khoảng 1.11%, dưới ngưỡng 2%. Các seed tại K đã chọn có likelihood và phân hoạch khác nhau đáng kể, phù hợp với các nghiệm cực trị địa phương khác nhau. ARI bất biến theo hoán vị ID, nên kết quả thấp không thể giải thích đơn thuần bằng đảo tên state. Hội tụ EM chưa bảo đảm một phân hoạch duy nhất.

Tỷ lệ high-risk trên mọi ngày thấp bị ảnh hưởng bởi số ngày không có sự kiện. Chỉ số có điều kiện tính trên 10 cặp có sự kiện của BTC và 9 cặp của ETH; một cặp ETH không có sự kiện nên không xác định Jaccard và được để trống. High-Vol ETH chỉ có **2/632 hard-label test**; state 0 có **0/632**, dù chiếm 819/1.891 hard-label train. Đây là hạn chế độ phủ và thay đổi occupancy ngoài mẫu; không đủ cơ sở khẳng định độ chính xác cảnh báo rủi ro hoặc hiệu quả giao dịch.

## Kiểm tra nâng cao

Đã fit RV14/RV21/RV30, mỗi cửa sổ 5 seed, giữ K của core. Cả ba cửa sổ dùng cùng **1.882 ngày train và 630 ngày validation**; scaler riêng chỉ học train. RV21 cũng refit trên mẫu chung để kiểm ảnh hưởng bỏ 9 ngày đầu. Không đọc test trong các phép fit/so sánh này và không thay model chính.

Median ARI theo seed của BTC với RV14/21/30 lần lượt khoảng **-0.0346 / -0.0346 / -0.1077**; ETH khoảng **0.9581 / 0.0052 / 0.9008**. ETH RV14/RV30 vẫn có seed lệch mạnh, min ARI lần lượt **0.0696 / 0**, nên chưa đạt toàn bộ tiêu chí ổn định đã khai báo. Không xếp hạng các cửa sổ bằng likelihood vì biến quan sát đã thay đổi. Kết quả chi tiết ở `C2_window_sensitivity.csv` và `C2_window_candidates.csv`.

## Xác minh và giao diện C3

Lần chạy cuối sử dụng Python 3.12.10, đúng 46 dependency trong lock. Đã chạy toàn bộ pytest trong tiến trình mới; kiểm 36 hash đầu ra, đối chiếu đầy đủ metadata cho hai tài sản, kiểm 12 score HMM/GMM trực tiếp bằng API thư viện, kiểm `pip check` và xác nhận đầu vào/source không đổi trong lúc xác minh. Các kiểm tra Data Leakage, Forward Filtering, Transition Matrix, Model Selection và GMM Benchmark đều nằm trong bộ test thực chạy.

Consumer giao diện đã tải filtered thật cho BTC/ETH, từ chối chế độ yêu cầu stable, rồi dùng chế độ nghiên cứu có xác nhận và ghi lại cảnh báo. Mỗi tài sản ghép được **632 return test** đúng ngày t+1 và cùng segment; tín hiệu đầu là 2025-01-24 cho return 2025-01-25, cuối return 2026-10-18. `as_of_utc` không muộn hơn đầu khoảng return. Phép tính minh họa `(1-prob_high_risk) * return_next_day` hữu hạn; không xuất kết luận lợi nhuận/NAV. Đây là xác nhận giao diện C2 cung cấp cho C3, chưa phải kiểm thử portfolio/dashboard C3 đầy đủ hoặc xác nhận của ba thành viên nhóm.

## Bằng chứng bàn giao

| Đường dẫn | Bằng chứng |
|---|---|
| `outputs/metadata/pytest_baseline_55.xml`, `.log` | Chạy lại 55 test trước sửa, 55 passed |
| `outputs/metadata/pytest_verification.xml`, `.log` | Lần chạy cuối 75 test, gồm đủ danh tính 55 test trước |
| `outputs/metadata/C2_verification.json` | Metadata audit, score audit, dependencies, hash, kết quả test và consumer |
| `outputs/metadata/C2_completion_run.json` | Các lệnh tái lập và exit code của core/diagnostics/verification |
| `outputs/metadata/C2_completion_pipeline.log`, `C2_window_diagnostics.log`, `C2_completion_verification.log` | Log Python nguyên bản của lần chạy cuối |
| `outputs/metadata/C2_C3_integration_check.json` | Chính sách cảnh báo, thời gian và số dòng consumer thực dùng |
| `outputs/tables/C2_seed_fit_diagnostics.csv`, `C2_state_coverage.csv` | Hội tụ, khác biệt nghiệm, occupancy và state hiếm |
| `outputs/metadata/C2_window_diagnostics.json` | Scaler train-only, scope không dùng test và hash chẩn đoán |

Chạy lại từ thư mục dự án: kích hoạt `$HOME\.venv`, cài `requirements-lock.txt`, rồi chạy `python scripts/complete_c2.py`. Danh mục đầy đủ B03/B04/B05/B07, xác suất, artifacts và ví dụ reader nằm trong [hướng dẫn C2](C2_huong_dan_va_nghiem_thu.md).

Đây là xác minh cục bộ có thể tái lập, không phải đánh giá bên thứ ba. Test đã được công bố từ lần chạy trước, không phải holdout mới. Chẩn đoán RV là phân tích bổ sung sau thực nghiệm. Chưa bootstrap khoảng tin cậy, kiểm định độ chính xác theo ground truth hay xác minh hiệu quả giao dịch; không triển khai NHHMM, Dune hoặc CoinGlass.
