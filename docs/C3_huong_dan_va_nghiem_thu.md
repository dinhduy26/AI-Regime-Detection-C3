# C3 Backtest regime-aware và dashboard Crypto Market Regime Radar

C3 dùng xác suất **filtered** đã kiểm chứng của C2 để mô phỏng chiến lược phân bổ phòng thủ theo regime, so sánh công bằng với buy-and-hold trên tập test, và dựng dashboard Streamlit 4 trang chỉ đọc kết quả đã xuất. Đây là mô phỏng nghiên cứu, không phải giao dịch thật và không phải khuyến nghị đầu tư.

## 1. Chạy lại

Chạy từ thư mục chứa `run_pipeline.py`, với môi trường đã cài `requirements-lock.txt` (file lock đã bổ sung `streamlit`, `plotly` và các gói phụ thuộc).

```bash
pip install -r requirements-lock.txt
python scripts/complete_c2.py                 # nếu chưa có đầu ra C2 (bị .gitignore, phải tái tạo)
python run_pipeline.py --steps backtest       # hoặc: python -m src.evaluation.backtest_run
pytest -q tests/test_c3_backtest.py tests/test_c3_dashboard.py
streamlit run dashboard/app.py                # mở http://localhost:8501
```

`run_pipeline.py` tự gọi `src/evaluation/backtest_run.py` ở bước `backtest` sau bước `models` của C2. Đầu ra C3 sinh tự động cũng được giữ local và bỏ qua bởi `.gitignore`, giống quy ước của C2.

## 2. Đặc tả chiến lược (C3-03)

Toàn bộ tham số nằm trong `config/backtest.yaml` (phiên bản 1.0.0). File này được tách khỏi `config/config.yaml` để không làm đổi hash cấu hình mà manifest C1/C2 đã ghi và `scripts/verify_c2.py` đối chiếu. Mọi giá trị được chỉ định trước theo Research & Technical Methodology (phần C3, mục 3.2-3.7) và không được chỉnh theo kết quả test.

| Thành phần | Quy định |
|---|---|
| Tín hiệu | `prob_high_risk` = P(S_t = k* \| X_1:t), xác suất filtered của state có volatility train lớn nhất do C2 chốt (BTC: state 0; ETH: state 2) |
| Đọc tín hiệu | `src.models.handoff.prepare_next_day_research_inputs(..., acknowledge_unstable=True)`: kiểm hash, schema, tổng xác suất, thời điểm `as_of_utc`, cùng segment, ghép tín hiệu ngày t với lợi suất ngày t+1 |
| Quy tắc tỷ trọng | w_t = 1 nếu p_risk,t < 0,5; ngược lại w_t = 0,25. Long/cash, không đòn bẩy, tiền mặt lợi suất 0 |
| Thời điểm | Tín hiệu cuối ngày UTC t quyết định tỷ trọng cho lợi suất t → t+1 |
| Phí | 0,10% × turnover; không gồm slippage, spread, thuế, tác động thị trường |
| Kỳ đầu | Cả hai chiến lược bắt đầu từ tiền mặt (w_-1 = 0) và chịu phí mở vị thế |
| Đối chứng | Buy-and-hold w = 1 trên cùng tài sản, cùng ngày, cùng cơ sở lợi suất và cùng quy tắc phí |
| Chỉ số | CR, CAGR theo số ngày lịch thực, volatility năm hóa √365, Sharpe sau phí với rf = 0, max drawdown, tổng turnover, tổng phí |

Công thức lõi (`src/evaluation/backtest.py`): Gross_{t+1} = w_t R_{t+1} + (1 - w_t) R^cash; Cost_{t+1} = f · TO_{t+1}; Net_{t+1} = Gross_{t+1} - Cost_{t+1}; NAV_{t+1} = NAV_t (1 + Net_{t+1}), NAV_0 = 1.

Ba kịch bản được báo riêng, không trộn:

| Kịch bản | Lợi suất | Turnover | Vai trò |
|---|---|---|---|
| `baseline_close_to_close` | C_{t+1}/C_t - 1 (simple_return của C1) | `target_change_approx`: \|w_t - w_{t-1}\| | Kết quả chính theo Methodology (idealized end-of-day exposure) |
| `crosscheck_drift_adjusted` | Như baseline | `drift_adjusted`: \|w_t - w_{t-}\|, với w_{t-} là tỷ trọng thực tế sau biến động giá | Kiểm tra chéo bắt buộc về phí |
| `crosscheck_next_open` | O_{t+2}/O_{t+1} - 1 | `target_change_approx` | Kiểm tra chéo bắt buộc về thời điểm khớp lệnh sau tín hiệu |

Với nến ngày Binance, open của ngày t+1 bắt đầu đúng lúc nến t đóng (00:00 UTC), nên kịch bản next-open vẫn giả định độ trễ bằng 0 và không có spread hay slippage. Kịch bản này có ít hơn một ngày so với baseline vì ngày test cuối chưa có open t+2; ngày đó bị loại khỏi cả hai chiến lược.

## 3. Kết quả trên tập test

Dữ liệu C1 đến hết 2026-10-18 UTC; tập test gồm 632 lợi suất ngày từ 2025-01-25 đến 2026-10-18 (631 ngày với kịch bản next-open). Mô hình C2: BTC `btc-5c260ade6a90e577` (K = 3, seed 1), ETH `eth-347a3c7d09e9edee` (K = 4, seed 1). Số liệu lấy từ `outputs/tables/B08_portfolio.csv`; lợi suất ghi theo phần trăm để dễ đọc, file CSV lưu dạng thập phân.

| Tài sản | Kịch bản | Chiến lược | NAV cuối | CAGR | Volatility năm | Sharpe | Max DD | Turnover | Tổng phí | Ngày phòng thủ |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BTC | baseline | Buy-and-hold | 0.7822 | -13.22% | 43.06% | -0.114 | -52.97% | 1.0000 | 0.100% | 0.0% |
| BTC | baseline | Regime-aware | 0.7338 | -16.37% | 41.33% | -0.225 | -55.09% | 4.0000 | 0.400% | 4.1% |
| BTC | drift-adjusted | Buy-and-hold | 0.7822 | -13.22% | 43.06% | -0.114 | -52.97% | 1.0000 | 0.100% | 0.0% |
| BTC | drift-adjusted | Regime-aware | 0.7337 | -16.37% | 41.33% | -0.225 | -55.10% | 4.0844 | 0.408% | 4.1% |
| BTC | next-open | Buy-and-hold | 0.7763 | -13.63% | 43.09% | -0.125 | -52.97% | 1.0000 | 0.100% | 0.0% |
| BTC | next-open | Regime-aware | 0.7282 | -16.76% | 41.36% | -0.236 | -55.09% | 4.0000 | 0.400% | 4.1% |
| ETH | baseline | Buy-and-hold | 0.7274 | -16.79% | 69.29% | 0.077 | -67.55% | 1.0000 | 0.100% | 0.0% |
| ETH | baseline | Regime-aware | 0.7218 | -17.16% | 69.02% | 0.069 | -66.27% | 4.0000 | 0.400% | 0.3% |
| ETH | drift-adjusted | Buy-and-hold | 0.7274 | -16.79% | 69.29% | 0.077 | -67.55% | 1.0000 | 0.100% | 0.0% |
| ETH | drift-adjusted | Regime-aware | 0.7218 | -17.16% | 69.02% | 0.069 | -66.27% | 3.9983 | 0.400% | 0.3% |
| ETH | next-open | Buy-and-hold | 0.7252 | -16.96% | 69.34% | 0.075 | -67.55% | 1.0000 | 0.100% | 0.0% |
| ETH | next-open | Regime-aware | 0.7196 | -17.33% | 69.08% | 0.066 | -66.27% | 4.0000 | 0.400% | 0.3% |

Diễn giải trong giới hạn của thiết kế:

- Trên giai đoạn test, chiến lược regime-aware **không cải thiện** NAV, CAGR hay Sharpe so với buy-and-hold ở cả hai tài sản. Kết luận giữ nguyên ở cả ba kịch bản, nên không phụ thuộc cách tính phí hay giả định khớp lệnh.
- **BTC** giảm tỷ trọng trong 26/632 ngày (4,1%), gồm hai giai đoạn: 21-23/03/2025 và 06-28/02/2026. Giai đoạn thứ hai bắt đầu sau khi nhịp giảm mạnh đầu tháng 2/2026 đã diễn ra, nên chiến lược giảm tỷ trọng đúng lúc thị trường hồi và bỏ lỡ một phần nhịp hồi. Volatility năm hóa thấp hơn khoảng 1,7 điểm phần trăm, nhưng max drawdown sâu hơn 2,1 điểm phần trăm.
- **ETH** chỉ giảm tỷ trọng trong 2/632 ngày (0,3%; 09/05/2025 và 23/02/2026), nên đường NAV gần như trùng buy-and-hold. Điều này nhất quán với báo cáo C2: state High-Vol của ETH chỉ có 2/632 hard-label trên test. Khác biệt max drawdown (-66,27% so với -67,55%) đến từ hai ngày này và không đủ cơ sở để kết luận chiến lược có tác dụng phòng thủ.
- Cả hai mô hình C2 bị đánh dấu `unstable` theo chẩn đoán seed (ARI trung vị giữa các seed rất thấp). Tín hiệu high-risk vì vậy phụ thuộc khởi tạo, và kết quả trên chỉ là một lần mô phỏng với mô hình đã khóa, không phải bằng chứng nhân quả về giá trị kinh tế của regime.
- Sharpe của ETH dương trong khi NAV giảm: trung bình số học của lợi suất ngày hơi dương nhưng volatility rất cao làm lợi suất hình học âm (volatility drag).

## 4. Đầu ra (C3-04 đến C3-07, C3-12)

| File | Nội dung |
|---|---|
| `outputs/tables/B08_portfolio.csv` | Khóa `asset, strategy, scenario, turnover_method`; CAGR, ann_volatility, Sharpe, max_drawdown, turnover, total_cost, n_days, start/end, share_days_risk_off, n_rebalances, tham số quy tắc, model_run_id, seed_stability_status, data_through_utc |
| `outputs/predictions/backtest_nav.csv` | Từng ngày: date_utc (ngày lợi suất), signal_date_utc, filtered_asof_utc, prob_high_risk, weight, pre_trade_weight, asset_return, gross, turnover, fee, net, nav, drawdown |
| `outputs/figures/H05_cumulative_nav.png/.svg` | NAV và drawdown test, buy-and-hold so với regime-aware, BTC và ETH, 300 dpi |
| `outputs/figures/H05_cumulative_nav_{btc,eth}.png/.svg` | Bản riêng từng tài sản |
| `outputs/figures/H06_dashboard*.png` | Ảnh chụp dashboard thật (tạo bằng `scripts/capture_dashboard.py`) |
| `outputs/metadata/backtest_manifest.json` | Trạng thái, cấu hình, provenance mô hình C2, cảnh báo unstable, giả định, hash đầu ra |
| `outputs/metadata/H06_capture.json` | Thời điểm chụp H06, URL, ngày dữ liệu |

## 5. Dashboard (C3-01, C3-02, C3-08 đến C3-11, C3-13)

`dashboard/result_service.py` là lớp đọc dữ liệu: chỉ đọc file đã xuất, đọc xác suất filtered qua reader kiểm chứng của C2, kiểm file smoothed đúng cờ hindsight, và báo file thiếu kèm lệnh khắc phục. `dashboard/ui.py` chứa bảng màu tông đất trầm, nền họa tiết tự vẽ (lưới, nến giá, đường dữ liệu), định dạng số kiểu Việt Nam (1.234,56), CSS và bố cục biểu đồ dùng chung; `.streamlit/config.toml` đặt theme. `dashboard/app.py` dựng 4 trang trên lưới cố định (hàng thẻ KPI, biểu đồ chính 8/12 cột, biểu đồ phụ 4/12 hoặc 6/12 cột, ba bậc chiều cao), chuyển bằng menu ở thanh bên (thanh bên còn có chọn BTC/ETH và ngày dữ liệu). Giải thích kỹ thuật nằm trong biểu tượng ⓘ ở góc mỗi thẻ; các bảng số chi tiết nằm trong mục thu gọn cuối trang. Các thống kê mô tả bổ sung (lợi suất sau khi vào regime, lợi suất theo tháng, độ biến động trượt, độ dài đợt regime) được tính trực tiếp từ file đã xuất:

| Trang | Nội dung | Nguồn |
|---|---|---|
| 1. Overview | Thẻ KPI (giá, regime hiện tại, số ngày ở regime, P(rủi ro cao), tỷ trọng theo quy tắc, RV21); giá 12 tháng tô nền regime kèm dải P(rủi ro cao); thước đo RV21 so với volatility từng state; dải regime toàn lịch sử của BTC và ETH đặt chồng nhau, kèm tỷ lệ ngày cùng ở regime rủi ro cao; tỷ trọng thời gian ở mỗi regime; hồ sơ từng regime | prices, features, regime_filtered (BTC, ETH), B04, B05, B08 |
| 2. Regime History | Chọn filtered hoặc smoothed (có cảnh báo hindsight), giai đoạn và khoảng ngày; giá thang log tô nền regime; lợi suất trung bình 5/10/20 ngày sau ngày vào từng regime so với mọi ngày; phân phối lợi suất ngày theo regime; sơ đồ chuyển trạng thái dạng mũi tên; độ dài từng đợt regime; bảng ma trận chuyển trong mục thu gọn | regime_filtered, regime_smoothed, features, B04, B05 |
| 3. Model Comparison | Thẻ KPI (K, log-score validation, log-score và số lần đổi state trên test của HMM so với GMM, độ ổn định seed); dải nhãn HMM và GMM đặt chồng nhau; log-score và flips theo giai đoạn; chọn K theo validation; bảng độ ổn định seed; B03 và B07 trong mục thu gọn | B03, B07, C2_stability_summary, regime_filtered, regime_gmm |
| 4. Portfolio & Risk | Chọn kịch bản; thẻ KPI so với buy-and-hold; NAV có tô vùng ngày phòng thủ; so sánh nhanh lợi nhuận và rủi ro; drawdown; heatmap lợi suất tháng × năm (regime-aware, buy-and-hold, chênh lệch); độ biến động trượt 60 ngày; phân phối lợi suất ngày; bảng B08, tải CSV, giả định trong mục thu gọn | B08, backtest_nav, backtest_manifest |

Nguyên tắc giao diện: hiển thị "Dữ liệu đến hết ngày" theo `data_through_utc` của từng tài sản và ghi rõ không phải realtime; nhãn cảnh báo khi mô hình C2 `unstable`; số state và nhãn theo đúng K của từng tài sản; nhãn "simulated" ở trang danh mục; không có biểu đồ drivers vì B06 (RQ3) chưa thực hiện; không gọi log-score là accuracy.

## 6. Nghiệm thu

`tests/test_c3_backtest.py` (16 test) và `tests/test_c3_dashboard.py` (8 test) đối chiếu các tiêu chí của Methodology mục 3.13 và Excel C3:

| Tiêu chí | Test |
|---|---|
| p_risk = 0,65 → w = 0,25; p_risk = 0,49 → w = 1 | `test_weight_rule_thresholds` |
| Ví dụ tay mục 3.8: gross 0,025; cost 0,00075; net 0,02425; NAV 1,02425 | `test_hand_example_section_3_8` |
| Ví dụ drift mục 3.6: 0,26829 và 1,829% NAV | `test_drift_example_section_3_6` |
| Turnover 0 thì phí 0; phí không làm tăng lợi suất | `test_zero_turnover_means_zero_fee_and_fee_never_adds_return` |
| Tín hiệu t chỉ áp dụng cho t+1; lợi suất tương lai không đổi NAV quá khứ | `test_signal_applies_to_next_period_only`, `test_nav_rows_use_filtered_signal_before_return` |
| Trọng số lấy từ xác suất filtered của file C2, không dùng smoothed | `test_nav_rows_use_filtered_signal_before_return` |
| NAV khớp NAV_t(1 + Net_{t+1}); B08 khớp chuỗi NAV; lợi suất khớp C1 | `test_nav_recurrence_and_drawdown`, `test_nav_reproduces_metrics_and_returns` |
| Buy-and-hold khớp công thức đóng (chỉ phí mở vị thế) | `test_buy_and_hold_matches_closed_form` |
| CAGR, volatility, Sharpe theo 365 ngày; MDD không dương; Sharpe không xác định khi s_g = 0 | `test_metrics_formulas` |
| Cùng giai đoạn, ngày và quy tắc phí cho hai chiến lược trong mỗi kịch bản | `test_same_period_and_rules_for_both_strategies` |
| Tham số chỉ định trước, không tune trên test | `test_backtest_config_is_prespecified` |
| Dashboard 4 trang chạy không lỗi trên dữ liệu thật; KPI trên UI khớp B08; định dạng số đúng | `test_app_renders_four_pages_without_errors`, `test_ui_kpis_match_exported_b08`, `test_vietnamese_number_format` |
| Cảnh báo unstable; số state đúng K; smoothed gắn nhãn hindsight; nhãn simulated | các test còn lại trong `test_c3_dashboard.py` |

## 7. Giới hạn

Backtest là mô phỏng lịch sử với một mô hình đã khóa, một ngưỡng và một cặp tỷ trọng cố định; không có rolling refit, volatility targeting hay kiểm định thống kê khác biệt hiệu suất. Phí 0,10% là giả định nghiên cứu, không phải biểu phí thực tế của Binance. Tập test đã được nhóm xem từ lần chạy C2 đầu tiên. Số ngày phòng thủ rất ít (đặc biệt với ETH) nên kết quả có độ bất định lớn và khả năng khái quát hạn chế. A-03 (kiểm thử end-to-end có biên bản và xác nhận của ba thành viên) vẫn cần nhóm thực hiện.
