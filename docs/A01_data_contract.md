# A-01 — Hợp đồng dữ liệu C1 → C2 → C3 → Research

Bản máy đọc của hợp đồng này là `config/schema.yaml`. Code C1 kiểm tra đầu ra theo file đó trước khi ghi. **C2 và C3 cần xác nhận** các mục dưới đây trước khi tích hợp.

## 1. Quy ước chung (đã chốt)

| Mục | Quy ước |
|---|---|
| Tài sản | `BTC` = BTCUSDT, `ETH` = ETHUSDT, Binance **Spot** |
| Tần suất | Nến 1 ngày, **UTC**. `date_utc` là ngày mở nến (00:00 đến 23:59:59.999 UTC) |
| Thời điểm biết thông tin | `available_at_utc` = lúc nến đóng = 00:00 UTC ngày hôm sau |
| Đơn vị return | Số thập phân trên ngày, **không phải %**. UI mới nhân 100 |
| Volume lõi | `quote_volume` (USDT). `base_volume` chỉ để đối soát |
| Định dạng | Dữ liệu dùng parquet, bảng cho bài báo dùng csv, manifest dùng json |
| Tên cột chuẩn | `log_return`, `rv21`, `log_dollar_volume`, `simple_return`. **Không dùng** `rv_21d`, `log_volume` |
| Mã bảng/hình | B01 = Descriptives, B02 = Sample Manifest, H01 = Price history |

## 2. Feature lõi đưa vào HMM (vector 3 chiều)

| Cột | Công thức | Đơn vị |
|---|---|---|
| `log_return` | ln(C_t / C_{t−1}) | thập phân/ngày |
| `rv21` | std mẫu (ddof = 1) của log_return từ t−20 đến t; không annualize | thập phân/ngày |
| `log_dollar_volume` | ln(1 + quote_volume_t) | log(1 + USDT) |

`simple_return` = C_t / C_{t−1} − 1 **chỉ dùng cho NAV của C3**, không đưa vào HMM.

## 3. Chia mẫu và chuẩn hóa

- Áp dụng riêng cho từng tài sản, sau khi bỏ warm-up.
- **Chế độ mặc định `fixed_dates`:** train ≤ 2023-05-05 < validation ≤ 2025-01-24 < test (đến ngày mới nhất). Hai mốc này được chốt bằng tỷ lệ 60/20/20 trên dữ liệu đến 2026-10-18 (1.891 / 630 / 632 ngày). Dữ liệu cập nhật hằng ngày chỉ nối vào cuối tập test, nên train, validation, scaler và mô hình không đổi.
- Chế độ dự phòng `ratio`: n_tr = ⌊0.6N⌋, n_val = ⌊0.2N⌋, phần còn lại là test; tính lại mỗi lần chạy.
- Tỷ lệ thực tế của từng tập (`share`) được ghi trong `split_manifest.json` và B02.
- Cột `split` nhận một trong ba giá trị `train`, `validation`, `test`.
- Scaler: mean và std **tổng thể (ddof = 0)** tính **chỉ trên train**, áp dụng cho cả ba tập. Không bao giờ fit lại scaler.
- Ngày cắt cụ thể và tham số scaler nằm trong `outputs/metadata/split_manifest.json`.

## 4. Ngày thiếu dữ liệu (gap)

- Không điền, không nội suy. Mỗi khoảng thiếu được ghi vào `gap_report.csv`.
- Mỗi đoạn ngày liên tục có một `segment_id`. Return và RV21 chỉ tính **bên trong** từng đoạn.
- Mặc định `gap_handling_method = longest_segment`: file feature chỉ chứa đoạn liên tục dài nhất.
- Nếu đổi sang `multi_segment`: **C2 phải huấn luyện HMM theo từng segment** (truyền `lengths` cho hmmlearn) và reset trạng thái ẩn ở đầu mỗi đoạn.

## 5. Bảng trách nhiệm

| File | Owner | Consumer | C2 xác nhận | C3 xác nhận |
|---|---|---|---|---|
| `prices_daily.parquet` | C1 | C2, C3 | ☐ | ☐ |
| `features_{asset}.parquet` | C1 | C2, C3 | ☐ | ☐ |
| `scaled_features_{asset}.parquet` + `scaler_{asset}.joblib` | C1 | C2 | ☐ | — |
| `split_manifest.json` | C1 | C2, C3 | ☐ | ☐ |
| `gap_report.csv` | C1 | C2, C3 | ☐ | ☐ |
| `regime_filtered.csv` (state_id, prob_state_0..K−1, K, as_of, model_id) | C2 | C3 | — | ☐ |
| `B03`–`B05`, `B07` | C2 | Research, C3 | — | ☐ |
| `B08_portfolio.csv`, `backtest_nav.csv`, H02–H06 | C3 | Research, UI | — | — |

## 6. Lưu ý cho C3 khi ghép dữ liệu backtest

Tín hiệu là filtered probability của ngày t, dùng để xác định tỷ trọng w_{t+1} cho `simple_return` của ngày t+1. Khi ghép, dùng `date_utc` trong `features_{asset}.parquet`. Ngày t+1 phải là ngày kề sau và cùng `segment_id`.

## 7. Ngày dữ liệu và giá tham khảo (cho C3)

- **Cập nhật hằng ngày:** sau 07:05 giờ Việt Nam (nến UTC hôm trước đã đóng) chạy `python run_pipeline.py --download`. Lệnh chỉ tải lại BTC/ETH; train, validation và scaler không đổi, ngày mới nối vào cuối tập test. Tự động hóa: `scripts/update_daily.bat` (Windows Task Scheduler) hoặc `scripts/update_daily.sh` (cron). Dữ liệu mở rộng chỉ tải lại khi cần: `python -m src.data.collect --skip-core --all-extensions`.
- `split_manifest.json → data_through_utc`: ngày nến UTC đã đóng gần nhất trong mẫu. Dashboard hiển thị "Dữ liệu đến hết ngày …" (Excel C3-08) và cảnh báo nếu ngày này cũ hơn hôm qua.
- `src/data/live_quote.py → get_live_quote(asset)`: giá Binance trực tiếp **chỉ để hiển thị**. Hàm không ghi file và không đưa vào feature hay backtest. Luôn kèm dòng chú thích `LIVE_QUOTE_NOTE`.
- Regime và backtest chỉ dùng nến ngày đã đóng. Đây là cập nhật theo ngày, không phải realtime (Excel C3-13, C3-14).

## 8. Đối chiếu tên giữa các tài liệu

Excel (bảng phân công) và tài liệu coder đặt tên khác nhau ở vài chỗ. Repo **theo Excel và sheet `du_lieu`**:

| Tài liệu khác ghi | Repo dùng | Ghi chú |
|---|---|---|
| `data/raw/btc_1d.csv`, `eth_1d.csv` (tài liệu chung §2.2.1) | `data/raw/btc_ohlcv.csv`, `eth_ohlcv.csv` | Theo Excel C1-03/C1-04 |
| `collection_config.yaml` (Excel C1-02) | `config/collection_config.yaml` **tự sinh** từ `config.yaml` mỗi lần chạy | Chỉ sửa `config.yaml`; không sửa tay file tự sinh |
| cột `symbol` trong prices (tài liệu chung) | có cả `asset` (BTC/ETH) và `symbol` (BTCUSDT/ETHUSDT) | Khóa là `(asset, date_utc)` theo sheet `du_lieu` |
| `rv_21d`, `log_volume` (bảng Code cũ) | `rv21`, `log_dollar_volume` | Theo tài liệu coder §1.6 |
| `config/settings.yaml` (tài liệu chung §3.1) | `config/config.yaml` | Theo Excel A-01 ("config.yaml") |
| `src/data/fetch.py` (tài liệu chung §3.1) | có `src/data/fetch.py` với `fetch_ohlcv()`; pipeline đầy đủ ở `collect.py` + `binance_collector.py` | Hàm giao tiếp đúng tên §3.2 |
| `tests/test_pipeline.py` (tài liệu chung §3.1) | có, kèm `tests/test_c1_acceptance.py` (10 tiêu chí đặc tả C1 §1.11) và `tests/test_extensions.py` (dữ liệu mở rộng) | 28 test |
| `build_features`, `chronological_split`, `fit_scaler` (§3.2) | có trong `src/features/build.py` | Kết quả trùng file đã lưu |

## 9. Dữ liệu mở rộng (RQ3, robustness, tài sản mở rộng)

Quy ước phân công: dữ liệu nào **một** trong ba tài liệu (Excel, tài liệu coder, blueprint) nhắc tới mà hai tài liệu còn lại không giao cho C1/C2/C3 thì **mặc định thuộc C1**. Vì vậy C1 tải đủ:

| Biến | File thô | Ghép vào | Căn cứ |
|---|---|---|---|
| VIX, SPX, DXY, US10Y, US2Y, US3M | `data/raw/macro_daily.csv` | `features_ext_*` (as-of) | Excel C1-08; blueprint §3.3 |
| STABLECOIN_SUPPLY, DEFI_TVL, MCAP_BTC/ETH, CMC200 | `data/raw/crypto_market_daily.csv` | `features_ext_*` (as-of) | Blueprint §2.4, §3.1, §3.2, §3.5; tài liệu coder §1.7 |
| Funding | `data/raw/derivatives_daily.csv` | `features_ext_*` (theo ngày UTC) | Excel C1-09 |
| Open interest lịch sử | `data/raw/open_interest_daily.csv` | `features_ext_*` (theo ngày UTC) | Excel C1-09; blueprint §2.4 |
| SOL, BNB | `data/raw/{sol,bnb}_ohlcv.csv` | `data/processed/extension/` | Blueprint §3.1 |

Lệnh tải: `python -m src.data.collect --skip-core --all-extensions`, sau đó `python run_pipeline.py`.

Quy tắc point-in-time: điểm dữ liệu ngày d chỉ dùng từ 00:00 UTC ngày d + 1 + lag (lag = 1). Funding và OI của ngày t đều đã có trước khi nến t đóng. Mỗi biến ghép as-of có cột `<biến>_available_at` và `<biến>_obs_date`; `tests/test_extensions.py` kiểm tra không có giá trị nào được dùng trước khi công bố và không có giá trị vô cực.

Biến tăng trưởng 7 ngày (`stablecoin_supply_growth_7d`, `defi_tvl_growth_7d`, `oi_growth_7d`) = ln(x_d / x_{d−7}); nếu x_d hoặc x_{d−7} ≤ 0 hoặc không có quan sát thì để trống (NaN), không nội suy.

Feature lõi HMM (`log_return`, `rv21`, `log_dollar_volume` của BTC/ETH) **không thay đổi**. Biến mở rộng chỉ dùng khi C2 làm RQ3 hoặc robustness (Excel C2-14). Trạng thái từng biến luôn ở `outputs/metadata/limitation_log.csv`; biến nào chưa tải được ghi rõ, không tạo số liệu thay thế.
