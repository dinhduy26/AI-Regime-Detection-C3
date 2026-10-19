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
| `tests/test_pipeline.py` (tài liệu chung §3.1) | có, kèm `tests/test_c1_acceptance.py` (10 tiêu chí đặc tả C1 §1.11) | 21 test |
| `build_features`, `chronological_split`, `fit_scaler` (§3.2) | có trong `src/features/build.py` | Kết quả trùng file đã lưu |

## 9. Trạng thái phần mở rộng (C1-08, C1-09)

RQ3 chỉ là mở rộng. Dữ liệu vĩ mô và funding **chưa được đưa vào mô hình**. Khi nhóm quyết định làm RQ3, chạy:

    python -m src.data.collect --skip-core --macro --derivatives
    python run_pipeline.py --steps features

Lệnh trên tạo `macro_daily.csv`, `derivatives_daily.csv` và `features_ext_{asset}.parquet`. Feature lõi của HMM không thay đổi.

Trạng thái hiện tại của từng biến mở rộng luôn được ghi trong `outputs/metadata/limitation_log.csv` (Excel C1-09: "derivatives_daily.csv hoặc limitation_log"). Biến nào chưa thu thập thì ghi rõ "chưa thu thập", không tạo số liệu thay thế.