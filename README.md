# Crypto Market Regime Radar

Nhận diện trạng thái thị trường (regime) BTC và ETH bằng Gaussian HMM, đối chứng GMM, backtest chiến lược và dashboard Streamlit.

> Đây là mô phỏng nghiên cứu, **không phải khuyến nghị đầu tư**.

| Nhóm | Phần việc | Thư mục chính |
|---|---|---|
| **C1** | Dữ liệu & feature | `src/data`, `src/features`, `src/evaluation/descriptives.py` |
| C2 | HMM/GMM & đánh giá | `src/models` |
| C3 | Backtest & dashboard | `src/evaluation`, `dashboard/` |

---

## 1. Cài đặt (một lần)

Yêu cầu Python 3.10 trở lên.

```bash
# vào thư mục dự án
cd crypto-regime-radar

# tạo môi trường ảo
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## 2. Tải dữ liệu thật (cần Internet)

Dự án **không cần API key**. Dữ liệu nến ngày của Binance Spot là API công khai.

```bash
python -m src.data.collect
```

Lệnh này tải BTCUSDT và ETHUSDT từ nến đầu tiên Binance có đến nến ngày gần nhất **đã đóng**, rồi ghi:
- `data/raw/btc_ohlcv.csv`
- `data/raw/eth_ohlcv.csv`
- `outputs/metadata/source_log.csv`

Nếu `api.binance.com` bị chặn, code tự thử lần lượt `api1/api2/api3` và `data-api.binance.vision`. Nếu vẫn lỗi, code chuyển sang kho file công khai của Binance. Bạn cũng có thể chọn thẳng nguồn đó:

```bash
python -m src.data.collect --method vision
```

Dữ liệu mở rộng (RQ3, robustness, tài sản mở rộng) — tải **tất cả** mà không tải lại BTC/ETH:

```bash
python -m src.data.collect --skip-core --all-extensions
python run_pipeline.py
```

| Nhóm | Biến | Nguồn | Căn cứ |
|---|---|---|---|
| Vĩ mô (`--macro`) | VIX, S&P 500, DXY | Yahoo Finance | Excel C1-08 |
| | Lợi suất trái phiếu Mỹ 10 năm, 2 năm, 3 tháng; term spread 10Y–2Y | FRED | Blueprint §3.3 |
| Crypto (`--crypto-market`) | Stablecoin supply + tăng trưởng 7 ngày | DefiLlama | Blueprint §2.4, §3.5; tài liệu coder §1.7 |
| | DeFi TVL + tăng trưởng 7 ngày | DefiLlama | Blueprint §3.2 |
| | Market cap BTC, ETH (lịch sử giới hạn ở bản miễn phí) | CoinGecko | Blueprint §3.2 |
| | CMC Crypto 200 Index (nếu Yahoo có) | Yahoo Finance | Blueprint §3.1 |
| Phái sinh (`--derivatives`) | Funding rate | Binance Futures API | Excel C1-09 |
| | Open interest **có lịch sử** + tăng trưởng 7 ngày | data.binance.vision (metrics) | Excel C1-09; blueprint §2.4 |
| Tài sản (`--extra-assets`) | SOL, BNB (xử lý giống hệt BTC/ETH) | Binance Spot | Blueprint §3.1 |

Mọi biến mở rộng được ghép **as-of** (chỉ dùng giá trị đã công bố trước khi nến ngày t đóng) vào `features_ext_{btc,eth}.parquet`, không đổi feature lõi của HMM. SOL/BNB ra thư mục `data/processed/extension/`. Trạng thái từng biến: `outputs/metadata/limitation_log.csv`. Bước tải open interest lịch sử mất vài phút vì phải tải từng file ngày.

## 3. Chạy pipeline

```bash
python run_pipeline.py            # clean -> features -> reports (dùng dữ liệu thô đã tải)
python run_pipeline.py --download # tải lại dữ liệu rồi chạy toàn bộ
pytest -q                         # kiểm thử nghiệm thu C1 trên dữ liệu thật
```

Khi C2 tạo `src/models/run.py` và C3 tạo `src/evaluation/backtest_run.py`, mỗi file có hàm `run(cfg)`, thì `run_pipeline.py` sẽ tự gọi thêm hai bước đó.

## 4. Cập nhật dữ liệu hằng ngày

Dữ liệu luôn là **nến ngày UTC đã đóng, mới nhất**. Nến của một ngày đóng lúc 07:00 sáng hôm sau theo giờ Việt Nam. Mỗi ngày, sau 07:05, chạy:

```bash
python run_pipeline.py --download
```

Mốc chia mẫu được **cố định theo ngày** (`split.mode: fixed_dates` trong `config/config.yaml`):

| Tập | Khoảng thời gian | Khi có dữ liệu mới |
|---|---|---|
| Train | đến hết 2023-05-05 | không đổi |
| Validation | 2023-05-06 → 2025-01-24 | không đổi |
| Test | từ 2025-01-25 đến **ngày mới nhất** | dài thêm mỗi ngày |

Nhờ vậy train, validation, scaler và mô hình của C2 không đổi khi cập nhật (Excel C2-04, C2-12, C2-13). Test `test_11` kiểm chứng điều này.

**Tự động hóa trên Windows:** mở Task Scheduler → Create Basic Task → chọn Daily lúc 07:05 → Action "Start a program" → chọn `scripts\update_daily.bat`. Nhật ký ghi vào `outputs/metadata/update_log.txt`. Trên macOS/Linux dùng `scripts/update_daily.sh` với cron.

**Ngày dữ liệu (as-of):** `split_manifest.json` và `run_manifest.json` có trường `data_through_utc`. Dashboard (C3) hiển thị trường này dưới dạng "Dữ liệu đến hết ngày …".

**Giá tham khảo trực tiếp cho dashboard:** dùng hàm `src/data/live_quote.py` (xem ví dụ trong file). Giá này chỉ để hiển thị, không lưu và không dùng để tính regime.

> Đây là cập nhật theo ngày, **không phải realtime** (Excel C3-13, C3-14).

## 5. Đối soát giá (kiểm tra dữ liệu có chuẩn không)

```bash
python -m src.data.crosscheck
```

Lệnh này thực hiện hai lớp kiểm tra và ghi kết quả vào `outputs/metadata/price_crosscheck.csv`:

| Lớp | So với | Kỳ vọng |
|---|---|---|
| A. Cùng nguồn | File lịch sử tháng do Binance công bố riêng trên data.binance.vision (các tháng nhiều biến động + tháng gần nhất) | **Khớp tuyệt đối** OHLCV |
| B. Nguồn khác | Yahoo Finance BTC-USD / ETH-USD (giá tổng hợp nhiều sàn, đơn vị USD) | Chênh lệch nhỏ (trung vị < 0,5%), tương quan return ngày > 0,98 |

Ngày nào lệch được liệt kê trong `price_crosscheck_days.csv` để xem tay. Kiểm tra bằng mắt: mở biểu đồ BTC/USDT trên binance.com, chọn khung **1D**, đặt múi giờ **UTC**, rê chuột vào một ngày rồi so OHLC với dòng tương ứng trong `data/raw/btc_ohlcv.csv`.

## 6. Đầu ra của C1

| File | Nội dung | Người dùng |
|---|---|---|
| `data/processed/prices_daily.parquet` | Giá sạch BTC + ETH, khóa `(asset, date_utc)` | C2, C3 |
| `data/processed/features_{btc,eth}.parquet` | `log_return, rv21, log_dollar_volume, simple_return, segment_id, split` và các cột tùy chọn | C2, C3 |
| `data/processed/scaled_features_{btc,eth}.parquet` | z-score, scaler fit **chỉ trên train** | C2 |
| `outputs/models/scaler_{btc,eth}.joblib` | Scaler đã fit | C2 |
| `outputs/metadata/split_manifest.json` | Chế độ chia, ngày cắt train/val/test, số dòng, tỷ lệ, warm-up, segment, tham số scaler, `data_through_utc` | C2, C3, Research |
| `outputs/metadata/gap_report.csv` | Các khoảng ngày thiếu và cách xử lý | C2, C3, Research |
| `outputs/metadata/data_quality.csv`, `excluded_rows.csv` | Báo cáo kiểm toán dữ liệu | QA |
| `outputs/metadata/source_log.csv` | Nguồn, endpoint, thời điểm tải, độ phủ thực tế | Research |
| `config/collection_config.yaml` | Cấu hình thu thập (tự sinh từ `config.yaml`, Excel C1-02) | Research, QA |
| `outputs/metadata/limitation_log.csv` | Trạng thái và giới hạn của biến mở rộng VIX/DXY/S&P/funding/OI (Excel C1-08/09) | Research, C2 |
| `outputs/metadata/data_dictionary.csv` | Từ điển biến | Research |
| `outputs/tables/B01_descriptives.csv` | Thống kê mô tả (full_sample và train) | Research §4.1 |
| `outputs/tables/B02_sample_manifest.csv` | Thông tin mẫu và nguồn | Research §3.1 |
| `outputs/figures/H01_price_history.png/.svg` | Lịch sử giá, 300 dpi | Research, dashboard |
| `outputs/metadata/run_manifest.json` | Thời điểm chạy, cấu hình, hash các file | QA |

Hợp đồng dữ liệu đầy đủ nằm ở `config/schema.yaml` và `docs/A01_data_contract.md`.

## 7. Tái lập kết quả (tài liệu chung §5.3)

| Mục | Giá trị |
|---|---|
| Python | 3.12.6 (máy chạy chính, Windows); yêu cầu ≥ 3.10 |
| Thư viện | Cài đúng phiên bản bằng `pip install -r requirements-lock.txt` |
| Nguồn giá | Binance Spot public API `/api/v3/klines`, nến 1d UTC đã đóng; đối soát với data.binance.vision và Yahoo (`price_crosscheck.csv`) |
| Ngày dữ liệu | `data_through_utc` trong `outputs/metadata/split_manifest.json` và `run_manifest.json` |
| Cấu hình | `config/config.yaml` (`version`, `config_hash` được ghi trong manifest) |
| Chia mẫu | `fixed_dates`: train ≤ 2023-05-05 < validation ≤ 2025-01-24 < test |
| Seed, phí | Không áp dụng cho C1 (C2/C3 ghi trong config của mình) |
| Lệnh chạy lại | `python run_pipeline.py` (dùng dữ liệu đã tải) hoặc `--download` (lấy mới nhất) |

**Hàm giao tiếp C1** (tài liệu chung §3.2), dành cho C2/C3 gọi trực tiếp:

```python
from src.data.fetch import fetch_ohlcv                       # fetch_ohlcv("BTCUSDT", "2024-01-01", "2024-12-31")
from src.features.build import build_features, chronological_split, fit_scaler
```

`tests/test_pipeline.py` kiểm tra các hàm này cho ra đúng kết quả như file đã lưu.

## 8. Cấu hình

Mọi tham số đều nằm trong `config/config.yaml`: ngày bắt đầu/kết thúc, tỷ lệ chia mẫu, cửa sổ RV, cách xử lý gap, bật/tắt dữ liệu mở rộng. Không viết cứng tham số trong code. Mỗi khi đổi tham số ảnh hưởng kết quả thì tăng `version`.

`collection.end_date: null` nghĩa là luôn lấy đến nến mới nhất đã đóng. Chỉ đổi mốc `split.train_end` / `split.validation_end` khi cả nhóm thống nhất, và phải tăng `version` vì scaler và mô hình sẽ phải fit lại.

**Khóa phiên bản thư viện:** sau khi cài xong trên máy chạy chính, chạy `pip freeze > requirements-lock.txt` rồi commit file đó. Thành viên khác cài bằng `pip install -r requirements-lock.txt` để có đúng phiên bản.

## 9. Cấu trúc thư mục

```text
config/        config.yaml, schema.yaml
data/          raw/ interim/ processed/
src/data/      collector Binance, vĩ mô, phái sinh; làm sạch
src/features/  feature, chia mẫu, scaler, ghép biến mở rộng
src/models/    (C2)
src/evaluation/descriptives.py (C1); backtest (C3)
dashboard/     (C3)
tests/         kiểm thử nghiệm thu
outputs/       tables/ figures/ metadata/ models/ predictions/
docs/          hợp đồng dữ liệu, giải thích phương pháp C1
scripts/       update_daily.bat / update_daily.sh (cập nhật hằng ngày)
run_pipeline.py
```
