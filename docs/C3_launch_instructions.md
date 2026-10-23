# Hướng dẫn chạy demo Crypto Market Regime Radar (C3-14)

Áp dụng cho Windows, macOS và Linux, Python 3.10 trở lên. Mọi lệnh chạy từ thư mục gốc dự án (thư mục chứa `run_pipeline.py`).

## 1. Cài đặt một lần

```bash
python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements-lock.txt
```

## 2. Tạo đầu ra (khi clone mới hoặc sau khi cập nhật dữ liệu)

Đầu ra C2 và C3 không được commit, nên máy mới phải chạy lại:

```bash
python run_pipeline.py --steps models backtest   # HMM/GMM, xác suất, B03-B05, B07, H02-H04; rồi B08, backtest_nav, H05
python scripts/complete_c2.py                    # (tùy chọn) thêm chẩn đoán RV14/RV30 và kiểm tra toàn bộ
```

Chạy `complete_c2.py` sau bước backtest: script này yêu cầu mọi test đạt và không test nào bị bỏ qua, nên nếu chạy trước khi có kết quả backtest thì các test C3 bị bỏ qua và script báo lỗi.

Cập nhật dữ liệu mới nhất (sau 07:05 giờ Việt Nam, khi nến ngày UTC đã đóng):

```bash
python run_pipeline.py --download          # tải lại, chạy C1 -> C2 -> C3
```

Mốc train/validation cố định, nên dữ liệu mới chỉ nối vào cuối tập test; mô hình C2 không đổi, còn backtest test dài thêm.

## 3. Mở dashboard

```bash
streamlit run dashboard/app.py
```

Trình duyệt mở `http://localhost:8501`. Chọn BTC hoặc ETH ở thanh bên; menu bên dưới chuyển giữa 4 trang Overview, Regime History, Model Comparison, Portfolio & Risk. Theme lấy từ `.streamlit/config.toml`, nên cần chạy lệnh từ thư mục gốc dự án. Nếu thiếu file, trang hiện danh sách file thiếu và lệnh khắc phục.

## 4. Kiểm tra trước khi demo

```bash
pytest -q
```

Toàn bộ test phải đạt, không lỗi. Riêng C3: `pytest -q tests/test_c3_backtest.py tests/test_c3_dashboard.py`.

## 5. Chụp H06 cho bài báo (tùy chọn)

Để server Streamlit chạy ở một cửa sổ, mở cửa sổ khác:

```bash
pip install playwright
python -m playwright install chromium
python scripts/capture_dashboard.py
```

Ảnh được ghi vào `outputs/figures/H06_dashboard.png` và `H06_dashboard_tab{1..4}_{btc,eth}.png`, thời điểm chụp ở `outputs/metadata/H06_capture.json`. Không cài được Playwright thì chụp màn hình thủ công và đặt đúng tên file trên.

## 6. Lỗi thường gặp

| Hiện tượng | Cách xử lý |
|---|---|
| `C2 model run is incomplete` hoặc thiếu `regime_filtered_*.csv` | Chạy `python scripts/complete_c2.py` |
| `C1 feature file changed since model run` | Dữ liệu C1 đã đổi sau lần chạy C2: chạy lại C2 rồi backtest |
| `unstable research model` | Mô hình C2 chưa đạt ổn định seed; `config/backtest.yaml` đã đặt `acknowledge_unstable: true` cho mục đích nghiên cứu, kiểm tra lại nếu bị sửa |
| Cổng 8501 bận | `streamlit run dashboard/app.py --server.port 8502` |
| `pip` báo lỗi đọc requirements trên Windows | Giữ file requirements ở dạng ASCII (xem README mục 9) |
