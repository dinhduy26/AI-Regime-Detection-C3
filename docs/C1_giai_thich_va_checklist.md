# C1: Giải thích quyết định và checklist nghiệm thu

Tài liệu này giúp người phụ trách C1 trả lời câu hỏi khi bảo vệ và đối chiếu từng nhiệm vụ trong bảng phân công với file code tương ứng.

## 1. Nhiệm vụ → file code → đầu ra

| Mã | Nhiệm vụ | File code | Đầu ra |
|---|---|---|---|
| A-01 | Hợp đồng tích hợp | `config/schema.yaml`, `docs/A01_data_contract.md` | Manifest schema duy nhất |
| C1-01 | Môi trường, thư mục | `requirements.txt`, `pyproject.toml`, `.env.example` | Repo chạy được |
| C1-02 | Nguồn, UTC, coverage | `config/config.yaml`, `src/data/collect.py`, `src/data/collection_meta.py` | `source_log.csv`, `config/collection_config.yaml` |
| C1-03 | Collector BTC | `src/data/binance_collector.py` | `data/raw/btc_ohlcv.csv` |
| C1-04 | Collector ETH | (cùng file, cùng schema) | `data/raw/eth_ohlcv.csv` |
| C1-05 | Làm sạch, QA, đối soát | `src/data/clean.py`, `src/data/crosscheck.py` | `prices_daily.parquet`, `gap_report.csv`, `data_quality.csv`, `price_crosscheck.csv` |
| C1-06 | Log return, RV21 | `src/features/build.py` | `features_{asset}.parquet` |
| C1-07 | Quote volume, Amihud, tương quan | `src/features/build.py` | `log_dollar_volume`, `amihud_illiq`, `btc_eth_corr21` |
| C1-08 | VIX/DXY/S&P, trái phiếu Mỹ; stablecoin, DeFi TVL, market cap, CMC200 | `src/data/macro_collector.py`, `src/data/crypto_market_collector.py`, `src/features/extended.py` | `macro_daily.csv`, `crypto_market_daily.csv`, `features_ext_{asset}.parquet` (gồm `spx_return_1d`, `dxy_return_1d`, `vix_change_1d`, `term_spread_10y_2y`, `*_growth_7d`) |
| C1-09 | Funding, open interest lịch sử | `src/data/derivatives_collector.py`, `src/data/collection_meta.py` | `derivatives_daily.csv`, `open_interest_daily.csv`, `limitation_log.csv` |
| (blueprint §3.1) | SOL, BNB | `src/features/extension_assets.py` | `data/processed/extension/*` |
| C1-10 | Ghép ma trận, chia mẫu | `src/features/build.py`, `src/features/pipeline.py` | `split_manifest.json` |
| C1-11 | Scaler chỉ trên train | `src/features/scaling.py` | `scaler_{asset}.joblib`, `scaled_features_{asset}.parquet` |
| C1-12 | Từ điển, B01/B02/H01 | `src/evaluation/descriptives.py` | `B01`, `B02`, `H01`, `data_dictionary.csv` |
| C1-13 | Pipeline chạy lại | `run_pipeline.py`, `scripts/update_daily.*` | `run_manifest.json`, `update_log.txt` |
| (hỗ trợ C3) | Giá tham khảo, ngày dữ liệu | `src/data/live_quote.py`, `data_through_utc` | Chỉ để hiển thị |
| Kiểm thử | 10 tiêu chí §1.11, hàm giao tiếp §3.2, dữ liệu mở rộng | `tests/test_c1_acceptance.py`, `tests/test_pipeline.py`, `tests/test_extensions.py` | `pytest -q` → 30 passed |

## 2. Các câu hỏi có thể bị hỏi khi bảo vệ

**Vì sao dùng quote volume (USDT) mà không dùng base volume?**
Base volume đo bằng số BTC hoặc ETH, nên đơn vị khác nhau giữa hai đồng và phụ thuộc giá. Ví dụ: 1.000 BTC năm 2018 và 1.000 BTC năm 2024 có giá trị khác hẳn nhau. Quote volume đo bằng USDT, phản ánh quy mô giao dịch theo giá trị và so sánh được giữa BTC với ETH.

**Vì sao lấy log của (1 + volume)?**
Volume lệch phải rất mạnh. Lấy log nén các ngày đột biến để phân phối gần chuẩn hơn, hợp với giả định Gaussian của HMM. Cộng 1 để tránh log(0).

**Vì sao RV21 dùng ddof = 1, còn scaler dùng ddof = 0?**
RV21 là *ước lượng* độ lệch chuẩn từ một mẫu nhỏ 21 quan sát, nên dùng mẫu số n − 1 để không bị chệch. Scaler chỉ là phép *biến đổi* chuẩn hóa theo đúng quy ước của `StandardScaler` (ddof = 0). Đặc tả yêu cầu không trộn hai đại lượng này. Test số 8 kiểm tra tham số đã lưu khớp với công thức ddof = 0.

**Vì sao không fit scaler trên toàn bộ dữ liệu?**
Nếu fit trên toàn mẫu thì mean và std đã "nhìn thấy" dữ liệu validation và test, tức là thông tin tương lai rò vào quá khứ (look-ahead bias). Scaler chỉ học từ train. Test số 8 thay toàn bộ dữ liệu val/test bằng giá trị khác và kiểm tra rằng tham số scaler không đổi.

**Gặp ngày thiếu dữ liệu thì xử lý thế nào?**
Code xử lý theo bốn bước:
1. Thử tải bù ngày đó từ chính Binance (kho data.binance.vision).
2. Nếu vẫn thiếu thì **không điền, không nội suy**, vì nội suy dùng giá của ngày sau, tức là dùng tương lai.
3. Chuỗi bị tách thành các đoạn liên tục (`segment_id`). Return và RV21 không bao giờ tính vắt qua khoảng thiếu, vì nếu vắt qua thì return của 2 ngày bị coi là return 1 ngày.
4. Mặc định HMM dùng đoạn liên tục dài nhất. Mọi thứ được ghi trong `gap_report.csv`.

**Làm sao chắc chắn không có look-ahead?**
- Mọi rolling window đều là trailing, chỉ nhìn từ t−20 đến t.
- Chỉ giữ nến đã đóng.
- Test số 4 cắt bỏ tương lai, hoặc bóp méo giá tương lai (nhân 1,7 lần), rồi kiểm tra rằng feature của mọi ngày trong quá khứ **giống hệt**. Test chạy ở 3 mốc cắt khác nhau.

**Vì sao chia 60/20/20 theo thời gian mà không chia ngẫu nhiên?**
Chuỗi thời gian có phụ thuộc theo thời gian. Chia ngẫu nhiên sẽ cho mô hình học từ năm 2025 rồi "dự báo" năm 2022. Ba tập phải nối tiếp nhau và không chồng lấn. Test chọn K chỉ dùng validation; tập test được giữ nguyên đến bước đánh giá cuối.

**Dữ liệu cập nhật mỗi ngày thì chia mẫu thế nào?**
Mốc chia được cố định theo ngày: train đến 05/05/2023, validation đến 24/01/2025. Hai mốc này được chốt bằng tỷ lệ 60/20/20 trên dữ liệu đến 18/10/2026. Ngày mới chỉ nối vào cuối tập test. Nếu chia lại 60/20/20 mỗi ngày thì một phần dữ liệu test sẽ bị chuyển sang validation, tức là test "bị đụng tới", và scaler cùng mô hình phải fit lại. Cách cố định mốc giữ được yêu cầu "frozen models + test" (Excel C2-12) mà dữ liệu vẫn mới nhất. Test `test_11` cắt bớt 30 và 120 ngày cuối của dữ liệu thật rồi kiểm tra rằng train, validation và scaler giống hệt.

**Vì sao không làm realtime?**
Mô hình được định nghĩa trên nến ngày (RV21, return ngày, volume ngày). Nến đang hình thành chưa phản ánh đủ một ngày, và đặc tả §1.2 cấm dùng nến này. Dữ liệu cập nhật mỗi sáng sau 07:00 giờ Việt Nam, khi nến UTC hôm trước đã đóng. Giá trực tiếp trên dashboard chỉ để tham khảo.

**Vì sao bỏ khoảng 21 dòng đầu (warm-up)?**
RV21 cần 21 log return, tức 22 giá đóng cửa liên tiếp. Các dòng chưa đủ cửa sổ có RV21 trống nên không đưa vào mô hình. Số dòng bị bỏ được **đếm thật** và ghi vào manifest, không ghi cứng là 21.

**Vì sao skewness và kurtosis tính theo moment không hiệu chỉnh?**
Theo đúng công thức đặc tả §1.9: skew = m3 / m2^1.5 và kurtosis = m4 / m2² (Pearson). Phân phối chuẩn có kurtosis bằng 3. Nếu cần excess kurtosis thì trừ 3, và bảng đã ghi rõ loại kurtosis đang dùng.

**Vì sao dữ liệu vĩ mô có độ trễ (lag)?**
Giá đóng cửa Mỹ của ngày d chỉ có sau khi phiên Mỹ đóng. Code dùng quy ước thận trọng: giá ngày d chỉ được dùng cho nến crypto từ ngày d+1 trở đi, ghép bằng quy tắc as-of `available_at ≤ thời điểm nến đóng`. Thứ Bảy và Chủ nhật dùng giá thứ Sáu, **không bao giờ** dùng giá thứ Hai.

**Open interest lấy lịch sử ở đâu?**
API Binance chỉ trả 30 ngày gần nhất, nên C1 lấy lịch sử từ file "metrics" (ảnh chụp 5 phút) mà Binance công bố trên data.binance.vision. Giá trị của ngày t là ảnh chụp cuối cùng trong ngày UTC t, tức đã có trước khi nến t đóng. Ngày nào Binance không có file thì để trống.

**Vì sao các biến mở rộng phải trễ 1 ngày?**
Dữ liệu vĩ mô, trái phiếu, stablecoin… được công bố sau khi ngày kết thúc. Để chắc chắn không dùng thông tin chưa công bố, giá trị của ngày d chỉ được dùng cho nến crypto từ ngày d+1. Test `test_ext_point_in_time` kiểm tra từng giá trị đã ghép.

**Vì sao lợi suất S&P 500 (và thay đổi VIX, DXY) không tính từ cột giá trong `features_ext`?**
Sau khi ghép theo lịch crypto 7 ngày, thứ Bảy và Chủ nhật lặp lại giá thứ Sáu. Nếu tính return trên chuỗi đó thì cuối tuần ra return giả bằng 0 và return thứ Hai bị lệch. Vì vậy C1 tính `spx_return_1d = ln(P_d / P_phiên trước)` trên **chuỗi phiên giao dịch Mỹ gốc**, rồi mới ghép as-of. Nếu nguồn thiếu nhiều phiên (phiên trước cách quá 5 ngày) thì để trống, không gộp nhiều phiên thành một return. Test `test_ext_session_returns_from_raw_sessions` tính lại từng giá trị từ `macro_daily.csv` gốc để kiểm chứng.

**Vì sao có biến bị thiếu hoặc lịch sử ngắn (CMC200, market cap, OI, stablecoin)?**
Đây là giới hạn của nguồn, không phải lỗi code. Yahoo đã không còn dữ liệu CMC200; CoinGecko miễn phí chỉ trả 365 ngày market cap; Binance chỉ có file open interest từ 09/2020 (BTC) và 12/2021 (ETH); DefiLlama trước 2020 theo dõi stablecoin chưa đầy đủ (cuối 2018 chỉ ghi khoảng 0,4 tỷ USD). Theo nguyên tắc "không tuyên bố có dữ liệu nếu nguồn không đáp ứng", C1 không tự tạo số liệu thay thế mà ghi rõ trạng thái từng biến trong `limitation_log.csv` và độ phủ trong `extension_coverage.csv`, để nhóm bài báo đưa vào phần giới hạn nghiên cứu.

**Dữ liệu có đáng tin không, kiểm tra thế nào?**
Hai lớp đối soát (`python -m src.data.crosscheck`): (A) so với file lịch sử Binance công bố riêng trên data.binance.vision, kết quả khớp tuyệt đối 0/244 dòng lệch; (B) so với Yahoo BTC-USD/ETH-USD (giá tổng hợp nhiều sàn), chênh lệch trung vị khoảng 0,09%, tương quan return ngày 0,991 (BTC) và 0,994 (ETH). Các ngày lệch lớn rơi vào giai đoạn 2017–2018, đợt USDT mất neo tháng 10/2018 và ngày 12/03/2020, đều giải thích được.

## 3. Checklist trước khi giao cho C2/C3

- [ ] `python -m src.data.collect` chạy thành công; `source_log.csv` có ngày bắt đầu/kết thúc thực tế
- [ ] `python run_pipeline.py` báo OK cho clean, features, reports
- [ ] `python -m src.data.collect --skip-core --all-extensions` rồi `python run_pipeline.py` để dữ liệu mở rộng mới nhất
- [ ] `pytest -q` báo **30 passed**, không còn test nào bị skip
- [ ] `python -m src.data.crosscheck` không có tháng nào lệch ở lớp A
- [ ] Đã xem `gap_report.csv` và `data_quality.csv`, đặc biệt các ngày có |log return| > 0,30 (đối chiếu với sự kiện thật)
- [ ] Đã xem hình H01 và `limitation_log.csv`
- [ ] Tạo `requirements-lock.txt` dạng ASCII (PowerShell: `pip freeze | Out-File -Encoding ascii requirements-lock.txt`) và commit file này
- [ ] Khi nén hoặc đưa repo lên GitHub, bỏ `.venv`, `__pycache__`, `.pytest_cache`
- [ ] (Tùy chọn) Đặt Task Scheduler chạy `scripts\update_daily.bat` lúc 07:05 mỗi ngày
- [ ] Mỗi lần C2 fit mô hình, ghi lại `data_through_utc` và `config_version` của lần chạy đó
- [ ] C2, C3 đọc mục 7 của `README.md` và ký xác nhận bảng trong `docs/A01_data_contract.md`
- [ ] Báo R2 cập nhật Methodology §3.7 theo cách chia mẫu cố định ngày (train ≤ 2023-05-05 < validation ≤ 2025-01-24 < test)
