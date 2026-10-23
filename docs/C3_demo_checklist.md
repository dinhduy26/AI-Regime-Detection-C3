# Checklist demo 3-5 phút (C3-14)

## Trước buổi demo

- [ ] Cài bằng `requirements-lock.txt`; `pytest -q` đạt toàn bộ.
- [ ] Đã chạy `python scripts/complete_c2.py` và `python run_pipeline.py --steps backtest` trên dữ liệu mới nhất.
- [ ] Ghi lại ngày dữ liệu (`data_through_utc`) để nói khi trình bày.
- [ ] Mở sẵn `streamlit run dashboard/app.py`; thử chuyển BTC/ETH và 4 trang trên menu thanh bên.
- [ ] Chuẩn bị video dự phòng hoặc ảnh H06 nếu máy demo không có mạng hoặc lỗi môi trường.

## Kịch bản trình bày

1. **Overview (30 giây):** ngày dữ liệu cuối (không phải realtime), regime filtered hiện tại, xác suất từng state, quy tắc w = 1 / 0,25. Nhắc banner `unstable` của C2.
2. **Regime History (60 giây):** giá tô nền theo state; chuyển sang Smoothed để cho thấy cảnh báo hindsight và vì sao backtest chỉ dùng filtered; ma trận chuyển và thời lượng kỳ vọng.
3. **Model Comparison (60 giây):** K chọn theo validation, test không tham gia; HMM so với GMM về log-score và flips; bảng độ ổn định seed.
4. **Portfolio & Risk (60-90 giây):** NAV và drawdown so với buy-and-hold ở kịch bản baseline, rồi hai kịch bản kiểm tra chéo; nêu trung thực rằng chiến lược không thắng buy-and-hold trên test.
5. **Kết (30 giây):** luồng dữ liệu C1 → C2 → C3, cách tái lập bằng một lệnh, giới hạn chính.

## Giới hạn cần nói rõ

- Cập nhật offline theo nến ngày UTC đã đóng, không phải realtime.
- Kết quả là mô phỏng lịch sử với phí giả định 0,10%, không gồm slippage, spread, thuế; không phải khuyến nghị đầu tư.
- Mô hình C2 nhạy với seed (unstable); số ngày phòng thủ trên test rất ít, đặc biệt với ETH.
- RQ3 (drivers), NHHMM và volatility targeting chưa thực hiện.

## Hướng cải tiến có thể đề xuất

- Ổn định mô hình: nhiều seed hơn, chọn theo đồng thuận giữa các seed, hoặc rolling refit có kiểm soát.
- Kiểm định khác biệt hiệu suất bằng block bootstrap theo khối ngày liên tục.
- Bổ sung kịch bản độ nhạy được khai báo trước (ngưỡng, phí) và tách rõ khỏi baseline.
