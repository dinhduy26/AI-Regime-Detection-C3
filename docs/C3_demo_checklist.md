# Checklist demo 3-5 phút (C3-14)

## Trước buổi demo

- [ ] Cài bằng `requirements-lock.txt`; `pytest -q` đạt toàn bộ.
- [ ] Đã chạy `python run_pipeline.py --steps models backtest` (hoặc `--download` để lấy dữ liệu mới nhất).
- [ ] Ghi lại ngày dữ liệu (`data_through_utc`) để nói khi trình bày.
- [ ] Mở sẵn `streamlit run dashboard/app.py`; thử chuyển BTC/ETH và 4 trang trên menu thanh bên.
- [ ] Chuẩn bị video dự phòng hoặc ảnh H06 nếu máy demo không có mạng hoặc lỗi môi trường.

## Kịch bản trình bày

1. **Overview (30-45 giây):** ngày dữ liệu cuối (không phải realtime), regime hiện tại và số ngày đã ở regime đó, P(rủi ro cao) và tỷ trọng theo quy tắc w = 1 / 0,25; thước đo RV21; dải regime BTC và ETH để thấy hai đồng có cùng lúc vào giai đoạn rủi ro hay không. Nhắc nhãn `unstable` của C2.
2. **Regime History (60 giây):** giá tô nền theo state; biểu đồ "Sau khi vào regime, giá đi đâu?" để trả lời câu hỏi regime có ích gì; phân phối lợi suất cho thấy High-Vol biến động rộng hơn; sơ đồ chuyển trạng thái; chuyển sang Smoothed để cho thấy cảnh báo hindsight và vì sao backtest chỉ dùng filtered.
3. **Model Comparison (60 giây):** dải HMM so với GMM (GMM đổi nhãn liên tục, HMM bền hơn); log-score và số lần đổi state; K chọn theo validation, test không tham gia; độ ổn định seed.
4. **Portfolio & Risk (60-90 giây):** NAV và drawdown so với buy-and-hold ở kịch bản baseline, heatmap lợi suất theo tháng và độ biến động trượt, rồi hai kịch bản kiểm tra chéo; nêu trung thực rằng chiến lược không thắng buy-and-hold trên test.
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
