# C2 HMM GMM và bàn giao xác suất regime

C2 đã triển khai mô hình Gaussian HMM riêng cho BTC và ETH, GMM đối chứng, lựa chọn bằng validation, đánh giá test sau khi khóa lựa chọn và xuất dữ liệu cho C3. Phần này dùng nguyên bộ dữ liệu C1 trong dự án; không tải lại giá, không thay mốc chia mẫu hoặc fit lại scaler. Nguồn dữ liệu và ngày chốt mẫu được ghi trong metadata.

## Chạy lại

Thư mục làm việc là thư mục chứa `run_pipeline.py`. Python yêu cầu từ 3.10; lần nghiệm thu dùng Python 3.12.10 trên Windows và môi trường dùng chung `$HOME\.venv`.

```powershell
& "$HOME\.venv\Scripts\Activate.ps1"
python -m pip install -r requirements-lock.txt
python scripts/complete_c2.py
```

Cũng có thể gọi `python -m src.models.run` để chỉ chạy core, `python -m src.models.diagnostics` để chẩn đoán cửa sổ và `python scripts/verify_c2.py` để xác minh. Các lệnh này không chạy lại C1. Cấu hình C2 nằm trong mục `models` của `config/config.yaml`, phiên bản 1.2.2. Mốc train/validation vẫn giữ nguyên như C1.

## Quy trình và lựa chọn mô hình

Ba đầu vào là `z_log_return`, `z_rv21`, `z_log_dollar_volume`. C2 đối chiếu từng vector chuẩn hóa với thống kê train trong manifest C1; không dùng `simple_return` làm feature. Scaler joblib của C1 được giữ nguyên và ghi hash để truy vết.

HMM dùng covariance diagonal, K=2/3/4, seeds 0–4, tối đa 500 vòng, tol=0.001 và variance floor 0.000001. Mỗi lần fit chỉ nhận train. Độ dài sequence được truyền vào thư viện; ngày thiếu hoặc đổi segment làm khởi tạo lại phân phối state. Ranh giới train → validation → test không làm reset nếu ngày vẫn liên tục.

`hmmlearn` coi chạm giới hạn số vòng là một điều kiện dừng. C2 kiểm thêm mức tăng likelihood cuối: chỉ chấp nhận gain trong khoảng [-0.000001, 0.001), variance hữu hạn không dưới sàn và mọi state có tỷ trọng nhãn hard-filtered train ít nhất 2%. Fit bị loại vẫn có dòng trong B03, kèm nguyên nhân.

Trong các fit hợp lệ, chọn average one-step predictive log-score validation cao nhất. Sai số hòa quy định trước là 1e-8; hòa thì chọn BIC train thấp hơn, sau đó K nhỏ hơn rồi seed nhỏ hơn. Không đổi tiêu chí theo kết quả test. Tham số và mapping train được ghi trong `model_{asset}_selection.json` trước đánh giá test. Sau khi có đủ metadata cuối cùng, C2 lưu `.joblib`, tải lại và so metadata/score; JSON cuối và manifest dùng cùng một đối tượng metadata đã chuẩn hóa, gồm khóa mapping dạng chuỗi như JSON.

GMM có cùng K với HMM đã chọn, covariance diagonal và cùng dữ liệu chuẩn hóa; thử 5 seed độc lập, chọn khởi tạo hội tụ có likelihood train cao nhất. Cluster ID của GMM độc lập với state ID của HMM.

## Kết quả lần nghiệm thu

Dữ liệu trong gói C1 kết thúc ngày 2026-10-18 UTC. Mỗi tài sản có 1.891 ngày train, 630 ngày validation và 632 ngày test. Train kết thúc 2023-05-05, validation kết thúc 2025-01-24; test bắt đầu 2025-01-25. Bảng dưới được lấy từ B03 và B07 của lần chạy, không phải ví dụ giả định.

| Tài sản | K HMM | Seed | Log score validation HMM | Log score test HMM | Log score test GMM |
|---|---:|---:|---:|---:|---:|
| BTC | 3 | 1 | -2.542605 | -2.395548 | -3.714177 |
| ETH | 4 | 1 | -2.187357 | -2.323816 | -3.430634 |

HMM gán mật độ có điều kiện theo lịch sử; GMM gán mật độ biên. Cùng biến, đơn vị và ngày quan sát cho phép đặt hai score cạnh nhau, nhưng đây không phải classification accuracy hay bằng chứng về hiệu quả chiến lược. C3 chưa có backtest trong repo này.

Nhãn được khóa bằng volatility log return có trọng số filtered trên train. State có volatility lớn nhất mang nhãn `High-Vol` và cờ `high_risk_state`; thấp nhất là `Low-Vol`. State còn lại giữ tên trung tính. BTC có high-risk state 0, ETH có high-risk state 2 trong lần chạy này. Không diễn giải trùng ID là cùng một regime giữa hai tài sản hoặc với GMM.

## Danh mục bàn giao

| File hoặc nhóm file | Nội dung |
|---|---|
| `outputs/tables/B03_model_selection.csv` | Đủ 30 ứng viên asset × K × seed, likelihood, AIC/BIC, validation score, điều kiện hợp lệ, lý do loại và lựa chọn |
| `outputs/tables/B04_regime_statistics.csv` | Thống kê filtered theo từng state và train/validation/test, share hard/soft, volatility ngày/năm, quote volume, run length và thời lượng mô hình |
| `outputs/tables/B05_transition_matrix.csv` | Xác suất từ state i sang j, persistence mô hình và thời lượng kỳ vọng theo ngày |
| `outputs/tables/B07_hmm_vs_gmm.csv` | Cùng ngày/cùng scale, log-score, số cặp ngày hợp lệ, flips/100 ngày và persistence thực nghiệm |
| `outputs/tables/GMM_initializations.csv` | Đủ 5 khởi tạo GMM mỗi tài sản, seed, hội tụ và lựa chọn |
| `outputs/tables/C2_seed_stability.csv` | ARI giữa các seed hợp lệ cùng K trên validation; không phụ thuộc hoán vị state ID |
| `outputs/tables/C2_stability_summary.csv` | Từng K, số seed/pair hợp lệ, min/median/mean/max ARI, bất đồng high-risk và trạng thái stable/unstable/thiếu bằng chứng |
| `outputs/predictions/regime_filtered_{btc,eth}.csv` | Xác suất nhân quả từng tài sản, K, mapping, high-risk, as_of, model ID, xác suất high-risk ngày kế tiếp |
| `outputs/predictions/regime_smoothed_{btc,eth}.csv` | Xác suất hindsight, `hindsight=true`, không có thời điểm tín hiệu để giao dịch |
| `outputs/predictions/regime_gmm_{btc,eth}.csv` | Posterior cụm GMM; mapping riêng |
| `outputs/predictions/regime_{filtered,smoothed,gmm}.csv` | File ghép hai tài sản; cột của state không tồn tại ở một tài sản để trống, phải đọc theo K |
| `outputs/models/hmm_{btc,eth}.joblib`, `gmm_{btc,eth}.joblib` | Model fit trên train và metadata cuối cùng, đã tải lại để kiểm tra |
| `outputs/metadata/model_{btc,eth}_selection.json` | Bản ghi khóa K/seed/mapping trước đánh giá test; tách rõ với metadata cuối |
| `outputs/metadata/model_{btc,eth}.json` | Mapping, transition, seed, train/scaler provenance, dấu thời gian khóa và kết quả kiểm tra |
| `outputs/metadata/model_manifest.json` | Trạng thái hoàn thành, cấu hình, phiên bản thư viện, hash đầu ra và giới hạn |
| `outputs/figures/H02_*`, `H03_*`, `H04_*` | Hình riêng BTC/ETH, PNG 300 dpi và SVG, chú thích source/causal/hindsight |
| `outputs/tables/C2_window_sensitivity.csv`, `C2_window_candidates.csv` | Chẩn đoán RV14/RV21/RV30 trên cùng mẫu train/validation, không thay core |
| `outputs/tables/C2_seed_fit_diagnostics.csv`, `C2_state_coverage.csv` | Hội tụ, độ phân tán likelihood và độ phủ state theo giai đoạn |
| `outputs/metadata/C2_C3_integration_check.json` | Kết quả consumer thực tế ghép tín hiệu t với return t+1 |

B04 dùng mean/volatility của log return theo trọng số xác suất, variance mẫu số bằng tổng trọng số. `share` là hard occupancy; `soft_share` là trung bình xác suất. Volatility năm nhân sqrt(365). Run length quan sát bị cắt tại ranh giới giai đoạn/segment; expected duration của mô hình là 1/(1-p_kk), có thể vô hạn khi p_kk=1. Persistence thực nghiệm trong B07 chỉ đếm cặp ngày liên tục bên trong giai đoạn, không được đồng nhất với p_kk.

## C3 tiếp nhận

```python
from src.models.handoff import load_filtered
from src.utils.config import load_config

signals, metadata = load_filtered(load_config(), "BTC", period="test", acknowledge_unstable=True)
k = metadata["K"]
high_risk = metadata["high_risk_state"]
```

Reader kiểm trạng thái `complete`, hash file, model ID, tổng xác suất và thời điểm tín hiệu. Dùng reader riêng cho từng tài sản vì BTC và ETH có thể khác K.

`metadata["seed_stability_status"]` cho biết cảnh báo độ ổn định của K được chọn. `load_filtered(..., require_stable=True)` từ chối cả hai mô hình hiện tại. Nếu dùng cho nghiên cứu, phải truyền rõ `acknowledge_unstable=True`; reader phát `UnstableModelWarning` và lưu quyết định vào `signals.attrs["c2_use_policy"]`. Không truyền xác nhận thì mô hình unstable bị từ chối. `require_stable=True` vẫn từ chối kể cả đã xác nhận nghiên cứu. `status=complete` chỉ nói pipeline hoàn thành, không bảo đảm mô hình stable.

`prepare_next_day_research_inputs(cfg, asset, acknowledge_unstable=True)` minh họa giao diện tiêu thụ: lấy filtered của t, ghép return thực của đúng ngày t+1 và cùng segment, yêu cầu `as_of_utc` không muộn hơn đầu khoảng return. Khi lấy test, hàm giữ tín hiệu ngày cuối validation cho return đầu test. `python -m src.models.integration` chạy consumer này cho BTC/ETH; đây không phải portfolio/NAV hoặc backtest đầy đủ của C3.

`date_utc=t` là ngày mở nến; `as_of_utc` lấy từ thời điểm sẵn có của giá C1, thường là 00:00 UTC ngày t+1. Đây là mốc dữ liệu đủ để tính tín hiệu cuối ngày, không phải bảo đảm độ trễ thực thi bằng 0. C3 dùng filtered của nến t để mô phỏng trọng số cho return ngày t+1 theo quy ước close-to-close; phải kiểm ngày kề nhau, cùng segment và thời điểm được phép dùng. Chỉ số `prob_high_risk_next_day` là alpha_t @ P, không phải dự báo giá.

`regime_smoothed.csv` phụ thuộc tương lai toàn segment và chỉ dùng đọc lịch sử. C3 không dùng file này cho backtest. Nếu model run thất bại, metadata có `status=failed`; không dùng các file còn lại từ lần chạy dở.

Model ID phụ thuộc dữ liệu train/validation, scaler, cấu hình mô hình, phiên bản thư viện và mã tính model. Nối thêm test không thay đổi danh tính mô hình nếu các thành phần này giữ nguyên. C2 hiện fit lại có seed cố định khi được gọi; không cập nhật tham số theo validation/test.

## Nghiệm thu và giới hạn

Đợt hoàn thiện có 75 test: đủ 55 trường hợp của giai đoạn trước và thêm 20 regression cases cho metadata, tải lại artifact, chính sách xác nhận nghiên cứu, schema giả mạo, thời điểm tín hiệu và scaler chẩn đoán. Các kiểm tra cũ được điều chỉnh theo giao diện xác nhận mới và kiểu khóa mapping JSON thống nhất, không bỏ trường hợp nào. C2 kiểm ví dụ forward hai state của đặc tả, đối chiếu smoothing với phép liệt kê xác suất chính xác và thư viện, ổn định số học khi xác suất rất nhỏ, prefix consistency, likelihood, gap reset, loại fit, fit không nhận test, mapping train-only, độc lập hai tài sản, schema/thời điểm và reader C3. Các bài kiểm xác suất dùng ví dụ toán học; kết quả nghiên cứu dùng dữ liệu C1 thực có.

Đã thực hiện chẩn đoán độ nhạy theo nhiều seed. K đang chọn của BTC có ARI trung vị khoảng -0.035005; ETH khoảng 0.026412 trên validation. Đây là bằng chứng nhãn regime phụ thuộc khởi tạo, không phải kết quả ổn định tốt. `C2_stability_summary.csv` báo `unstable` cho cả hai, đồng thời vẫn báo dòng K không có fit hợp lệ. Không thay K/seed để làm đẹp độ ổn định hoặc điểm test.

Các ngưỡng cảnh báo được bổ sung sau lần chạy đầu: tối thiểu 3 seed hợp lệ, median ARI >= 0.8 và min ARI >= 0.5; phải có đầy đủ các cặp seed. Đây là quy ước mô tả vận hành trong `models.stability`, không phải kiểm định ý nghĩa thống kê hoặc ngưỡng được đăng ký trước thí nghiệm ban đầu. Thay ngưỡng không ảnh hưởng lựa chọn mô hình. Bảng pair còn đo tỷ lệ bất đồng nhãn high-risk và MAE xác suất high-risk; state high-risk của mỗi seed được xác định riêng bằng volatility train.

`scripts/verify_c2.py` khởi chạy pytest trong tiến trình Python mới, không dùng pytest cache, ghi JUnit/log và `outputs/metadata/C2_verification.json`. Script đối chiếu danh tính với cả baseline 47 và baseline 55, không cho test bỏ qua, kiểm hash artifact/source/config, đối chiếu toàn bộ metadata HMM/GMM/JSON/manifest và CSV, kiểm 12 score B07 bằng likelihood prefix của HMM và API GMM trực tiếp, kiểm đúng 46 dependency đã pin và `pip check`, kiểm dữ liệu/source không thay đổi trong lúc xác minh. Consumer C3 được chạy thêm và lưu bằng chứng chính sách cảnh báo. Đây là xác minh cục bộ trên cùng máy/môi trường, không phải đánh giá bên thứ ba hoặc máy thứ hai.

Danh tính 47/55 test được lưu trong fixture `tests/fixtures/c2_test_inventory.json`, có hash của bằng chứng lịch sử. Log baseline có thể giữ local để đối chiếu thêm, nhưng không bắt buộc khi clone repository. Model, dự đoán, bảng/hình và log C2 sinh tự động được `.gitignore` loại khỏi commit; chạy `complete_c2.py` để tái tạo từ đầu vào C1 hiện có. Quy trình chuẩn bị commit và lint nằm trong [C2_commit.md](C2_commit.md).

Đã chạy thêm RV14/RV21/RV30 với 5 seed, giữ K của core, dùng cùng 1.882 ngày train và 630 ngày validation để so độ nhạy. Scaler từng phương án chỉ học từ train, không ghi đè scaler C1; không dùng test, không thay model chính. RV21 cũng được refit trên mẫu chung để cho thấy ảnh hưởng bỏ 9 ngày đầu. Không so likelihood giữa các cửa sổ như cùng biến quan sát. Cả sáu nhóm asset × window vẫn có ít nhất một tiêu chí ổn định chưa đạt. Chưa bootstrap hoặc kiểm định khác biệt có khoảng tin cậy.

Tỷ lệ bất đồng high-risk trên toàn bộ ngày thấp chưa đủ chứng minh tín hiệu đáng tin: khi chỉ xét các ngày ít nhất một seed báo high-risk, tỷ lệ bất đồng trung bình là 39% BTC và khoảng 74.74% ETH (ETH có 9 cặp có sự kiện, một cặp không có sự kiện để tính). ETH High-Vol chỉ có 2/632 hard-label trên test; state 0 không có hard-label test. Xem thêm `C2_state_coverage.csv`. Đây là hạn chế đáng kể khi đánh giá sự kiện hiếm, không phải bằng chứng về độ chính xác cảnh báo.

Seed có thể rơi vào cực trị địa phương khác nhau; xem toàn bộ B03 và ARI, không chỉ dòng thắng. Các thống kê return/volatility train đã dùng feature đó để fit nên không phải kiểm định độc lập. Val/test là mô tả ngoài mẫu dưới mô hình và mapping đã khóa. Test đã được báo ở lần chạy đầu; đợt này không phải một holdout mới chưa từng xem. Các thay đổi bổ sung chẩn đoán/xác minh, giữ tiêu chí chọn mô hình và không điều chỉnh theo test.

RQ3, NHHMM, K-means và B06 chưa thực hiện. C2 không xuất hệ số/p-value giả. Khối lượng giao dịch không đo đầy đủ liquidity. Rolling RV21 có cửa sổ chồng lấn nên giả định độc lập có điều kiện chỉ là xấp xỉ.

A-02 mới hoàn tất kiểm tra kỹ thuật C1 → C2 và khả năng đọc tín hiệu của C3. Buổi ghép C3/backtest thực tế, biên bản và xác nhận của cả ba thành viên vẫn cần nhóm thực hiện; không coi mã kiểm thử là chữ ký nghiệm thu của nhóm.

Tham khảo API triển khai: [hmmlearn GaussianHMM và convergence monitor](https://hmmlearn.readthedocs.io/en/stable/api.html), [scikit-learn GaussianMixture](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html).
