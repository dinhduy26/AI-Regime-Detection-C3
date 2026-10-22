# Chuẩn bị commit C2

Chỉ đưa source, test, cấu hình, dependency và tài liệu vào commit. Model, xác suất, bảng/hình và log C2 vẫn giữ local, được `.gitignore` loại khỏi commit và tái tạo bằng `python scripts/complete_c2.py`. Không thay dữ liệu hay source C1/C3.

## Kiểm tra trước commit

```powershell
& "$HOME\.venv\Scripts\Activate.ps1"
python -m pip install -r requirements-dev.txt
python -m ruff check --select E4,E7,E9,F,I src/models scripts/complete_c2.py scripts/verify_c2.py tests/test_c2_models.py tests/test_c2_stability.py tests/test_c2_completion.py
python -m ruff format --check src/models scripts/complete_c2.py scripts/verify_c2.py tests/test_c2_models.py tests/test_c2_stability.py tests/test_c2_completion.py
python scripts/complete_c2.py
```

Nghiệm thu hiện có 75 test, gồm đủ 55 trường hợp lịch sử. Fixture `tests/fixtures/c2_test_inventory.json` lưu danh tính và hash bằng chứng lịch sử; không cần log máy cũ để xác minh trên checkout mới. Đầu vào C1 vẫn cần có theo hướng dẫn dự án.

## Phạm vi stage trong repository Git của nhóm

```powershell
git add .gitignore README.md config/config.yaml requirements.txt requirements-lock.txt requirements-dev.txt
git add src/models scripts/complete_c2.py scripts/verify_c2.py
git add tests/test_c2_models.py tests/test_c2_stability.py tests/test_c2_completion.py tests/fixtures/c2_test_inventory.json
git add docs/C2_commit.md docs/C2_huong_dan_va_nghiem_thu.md docs/C2_nghiem_thu_hoan_thien.md
git diff --cached --check
git diff --cached --stat
```

Nếu tài liệu cũ `docs/C2_kiem_tra_lai.md` đã được track, stage việc xóa file đó. Nếu artifacts đã được track từ trước, `.gitignore` không tự bỏ chúng khỏi index; kiểm tra danh sách stage trước khi commit. Chạy các lệnh stage trong checkout Git của nhóm.

Gợi ý commit message: `feat(c2): add validated HMM/GMM regime pipeline`.

Giới hạn nghiên cứu vẫn giữ nguyên: BTC/ETH còn cảnh báo seed sensitivity; kiểm thử code đạt không đồng nghĩa mô hình ổn định hoặc chiến lược có lợi nhuận. Báo cáo kết quả nằm trong [C2_nghiem_thu_hoan_thien.md](C2_nghiem_thu_hoan_thien.md).
