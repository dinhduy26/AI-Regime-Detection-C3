#!/usr/bin/env bash
# Cập nhật dữ liệu hằng ngày (macOS/Linux). Lên lịch bằng cron, ví dụ 07:05 giờ VN:
#   5 7 * * * /đường/dẫn/crypto-regime-radar/scripts/update_daily.sh
cd "$(dirname "$0")/.." || exit 1
source .venv/bin/activate
{
  echo "===== $(date) ====="
  python run_pipeline.py --download
  python -m pytest -q
} >> outputs/metadata/update_log.txt 2>&1
