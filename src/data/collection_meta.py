"""Tệp mô tả việc thu thập dữ liệu, đặt tên đúng theo Excel.

1) config/collection_config.yaml (Excel C1-02)
   TỰ SINH từ mục `collection` + `assets` của config/config.yaml mỗi lần chạy, kèm thời điểm
   và hash cấu hình. Không sửa tay file này; muốn đổi tham số thì sửa config.yaml.
   Làm vậy để có đúng tên file Excel yêu cầu mà không có hai nơi khai báo tham số.

2) outputs/metadata/limitation_log.csv (Excel C1-08 / C1-09)
   Ghi trạng thái biến MỞ RỘNG (VIX, S&P 500, DXY, funding, open interest):
   - nếu đã thu thập: giữ dòng do collector ghi (độ phủ, giới hạn lịch sử thật)
   - nếu chưa: ghi "chưa thu thập — RQ3 là mở rộng", tuyệt đối không tạo số liệu.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import yaml

from src.utils.config import ROOT, path_of

EXT_VARS = {
    "vix": ("C1-08", "macro_daily.csv"), "spx": ("C1-08", "macro_daily.csv"),
    "dxy": ("C1-08", "macro_daily.csv"),
    "funding_rate": ("C1-09", "derivatives_daily.csv"),
    "open_interest": ("C1-09", "btc_open_interest_recent.csv"),
}
LIMIT_COLS = ["task", "asset", "variable", "status", "reason", "updated_at_utc"]


def write_collection_config(cfg: dict) -> None:
    snap = {
        "_ghi_chu": "TỰ SINH từ config/config.yaml — KHÔNG sửa tay file này",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_config": "config/config.yaml",
        "config_version": cfg["version"], "config_hash": cfg["_config_hash"],
        "source": "Binance Spot (REST /api/v3/klines; dự phòng data.binance.vision)",
        "market": "spot", "timezone": "UTC", "candle_rule": "chỉ nến đã đóng",
        "assets": cfg["assets"], "collection": cfg["collection"],
    }
    with open(ROOT / "config" / "collection_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(snap, f, allow_unicode=True, sort_keys=False)


def write_limitation_log(cfg: dict) -> pd.DataFrame:
    meta, raw = path_of(cfg, "metadata"), path_of(cfg, "raw")
    path = meta / "limitation_log.csv"
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    old = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=LIMIT_COLS)
    if "task" not in old.columns:  # dòng do derivatives_collector ghi (định dạng cũ)
        old.insert(0, "task", "C1-09")
    rows = []
    macro = pd.read_csv(raw / "macro_daily.csv") if (raw / "macro_daily.csv").exists() else None
    for var, (task, fname) in EXT_VARS.items():
        # giữ dòng giới hạn THẬT do collector ghi; bỏ dòng "chưa thu thập" cũ để đánh giá lại
        kept = old[(old["variable"] == var) & (old["status"] != "chưa thu thập")]
        if len(kept):
            rows += kept.to_dict(orient="records")
            continue
        if task == "C1-08" and macro is not None and (macro["series"].str.lower() == var).any():
            s = macro[macro["series"].str.lower() == var]
            rows.append({"task": task, "asset": "ALL", "variable": var, "status": "đã thu thập",
                         "reason": f"{s['date_local'].min()} -> {s['date_local'].max()}, "
                                   f"{len(s)} phiên; nguồn {s['source'].iloc[0]}; "
                                   "ghép as-of theo available_at", "updated_at_utc": now})
        elif var == "funding_rate" and (raw / fname).exists():
            d = pd.read_csv(raw / fname)
            rows.append({"task": task, "asset": "BTC/ETH", "variable": var, "status": "đã thu thập",
                         "reason": f"{d['date_utc'].min()} -> {d['date_utc'].max()} (lịch sử perpetual)",
                         "updated_at_utc": now})
        else:
            rows.append({"task": task, "asset": "ALL", "variable": var, "status": "chưa thu thập",
                         "reason": "RQ3 là mở rộng tùy chọn; chưa được nhóm phê duyệt nên chưa tải. "
                                   "Không tạo số liệu thay thế. Chạy: python -m src.data.collect "
                                   "--skip-core --macro --derivatives",
                         "updated_at_utc": now})
    out = pd.DataFrame(rows)
    for c in LIMIT_COLS:
        if c not in out.columns:
            out[c] = now if c == "updated_at_utc" else ""
    out[LIMIT_COLS].to_csv(path, index=False)
    return out
