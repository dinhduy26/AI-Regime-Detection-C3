"""Tệp mô tả việc thu thập dữ liệu, đặt tên đúng theo Excel.

1) config/collection_config.yaml (Excel C1-02)
   TỰ SINH từ mục `collection` + `assets` của config/config.yaml mỗi lần chạy, kèm thời điểm
   và hash cấu hình. Không sửa tay file này; muốn đổi tham số thì sửa config.yaml.
   Làm vậy để có đúng tên file Excel yêu cầu mà không có hai nơi khai báo tham số.

2) outputs/metadata/limitation_log.csv (Excel C1-08 / C1-09)
   Trạng thái MỌI biến mở rộng có trong 3 tài liệu (Excel, tài liệu coder, blueprint): vĩ mô,
   trái phiếu Mỹ, stablecoin, DeFi TVL, market cap, crypto index, funding, OI, SOL, BNB.
   Tính lại từ file thật mỗi lần chạy; chưa có thì ghi "chưa thu thập", không tạo số liệu.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import yaml

from src.utils.config import ROOT, path_of

# biến mở rộng: (mã nhiệm vụ, file raw, tên series trong file hoặc None, căn cứ tài liệu)
EXT_VARS = {
    "vix": ("C1-08", "macro_daily.csv", "VIX", "Excel C1-08; blueprint §2.4, §3.3"),
    "spx": ("C1-08", "macro_daily.csv", "SPX", "Excel C1-08; blueprint §3.3"),
    "dxy": ("C1-08", "macro_daily.csv", "DXY", "Excel C1-08; blueprint §2.4, §3.3"),
    "us10y": ("C1-08", "macro_daily.csv", "US10Y", "Blueprint §3.3 (US Treasury yields)"),
    "us2y": ("C1-08", "macro_daily.csv", "US2Y", "Blueprint §3.3 (US Treasury yields)"),
    "us3m": ("C1-08", "macro_daily.csv", "US3M", "Blueprint §3.3 (US Treasury yields)"),
    "stablecoin_supply": ("C1-08", "crypto_market_daily.csv", "STABLECOIN_SUPPLY",
                          "Blueprint §2.4, §3.2, §3.5; tài liệu coder §1.7"),
    "defi_tvl": ("C1-08", "crypto_market_daily.csv", "DEFI_TVL", "Blueprint §3.2 (DefiLlama TVL)"),
    "market_cap_btc": ("C1-08", "crypto_market_daily.csv", "MCAP_BTC", "Blueprint §3.2 (CoinGecko)"),
    "market_cap_eth": ("C1-08", "crypto_market_daily.csv", "MCAP_ETH", "Blueprint §3.2 (CoinGecko)"),
    "crypto_index_cmc200": ("C1-08", "crypto_market_daily.csv", "CMC200", "Blueprint §3.1 (crypto market index)"),
    "funding_rate": ("C1-09", "derivatives_daily.csv", None, "Excel C1-09; blueprint §2.4"),
    "open_interest": ("C1-09", "open_interest_daily.csv", None, "Excel C1-09; blueprint §2.4, §3.2"),
    "sol_ohlcv": ("C1-03/04 mở rộng", "sol_ohlcv.csv", None, "Blueprint §3.1 (tài sản mở rộng)"),
    "bnb_ohlcv": ("C1-03/04 mở rộng", "bnb_ohlcv.csv", None, "Blueprint §3.1 (tài sản mở rộng)"),
}
HOW_TO = "python -m src.data.collect --skip-core --all-extensions"
LIMIT_COLS = ["task", "asset", "variable", "status", "reason", "basis", "updated_at_utc"]


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


def _coverage(df: pd.DataFrame, date_col: str) -> str:
    return f"{df[date_col].min()} -> {df[date_col].max()}, {len(df)} dòng"


def write_limitation_log(cfg: dict) -> pd.DataFrame:
    """Trạng thái từng biến mở rộng, tính lại TỪ FILE THẬT mỗi lần chạy (không giữ trạng thái cũ).
    Nếu file chưa có: dùng lý do lỗi collector đã ghi (nếu có), nếu không thì "chưa thu thập"."""
    meta, raw = path_of(cfg, "metadata"), path_of(cfg, "raw")
    path = meta / "limitation_log.csv"
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    old = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=LIMIT_COLS)
    src_log = meta / "source_log.csv"
    srclog = pd.read_csv(src_log, dtype=str) if src_log.exists() else pd.DataFrame(columns=["asset", "notes"])
    reliable = cfg.get("crypto_market", {}).get("reliable_from") or {}
    cache: dict[str, pd.DataFrame | None] = {}
    rows = []
    for var, (task, fname, series, basis) in EXT_VARS.items():
        if fname not in cache:
            cache[fname] = pd.read_csv(raw / fname) if (raw / fname).exists() else None
        df = cache[fname]
        if df is not None and series is not None:
            df = df[df["series"] == series]
        if df is not None and len(df):
            date_col = "date_local" if "date_local" in df.columns else "date_utc"
            src = df["source"].iloc[0] if "source" in df.columns else fname
            note = ""
            if var.startswith("market_cap"):
                note = " — CoinGecko miễn phí giới hạn lịch sử; chỉ dùng mô tả/robustness"
            if var == "open_interest":
                note = " — lịch sử từ data.binance.vision (ảnh chụp hợp lệ cuối ngày UTC; bỏ ảnh chụp OI <= 0)"
            if reliable.get(series):
                note += (f" — trước {reliable[series]} nguồn theo dõi chưa đầy đủ nên để trống trong "
                         "features_ext (dữ liệu thô giữ nguyên)")
            rows.append({"task": task, "asset": ",".join(sorted(df["asset"].unique())) if "asset" in df.columns
                         else "ALL", "variable": var, "status": "đã thu thập",
                         "reason": f"{_coverage(df, date_col)}; nguồn {src}{note}", "basis": basis,
                         "updated_at_utc": now})
            continue
        fail = old[(old.get("variable") == var) & (old.get("status") == "không dùng")] if len(old) else old
        tried = srclog[(srclog["asset"].astype(str) == (series or "")) &
                       srclog["notes"].astype(str).str.startswith("KHÔNG THU THẬP ĐƯỢC")] if series else srclog.iloc[0:0]
        if len(tried):
            rows.append({"task": task, "asset": "ALL", "variable": var, "status": "không dùng",
                         "reason": tried["notes"].iloc[-1], "basis": basis, "updated_at_utc": now})
        elif len(fail):
            rows += [{**r, "basis": basis, "updated_at_utc": now} for r in fail.to_dict(orient="records")]
        else:
            rows.append({"task": task, "asset": "ALL", "variable": var, "status": "chưa thu thập",
                         "reason": f"Mở rộng tùy chọn, chưa tải. Không tạo số liệu thay thế. Chạy: {HOW_TO}",
                         "basis": basis, "updated_at_utc": now})
    out = pd.DataFrame(rows)
    for c in LIMIT_COLS:
        if c not in out.columns:
            out[c] = ""
    out[LIMIT_COLS].to_csv(path, index=False)
    return out
