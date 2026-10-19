"""Hàm giao tiếp C1 theo tài liệu chung §3.2:  fetch_ohlcv(symbol, start_utc, end_utc) -> DataFrame

Dùng khi cần lấy nhanh nến ngày Binance Spot trong một khoảng (ví dụ C2/C3 kiểm tra).
Pipeline chính vẫn dùng `python -m src.data.collect` (có source_log, tải bù, dự phòng).
Chỉ trả nến ĐÃ ĐÓNG; không ghi file.
"""
from __future__ import annotations

import copy

import pandas as pd

from src.data.binance_collector import fetch_klines_api, finalize_raw
from src.utils.config import load_config


def fetch_ohlcv(symbol: str, start_utc: str, end_utc: str | None = None,
                cfg: dict | None = None) -> pd.DataFrame:
    """Nến ngày UTC đã đóng của `symbol` (vd. "BTCUSDT") trong [start_utc, end_utc] (YYYY-MM-DD).

    Cột: asset, symbol, date_utc, open_time_ms, close_time_ms, open, high, low, close,
         base_volume, quote_volume, n_trades, taker_buy_*, source, source_url, fetched_at_utc
    """
    cfg = copy.deepcopy(cfg or load_config())
    cfg["collection"]["start_date"] = start_utc
    cfg["collection"]["end_date"] = end_utc
    asset = next((a for a, v in cfg["assets"].items() if v["symbol"] == symbol), symbol)
    df, url = fetch_klines_api(symbol, cfg)
    df = finalize_raw(df, asset, symbol, "Binance Spot REST /api/v3/klines", url, cfg)
    return df[df["date_utc"] >= start_utc].reset_index(drop=True)