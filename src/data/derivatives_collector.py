"""C1-09 [TÙY CHỌN / MỞ RỘNG RQ3] — Funding rate và Open Interest (Binance USDⓈ-M Futures).

- Funding rate: GET /fapi/v1/fundingRate (lịch sử từ khi hợp đồng perpetual ra mắt, ~09/2019 với BTC).
  Thanh toán mỗi 8 giờ (00:00, 08:00, 16:00 UTC). Tổng hợp theo ngày UTC:
      funding_sum  = tổng các funding trong ngày
      funding_mean = trung bình, funding_n = số lần thanh toán thực có
  Mọi lần thanh toán trong ngày t đều xảy ra trước khi nến ngày t đóng (24:00 UTC),
  nên giá trị ngày t được coi là có sẵn tại cuối ngày t.
- Open interest: GET /futures/data/openInterestHist CHỈ trả khoảng 30 ngày gần nhất
  -> không đủ lịch sử cho mô hình; vẫn tải để ghi giới hạn vào limitation_log.
Không điền giả số; ngày không có dữ liệu để trống.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from src.data.binance_collector import SourceUnavailable, _get_with_retry, _to_ms
from src.utils.config import get_logger

log = get_logger("derivatives")


def _funding_history(symbol: str, cfg: dict) -> pd.DataFrame:
    base = cfg["derivatives"]["futures_base_url"]
    url = f"{base}/fapi/v1/fundingRate"
    session = requests.Session()
    cursor, rows = _to_ms(cfg["derivatives"]["funding_start_date"]), []
    now = int(datetime.now(timezone.utc).timestamp() * 1000)
    while cursor < now:
        batch = _get_with_retry(session, url, {"symbol": symbol, "startTime": cursor,
                                               "limit": 1000}, cfg)
        if not batch:
            break
        rows.extend(batch)
        cursor = int(batch[-1]["fundingTime"]) + 1
        if len(batch) < 1000:
            break
        time.sleep(cfg["collection"]["sleep_between_requests"])
    df = pd.DataFrame(rows)
    df["fundingRate"] = pd.to_numeric(df["fundingRate"])
    df["funding_time_utc"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    return df


def _open_interest(symbol: str, cfg: dict) -> pd.DataFrame:
    base = cfg["derivatives"]["futures_base_url"]
    url = f"{base}/futures/data/openInterestHist"
    rows = _get_with_retry(requests.Session(), url,
                           {"symbol": symbol, "period": "1d", "limit": 500}, cfg)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["date_utc"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.strftime("%Y-%m-%d")
    df["open_interest"] = pd.to_numeric(df["sumOpenInterest"])
    df["open_interest_value_usdt"] = pd.to_numeric(df["sumOpenInterestValue"])
    return df[["date_utc", "open_interest", "open_interest_value_usdt"]]


def collect_derivatives(cfg: dict, raw_dir: Path) -> tuple[list[dict], list[dict]]:
    fetched = datetime.now(timezone.utc)
    today = fetched.strftime("%Y-%m-%d")
    frames, src_logs, limits = [], [], []
    for asset, symbol in cfg["derivatives"]["symbols"].items():
        try:
            f = _funding_history(symbol, cfg)
            f["date_utc"] = f["funding_time_utc"].dt.strftime("%Y-%m-%d")
            f = f[f["date_utc"] < today]  # ngày hiện tại chưa đủ 3 lần thanh toán
            daily = (f.groupby("date_utc")
                       .agg(funding_sum=("fundingRate", "sum"),
                            funding_mean=("fundingRate", "mean"),
                            funding_n=("fundingRate", "size"),
                            last_funding_utc=("funding_time_utc", "max"))
                       .reset_index())
            daily.insert(0, "asset", asset)
            frames.append(daily)
            src_logs.append({"asset": f"{asset}_funding", "symbol": symbol,
                             "source": "Binance USDT-M Futures /fapi/v1/fundingRate",
                             "endpoint": cfg["derivatives"]["futures_base_url"], "interval": "8h -> 1d",
                             "timezone": "UTC", "requested_start": cfg["derivatives"]["funding_start_date"],
                             "actual_start": daily["date_utc"].min(), "actual_end": daily["date_utc"].max(),
                             "n_rows": len(daily), "fetched_at_utc": fetched.isoformat(timespec="seconds"),
                             "notes": f"{int((daily['funding_n'] != 3).sum())} ngày không đủ 3 lần thanh toán"})
        except SourceUnavailable as e:
            limits.append({"task": "C1-09", "asset": asset, "variable": "funding_rate",
                           "status": "không dùng", "reason": str(e)})
        try:
            oi = _open_interest(symbol, cfg)
            limits.append({"task": "C1-09", "asset": asset, "variable": "open_interest",
                           "status": "không dùng cho mô hình",
                           "reason": f"API chỉ trả {len(oi)} ngày gần nhất "
                                     f"({oi['date_utc'].min() if len(oi) else '-'} -> "
                                     f"{oi['date_utc'].max() if len(oi) else '-'}); không đủ lịch sử"})
            if len(oi):
                oi.insert(0, "asset", asset)
                oi.to_csv(raw_dir / f"{asset.lower()}_open_interest_recent.csv", index=False)
        except SourceUnavailable as e:
            limits.append({"task": "C1-09", "asset": asset, "variable": "open_interest",
                           "status": "không dùng", "reason": str(e)})
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(raw_dir / "derivatives_daily.csv", index=False)
    return src_logs, limits
