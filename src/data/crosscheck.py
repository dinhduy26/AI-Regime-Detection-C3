"""Đối soát giá (C1-05: "đối soát cùng nguồn"; tài liệu chung §2.2.2: "so sánh overlapping close").

Chạy (cần Internet):   python -m src.data.crosscheck

Hai lớp kiểm tra, KHÔNG sửa dữ liệu, chỉ ghi báo cáo:
 (A) CÙNG NGUỒN, KÊNH ĐỘC LẬP — so data/raw/*_ohlcv.csv (tải qua REST API) với file lịch sử
     tháng trên data.binance.vision (Binance công bố riêng). Kỳ vọng: KHỚP TUYỆT ĐỐI OHLCV.
     Kiểm tra các tháng có biến động lớn + tháng gần nhất đã kết thúc.
 (B) NGUỒN KHÁC — so giá đóng cửa với Yahoo Finance BTC-USD / ETH-USD (giá tổng hợp nhiều sàn,
     USD chứ không phải USDT). Kỳ vọng: KHÔNG khớp tuyệt đối, chênh lệch nhỏ (thường < 1%),
     lợi suất ngày tương quan rất cao. Ngày chênh lệch lớn được liệt kê để xem tay.
Báo cáo: outputs/metadata/price_crosscheck.csv và price_crosscheck_days.csv
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

from src.data.binance_collector import _parse_vision_zip
from src.utils.config import get_logger, load_config, path_of

log = get_logger("crosscheck")
# Các tháng có sự kiện lớn (kiểm tra kỹ vùng biến động) — tháng gần nhất được thêm tự động
EVENT_MONTHS = ["2018-03", "2020-03", "2021-05", "2021-11", "2022-06", "2022-11", "2024-03"]
OHLCV = ["open", "high", "low", "close", "base_volume", "quote_volume"]
YAHOO = {"BTC": "BTC-USD", "ETH": "ETH-USD"}


def _last_full_month() -> str:
    first = datetime.now(timezone.utc).date().replace(day=1)
    return (pd.Timestamp(first) - pd.Timedelta(days=1)).strftime("%Y-%m")


def check_vision(asset: str, raw: pd.DataFrame, cfg: dict) -> tuple[dict, pd.DataFrame]:
    symbol = cfg["assets"][asset]["symbol"]
    base = cfg["collection"]["vision_base_url"].rstrip("/")
    frames, months_ok = [], []
    for m in EVENT_MONTHS + [_last_full_month()]:
        r = requests.get(f"{base}/{symbol}/1d/{symbol}-1d-{m}.zip", timeout=30)
        if r.status_code != 200:
            log.warning("vision %s %s: HTTP %s", symbol, m, r.status_code)
            continue
        v = _parse_vision_zip(r.content)
        v["date_utc"] = pd.to_datetime(v["open_time_ms"], unit="ms", utc=True).dt.strftime("%Y-%m-%d")
        frames.append(v)
        months_ok.append(m)
    vis = pd.concat(frames, ignore_index=True)
    m = raw.merge(vis[["date_utc"] + OHLCV], on="date_utc", suffixes=("", "_vision"))
    diff = pd.concat({c: (m[c] - m[f"{c}_vision"]).abs() for c in OHLCV}, axis=1)
    bad = m[(diff > 1e-8 * m[OHLCV].abs().clip(lower=1)).any(axis=1)]
    days = bad[["date_utc"] + OHLCV + [f"{c}_vision" for c in OHLCV]].assign(asset=asset, check="A_vision")
    return {"asset": asset, "check": "A: REST API vs data.binance.vision (cùng nguồn Binance)",
            "months": " ".join(months_ok), "n_days_compared": len(m), "n_mismatch": len(bad),
            "max_abs_diff_close": float(diff["close"].max()) if len(m) else np.nan,
            "verdict": "KHỚP TUYỆT ĐỐI" if len(m) and not len(bad) else "CẦN XEM price_crosscheck_days.csv"
            }, days


def check_yahoo(asset: str, raw: pd.DataFrame, cfg: dict) -> tuple[dict, pd.DataFrame]:
    import yfinance as yf

    y = yf.download(YAHOO[asset], start=raw["date_utc"].min(), interval="1d", auto_adjust=False,
                    progress=False, multi_level_index=False)
    y = y[["Close"]].rename(columns={"Close": "close_yahoo"}).reset_index()
    y["date_utc"] = pd.to_datetime(y["Date"]).dt.strftime("%Y-%m-%d")   # Yahoo crypto: ngày UTC
    m = raw[["date_utc", "close"]].merge(y[["date_utc", "close_yahoo"]], on="date_utc").dropna()
    m["diff_pct"] = (m["close"] / m["close_yahoo"] - 1) * 100
    r_b = np.log(m["close"]).diff()
    r_y = np.log(m["close_yahoo"]).diff()
    big = m[m["diff_pct"].abs() > 2].assign(asset=asset, check="B_yahoo")
    return {"asset": asset, "check": f"B: Binance USDT vs Yahoo {YAHOO[asset]} (nguồn khác, USD)",
            "months": f"{m['date_utc'].min()} -> {m['date_utc'].max()}", "n_days_compared": len(m),
            "n_mismatch": len(big),
            "median_abs_diff_pct": round(float(m["diff_pct"].abs().median()), 4),
            "p99_abs_diff_pct": round(float(m["diff_pct"].abs().quantile(0.99)), 4),
            "corr_daily_log_return": round(float(r_b.corr(r_y)), 5),
            "verdict": ("NHẤT QUÁN" if m["diff_pct"].abs().median() < 0.5 and r_b.corr(r_y) > 0.98
                        else "CẦN XEM")}, big


def main() -> None:
    cfg = load_config()
    raw_dir, meta = path_of(cfg, "raw"), path_of(cfg, "metadata")
    summary, days = [], []
    for asset in cfg["assets"]:
        raw = pd.read_csv(raw_dir / f"{asset.lower()}_ohlcv.csv", dtype={"date_utc": str})
        for fn in (check_vision, check_yahoo):
            try:
                s, d = fn(asset, raw, cfg)
                summary.append(s)
                days.append(d)
            except Exception as e:  # noqa: BLE001
                summary.append({"asset": asset, "check": fn.__name__, "verdict": f"LỖI: {e}"})
    out = pd.DataFrame(summary)
    out.to_csv(meta / "price_crosscheck.csv", index=False)
    if days:
        pd.concat(days, ignore_index=True).to_csv(meta / "price_crosscheck_days.csv", index=False)
    pd.set_option("display.width", 200)
    print(out.drop(columns=["months"], errors="ignore").to_string(index=False))
    print("\nĐã ghi outputs/metadata/price_crosscheck.csv và price_crosscheck_days.csv")


if __name__ == "__main__":
    main()