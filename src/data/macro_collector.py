"""C1-08 [TÙY CHỌN / MỞ RỘNG RQ3] — VIX, S&P 500, DXY và lợi suất trái phiếu Mỹ.

Nguồn chính: Yahoo Finance qua thư viện yfinance (VIX, S&P 500, DXY).
Lợi suất trái phiếu Mỹ (blueprint §3.3): FRED DGS10 / DGS2 / DGS3MO, đơn vị %/năm.
Dự phòng   : FRED (VIXCLS, SP500) — tải CSV công khai. DXY không có trên FRED
             (DTWEXBGS là chỉ số khác) -> nếu Yahoo lỗi thì GHI GIỚI HẠN, không thay thế ngầm.

Quy tắc thời điểm sẵn có (point-in-time):
- Giá đóng cửa Mỹ của phiên ngày d được coi là CÓ SẴN từ
      available_at_utc = (d + 1 ngày) 00:00 UTC + availability_lag_days
  Với lag = 1 (mặc định), nến crypto ngày d+1 mới được dùng giá ngày d.
- Phiên của ngày hôm nay (giờ New York) chưa đóng cửa -> bị loại khỏi dữ liệu thô.
- Cuối tuần/ngày lễ không có giá mới -> khi ghép, chỉ lấy giá GẦN NHẤT ĐÃ CÓ SẴN (as-of),
  tuyệt đối không lấy giá thứ Hai cho thứ Bảy/Chủ nhật.
"""
from __future__ import annotations

import io
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from src.utils.config import get_logger

log = get_logger("macro")

FRED_FALLBACK = {"VIX": "VIXCLS", "SPX": "SP500"}


def _from_yahoo(ticker: str, start: str) -> pd.DataFrame:
    import yfinance as yf  # import muộn: chỉ cần khi bật macro

    df = yf.download(ticker, start=start, interval="1d", auto_adjust=False,
                     progress=False, multi_level_index=False)
    if df is None or df.empty:
        raise RuntimeError(f"Yahoo không trả dữ liệu cho {ticker}")
    out = df[["Close"]].rename(columns={"Close": "value"}).reset_index()
    out = out.rename(columns={"Date": "date_local"})
    out["date_local"] = pd.to_datetime(out["date_local"]).dt.strftime("%Y-%m-%d")
    return _drop_unclosed(out.dropna(subset=["value"]))


def _drop_unclosed(df: pd.DataFrame) -> pd.DataFrame:
    """Bỏ phiên của NGÀY HÔM NAY theo giờ New York: phiên đó chưa đóng cửa
    (DXY giao dịch gần 24h nên Yahoo trả cả giá đang chạy)."""
    today_ny = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
    return df[df["date_local"] < today_ny]


def _from_fred(series_id: str, start: str) -> pd.DataFrame:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = ["date_local", "value"]
    df["value"] = pd.to_numeric(df["value"], errors="coerce")  # FRED dùng "." cho ngày nghỉ
    df = df.dropna(subset=["value"])
    return _drop_unclosed(df[df["date_local"] >= start])


def collect_macro(cfg: dict, raw_dir: Path) -> list[dict]:
    m = cfg["macro"]
    lag = int(m["availability_lag_days"])
    frames, logs = [], []
    fetched = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for name, ticker in m["tickers"].items():
        source, note = f"Yahoo Finance ({ticker})", ""
        try:
            df = _from_yahoo(ticker, m["start_date"])
        except Exception as e:  # noqa: BLE001
            log.warning("Yahoo lỗi cho %s: %s", name, e)
            if name in FRED_FALLBACK:
                df = _from_fred(FRED_FALLBACK[name], m["start_date"])
                source = f"FRED ({FRED_FALLBACK[name]})"
                note = f"Yahoo lỗi ({e}); dùng FRED"
            else:
                logs.append({"asset": name, "symbol": ticker, "source": "Yahoo Finance",
                             "endpoint": "", "interval": "1d", "timezone": "US/Eastern",
                             "requested_start": m["start_date"], "actual_start": "",
                             "actual_end": "", "n_rows": 0, "fetched_at_utc": fetched,
                             "notes": f"KHÔNG THU THẬP ĐƯỢC: {e}. Không thay bằng chỉ số khác."})
                continue
        df["series"] = name
        df["source"] = source
        df["available_at_utc"] = (pd.to_datetime(df["date_local"]) + pd.Timedelta(days=1 + lag)
                                  ).dt.strftime("%Y-%m-%dT00:00:00+00:00")
        df["fetched_at_utc"] = fetched
        frames.append(df[["series", "date_local", "value", "available_at_utc", "source",
                          "fetched_at_utc"]])
        logs.append({"asset": name, "symbol": ticker, "source": source, "endpoint": "",
                     "interval": "1d (ngày giao dịch Mỹ)", "timezone": "US/Eastern",
                     "requested_start": m["start_date"], "actual_start": df["date_local"].min(),
                     "actual_end": df["date_local"].max(), "n_rows": len(df),
                     "fetched_at_utc": fetched,
                     "notes": f"available_at = ngày + {1 + lag} ngày 00:00 UTC. {note}".strip()})
        log.info("%s: %d phiên (%s -> %s)", name, len(df), df["date_local"].min(),
                 df["date_local"].max())
    # ---- Lợi suất trái phiếu Mỹ từ FRED (cùng quy tắc thời điểm sẵn có) ----
    for name, series_id in m.get("fred_series", {}).items():
        try:
            df = _from_fred(series_id, m["start_date"])
        except Exception as e:  # noqa: BLE001
            logs.append({"asset": name, "symbol": series_id, "source": "FRED", "endpoint": "",
                         "interval": "1d", "timezone": "US/Eastern",
                         "requested_start": m["start_date"], "actual_start": "", "actual_end": "",
                         "n_rows": 0, "fetched_at_utc": fetched,
                         "notes": f"KHÔNG THU THẬP ĐƯỢC: {e}"})
            continue
        df["series"] = name
        df["source"] = f"FRED ({series_id})"
        df["available_at_utc"] = (pd.to_datetime(df["date_local"]) + pd.Timedelta(days=1 + lag)
                                  ).dt.strftime("%Y-%m-%dT00:00:00+00:00")
        df["fetched_at_utc"] = fetched
        frames.append(df[["series", "date_local", "value", "available_at_utc", "source",
                          "fetched_at_utc"]])
        logs.append({"asset": name, "symbol": series_id, "source": f"FRED ({series_id})",
                     "endpoint": f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}",
                     "interval": "1d (ngày làm việc Mỹ)", "timezone": "US/Eastern",
                     "requested_start": m["start_date"], "actual_start": df["date_local"].min(),
                     "actual_end": df["date_local"].max(), "n_rows": len(df), "fetched_at_utc": fetched,
                     "notes": f"đơn vị %/năm; available_at = ngày + {1 + lag} ngày 00:00 UTC"})
        log.info("%s (%s): %d ngày (%s -> %s)", name, series_id, len(df), df["date_local"].min(),
                 df["date_local"].max())

    if frames:
        out = pd.concat(frames, ignore_index=True)
        out.to_csv(raw_dir / "macro_daily.csv", index=False)
    return logs
