"""[MỞ RỘNG RQ3 / ROBUSTNESS] Biến cấp thị trường crypto (blueprint §3.1–3.2, §3.5; tài liệu coder §1.7).

| series            | Nguồn (API công khai, miễn phí)                         | Đơn vị |
|-------------------|----------------------------------------------------------|--------|
| STABLECOIN_SUPPLY | DefiLlama stablecoins.llama.fi/stablecoincharts/all      | USD (tổng lưu hành, mọi loại peg quy USD) |
| DEFI_TVL          | DefiLlama api.llama.fi/v2/historicalChainTvl             | USD |
| MCAP_BTC, MCAP_ETH| CoinGecko /coins/{id}/market_chart                       | USD |
| CMC200            | Yahoo Finance ^CMC200 (CMC Crypto 200 Index)             | điểm chỉ số |

Quy tắc thời điểm sẵn có (giống macro, thận trọng):
    available_at_utc = (ngày d + 1 + availability_lag_days) 00:00 UTC
Điểm của NGÀY HÔM NAY (UTC) chưa trọn ngày -> bị loại. Không điền, không nội suy.
CoinGecko bản miễn phí chỉ trả lịch sử giới hạn (thường 365 ngày) -> ghi đúng độ phủ thật.
Ghi ra: data/raw/crypto_market_daily.csv (series, date_local, value, available_at_utc, source, fetched_at_utc)
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from src.utils.config import get_logger

log = get_logger("crypto-market")
COLS = ["series", "date_local", "value", "available_at_utc", "source", "fetched_at_utc"]


def _today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _get_json(url: str, params: dict | None = None, headers: dict | None = None, tries: int = 4):
    last = ""
    for i in range(tries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=60)
            if r.status_code == 200:
                return r.json()
            last = f"HTTP {r.status_code}: {r.text[:150]}"
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(2 * 2 ** i)
                continue
            break
        except requests.RequestException as e:
            last = repr(e)
            time.sleep(2 * 2 ** i)
    raise RuntimeError(last)


def _stablecoins(cfg: dict) -> pd.DataFrame:
    rows = _get_json(cfg["crypto_market"]["stablecoins_url"])
    out = []
    for r in rows:
        usd = r.get("totalCirculatingUSD") or {}
        total = sum(float(v) for v in usd.values() if v is not None)
        d = datetime.fromtimestamp(int(r["date"]), timezone.utc).strftime("%Y-%m-%d")
        out.append({"date_local": d, "value": total})
    return pd.DataFrame(out)


def _defi_tvl(cfg: dict) -> pd.DataFrame:
    rows = _get_json(cfg["crypto_market"]["defi_tvl_url"])
    return pd.DataFrame([{"date_local": datetime.fromtimestamp(int(r["date"]), timezone.utc)
                          .strftime("%Y-%m-%d"), "value": float(r["tvl"])} for r in rows])


def _coingecko_mcap(coin_id: str, cfg: dict) -> tuple[pd.DataFrame, str]:
    base = cfg["crypto_market"]["coingecko_url"]
    key = os.environ.get("COINGECKO_DEMO_API_KEY")       # tùy chọn; không bắt buộc
    headers = {"x-cg-demo-api-key": key} if key else None
    note = ""
    for days in ("max", "365"):
        try:
            j = _get_json(f"{base}/coins/{coin_id}/market_chart",
                          {"vs_currency": "usd", "days": days, "interval": "daily"}, headers)
            note = f"days={days}" + ("" if days == "max" else " (bản miễn phí giới hạn lịch sử)")
            break
        except RuntimeError as e:
            note = str(e)
            j = None
    if not j:
        raise RuntimeError(note)
    df = pd.DataFrame(j["market_caps"], columns=["ts", "value"])
    df["date_local"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.strftime("%Y-%m-%d")
    return df.drop_duplicates("date_local", keep="first")[["date_local", "value"]], note


def _yahoo_index(ticker: str, start: str) -> pd.DataFrame:
    import yfinance as yf

    df = yf.download(ticker, start=start, interval="1d", auto_adjust=False, progress=False,
                     multi_level_index=False)
    if df is None or df.empty:
        raise RuntimeError(f"Yahoo không trả dữ liệu cho {ticker}")
    out = df[["Close"]].rename(columns={"Close": "value"}).reset_index().rename(columns={"Date": "date_local"})
    out["date_local"] = pd.to_datetime(out["date_local"]).dt.strftime("%Y-%m-%d")
    return out.dropna(subset=["value"])


def collect_crypto_market(cfg: dict, raw_dir: Path) -> list[dict]:
    c = cfg["crypto_market"]
    lag, start, today = int(c["availability_lag_days"]), c["start_date"], _today_utc()
    fetched = datetime.now(timezone.utc).isoformat(timespec="seconds")
    jobs = [("STABLECOIN_SUPPLY", "DefiLlama stablecoins", c["stablecoins_url"], lambda: (_stablecoins(cfg), "")),
            ("DEFI_TVL", "DefiLlama DeFi TVL", c["defi_tvl_url"], lambda: (_defi_tvl(cfg), ""))]
    for a, cid in c["coingecko_ids"].items():
        jobs.append((f"MCAP_{a}", f"CoinGecko market cap ({cid})", f"{c['coingecko_url']}/coins/{cid}/market_chart",
                     lambda cid=cid: _coingecko_mcap(cid, cfg)))
    for name, tk in c["crypto_index_tickers"].items():
        jobs.append((name, f"Yahoo Finance ({tk})", tk, lambda tk=tk: (_yahoo_index(tk, start), "")))

    frames, logs = [], []
    for name, source, endpoint, fn in jobs:
        try:
            df, note = fn()
            df = df[(df["date_local"] >= start) & (df["date_local"] < today)]   # bỏ điểm hôm nay
            df = df.dropna(subset=["value"]).drop_duplicates("date_local", keep="last")
            n_nonpos = int((df["value"] <= 0).sum())
            df = df[df["value"] > 0]          # 0 = nguồn chưa theo dõi, KHÔNG phải giá trị thật
            if n_nonpos:
                note = (note + f" bỏ {n_nonpos} ngày giá trị <= 0 (nguồn chưa theo dõi)").strip()
            if df.empty:
                raise RuntimeError("không có dữ liệu trong khoảng yêu cầu")
        except Exception as e:  # noqa: BLE001
            log.warning("%s: KHÔNG thu thập được (%s)", name, e)
            logs.append({"asset": name, "symbol": endpoint, "source": source, "endpoint": endpoint,
                         "interval": "1d", "timezone": "UTC", "requested_start": start,
                         "actual_start": "", "actual_end": "", "n_rows": 0, "fetched_at_utc": fetched,
                         "notes": f"KHÔNG THU THẬP ĐƯỢC: {e}. Không thay bằng nguồn khác."})
            continue
        df = df.assign(series=name, source=source, fetched_at_utc=fetched,
                       available_at_utc=(pd.to_datetime(df["date_local"]) + pd.Timedelta(days=1 + lag))
                       .dt.strftime("%Y-%m-%dT00:00:00+00:00"))
        frames.append(df[COLS])
        logs.append({"asset": name, "symbol": endpoint, "source": source, "endpoint": endpoint,
                     "interval": "1d", "timezone": "UTC", "requested_start": start,
                     "actual_start": df["date_local"].min(), "actual_end": df["date_local"].max(),
                     "n_rows": len(df), "fetched_at_utc": fetched,
                     "notes": f"available_at = ngày + {1 + lag} ngày 00:00 UTC. {note}".strip()})
        log.info("%s: %d ngày (%s -> %s) %s", name, len(df), df["date_local"].min(),
                 df["date_local"].max(), note)
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(raw_dir / "crypto_market_daily.csv", index=False)
    return logs
