"""C1-09 [TÙY CHỌN / MỞ RỘNG RQ3] — Funding rate và Open Interest (Binance USDⓈ-M Futures).

- Funding rate: GET /fapi/v1/fundingRate (lịch sử từ khi hợp đồng perpetual ra mắt, ~09/2019 với BTC).
  Thanh toán mỗi 8 giờ (00:00, 08:00, 16:00 UTC). Tổng hợp theo ngày UTC:
      funding_sum  = tổng các funding trong ngày
      funding_mean = trung bình, funding_n = số lần thanh toán thực có
  Mọi lần thanh toán trong ngày t đều xảy ra trước khi nến ngày t đóng (24:00 UTC),
  nên giá trị ngày t được coi là có sẵn tại cuối ngày t.
- Open interest: API /futures/data/openInterestHist chỉ trả ~30 ngày, nên lấy LỊCH SỬ từ file
  "metrics" (ảnh chụp 5 phút) trên data.binance.vision/data/futures/um/daily/metrics/.
  Giá trị ngày t = ẢNH CHỤP HỢP LỆ CUỐI CÙNG trong ngày UTC t (đã có trước khi nến t đóng).
  Làm sạch: bỏ dòng lặp y hệt; bỏ ảnh chụp cùng giờ nhưng khác số (không đoán bản đúng);
  bỏ ảnh chụp OI <= 0 (lỗi dữ liệu — OI của BTC/ETH perpetual không thể bằng 0).
  Ngày nào không có file -> để trống, ghi số ngày thiếu vào source_log.
Không điền giả số; ngày không có dữ liệu để trống.
"""
from __future__ import annotations

import io
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
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


def _oi_one_day(symbol: str, day: str, cfg: dict) -> dict | None:
    """Tải file metrics của 1 ngày; trả về ảnh chụp cuối cùng THUỘC ngày UTC đó."""
    url = f"{cfg['derivatives']['oi_metrics_url']}/{symbol}/{symbol}-metrics-{day}.zip"
    for i in range(3):
        try:
            r = requests.get(url, timeout=30)
            break
        except requests.RequestException:
            time.sleep(1.5 * 2 ** i)
    else:
        return None
    if r.status_code != 200:
        return None
    try:
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            m = pd.read_csv(z.open(z.namelist()[0]))
        m["create_time"] = pd.to_datetime(m["create_time"], utc=True)
        m["sum_open_interest"] = pd.to_numeric(m["sum_open_interest"])
        m["sum_open_interest_value"] = pd.to_numeric(m["sum_open_interest_value"])
    except (zipfile.BadZipFile, KeyError, ValueError) as e:
        log.warning("%s %s: file metrics lỗi định dạng (%s) -> bỏ qua ngày này", symbol, day, e)
        return None
    m = m[m["create_time"].dt.strftime("%Y-%m-%d") == day]
    n_raw = len(m)
    m = m.drop_duplicates(subset=["create_time", "sum_open_interest", "sum_open_interest_value"])
    n_conflict = int(m["create_time"].duplicated(keep=False).sum())   # cùng giờ nhưng khác số
    m = m[~m["create_time"].duplicated(keep=False)]                     # không đoán bản đúng
    valid = m[(m["sum_open_interest"] > 0) & (m["sum_open_interest_value"] > 0)].sort_values("create_time")
    if valid.empty:
        return None
    last = valid.iloc[-1]
    return {"date_utc": day, "open_interest": float(last["sum_open_interest"]),
            "open_interest_value_usdt": float(last["sum_open_interest_value"]),
            "n_snapshots": len(m), "n_duplicate_rows": n_raw - len(m) - n_conflict,
            "n_conflicting_rows": n_conflict, "n_invalid_snapshots": len(m) - len(valid),
            "last_snapshot_utc": last["create_time"].isoformat()}


def _first_available_day(symbol: str, cfg: dict, end: datetime) -> datetime | None:
    """Dò từng tháng (ngày 1, rồi ngày 15) từ oi_start_date tới khi gặp file đầu tiên."""
    d = datetime.strptime(cfg["derivatives"]["oi_start_date"], "%Y-%m-%d")
    while d <= end:
        for day in (d, d.replace(day=15)):
            if _oi_one_day(symbol, day.strftime("%Y-%m-%d"), cfg):
                # lùi lại tối đa 31 ngày để không bỏ sót các ngày đầu
                back = day - timedelta(days=31)
                return max(back, datetime.strptime(cfg["derivatives"]["oi_start_date"], "%Y-%m-%d"))
        d = (d.replace(day=28) + timedelta(days=4)).replace(day=1)
    return None


def _oi_history(symbol: str, cfg: dict) -> tuple[pd.DataFrame, int]:
    yesterday = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=1)
    start = _first_available_day(symbol, cfg, yesterday)
    if start is None:
        raise SourceUnavailable(f"data.binance.vision không có file metrics cho {symbol}")
    days = [(start + timedelta(days=i)).strftime("%Y-%m-%d")
            for i in range((yesterday - start).days + 1)]
    log.info("%s: tải open interest %d ngày (%s -> %s) từ data.binance.vision ...",
             symbol, len(days), days[0], days[-1])
    rows, done, step = [], 0, max(1, len(days) // 10)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=int(cfg["derivatives"]["oi_download_workers"])) as ex:
        futures = [ex.submit(_oi_one_day, symbol, d, cfg) for d in days]
        for fut in as_completed(futures):
            r = fut.result()
            if r:
                rows.append(r)
            done += 1
            if done % step == 0 or done == len(days):
                el = time.time() - t0
                log.info("%s open interest: %d/%d ngày (%d%%), đã chạy %.0fs, còn khoảng %.0fs",
                         symbol, done, len(days), 100 * done // len(days), el,
                         el / done * (len(days) - done))
    df = pd.DataFrame(rows).sort_values("date_utc").reset_index(drop=True)
    first = df["date_utc"].min()
    n_missing = len([d for d in days if d >= first]) - len(df)
    return df, n_missing


def collect_derivatives(cfg: dict, raw_dir: Path) -> tuple[list[dict], list[dict]]:
    fetched = datetime.now(timezone.utc)
    today = fetched.strftime("%Y-%m-%d")
    frames, oi_frames, src_logs, limits = [], [], [], []
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
            oi, n_missing = _oi_history(symbol, cfg)
            oi.insert(0, "asset", asset)
            oi_frames.append(oi)
            src_logs.append({"asset": f"{asset}_open_interest", "symbol": symbol,
                             "source": "Binance USDT-M Futures metrics (data.binance.vision)",
                             "endpoint": cfg["derivatives"]["oi_metrics_url"], "interval": "5m -> 1d",
                             "timezone": "UTC", "requested_start": cfg["derivatives"]["oi_start_date"],
                             "actual_start": oi["date_utc"].min(), "actual_end": oi["date_utc"].max(),
                             "n_rows": len(oi), "fetched_at_utc": fetched.isoformat(timespec="seconds"),
                             "notes": f"giá trị ngày = ảnh chụp cuối ngày UTC; {n_missing} ngày không có file"})
        except SourceUnavailable as e:
            limits.append({"task": "C1-09", "asset": asset, "variable": "open_interest",
                           "status": "không dùng", "reason": str(e)})
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(raw_dir / "derivatives_daily.csv", index=False)
    if oi_frames:
        pd.concat(oi_frames, ignore_index=True).to_csv(raw_dir / "open_interest_daily.csv", index=False)
    return src_logs, limits
