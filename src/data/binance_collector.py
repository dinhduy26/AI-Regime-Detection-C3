"""C1-03 / C1-04 — Thu thập OHLCV nến ngày BTCUSDT, ETHUSDT từ Binance Spot.

Nguồn chính : REST API công khai  GET /api/v3/klines  (không cần API key)
Nguồn dự phòng: kho file lịch sử công khai https://data.binance.vision (cùng provenance Binance)

Nguyên tắc:
- Phân trang theo startTime, mỗi request tối đa 1000 nến.
- Retry + exponential backoff khi lỗi mạng / 429 / 5xx; tôn trọng header Retry-After.
- CHỈ giữ nến đã đóng: close_time < thời điểm tải. Nến của ngày hiện tại bị loại.
- Không điền, không nội suy. Dữ liệu thô được lưu nguyên trạng kèm thời điểm tải.

Cột của một kline Binance (theo tài liệu API):
 0 open_time(ms) 1 open 2 high 3 low 4 close 5 volume(base) 6 close_time(ms)
 7 quote_asset_volume 8 number_of_trades 9 taker_buy_base 10 taker_buy_quote 11 ignore
"""
from __future__ import annotations

import io
import time
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

from src.utils.config import get_logger

log = get_logger("collector")

DAY_MS = 86_400_000
KLINE_COLS = [
    "open_time_ms", "open", "high", "low", "close", "base_volume", "close_time_ms",
    "quote_volume", "n_trades", "taker_buy_base_volume", "taker_buy_quote_volume", "ignore",
]
NUMERIC_COLS = ["open", "high", "low", "close", "base_volume", "quote_volume",
                "taker_buy_base_volume", "taker_buy_quote_volume"]


class SourceUnavailable(RuntimeError):
    """Không truy cập được nguồn sau khi đã thử lại."""


def _to_ms(date_str: str) -> int:
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def _get_with_retry(session: requests.Session, url: str, params: dict, cfg: dict):
    """GET có retry. Trả về (json, None) hoặc ném lỗi sau max_retries."""
    c = cfg["collection"]
    last_err = None
    for attempt in range(c["max_retries"]):
        try:
            r = session.get(url, params=params, timeout=c["timeout_seconds"])
            if r.status_code == 200:
                return r.json()
            if r.status_code in (403, 451):
                # Bị chặn theo khu vực -> không retry endpoint này, chuyển endpoint khác
                raise SourceUnavailable(f"HTTP {r.status_code} (bị chặn khu vực) tại {url}")
            if r.status_code in (418, 429):
                wait = float(r.headers.get("Retry-After", c["backoff_base_seconds"] * 2 ** attempt))
                log.warning("Bị giới hạn tốc độ (HTTP %s), chờ %.1fs", r.status_code, wait)
                time.sleep(wait)
                continue
            last_err = f"HTTP {r.status_code}: {r.text[:200]}"
        except requests.RequestException as e:
            last_err = repr(e)
        wait = c["backoff_base_seconds"] * 2 ** attempt
        log.warning("Lỗi '%s' — thử lại lần %d sau %.1fs", last_err, attempt + 1, wait)
        time.sleep(wait)
    raise SourceUnavailable(f"Hết số lần thử tại {url}: {last_err}")


def _klines_to_frame(rows: list) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=KLINE_COLS)
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["open_time_ms"] = df["open_time_ms"].astype("int64")
    df["close_time_ms"] = df["close_time_ms"].astype("int64")
    df["n_trades"] = pd.to_numeric(df["n_trades"], errors="coerce").astype("Int64")
    return df.drop(columns=["ignore"])


def fetch_klines_api(symbol: str, cfg: dict) -> tuple[pd.DataFrame, str]:
    """Tải toàn bộ nến ngày qua REST API, thử lần lượt các base_url. Trả về (df, endpoint)."""
    c = cfg["collection"]
    start_ms = _to_ms(c["start_date"])
    now_ms = _now_ms()
    end_ms = now_ms if not c.get("end_date") else _to_ms(c["end_date"]) + DAY_MS - 1

    errors = []
    for base in c["base_urls"]:
        url = f"{base}/api/v3/klines"
        session = requests.Session()
        frames, cursor = [], start_ms
        try:
            while cursor <= end_ms:
                params = {"symbol": symbol, "interval": c["interval"], "startTime": cursor,
                          "endTime": end_ms, "limit": c["request_limit"]}
                rows = _get_with_retry(session, url, params, cfg)
                if not rows:
                    break
                frames.append(_klines_to_frame(rows))
                last_open = int(rows[-1][0])
                log.info("%s: %d nến, đến %s", symbol, len(rows),
                         datetime.fromtimestamp(last_open / 1000, timezone.utc).date())
                cursor = last_open + DAY_MS
                if len(rows) < c["request_limit"]:
                    break
                time.sleep(c["sleep_between_requests"])
        except SourceUnavailable as e:
            errors.append(str(e))
            log.warning("Endpoint %s không dùng được: %s", base, e)
            continue
        if frames:
            df = pd.concat(frames, ignore_index=True)
            return df, url
        errors.append(f"{url}: không trả về dữ liệu")
    raise SourceUnavailable("Tất cả endpoint REST đều thất bại:\n  " + "\n  ".join(errors))


def _parse_vision_zip(content: bytes) -> pd.DataFrame:
    """Đọc 1 file zip kline của data.binance.vision thành DataFrame chuẩn."""
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        raw = pd.read_csv(z.open(z.namelist()[0]), header=None)
    if not str(raw.iloc[0, 0]).isdigit():  # một số file có dòng tiêu đề
        raw = raw.iloc[1:]
    raw = raw.iloc[:, :12].copy()
    raw.columns = KLINE_COLS
    for col in ("open_time_ms", "close_time_ms"):
        v = pd.to_numeric(raw[col]).astype("int64")
        raw[col] = v.where(v < 10**14, v // 1000)  # từ 2025 file dùng micro-giây -> mili-giây
    return _klines_to_frame(raw.values.tolist())


def fetch_klines_vision(symbol: str, cfg: dict) -> tuple[pd.DataFrame, str]:
    """Nguồn dự phòng: tải file zip tháng (và file ngày cho tháng hiện tại) từ data.binance.vision.

    Lưu ý: từ 2025-01-01 Binance đổi timestamp trong file spot sang micro-giây;
    hàm tự nhận diện và quy đổi về mili-giây.
    """
    c = cfg["collection"]
    monthly = c["vision_base_url"].rstrip("/")
    daily = monthly.replace("/monthly/", "/daily/")
    start = datetime.strptime(c["start_date"], "%Y-%m-%d").date().replace(day=1)
    today = datetime.now(timezone.utc).date()
    end = today if not c.get("end_date") else datetime.strptime(c["end_date"], "%Y-%m-%d").date()
    session = requests.Session()

    def _read_zip(url: str) -> pd.DataFrame | None:
        r = session.get(url, timeout=c["timeout_seconds"])
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return _parse_vision_zip(r.content)

    frames = []
    month = start
    while month <= end.replace(day=1):
        tag = f"{month:%Y-%m}"
        df = _read_zip(f"{monthly}/{symbol}/1d/{symbol}-1d-{tag}.zip")
        if df is not None:
            frames.append(df)
            log.info("vision %s %s: %d nến", symbol, tag, len(df))
        elif month.year == end.year and month.month == end.month or month >= today.replace(day=1):
            # Tháng chưa kết thúc -> chưa có file tháng, tải từng file ngày
            d = month
            while d <= end and d.month == month.month:
                dd = _read_zip(f"{daily}/{symbol}/1d/{symbol}-1d-{d:%Y-%m-%d}.zip")
                if dd is not None:
                    frames.append(dd)
                d += timedelta(days=1)
        month = (month.replace(day=28) + timedelta(days=4)).replace(day=1)
    if not frames:
        raise SourceUnavailable(f"data.binance.vision không có file cho {symbol}")
    return pd.concat(frames, ignore_index=True), monthly


def _missing_dates(dates: pd.Series) -> list[str]:
    """Các ngày UTC nằm giữa ngày đầu và ngày cuối nhưng không có nến."""
    d = pd.to_datetime(dates.drop_duplicates())
    if d.empty:
        return []
    full = pd.date_range(d.min(), d.max(), freq="D")
    return [x.strftime("%Y-%m-%d") for x in full.difference(d)]


def refill_from_vision(symbol: str, dates: list[str], cfg: dict) -> pd.DataFrame:
    """Thử tải từng ngày thiếu từ file ngày của data.binance.vision. Không có thì bỏ qua."""
    daily = cfg["collection"]["vision_base_url"].rstrip("/").replace("/monthly/", "/daily/")
    session, frames = requests.Session(), []
    for d in dates:
        url = f"{daily}/{symbol}/1d/{symbol}-1d-{d}.zip"
        try:
            r = session.get(url, timeout=cfg["collection"]["timeout_seconds"])
        except requests.RequestException:
            continue
        if r.status_code != 200:
            continue
        frames.append(_parse_vision_zip(r.content))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=KLINE_COLS[:-1])


def finalize_raw(df: pd.DataFrame, asset: str, symbol: str, source: str, url: str,
                 cfg: dict) -> pd.DataFrame:
    """Chuẩn hóa dữ liệu thô: gắn provenance, bỏ nến chưa đóng, lọc end_date, sắp xếp."""
    fetched_at = datetime.now(timezone.utc)
    fetched_ms = int(fetched_at.timestamp() * 1000)
    n0 = len(df)
    df = df[df["close_time_ms"] < fetched_ms].copy()  # chỉ nến ĐÃ ĐÓNG
    n_unclosed = n0 - len(df)
    df["date_utc"] = pd.to_datetime(df["open_time_ms"], unit="ms", utc=True).dt.strftime("%Y-%m-%d")
    end_date = cfg["collection"].get("end_date")
    if end_date:
        df = df[df["date_utc"] <= end_date]
    df.insert(0, "asset", asset)
    df.insert(1, "symbol", symbol)
    df["source"] = source
    df["source_url"] = url
    df["fetched_at_utc"] = fetched_at.isoformat(timespec="seconds")
    df = df.sort_values("open_time_ms").reset_index(drop=True)
    cols = ["asset", "symbol", "date_utc", "open_time_ms", "close_time_ms", "open", "high", "low",
            "close", "base_volume", "quote_volume", "n_trades", "taker_buy_base_volume",
            "taker_buy_quote_volume", "source", "source_url", "fetched_at_utc"]
    out = df[cols].copy()
    out.attrs["n_unclosed_dropped"] = n_unclosed
    return out


def collect_asset(asset: str, cfg: dict, raw_dir: Path, method: str = "api") -> dict:
    """Tải 1 tài sản, lưu data/raw/{asset}_ohlcv.csv, trả về 1 dòng cho source_log."""
    symbol = cfg["assets"][asset]["symbol"]
    notes = []
    if method == "api":
        try:
            df, url = fetch_klines_api(symbol, cfg)
            source = "Binance Spot REST /api/v3/klines"
        except SourceUnavailable as e:
            log.error("REST API thất bại, chuyển sang nguồn dự phòng data.binance.vision\n%s", e)
            notes.append("REST thất bại -> dùng data.binance.vision; lý do: " + str(e).replace("\n", " | "))
            df, url = fetch_klines_vision(symbol, cfg)
            source = "Binance Spot public data (data.binance.vision)"
    else:
        df, url = fetch_klines_vision(symbol, cfg)
        source = "Binance Spot public data (data.binance.vision)"

    df = finalize_raw(df, asset, symbol, source, url, cfg)
    n_unclosed = df.attrs.get("n_unclosed_dropped", 0)

    # Tải bù ngày thiếu từ CÙNG NGUỒN Binance (file ngày trên data.binance.vision)
    missing = _missing_dates(df["date_utc"])
    if missing:
        log.warning("%s thiếu %d ngày trong lịch UTC -> thử tải bù từ data.binance.vision",
                    asset, len(missing))
        refill = refill_from_vision(symbol, missing, cfg)
        if len(refill):
            refill = finalize_raw(refill, asset, symbol,
                                  "Binance Spot public data (data.binance.vision) — tải bù",
                                  cfg["collection"]["vision_base_url"], cfg)
            df = (pd.concat([df, refill], ignore_index=True)
                    .sort_values("open_time_ms").reset_index(drop=True))
        notes.append(f"thiếu {len(missing)} ngày qua REST; tải bù được {len(refill)} ngày "
                     f"từ data.binance.vision; còn thiếu {len(missing) - len(refill)}")
    df.attrs["n_unclosed_dropped"] = n_unclosed
    if df.attrs.get("n_unclosed_dropped"):
        notes.append(f"loại {df.attrs['n_unclosed_dropped']} nến chưa đóng")
    out = raw_dir / f"{asset.lower()}_ohlcv.csv"
    df.to_csv(out, index=False)
    log.info("Đã lưu %s (%d dòng, %s -> %s)", out, len(df), df["date_utc"].min(), df["date_utc"].max())
    if df["date_utc"].min() > cfg["collection"]["start_date"]:
        notes.append(f"nguồn chỉ có dữ liệu từ {df['date_utc'].min()} "
                     f"(yêu cầu từ {cfg['collection']['start_date']})")
    return {
        "asset": asset, "symbol": symbol, "source": source, "endpoint": url,
        "interval": cfg["collection"]["interval"], "timezone": "UTC",
        "requested_start": cfg["collection"]["start_date"],
        "actual_start": df["date_utc"].min(), "actual_end": df["date_utc"].max(),
        "n_rows": len(df), "fetched_at_utc": df["fetched_at_utc"].iloc[0],
        "notes": "; ".join(notes),
    }
