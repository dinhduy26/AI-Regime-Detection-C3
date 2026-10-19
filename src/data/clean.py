"""C1-05 — Làm sạch và kiểm toán OHLCV.

Đầu vào : data/raw/{btc,eth}_ohlcv.csv  (dữ liệu thật đã tải bằng src.data.collect)
Đầu ra  : data/processed/prices_daily.parquet
          outputs/metadata/gap_report.csv, data_quality.csv, excluded_rows.csv

Quy tắc (theo đặc tả C1 §1.2):
- Một quan sát hợp lệ: O,H,L,C > 0; H >= max(O,C,L); L <= min(O,C,H); volume >= 0; không null.
- Nến phải bắt đầu đúng 00:00 UTC và kéo dài đúng 1 ngày.
- Khóa (asset, date_utc) duy nhất. Trùng giống hệt -> giữ 1. Trùng mâu thuẫn -> KHÔNG đoán,
  coi như ngày thiếu và ghi lý do.
- Ngày thiếu: KHÔNG điền, KHÔNG nội suy. Ghi gap_report; chuỗi bị tách thành các đoạn
  liên tục, mỗi đoạn có segment_id.
- Cú nhảy |log return| lớn chỉ được GHI NHẬN, không tự xóa (crypto có thể biến động thật).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.collection_meta import write_collection_config, write_limitation_log
from src.utils.config import ROOT, check_columns, get_logger, load_schema, path_of

log = get_logger("clean")
DAY_MS = 86_400_000


def load_raw(asset: str, raw_dir) -> pd.DataFrame:
    path = raw_dir / f"{asset.lower()}_ohlcv.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Chưa có {path.relative_to(ROOT)}. Hãy chạy `python -m src.data.collect` "
            "trên máy có Internet để tải dữ liệu thật trước.")
    return pd.read_csv(path, dtype={"date_utc": str})


def validate_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tách dòng hợp lệ / không hợp lệ. Trả về (valid, excluded có cột reason)."""
    reasons = pd.Series("", index=df.index)

    def flag(mask, text):
        nonlocal reasons
        reasons = reasons.where(~mask, reasons + text + "; ")

    price = ["open", "high", "low", "close"]
    flag(df[price + ["base_volume", "quote_volume"]].isna().any(axis=1), "giá trị null")
    flag((df[price] <= 0).any(axis=1), "giá <= 0")
    flag(df["high"] < df[["open", "close", "low"]].max(axis=1), "high < max(O,C,L)")
    flag(df["low"] > df[["open", "close", "high"]].min(axis=1), "low > min(O,C,H)")
    flag((df["base_volume"] < 0) | (df["quote_volume"] < 0), "volume âm")
    flag(df["open_time_ms"] % DAY_MS != 0, "open_time không đúng 00:00 UTC")
    flag(df["close_time_ms"] - df["open_time_ms"] != DAY_MS - 1, "nến không dài đúng 1 ngày")

    bad = reasons != ""
    excluded = df[bad].copy()
    excluded["reason"] = reasons[bad].str.rstrip("; ")
    return df[~bad].copy(), excluded


def resolve_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Giữ 1 bản cho trùng giống hệt; loại toàn bộ ngày có trùng mâu thuẫn."""
    value_cols = ["open", "high", "low", "close", "base_volume", "quote_volume"]
    n_exact = int(df.duplicated(subset=["date_utc"] + value_cols).sum())
    df = df.drop_duplicates(subset=["date_utc"] + value_cols, keep="first")
    conflict = df["date_utc"].duplicated(keep=False)
    excluded = df[conflict].copy()
    excluded["reason"] = "trùng ngày nhưng giá trị khác nhau — không đoán bản đúng"
    return df[~conflict].copy(), excluded, n_exact


def build_gap_report(asset: str, dates: pd.Series, method: str) -> tuple[pd.DataFrame, pd.Series]:
    """Tìm các khoảng ngày thiếu và gán segment_id cho từng ngày có dữ liệu."""
    d = pd.to_datetime(dates).sort_values().reset_index(drop=True)
    step = d.diff().dt.days
    seg = (step.fillna(1) != 1).cumsum().astype(int)  # tăng 1 sau mỗi khoảng đứt
    rows = []
    for i in np.where(step.fillna(1) > 1)[0]:
        before, after = d[i - 1], d[i]
        rows.append({
            "asset": asset,
            "missing_from": (before + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            "missing_to": (after - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            "gap_length": int(step[i] - 1),
            "last_before": before.strftime("%Y-%m-%d"),
            "first_after": after.strftime("%Y-%m-%d"),
            "handling": ("không điền/không nội suy; đứt chuỗi; không tính return/RV21 qua gap; "
                         + ("HMM chỉ dùng đoạn liên tục dài nhất" if method == "longest_segment"
                            else "giữ đoạn với segment_id riêng, C2 reset trạng thái ẩn đầu đoạn")),
        })
    cols = ["asset", "missing_from", "missing_to", "gap_length", "last_before", "first_after",
            "handling"]
    return pd.DataFrame(rows, columns=cols), pd.Series(seg.values, index=d.dt.strftime("%Y-%m-%d"))


def clean_asset(asset: str, cfg: dict, raw_dir) -> dict:
    raw = load_raw(asset, raw_dir)
    n_raw = len(raw)
    valid, excl_invalid = validate_rows(raw)
    valid, excl_dup, n_exact_dup = resolve_duplicates(valid)
    valid = valid.sort_values("date_utc").reset_index(drop=True)

    method = cfg["cleaning"]["gap_handling_method"]
    gaps, seg_map = build_gap_report(asset, valid["date_utc"], method)
    valid["segment_id"] = valid["date_utc"].map(seg_map).astype(int)
    # Thời điểm thông tin của nến t được coi là đã biết: khi nến đóng = 00:00 UTC ngày t+1
    valid["available_at_utc"] = (pd.to_datetime(valid["close_time_ms"] + 1, unit="ms", utc=True)
                                 .dt.strftime("%Y-%m-%dT%H:%M:%S+00:00"))

    # Kiểm toán bổ sung (chỉ ghi nhận)
    same_seg = valid["segment_id"].eq(valid["segment_id"].shift())
    lr = np.log(valid["close"] / valid["close"].shift()).where(same_seg)
    thr = cfg["cleaning"]["extreme_abs_log_return"]
    extreme = valid.loc[lr.abs() > thr, "date_utc"].tolist()
    zero_q = valid.loc[valid["quote_volume"] == 0, "date_utc"].tolist()
    n_missing_days = int(gaps["gap_length"].sum()) if len(gaps) else 0

    quality = [
        {"check": "số dòng thô", "n_flagged": n_raw, "detail": "", "action": "—"},
        {"check": "dòng không hợp lệ (OHLC/volume/thời gian)", "n_flagged": len(excl_invalid),
         "detail": "; ".join(excl_invalid["date_utc"].astype(str).head(20)),
         "action": "loại, ghi excluded_rows.csv"},
        {"check": "trùng (asset,date) giống hệt", "n_flagged": n_exact_dup, "detail": "",
         "action": "giữ 1 bản"},
        {"check": "trùng (asset,date) mâu thuẫn", "n_flagged": len(excl_dup),
         "detail": "; ".join(excl_dup["date_utc"].astype(str).unique()[:20]),
         "action": "loại cả ngày, coi như thiếu"},
        {"check": "ngày thiếu trong lịch UTC", "n_flagged": n_missing_days,
         "detail": f"{len(gaps)} khoảng đứt", "action": "không điền; xem gap_report.csv"},
        {"check": "số đoạn liên tục", "n_flagged": int(valid["segment_id"].nunique()), "detail": "",
         "action": f"gap_handling_method={method}"},
        {"check": f"|log return| > {thr}", "n_flagged": len(extreme), "detail": "; ".join(extreme),
         "action": "chỉ ghi nhận, giữ nguyên (biến động thật cần kiểm tra thủ công)"},
        {"check": "quote_volume = 0", "n_flagged": len(zero_q), "detail": "; ".join(zero_q),
         "action": "giữ; log_dollar_volume = ln(1+0) = 0; amihud để trống"},
        {"check": "nguồn dữ liệu", "n_flagged": int(valid["source"].nunique()),
         "detail": " | ".join(valid["source"].unique()), "action": "ghi provenance"},
        {"check": "số dòng sạch", "n_flagged": len(valid),
         "detail": f"{valid['date_utc'].min()} -> {valid['date_utc'].max()}", "action": "—"},
    ]
    for q in quality:
        q["asset"] = asset

    excluded = pd.concat([excl_invalid, excl_dup], ignore_index=True)
    log.info("%s: thô %d -> sạch %d dòng, %d khoảng đứt (%d ngày thiếu)",
             asset, n_raw, len(valid), len(gaps), n_missing_days)
    return {"prices": valid, "gaps": gaps, "quality": quality, "excluded": excluded,
            "n_raw": n_raw, "n_missing_days": n_missing_days}


def run(cfg: dict) -> dict[str, dict]:
    raw_dir, proc_dir, meta_dir = path_of(cfg, "raw"), path_of(cfg, "processed"), path_of(cfg, "metadata")
    schema = load_schema()["files"]
    results = {a: clean_asset(a, cfg, raw_dir) for a in cfg["assets"]}

    cols = schema["prices_daily"]["columns"]
    prices = pd.concat([r["prices"] for r in results.values()], ignore_index=True)
    check_columns(prices, cols, "prices_daily")
    prices = prices[cols].sort_values(["asset", "date_utc"]).reset_index(drop=True)
    assert not prices.duplicated(["asset", "date_utc"]).any(), "khóa (asset,date_utc) bị trùng"
    prices.to_parquet(proc_dir / "prices_daily.parquet", index=False)

    pd.concat([r["gaps"] for r in results.values()], ignore_index=True).to_csv(
        meta_dir / "gap_report.csv", index=False)
    q = pd.DataFrame([row for r in results.values() for row in r["quality"]])
    q[schema["data_quality"]["columns"]].to_csv(meta_dir / "data_quality.csv", index=False)
    pd.concat([r["excluded"] for r in results.values()], ignore_index=True).to_csv(
        meta_dir / "excluded_rows.csv", index=False)
    write_collection_config(cfg)
    write_limitation_log(cfg)
    log.info("Đã ghi prices_daily.parquet, gap_report.csv, data_quality.csv, excluded_rows.csv, "
             "limitation_log.csv, config/collection_config.yaml")
    return results
