"""[MỞ RỘNG RQ3] Ghép biến vĩ mô / thị trường crypto / phái sinh theo thời điểm sẵn có (as-of).

Quy tắc (đặc tả C1 §1.7):
    Z_t^asof = Z_{tau*(t)},   tau*(t) = max{ tau : available_at(Z_tau) <= decision_time_t }
với decision_time_t = thời điểm nến crypto ngày t đóng = 00:00 UTC ngày t+1.
Không backfill; ngày chưa có giá trị nào đã sẵn có thì để trống.
File riêng features_ext_{asset}.parquet — KHÔNG thay đổi feature lõi của HMM.

Nguồn (data/raw/):
  macro_daily.csv          VIX, SPX, DXY, US10Y, US2Y, US3M         (series, date_local, value, available_at_utc)
  crypto_market_daily.csv  STABLECOIN_SUPPLY, DEFI_TVL, MCAP_*, CMC200 (cùng schema)
  derivatives_daily.csv    funding theo ngày UTC                     (asset, date_utc, ...)
  open_interest_daily.csv  OI ảnh chụp cuối ngày UTC                 (asset, date_utc, ...)
Biến dẫn xuất tính TRÊN CHUỖI GỐC trước khi ghép (không dùng thông tin sau ngày quan sát):
  stablecoin_supply_growth_7d = ln(S_d / S_{d-7}),  defi_tvl_growth_7d tương tự,
  oi_growth_7d = ln(OI_t / OI_{t-7}),  term_spread_10y_2y = us10y - us2y (cùng thời điểm sẵn có),
  spx_return_1d = ln(SPX_d / SPX_phiên trước), dxy_return_1d tương tự, vix_change_1d = VIX_d - VIX_phiên trước
  (tính trên chuỗi PHIÊN GIAO DỊCH Mỹ gốc; ghép as-of nên thứ Bảy/CN/thứ Hai mang return phiên
  thứ Sáu — dùng cột <biến>_obs_date nếu chỉ muốn lấy ngày có quan sát mới).
Mỗi biến ghép as-of có thêm cột <tên>_available_at để kiểm chứng point-in-time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.config import get_logger

log = get_logger("extended")
SERIES_FILES = ["macro_daily.csv", "crypto_market_daily.csv"]
GROWTH_SERIES = {"STABLECOIN_SUPPLY": "stablecoin_supply_growth_7d", "DEFI_TVL": "defi_tvl_growth_7d"}
# Biến thay đổi 1 phiên của thị trường Mỹ (blueprint §3.3 "S&P 500 returns", VIX, DXY).
# Tính trên CHUỖI PHIÊN GIAO DỊCH GỐC (phiên trước = quan sát liền trước), KHÔNG tính trên lịch
# crypto 7 ngày (nếu tính sau khi ghép, thứ Bảy/CN lặp giá thứ Sáu -> return giả bằng 0).
#   kind "log":  ln(P_d / P_prev)      kind "diff": P_d - P_prev (điểm)
SESSION_CHANGES = {"SPX": ("spx_return_1d", "log"), "DXY": ("dxy_return_1d", "log"),
                   "VIX": ("vix_change_1d", "diff")}
# Nếu phiên trước cách quá số ngày lịch này (nguồn bị thiếu nhiều phiên) -> để trống, không gộp
# nhiều phiên thành 1 return. 5 ngày đủ cho cuối tuần + nghỉ lễ Mỹ (vd. thứ Sáu -> thứ Ba).
SESSION_MAX_GAP_DAYS = 5


def _session_change(dates: pd.Series, values: pd.Series, kind: str, max_gap: int) -> pd.Series:
    """Thay đổi so với phiên giao dịch LIỀN TRƯỚC trên chuỗi gốc (đã sắp xếp theo ngày).
    Trống nếu: không có phiên trước, phiên trước cách > max_gap ngày, hoặc (kind=log) giá <= 0."""
    d = pd.to_datetime(dates).to_numpy()
    v = values.to_numpy(dtype=float)
    out = np.full(len(v), np.nan)
    if len(v) < 2:
        return pd.Series(out, index=values.index)
    gap = (d[1:] - d[:-1]) / np.timedelta64(1, "D")
    cur, prev = v[1:], v[:-1]
    ok = np.isfinite(cur) & np.isfinite(prev) & (gap <= max_gap)
    if kind == "log":
        ok &= (cur > 0) & (prev > 0)
        out[1:][ok] = np.log(cur[ok] / prev[ok])
    else:
        out[1:][ok] = cur[ok] - prev[ok]
    return pd.Series(out, index=values.index)


def _calendar_growth(dates: pd.Series, values: pd.Series, k: int) -> pd.Series:
    """ln(v_d / v_{d-k}) theo LỊCH (d-k phải có quan sát thật), không nội suy.
    Giá trị <= 0 ở ngày d hoặc d-k (vd. TVL/OI bằng 0 ở giai đoạn đầu) -> để trống, không ra vô cực."""
    s = pd.Series(values.to_numpy(dtype=float), index=pd.to_datetime(dates))
    cur = s.to_numpy()
    prev = s.reindex(s.index - pd.Timedelta(days=k)).to_numpy()
    ok = (cur > 0) & (prev > 0)
    out = np.full(len(cur), np.nan)
    out[ok] = np.log(cur[ok] / prev[ok])
    return pd.Series(out, index=values.index)


def _asof(base: pd.DataFrame, s: pd.DataFrame, name: str) -> pd.DataFrame:
    s = s.sort_values("available_at_utc").rename(columns={
        "value": name, "date_local": f"{name}_obs_date", "available_at_utc": f"{name}_available_at"})
    s[f"{name}_available_at"] = s[f"{name}_available_at"].astype("datetime64[ns, UTC]")
    out = pd.merge_asof(base, s[[f"{name}_available_at", f"{name}_obs_date", name]],
                        left_on="decision_time_utc", right_on=f"{name}_available_at", direction="backward")
    return out


def build_extended(feat: pd.DataFrame, raw_dir, cfg: dict | None = None) -> tuple[pd.DataFrame | None, list[dict]]:
    asset = feat["asset"].iloc[0]
    k = int((cfg or {}).get("crypto_market", {}).get("growth_window_days", 7))
    base = feat[["asset", "date_utc", "split"]].copy()
    base["decision_time_utc"] = ((pd.to_datetime(base["date_utc"], utc=True) + pd.Timedelta(days=1))
                                 .astype("datetime64[ns, UTC]"))
    base = base.sort_values("decision_time_utc")
    added_cols: list[tuple[str, str]] = []

    # ---- chuỗi dạng (series, date_local, value, available_at_utc): macro + crypto market ----
    for fname in SERIES_FILES:
        path = raw_dir / fname
        if not path.exists():
            continue
        data = pd.read_csv(path)
        data["available_at_utc"] = pd.to_datetime(data["available_at_utc"], utc=True)
        reliable = ((cfg or {}).get("crypto_market", {}).get("reliable_from") or {})
        for series, s in data.groupby("series"):
            s = s.sort_values("date_local").drop_duplicates("date_local", keep="last")
            if fname == "crypto_market_daily.csv":
                s = s[s["value"] > 0]                                   # 0 = nguồn chưa theo dõi
                if reliable.get(series):
                    s = s[s["date_local"] >= reliable[series]]          # trước mốc: độ phủ chưa đủ
            name = str(series).lower()
            base = _asof(base, s[["date_local", "value", "available_at_utc"]], name)
            added_cols.append((name, f"{fname}: {series}"))
            if series in GROWTH_SERIES:
                g = s.assign(value=_calendar_growth(s["date_local"], s["value"], k)).dropna(subset=["value"])
                gname = GROWTH_SERIES[series].replace("_7d", f"_{k}d")
                base = _asof(base, g[["date_local", "value", "available_at_utc"]], gname)
                added_cols.append((gname, f"ln({series}_d / {series}_(d-{k}))"))
            if fname == "macro_daily.csv" and series in SESSION_CHANGES:
                cname, kind = SESSION_CHANGES[series]
                s = s.dropna(subset=["value"])
                c = s.assign(value=_session_change(s["date_local"], s["value"], kind,
                                                   SESSION_MAX_GAP_DAYS))
                # GIỮ dòng NaN: phiên mới không tính được return -> ngày crypto đó để trống,
                # không mang return cũ của phiên trước ra dùng tiếp.
                base = _asof(base, c[["date_local", "value", "available_at_utc"]], cname)
                rule = (f"ln({series}_d / {series}_phiên trước)" if kind == "log"
                        else f"{series}_d - {series}_phiên trước (điểm)")
                added_cols.append((cname, rule + "; tính trên chuỗi phiên Mỹ gốc rồi ghép as-of"))

    if {"us10y", "us2y"} <= set(base.columns):
        # chỉ tính khi hai kỳ hạn là CÙNG một ngày quan sát (không trộn ngày)
        same = base["us10y_obs_date"] == base["us2y_obs_date"]
        base["term_spread_10y_2y"] = (base["us10y"] - base["us2y"]).where(same)
        added_cols.append(("term_spread_10y_2y", "us10y - us2y (cùng ngày quan sát)"))

    # ---- theo ngày UTC của tài sản: funding, open interest (có sẵn trước khi nến t đóng) ----
    dpath = raw_dir / "derivatives_daily.csv"
    if dpath.exists():
        d = pd.read_csv(dpath)
        d = d[d["asset"] == asset][["date_utc", "funding_sum", "funding_mean", "funding_n"]]
        base = base.merge(d, on="date_utc", how="left")
        added_cols.append(("funding_sum", "tổng funding đã thanh toán trong ngày UTC t"))
    opath = raw_dir / "open_interest_daily.csv"
    if opath.exists():
        o = pd.read_csv(opath)
        o = o[(o["asset"] == asset) & (o["open_interest_value_usdt"] > 0)].sort_values("date_utc")
        o["oi_growth_7d".replace("_7d", f"_{k}d")] = _calendar_growth(
            o["date_utc"], o["open_interest_value_usdt"], k)
        keep = ["date_utc", "open_interest", "open_interest_value_usdt", f"oi_growth_{k}d"]
        base = base.merge(o[keep], on="date_utc", how="left")
        added_cols += [("open_interest_value_usdt", "OI ảnh chụp cuối ngày UTC t"),
                       (f"oi_growth_{k}d", f"ln(OI_t / OI_(t-{k}))")]

    if not added_cols:
        return None, []
    base = base.sort_values("date_utc").reset_index(drop=True)
    coverage = [{"asset": asset, "variable": c, "first_available": base.loc[base[c].notna(), "date_utc"].min(),
                 "n_non_null": int(base[c].notna().sum()), "n_rows": len(base), "rule": rule}
                for c, rule in added_cols]
    return base, coverage