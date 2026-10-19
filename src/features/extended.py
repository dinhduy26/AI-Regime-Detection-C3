"""[MỞ RỘNG RQ3] Ghép biến vĩ mô / phái sinh theo thời điểm sẵn có (as-of).

Quy tắc (đặc tả C1 §1.7):
    Z_t^asof = Z_{tau*(t)},   tau*(t) = max{ tau : available_at(Z_tau) <= decision_time_t }
với decision_time_t = thời điểm nến crypto ngày t đóng = 00:00 UTC ngày t+1.
Không backfill; ngày chưa có giá trị nào đã sẵn có thì để trống.
File riêng features_ext_{asset}.parquet — KHÔNG thay đổi feature lõi của HMM.
"""
from __future__ import annotations

import pandas as pd

from src.utils.config import get_logger

log = get_logger("extended")


def build_extended(feat: pd.DataFrame, raw_dir) -> tuple[pd.DataFrame | None, list[dict]]:
    asset = feat["asset"].iloc[0]
    base = feat[["asset", "date_utc", "split"]].copy()
    base["decision_time_utc"] = ((pd.to_datetime(base["date_utc"], utc=True) + pd.Timedelta(days=1))
                                 .astype("datetime64[ns, UTC]"))
    base = base.sort_values("decision_time_utc")
    coverage, added = [], False

    macro_path = raw_dir / "macro_daily.csv"
    if macro_path.exists():
        macro = pd.read_csv(macro_path)
        macro["available_at_utc"] = (pd.to_datetime(macro["available_at_utc"], utc=True)
                                     .astype("datetime64[ns, UTC]"))
        for name, s in macro.groupby("series"):
            s = s.sort_values("available_at_utc")[["available_at_utc", "date_local", "value"]]
            s = s.rename(columns={"value": name.lower(), "date_local": f"{name.lower()}_obs_date"})
            base = pd.merge_asof(base, s, left_on="decision_time_utc", right_on="available_at_utc",
                                 direction="backward").drop(columns="available_at_utc")
            coverage.append({"asset": asset, "variable": name.lower(),
                             "first_available": base.loc[base[name.lower()].notna(), "date_utc"].min(),
                             "n_non_null": int(base[name.lower()].notna().sum()),
                             "rule": "as-of available_at <= close of UTC candle"})
            added = True

    deriv_path = raw_dir / "derivatives_daily.csv"
    if deriv_path.exists():
        d = pd.read_csv(deriv_path)
        d = d[d["asset"] == asset][["date_utc", "funding_sum", "funding_mean", "funding_n"]]
        base = base.merge(d, on="date_utc", how="left")   # mọi funding ngày t đã trả trước khi nến t đóng
        coverage.append({"asset": asset, "variable": "funding_sum",
                         "first_available": base.loc[base["funding_sum"].notna(), "date_utc"].min(),
                         "n_non_null": int(base["funding_sum"].notna().sum()),
                         "rule": "tổng funding đã thanh toán trong ngày UTC t"})
        added = True

    if not added:
        return None, []
    return base.sort_values("date_utc").reset_index(drop=True), coverage
