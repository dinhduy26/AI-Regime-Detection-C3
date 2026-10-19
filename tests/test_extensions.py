"""Kiểm thử dữ liệu MỞ RỘNG (vĩ mô, trái phiếu, stablecoin, TVL, market cap, funding, OI, SOL/BNB).
Chạy trên dữ liệu THẬT; phần nào chưa tải thì test tự bỏ qua."""
import joblib
import numpy as np
import pandas as pd
import pytest

from src.data.collection_meta import EXT_VARS
from src.features.build import compute_asset_features, select_and_split
from src.utils.config import ROOT, path_of, read_json


def _ext_files(cfg):
    proc = path_of(cfg, "processed")
    files = {a: proc / f"features_ext_{a.lower()}.parquet" for a in cfg["assets"]}
    files = {a: p for a, p in files.items() if p.exists()}
    if not files:
        pytest.skip("chưa có features_ext (chưa tải dữ liệu mở rộng)")
    return {a: pd.read_parquet(p) for a, p in files.items()}


def test_ext_point_in_time(cfg):
    """Mọi giá trị ghép as-of phải có sẵn TRƯỚC hoặc ĐÚNG lúc nến ngày t đóng."""
    for a, e in _ext_files(cfg).items():
        avail = [c for c in e.columns if c.endswith("_available_at")]
        assert avail, f"{a}: không có cột _available_at để kiểm chứng"
        for c in avail:
            ok = e[c].isna() | (e[c] <= e["decision_time_utc"])
            assert ok.all(), f"{a}.{c}: có giá trị dùng trước khi công bố"
            obs = c.replace("_available_at", "_obs_date")
            m = e[obs].notna()
            assert (e.loc[m, obs] <= e.loc[m, "date_utc"]).all(), f"{a}.{obs}: ngày quan sát sau ngày t"


def test_ext_rows_match_core(cfg, feats):
    for a, e in _ext_files(cfg).items():
        assert e["date_utc"].tolist() == feats[a]["date_utc"].tolist()
        assert e["split"].tolist() == feats[a]["split"].tolist()


def test_ext_growth_is_backward_looking(cfg):
    """Tăng trưởng 7 ngày của OI phải tính từ OI hiện tại và OI 7 ngày TRƯỚC."""
    for e in _ext_files(cfg).values():
        if "open_interest_value_usdt" not in e.columns:
            pytest.skip("chưa có open interest")
        k = int(cfg["crypto_market"]["growth_window_days"])
        s = e.set_index(pd.to_datetime(e["date_utc"]))["open_interest_value_usdt"]
        cur, prev = s.to_numpy(), s.reindex(s.index - pd.Timedelta(days=k)).to_numpy()
        ok = (cur > 0) & (prev > 0)
        expect = np.full(len(cur), np.nan)
        expect[ok] = np.log(cur[ok] / prev[ok])
        got = e[f"oi_growth_{k}d"].to_numpy()
        both = ~np.isnan(expect) & ~np.isnan(got)
        np.testing.assert_allclose(got[both], expect[both], rtol=1e-12)


def test_ext_session_returns_from_raw_sessions(cfg):
    """spx_return_1d / dxy_return_1d / vix_change_1d phải bằng thay đổi so với PHIÊN GIAO DỊCH
    LIỀN TRƯỚC trong macro_daily.csv gốc (không phải ngày crypto trước), và không có return giả = 0
    do lặp giá cuối tuần."""
    raw = path_of(cfg, "raw") / "macro_daily.csv"
    if not raw.exists():
        pytest.skip("chưa có macro_daily.csv")
    m = pd.read_csv(raw).dropna(subset=["value"])
    for a, e in _ext_files(cfg).items():
        for series, (col, kind) in {"SPX": ("spx_return_1d", "log"), "DXY": ("dxy_return_1d", "log"),
                                    "VIX": ("vix_change_1d", "diff")}.items():
            if col not in e.columns:
                continue
            s = (m[m["series"] == series].sort_values("date_local")
                 .drop_duplicates("date_local", keep="last").reset_index(drop=True))
            prev_v, prev_d = s["value"].shift(1), pd.to_datetime(s["date_local"]).shift(1)
            gap = (pd.to_datetime(s["date_local"]) - prev_d).dt.days
            expect = np.log(s["value"] / prev_v) if kind == "log" else s["value"] - prev_v
            expect = pd.Series(expect.where(gap <= 5).to_numpy(), index=s["date_local"])
            got = e.dropna(subset=[col])
            np.testing.assert_allclose(got[col].to_numpy(),
                                       expect.reindex(got[f"{col}_obs_date"]).to_numpy(),
                                       rtol=1e-12, atol=1e-15, err_msg=f"{a}.{col}")
            # mỗi giá trị phải đến từ đúng ngày quan sát của chính chuỗi mức
            lvl = series.lower()
            if f"{lvl}_obs_date" in e.columns:
                same = got[f"{lvl}_obs_date"] == got[f"{col}_obs_date"]
                assert same.mean() > 0.99, f"{a}.{col}: lệch ngày quan sát so với {lvl}"


def test_ext_no_infinite_values(cfg):
    """Không có giá trị vô cực (vd. ln(x/0)) trong biến mở rộng — C2 dùng được trực tiếp."""
    for a, e in _ext_files(cfg).items():
        num = e.select_dtypes("number")
        bad = [c for c in num.columns if np.isinf(num[c]).any()]
        assert not bad, f"{a}: có giá trị vô cực ở {bad}"


def test_ext_invalid_values_blanked(cfg):
    """OI <= 0 (lỗi nguồn) và chuỗi crypto trước mốc tin cậy phải để trống trong features_ext."""
    rel = cfg["crypto_market"].get("reliable_from") or {}
    for a, e in _ext_files(cfg).items():
        if "open_interest_value_usdt" in e.columns:
            v = e["open_interest_value_usdt"].dropna()
            assert (v > 0).all(), f"{a}: còn OI <= 0"
        for series, start in rel.items():
            col = series.lower()
            if start and col in e.columns:
                obs = e[f"{col}_obs_date"]
                assert (obs.dropna() >= start).all(), f"{a}: {col} dùng quan sát trước {start}"
        for col in ("stablecoin_supply", "defi_tvl"):
            if col in e.columns:
                assert (e[col].dropna() > 0).all(), f"{a}: {col} còn giá trị <= 0"


def test_limitation_log_covers_every_extension_var(cfg):
    p = path_of(cfg, "metadata") / "limitation_log.csv"
    if not p.exists():
        pytest.skip("chưa có limitation_log")
    lg = pd.read_csv(p)
    assert set(EXT_VARS) <= set(lg["variable"]), set(EXT_VARS) - set(lg["variable"])
    assert lg["status"].isin(["đã thu thập", "chưa thu thập", "không dùng"]).all()


def test_raw_extension_no_unfinished_day(cfg):
    raw = path_of(cfg, "raw")
    for fname in ("macro_daily.csv", "crypto_market_daily.csv"):
        if not (raw / fname).exists():
            continue
        d = pd.read_csv(raw / fname)
        fetched_day = pd.to_datetime(d["fetched_at_utc"], utc=True).dt.strftime("%Y-%m-%d")
        assert (d["date_local"] <= fetched_day).all(), f"{fname}: có dữ liệu ngày tương lai"
        assert not d.duplicated(["series", "date_local"]).any(), f"{fname}: trùng (series, ngày)"


# ---------------- Tài sản mở rộng SOL / BNB ----------------
def _ext_assets(cfg):
    m = path_of(cfg, "metadata") / "split_manifest_ext.json"
    if not m.exists():
        pytest.skip("chưa tải SOL/BNB")
    return read_json(m)


def test_extension_assets_same_rules_as_core(cfg):
    man = _ext_assets(cfg)
    proc = ROOT / cfg["paths"]["processed"] / "extension"
    prices = pd.read_parquet(proc / "prices_daily_ext.parquet")
    assert not prices.duplicated(["asset", "date_utc"]).any()
    core = cfg["features"]["core"]
    for a in man["assets"]:
        f = pd.read_parquet(proc / f"features_{a.lower()}.parquet")
        again, _ = select_and_split(compute_asset_features(prices[prices["asset"] == a], cfg), cfg)
        assert again["date_utc"].tolist() == f["date_utc"].tolist()
        assert f[core].notna().all().all() and f["date_utc"].is_unique
        order = f["split"].map({"train": 0, "validation": 1, "test": 2})
        assert order.is_monotonic_increasing
        sc = joblib.load(ROOT / cfg["paths"]["models"] / "extension" / f"scaler_{a.lower()}.joblib")["scaler"]
        tr = f[f["split"] == "train"][core]
        np.testing.assert_allclose(sc.mean_, tr.mean().to_numpy())
        np.testing.assert_allclose(sc.scale_, tr.std(ddof=0).to_numpy())