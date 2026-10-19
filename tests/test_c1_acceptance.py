"""Kiểm thử nghiệm thu C1 — đánh số theo đặc tả C1 §1.11 (10 tiêu chí).
Chạy:  pytest -q
"""
import math

import joblib
import numpy as np
import pandas as pd
import pytest

from src.features.build import compute_asset_features, select_and_split
from src.features.scaling import fit_transform
from src.utils.config import ROOT, load_schema, path_of

DAY_MS = 86_400_000


# ---------- 1. Khóa duy nhất, nến đã đóng, UTC chuẩn ----------
def test_01_unique_key_closed_utc(prices, raw):
    assert not prices.duplicated(["asset", "date_utc"]).any()
    for a, r in raw.items():
        fetched_ms = pd.to_datetime(r["fetched_at_utc"], utc=True).map(lambda x: x.timestamp() * 1000)
        assert (r["close_time_ms"] < fetched_ms).all(), f"{a}: có nến chưa đóng"
        assert (r["open_time_ms"] % DAY_MS == 0).all(), f"{a}: open_time không phải 00:00 UTC"
        d = pd.to_datetime(r["open_time_ms"], unit="ms", utc=True).dt.strftime("%Y-%m-%d")
        assert (d == r["date_utc"]).all()


# ---------- 2. OHLC hợp lệ, volume không âm, dòng bị loại có lý do ----------
def test_02_ohlc_valid(prices, cfg):
    p = prices
    assert (p[["open", "high", "low", "close"]] > 0).all().all()
    assert (p["high"] >= p[["open", "close", "low"]].max(axis=1)).all()
    assert (p["low"] <= p[["open", "close", "high"]].min(axis=1)).all()
    assert (p[["base_volume", "quote_volume"]] >= 0).all().all()
    excl = path_of(cfg, "metadata") / "excluded_rows.csv"
    if excl.exists() and excl.stat().st_size > 1:
        e = pd.read_csv(excl)
        if len(e):
            assert e["reason"].notna().all() and (e["reason"].astype(str) != "").all()


# ---------- 3. Không có return 1 ngày vắt qua khoảng trống ----------
def test_03_no_return_across_gap(prices, feats, cfg):
    assert (path_of(cfg, "metadata") / "gap_report.csv").exists()
    for a, f in feats.items():
        p = prices[prices["asset"] == a].set_index("date_utc")
        prev = (pd.to_datetime(f["date_utc"]) - pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d")
        assert prev.isin(p.index).all(), f"{a}: có return mà ngày trước không tồn tại"
        assert (p.loc[prev, "segment_id"].to_numpy() == f["segment_id"].to_numpy()).all()
        # mọi ngày trong features thuộc đoạn liên tục: ngày liền nhau trong cùng segment
        for _, seg in f.groupby("segment_id"):
            assert (pd.to_datetime(seg["date_utc"]).diff().dropna() == pd.Timedelta(days=1)).all()


# ---------- 4. Feature ngày t không đổi khi thay dữ liệu sau t ----------
@pytest.mark.parametrize("cut_frac", [0.3, 0.6, 0.9])
def test_04_point_in_time(prices, cfg, cut_frac):
    for a in cfg["assets"]:
        p = prices[prices["asset"] == a].sort_values("date_utc").reset_index(drop=True)
        cut = p["date_utc"].iloc[int(len(p) * cut_frac)]
        base = compute_asset_features(p, cfg).set_index("date_utc")
        # (a) cắt bỏ toàn bộ tương lai
        trunc = compute_asset_features(p[p["date_utc"] <= cut], cfg).set_index("date_utc")
        # (b) bóp méo tương lai: nhân giá & volume sau t với hệ số
        pert = p.copy()
        fut = pert["date_utc"] > cut
        pert.loc[fut, ["open", "high", "low", "close"]] *= 1.7
        pert.loc[fut, "quote_volume"] *= 0.2
        pert = compute_asset_features(pert, cfg).set_index("date_utc")
        cols = ["log_return", "simple_return", "rv21", "log_dollar_volume", "rv7", "rv30"]
        past = base.index[base.index <= cut]
        pd.testing.assert_frame_equal(base.loc[past, cols], trunc.loc[past, cols])
        pd.testing.assert_frame_equal(base.loc[past, cols], pert.loc[past, cols])


# ---------- 5. Công thức return: 100 -> 105 ; và khớp trên dữ liệu thật ----------
def test_05_return_formulas(prices, feats):
    # Ví dụ số học trong đặc tả (không phải dữ liệu):
    assert math.isclose(math.log(105 / 100), 0.04879016, abs_tol=1e-8)
    assert math.isclose(105 / 100 - 1, 0.05, abs_tol=1e-12)
    for a, f in feats.items():
        p = prices[prices["asset"] == a].set_index("date_utc")["close"]
        prev = (pd.to_datetime(f["date_utc"]) - pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d")
        c_t, c_prev = p.loc[f["date_utc"]].to_numpy(), p.loc[prev].to_numpy()
        np.testing.assert_allclose(f["log_return"], np.log(c_t / c_prev), rtol=0, atol=1e-12)
        np.testing.assert_allclose(f["simple_return"], c_t / c_prev - 1, rtol=0, atol=1e-12)
        np.testing.assert_allclose(f["simple_return"], np.expm1(f["log_return"]), atol=1e-12)


# ---------- 6. RV21: chưa đủ 21 return thì trống; chỉ dùng t-20..t; ddof=1 ----------
def test_06_rv21(prices, cfg):
    for a in cfg["assets"]:
        p = prices[prices["asset"] == a]
        f = compute_asset_features(p, cfg)
        for _, seg in f.groupby("segment_id"):
            seg = seg.reset_index(drop=True)
            assert seg["rv21"].iloc[: min(21, len(seg))].isna().all()  # 22 close -> rv21 đầu tiên
            if len(seg) > 21:
                assert seg["rv21"].iloc[21:].notna().all()
                lr = seg["log_return"].to_numpy()
                for i in np.linspace(21, len(seg) - 1, 15).astype(int):
                    assert math.isclose(seg["rv21"].iloc[i], np.std(lr[i - 20:i + 1], ddof=1),
                                        rel_tol=1e-10)


# ---------- 7. train + val + test = tổng hợp lệ; không chồng lấn; đúng thứ tự ----------
def test_07_split(feats, manifest, cfg):
    sp = cfg["split"]
    for a, f in feats.items():
        info = manifest["assets"][a]
        N = len(f)
        assert N == info["n_eligible"]
        n_tr, n_val = (f["split"] == "train").sum(), (f["split"] == "validation").sum()
        assert n_tr + n_val + (f["split"] == "test").sum() == N
        if sp["mode"] == "ratio":
            assert n_tr == math.floor(0.6 * N) and n_val == math.floor(0.2 * N)
        else:  # fixed_dates: đúng mốc ngày trong config
            assert f.loc[f["split"] == "train", "date_utc"].max() <= sp["train_end"]
            assert f.loc[f["split"] == "validation", "date_utc"].min() > sp["train_end"]
            assert f.loc[f["split"] == "validation", "date_utc"].max() <= sp["validation_end"]
            assert f.loc[f["split"] == "test", "date_utc"].min() > sp["validation_end"]
        assert f["date_utc"].is_monotonic_increasing and f["date_utc"].is_unique
        order = f["split"].map({"train": 0, "validation": 1, "test": 2})
        assert order.is_monotonic_increasing
        assert f[["log_return", "rv21", "log_dollar_volume"]].notna().all().all()


# ---------- 8. Scaler chỉ phụ thuộc train ----------
def test_08_scaler_train_only(feats, scaled, cfg, tmp_path):
    core = cfg["features"]["core"]
    for a, f in feats.items():
        saved = joblib.load(ROOT / f"outputs/models/scaler_{a.lower()}.joblib")["scaler"]
        tr = f[f["split"] == "train"][core]
        np.testing.assert_allclose(saved.mean_, tr.mean().to_numpy())
        np.testing.assert_allclose(saved.scale_, tr.std(ddof=0).to_numpy())
        # thay toàn bộ val/test -> tham số scaler phải giữ nguyên
        alt = f.copy()
        mask = alt["split"] != "train"
        alt.loc[mask, core] = alt.loc[mask, core] * 3.0 + 10.0
        _, params = fit_transform(alt, core, tmp_path / "s.joblib")
        np.testing.assert_allclose(list(params["mean"].values()), saved.mean_)
        np.testing.assert_allclose(list(params["scale"].values()), saved.scale_)
        # file scaled khớp công thức
        z = (f[core].to_numpy() - saved.mean_) / saved.scale_
        np.testing.assert_allclose(scaled[a][[f"z_{c}" for c in core]].to_numpy(), z, atol=1e-12)


# ---------- 9. Cùng schema BTC/ETH; volume lõi là QUOTE volume ----------
def test_09_schema_and_quote_volume(feats, prices):
    schema = load_schema()["files"]["features"]["columns"]
    cols = [list(f.columns) for f in feats.values()]
    assert all(c == cols[0] for c in cols)
    for a, f in feats.items():
        assert all(c in f.columns for c in schema)
        q = prices[prices["asset"] == a].set_index("date_utc").loc[f["date_utc"], "quote_volume"]
        np.testing.assert_allclose(f["log_dollar_volume"], np.log1p(q.to_numpy()))
        assert "rv_21d" not in f.columns and "log_volume" not in f.columns


# ---------- 10. B01/B02/H01 thực tế, truy vết được nguồn ----------
def test_10_handoff_outputs(cfg, prices, manifest):
    t, fg, m = path_of(cfg, "tables"), path_of(cfg, "figures"), path_of(cfg, "metadata")
    for p in (t / "B01_descriptives.csv", t / "B02_sample_manifest.csv",
              fg / "H01_price_history.png", m / "data_dictionary.csv", m / "source_log.csv",
              m / "limitation_log.csv", ROOT / "config" / "collection_config.yaml"):
        assert p.exists() and p.stat().st_size > 0, f"thiếu {p.name}"
    b02 = pd.read_csv(t / "B02_sample_manifest.csv")
    src = pd.read_csv(m / "source_log.csv", dtype=str)
    for _, r in b02.iterrows():
        p = prices[prices["asset"] == r["asset"]]
        assert r["start"] == p["date_utc"].min() and r["end"] == p["date_utc"].max()
        s = src[src["asset"] == r["asset"]].iloc[0]
        assert s["actual_start"] <= r["start"] and r["end"] <= s["actual_end"]
        assert r["n_train"] + r["n_val"] + r["n_test"] == r["n_eligible"]
    b01 = pd.read_csv(t / "B01_descriptives.csv")
    assert set(b01["period"]) == {"full_sample", "train"}


# ---------- Bổ sung: select_and_split chạy lại cho kết quả giống file đã lưu ----------
def test_split_reproducible(prices, feats, cfg):
    for a, f in feats.items():
        again, _ = select_and_split(compute_asset_features(prices[prices["asset"] == a], cfg), cfg)
        assert again["date_utc"].tolist() == f["date_utc"].tolist()
        assert again["split"].tolist() == f["split"].tolist()


# ---------- 11. Cập nhật dữ liệu mới KHÔNG làm đổi train / validation / scaler ----------
@pytest.mark.parametrize("drop_days", [30, 120])
def test_11_daily_update_keeps_split_and_scaler(prices, cfg, tmp_path, drop_days):
    """Mô phỏng 'hôm nay' và 'drop_days ngày trước' bằng cách cắt bớt dữ liệu THẬT ở cuối.
    Với split.mode = fixed_dates: train, validation và tham số scaler phải giống hệt;
    phần dữ liệu thêm vào chỉ nằm trong tập test."""
    if cfg["split"]["mode"] != "fixed_dates":
        pytest.skip("chỉ áp dụng cho split.mode = fixed_dates")
    core = cfg["features"]["core"]
    for a in cfg["assets"]:
        p = prices[prices["asset"] == a].sort_values("date_utc")
        cut = p["date_utc"].iloc[-1 - drop_days]
        if cut <= cfg["split"]["validation_end"]:
            pytest.skip("dữ liệu chưa vượt mốc validation đủ xa")
        now, _ = select_and_split(compute_asset_features(p, cfg), cfg)
        before, _ = select_and_split(compute_asset_features(p[p["date_utc"] <= cut], cfg), cfg)
        for part in ("train", "validation"):
            pd.testing.assert_frame_equal(
                now[now["split"] == part].reset_index(drop=True),
                before[before["split"] == part].reset_index(drop=True))
        _, p_now = fit_transform(now, core, tmp_path / "a.joblib")
        _, p_before = fit_transform(before, core, tmp_path / "b.joblib")
        assert p_now["mean"] == p_before["mean"] and p_now["scale"] == p_before["scale"]
        extra = now[now["date_utc"] > cut]
        assert (extra["split"] == "test").all() and len(extra) == drop_days


# ---------- 12. Có ngày dữ liệu (as-of) cho dashboard ----------
def test_12_data_through(manifest, prices):
    for a, info in manifest["assets"].items():
        assert info["data_through_utc"] == info["test"]["end"]
        assert info["latest_date_in_sample"], f"{a}: ngày mới nhất không nằm trong mẫu"
        assert info["data_through_utc"] == prices.loc[prices["asset"] == a, "date_utc"].max()
