"""Test tối thiểu C1 theo tài liệu chung §3.1: UTC, n_unique, no_nan_after_warmup, train-only scaler
+ kiểm tra các hàm giao tiếp §3.2 cho đúng kết quả đã lưu. Chạy trên dữ liệu THẬT (skip nếu chưa có)."""
import joblib
import numpy as np
import pandas as pd

from src.features.build import build_features, chronological_split, fit_scaler
from src.utils.config import ROOT


def test_utc(raw, prices):
    for r in raw.values():
        ot = pd.to_datetime(r["open_time_ms"], unit="ms", utc=True)
        assert (ot.dt.hour == 0).all() and (ot.dt.minute == 0).all()        # nến mở 00:00 UTC
        assert (ot.dt.strftime("%Y-%m-%d") == r["date_utc"]).all()
    assert prices["available_at_utc"].str.endswith("T00:00:00+00:00").all()


def test_n_unique(prices, feats):
    assert not prices.duplicated(["asset", "date_utc"]).any()
    for f in feats.values():
        assert f["date_utc"].is_unique


def test_no_nan_after_warmup(feats, cfg):
    for f in feats.values():
        assert f[cfg["features"]["core"] + ["simple_return"]].notna().all().all()


def test_train_only_scaler(feats, cfg):
    for a, f in feats.items():
        saved = joblib.load(ROOT / f"outputs/models/scaler_{a.lower()}.joblib")["scaler"]
        s = fit_scaler(chronological_split(f)["train"], cfg)
        np.testing.assert_array_equal(s.mean_, saved.mean_)
        np.testing.assert_array_equal(s.scale_, saved.scale_)


def test_interface_matches_saved_files(prices, feats, cfg):
    built = build_features(prices, cfg)
    for a, f in feats.items():
        b = built[built["asset"] == a][f.columns].reset_index(drop=True)
        pd.testing.assert_frame_equal(b, f.reset_index(drop=True), check_exact=True)
        parts = chronological_split(f)
        assert sum(len(p) for p in parts.values()) == len(f)