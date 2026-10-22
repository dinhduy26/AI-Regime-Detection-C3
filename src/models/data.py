"""Validate C1 handoff without changing its data, split or scaler."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.inference import sequence_starts
from src.utils.config import file_hash, path_of, read_json

CORE = ["log_return", "rv21", "log_dollar_volume"]
ZCOLS = [f"z_{c}" for c in CORE]
PERIODS = ("train", "validation", "test")


def load_asset(cfg, asset):
    proc, meta = path_of(cfg, "processed"), path_of(cfg, "metadata")
    manifest = read_json(meta / "split_manifest.json")
    if manifest["core_features"] != CORE or cfg["features"]["core"] != CORE:
        raise ValueError("C2 core model requires exactly the three agreed C1 features")
    info = manifest["assets"][asset]
    scaler = info["scaler"]
    if scaler["fit_on"] != "train" or scaler["std_ddof"] != 0 or scaler["features"] != CORE:
        raise ValueError("Scaler must use train only, ddof=0, and the agreed feature order")
    feature_path = proc / f"features_{asset.lower()}.parquet"
    scaled_path = proc / f"scaled_features_{asset.lower()}.parquet"
    feat, scaled = pd.read_parquet(feature_path), pd.read_parquet(scaled_path)
    keys = ["asset", "date_utc", "segment_id", "split"]
    if len(feat) != len(scaled) or not feat[keys].equals(scaled[keys]):
        raise ValueError("C1 raw/scaled feature keys or row order disagree")
    if set(feat["asset"]) != {asset}:
        raise ValueError("Asset contamination in C1 feature handoff")
    sequence_starts(feat)
    rank = feat["split"].map({p: i for i, p in enumerate(PERIODS)})
    if rank.isna().any() or not rank.is_monotonic_increasing:
        raise ValueError("Splits must be chronological train -> validation -> test")
    for period in PERIODS:
        part = feat.loc[feat["split"] == period]
        if len(part) == 0 or len(part) != info[period]["n"]:
            raise ValueError(f"Empty or mismatched {period} count in C1 manifest")
        dates = pd.to_datetime(part["date_utc"], utc=True)
        if (
            dates.min().strftime("%Y-%m-%d") != info[period]["start"]
            or dates.max().strftime("%Y-%m-%d") != info[period]["end"]
        ):
            raise ValueError(f"{period} dates disagree with C1 manifest")
    tr = feat.loc[feat["split"] == "train", CORE].to_numpy(dtype=float)
    mean, std = tr.mean(axis=0), tr.std(axis=0, ddof=0)
    scale = np.where(std == 0, 1, std)
    if not np.allclose(mean, [scaler["mean"][c] for c in CORE]) or not np.allclose(
        scale, [scaler["scale"][c] for c in CORE]
    ):
        raise ValueError("C1 scaler metadata does not match train-only statistics")
    z = scaled[ZCOLS].to_numpy(dtype=float)
    if not np.isfinite(z).all() or not np.allclose(z, (feat[CORE].to_numpy() - mean) / scale):
        raise ValueError("C1 scaled features do not match the frozen train scaler")
    frame = feat.copy()
    frame[ZCOLS] = z
    prices = pd.read_parquet(proc / "prices_daily.parquet")
    prices = prices.loc[prices["asset"] == asset, ["asset", "date_utc", "close", "available_at_utc"]]
    frame = frame.merge(prices, on=["asset", "date_utc"], how="left", validate="one_to_one", sort=False)
    dates = pd.to_datetime(frame["date_utc"], utc=True)
    available = pd.to_datetime(frame["available_at_utc"], utc=True)
    if available.isna().any() or (available < dates + pd.Timedelta(days=1)).any():
        raise ValueError("Features cannot be used before the daily candle closes")
    if not np.isfinite(frame[CORE + ["close", "simple_return"]].to_numpy(dtype=float)).all():
        raise ValueError("C1 price/feature handoff contains nonfinite values")
    if (frame["close"] <= 0).any():
        raise ValueError("C1 close prices must be positive")
    return (
        frame,
        info,
        {
            "features_sha256": file_hash(feature_path),
            "scaled_features_sha256": file_hash(scaled_path),
            "split_manifest_sha256": file_hash(meta / "split_manifest.json"),
            "scaler_sha256": file_hash(path_of(cfg, "models") / f"scaler_{asset.lower()}.joblib"),
            "data_through_utc": info["data_through_utc"],
            "c1_config_version": manifest["config_version"],
            "c1_config_hash": manifest["config_hash"],
        },
    )
