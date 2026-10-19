"""Chạy toàn bộ khối feature: prices_daily -> features_{asset}, scaled, scaler, split_manifest."""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from src.features.build import add_btc_eth_corr, compute_asset_features, select_and_split
from src.features.extended import build_extended
from src.features.scaling import fit_transform
from src.utils.config import ROOT, check_columns, file_hash, get_logger, load_schema, path_of, write_json

log = get_logger("feature-pipeline")


def run(cfg: dict) -> dict:
    proc, meta, models, raw = (path_of(cfg, k) for k in ("processed", "metadata", "models", "raw"))
    schema = load_schema()["files"]
    prices_path = proc / "prices_daily.parquet"
    if not prices_path.exists():
        raise FileNotFoundError("Chưa có prices_daily.parquet — chạy bước clean trước.")
    prices = pd.read_parquet(prices_path)
    core = cfg["features"]["core"]

    full = {a: compute_asset_features(prices[prices["asset"] == a], cfg) for a in cfg["assets"]}
    add_btc_eth_corr(full, cfg)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_version": cfg["version"], "config_hash": cfg["_config_hash"],
        "gap_handling_method": cfg["cleaning"]["gap_handling_method"],
        "core_features": core, "split_config": {k: v for k, v in cfg["split"].items()},
        "prices_daily_hash": file_hash(prices_path), "assets": {},
    }
    feat_cols = schema["features"]["columns"]
    opt_cols = schema["features"]["optional_columns"]
    coverage_all = []

    for asset, df in full.items():
        feat, info = select_and_split(df, cfg)
        check_columns(feat, feat_cols, f"features_{asset}")
        keep_cols = feat_cols + [c for c in opt_cols if c in feat.columns]
        feat = feat[keep_cols]
        feat.to_parquet(proc / f"features_{asset.lower()}.parquet", index=False)

        scaled, params = fit_transform(feat, core, models / f"scaler_{asset.lower()}.joblib")
        scaled.to_parquet(proc / f"scaled_features_{asset.lower()}.parquet", index=False)

        info["scaler"] = params | {"path": f"outputs/models/scaler_{asset.lower()}.joblib"}
        info["files"] = {"features": f"data/processed/features_{asset.lower()}.parquet",
                         "scaled_features": f"data/processed/scaled_features_{asset.lower()}.parquet"}
        manifest["assets"][asset] = info
        log.info("%s: N=%d (train %d | val %d | test %d), warm-up loại %d",
                 asset, info["n_eligible"], info["train"]["n"], info["validation"]["n"],
                 info["test"]["n"], info["n_warmup_dropped"])

        ext, cov = build_extended(feat, raw)
        if ext is not None:
            ext.to_parquet(proc / f"features_ext_{asset.lower()}.parquet", index=False)
            coverage_all += cov

    manifest["data_through_utc"] = {a: i["data_through_utc"] for a, i in manifest["assets"].items()}
    if coverage_all:
        pd.DataFrame(coverage_all).to_csv(meta / "extension_coverage.csv", index=False)
    write_json(manifest, meta / "split_manifest.json")
    log.info("Đã ghi %s", (meta / "split_manifest.json").relative_to(ROOT))
    return manifest
