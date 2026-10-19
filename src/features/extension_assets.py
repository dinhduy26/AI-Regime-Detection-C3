"""[MỞ RỘNG] Tài sản mở rộng SOL, BNB (blueprint §3.1) — dùng ĐÚNG các hàm của BTC/ETH:
làm sạch (clean_asset) -> feature (compute_asset_features) -> chia mẫu (select_and_split, cùng mốc
fixed_dates) -> scaler chỉ trên train (fit_transform).

Đầu ra tách riêng để KHÔNG đổi hợp đồng dữ liệu lõi BTC/ETH:
  data/processed/extension/prices_daily_ext.parquet
  data/processed/extension/features_{asset}.parquet, scaled_features_{asset}.parquet
  outputs/models/extension/scaler_{asset}.joblib
  outputs/metadata/split_manifest_ext.json, gap_report_ext.csv, data_quality_ext.csv
  outputs/tables/B01_descriptives_ext.csv
Tài sản chưa tải (không có data/raw/{asset}_ohlcv.csv) thì bỏ qua và ghi vào limitation_log.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from src.data.clean import clean_asset
from src.evaluation.descriptives import make_b01
from src.features.build import compute_asset_features, select_and_split
from src.features.scaling import fit_transform
from src.utils.config import ROOT, get_logger, load_schema, path_of, write_json

log = get_logger("extension-assets")


def run(cfg: dict) -> dict:
    raw, meta, tables = path_of(cfg, "raw"), path_of(cfg, "metadata"), path_of(cfg, "tables")
    out_dir = ROOT / cfg["paths"]["processed"] / "extension"
    model_dir = ROOT / cfg["paths"]["models"] / "extension"
    schema = load_schema()["files"]
    assets = [a for a in cfg.get("extension_assets", {}) if (raw / f"{a.lower()}_ohlcv.csv").exists()]
    if not assets:
        log.info("Chưa có dữ liệu tài sản mở rộng (chạy: python -m src.data.collect --skip-core --extra-assets)")
        return {}
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    manifest = {"created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "config_version": cfg["version"], "config_hash": cfg["_config_hash"],
                "split_config": dict(cfg["split"]), "core_features": cfg["features"]["core"], "assets": {}}
    prices_all, gaps_all, quality_all, feats = [], [], [], {}
    for a in assets:
        res = clean_asset(a, cfg, raw)
        prices = res["prices"][schema["prices_daily"]["columns"]]
        prices_all.append(prices)
        gaps_all.append(res["gaps"])
        quality_all += res["quality"]
        feat, info = select_and_split(compute_asset_features(prices, cfg), cfg)
        cols = schema["features"]["columns"] + [c for c in schema["features"]["optional_columns"]
                                                if c in feat.columns]
        feat = feat[cols]
        feat.to_parquet(out_dir / f"features_{a.lower()}.parquet", index=False)
        scaled, params = fit_transform(feat, cfg["features"]["core"], model_dir / f"scaler_{a.lower()}.joblib")
        scaled.to_parquet(out_dir / f"scaled_features_{a.lower()}.parquet", index=False)
        info["scaler"] = params
        manifest["assets"][a] = info
        feats[a] = feat
        log.info("%s: %s -> %s, N=%d (train %d | val %d | test %d)", a, prices["date_utc"].min(),
                 prices["date_utc"].max(), info["n_eligible"], info["train"]["n"],
                 info["validation"]["n"], info["test"]["n"])

    pd.concat(prices_all, ignore_index=True).to_parquet(out_dir / "prices_daily_ext.parquet", index=False)
    pd.concat(gaps_all, ignore_index=True).to_csv(meta / "gap_report_ext.csv", index=False)
    pd.DataFrame(quality_all)[schema["data_quality"]["columns"]].to_csv(meta / "data_quality_ext.csv", index=False)
    make_b01(feats)[schema["B01_descriptives"]["columns"]].to_csv(tables / "B01_descriptives_ext.csv", index=False)
    manifest["data_through_utc"] = {a: i["data_through_utc"] for a, i in manifest["assets"].items()}
    write_json(manifest, meta / "split_manifest_ext.json")
    return manifest
