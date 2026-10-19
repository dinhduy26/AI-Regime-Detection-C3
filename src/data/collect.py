"""Lệnh tải dữ liệu thật (chạy trên máy có Internet).

    python -m src.data.collect                    # BTC + ETH từ Binance REST API
    python -m src.data.collect --method vision    # dùng kho file data.binance.vision
    python -m src.data.collect --macro --derivatives   # thêm dữ liệu mở rộng RQ3
    python -m src.data.collect --skip-core --all-extensions   # MỌI dữ liệu mở rộng, không tải lại BTC/ETH

Kết quả: data/raw/btc_ohlcv.csv, data/raw/eth_ohlcv.csv,
         outputs/metadata/source_log.csv  (+ macro_daily.csv, derivatives_daily.csv nếu bật)
"""
from __future__ import annotations

import argparse

import pandas as pd

from src.data.binance_collector import collect_asset
from src.data.collection_meta import write_collection_config, write_limitation_log
from src.utils.config import ROOT, get_logger, load_config, path_of

log = get_logger("collect")


def _upsert_source_log(rows: list[dict], meta_dir) -> None:
    path = meta_dir / "source_log.csv"
    new = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path, dtype=str)
        old = old[~old["asset"].isin(new["asset"].astype(str))]
        new = pd.concat([old, new.astype(str)], ignore_index=True)
    new.to_csv(path, index=False)
    log.info("Cập nhật %s", path.relative_to(ROOT))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Tải dữ liệu thật cho C1")
    ap.add_argument("--config", default=None)
    ap.add_argument("--method", choices=["api", "vision"], default="api")
    ap.add_argument("--assets", nargs="*", default=None, help="mặc định: tất cả trong config")
    ap.add_argument("--macro", action="store_true", help="tải VIX/SPX/DXY (mở rộng)")
    ap.add_argument("--derivatives", action="store_true", help="tải funding/OI (mở rộng)")
    ap.add_argument("--crypto-market", action="store_true",
                    help="tải stablecoin supply, DeFi TVL (DefiLlama), market cap (CoinGecko), CMC200")
    ap.add_argument("--extra-assets", action="store_true", help="tải SOL, BNB (tài sản mở rộng)")
    ap.add_argument("--all-extensions", action="store_true",
                    help="= --macro --derivatives --crypto-market --extra-assets")
    ap.add_argument("--skip-core", action="store_true",
                    help="không tải lại BTC/ETH, chỉ tải dữ liệu mở rộng")
    args = ap.parse_args(argv)
    if args.all_extensions:
        args.macro = args.derivatives = args.crypto_market = args.extra_assets = True

    cfg = load_config(args.config)
    raw_dir, meta_dir = path_of(cfg, "raw"), path_of(cfg, "metadata")
    assets = args.assets or list(cfg["assets"])

    rows = [] if args.skip_core else [collect_asset(a, cfg, raw_dir, method=args.method) for a in assets]

    if args.extra_assets:
        rows += [collect_asset(a, cfg, raw_dir, method=args.method) for a in cfg["extension_assets"]]

    if args.crypto_market or cfg["crypto_market"]["enabled"]:
        from src.data.crypto_market_collector import collect_crypto_market
        rows += collect_crypto_market(cfg, raw_dir)

    if args.macro or cfg["macro"]["enabled"]:
        from src.data.macro_collector import collect_macro
        rows += collect_macro(cfg, raw_dir)

    if args.derivatives or cfg["derivatives"]["enabled"]:
        from src.data.derivatives_collector import collect_derivatives
        src_rows, limits = collect_derivatives(cfg, raw_dir)
        rows += src_rows
        if limits:
            pd.DataFrame(limits).to_csv(meta_dir / "limitation_log.csv", index=False)

    write_collection_config(cfg)
    write_limitation_log(cfg)

    if not rows:
        print("Không có gì để tải.")
        return
    _upsert_source_log(rows, meta_dir)
    print("\n=== TÓM TẮT ===")
    print(pd.DataFrame(rows)[["asset", "source", "actual_start", "actual_end", "n_rows"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
