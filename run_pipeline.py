"""C1-13 — Chạy lại toàn bộ pipeline từ một lệnh.

    python run_pipeline.py                 # dùng dữ liệu thô đã có (cache), chạy clean -> features -> reports
    python run_pipeline.py --download      # tải lại dữ liệu thật từ Binance rồi chạy tiếp
    python run_pipeline.py --steps clean features      # chỉ chạy một số bước
    python run_pipeline.py --download --macro --derivatives   # kèm dữ liệu mở rộng RQ3

Thứ tự bước: collect -> clean -> features -> reports -> models (C2) -> backtest (C3)
Bước models/backtest được gọi tự động KHI module của C2/C3 đã có hàm run(cfg):
    src/models/run.py        (C2)
    src/evaluation/backtest_run.py  (C3)
Mỗi lần chạy ghi outputs/metadata/run_manifest.json (thời điểm, cấu hình, trạng thái, hash file).
"""
from __future__ import annotations

import argparse
import importlib
import sys
import time
import traceback
from datetime import datetime, timezone

from src.utils.config import ROOT, file_hash, get_logger, load_config, path_of, read_json, write_json

log = get_logger("pipeline")
ALL_STEPS = ["collect", "clean", "features", "reports", "models", "backtest"]
OPTIONAL_HOOKS = {"models": "src.models.run", "backtest": "src.evaluation.backtest_run"}


def raw_ready(cfg) -> bool:
    raw = path_of(cfg, "raw")
    return all((raw / f"{a.lower()}_ohlcv.csv").exists() for a in cfg["assets"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Crypto Market Regime Radar — pipeline")
    ap.add_argument("--config", default=None)
    ap.add_argument("--download", action="store_true", help="tải lại dữ liệu thật dù đã có cache")
    ap.add_argument("--method", choices=["api", "vision"], default="api")
    ap.add_argument("--macro", action="store_true")
    ap.add_argument("--derivatives", action="store_true")
    ap.add_argument("--steps", nargs="*", default=None, choices=ALL_STEPS)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    steps = args.steps or ALL_STEPS
    status, t_all = {}, time.time()

    for step in steps:
        t0 = time.time()
        try:
            if step == "collect":
                if raw_ready(cfg) and not args.download:
                    status[step] = "bỏ qua — đã có dữ liệu thô (dùng --download để tải lại)"
                    log.info("collect: %s", status[step])
                    continue
                from src.data import collect
                cli = ["--method", args.method] + (["--macro"] if args.macro else []) \
                    + (["--derivatives"] if args.derivatives else [])
                if args.config:
                    cli += ["--config", args.config]
                collect.main(cli)
            elif step == "clean":
                from src.data import clean
                clean.run(cfg)
            elif step == "features":
                from src.features import pipeline
                pipeline.run(cfg)
            elif step == "reports":
                from src.evaluation import descriptives
                descriptives.run(cfg)
            else:  # bước của C2/C3: chỉ chạy nếu module đã tồn tại
                try:
                    mod = importlib.import_module(OPTIONAL_HOOKS[step])
                except ModuleNotFoundError:
                    status[step] = f"chưa có — {OPTIONAL_HOOKS[step]} chưa được tạo"
                    log.info("%s: %s", step, status[step])
                    continue
                mod.run(cfg)
            status[step] = f"OK ({time.time() - t0:.1f}s)"
        except Exception as e:  # noqa: BLE001
            status[step] = f"LỖI: {e}"
            log.error("Bước %s lỗi:\n%s", step, traceback.format_exc())
            break

    outputs = {}
    for folder in ("data/processed", "outputs/tables", "outputs/figures", "outputs/metadata",
                   "outputs/models", "data/raw"):
        for p in sorted((ROOT / folder).glob("*")):
            if p.is_file() and p.name not in (".gitkeep", "run_manifest.json"):
                outputs[p.relative_to(ROOT).as_posix()] = file_hash(p)
    split_path = path_of(cfg, "metadata") / "split_manifest.json"
    data_through = read_json(split_path).get("data_through_utc") if split_path.exists() else None
    write_json({
        "run_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_through_utc": data_through, "split_mode": cfg["split"].get("mode", "ratio"),
        "duration_s": round(time.time() - t_all, 1),
        "python": sys.version.split()[0],
        "config_path": cfg["_config_path"], "config_version": cfg["version"],
        "config_hash": cfg["_config_hash"],
        "steps": status, "output_hashes": outputs,
    }, path_of(cfg, "metadata") / "run_manifest.json")

    print("\n=== KẾT QUẢ PIPELINE ===")
    for k, v in status.items():
        print(f"  {k:<9} {v}")
    if data_through:
        print(f"  Dữ liệu (nến ngày UTC đã đóng) đến hết: {data_through}  | split: {cfg['split'].get('mode')}")
    return 1 if any(str(v).startswith("LỖI") for v in status.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
