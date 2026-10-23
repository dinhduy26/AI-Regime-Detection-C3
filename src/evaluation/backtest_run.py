"""C3 entry point: python -m src.evaluation.backtest_run

Đọc xác suất FILTERED đã kiểm chứng của C2 (qua src.models.handoff), giá/lợi suất của C1,
chạy backtest regime-aware và buy-and-hold trên tập test theo ba kịch bản đã khai báo
trong config/backtest.yaml, rồi xuất:

    outputs/tables/B08_portfolio.csv            chỉ số hiệu suất theo asset x strategy x scenario
    outputs/predictions/backtest_nav.csv        chuỗi NAV từng ngày (đọc bởi dashboard)
    outputs/figures/H05_cumulative_nav*.png/svg NAV test, buy-and-hold vs regime-aware
    outputs/metadata/backtest_manifest.json     cấu hình, provenance, giả định, hash đầu ra

Được run_pipeline.py gọi tự động qua hàm run(cfg). Đây là MÔ PHỎNG lịch sử, không phải
giao dịch thật và không phải khuyến nghị đầu tư.
"""

from __future__ import annotations

import argparse
import platform
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.evaluation.backtest import performance_metrics, regime_weights, simulate
from src.models.handoff import UnstableModelWarning, prepare_next_day_research_inputs
from src.utils.config import ROOT, file_hash, get_logger, load_config, path_of, read_json, write_json

log = get_logger("C3-backtest")
BACKTEST_CONFIG = ROOT / "config" / "backtest.yaml"
STRATEGIES = ("buy_and_hold", "regime_aware")
STRATEGY_LABELS = {"buy_and_hold": "Buy-and-Hold", "regime_aware": "Regime-aware"}


def load_backtest_config(path=None):
    path = Path(path) if path else BACKTEST_CONFIG
    with open(path, encoding="utf-8") as f:
        bt = yaml.safe_load(f)
    bt["_path"] = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)
    bt["_hash"] = file_hash(path)
    a = bt["allocation"]
    if not (0 < a["risk_threshold"] < 1 and 0 <= a["weight_risk_off"] <= a["weight_risk_on"] <= 1):
        raise ValueError("backtest.yaml: ngưỡng/tỷ trọng không hợp lệ")
    if bt["period"] not in ("train", "validation", "test"):
        raise ValueError("backtest.yaml: period phải là train/validation/test")
    for name, sc in bt["scenarios"].items():
        if sc["return_basis"] not in ("close_to_close", "next_open"):
            raise ValueError(f"{name}: return_basis không hợp lệ")
    return bt


def build_inputs(cfg, bt, asset):
    """Ghép tín hiệu filtered ngày t với lợi suất kỳ kế tiếp cho cả hai cơ sở lợi suất.

    close_to_close: R = C_{t+1}/C_t - 1 (simple_return của C1 tại ngày t+1).
    next_open:      R = O_{t+2}/O_{t+1} - 1 (khớp ở giá open ngay sau khi có tín hiệu).
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UnstableModelWarning)
        aligned, metadata = prepare_next_day_research_inputs(
            cfg, asset, period=bt["period"], acknowledge_unstable=bt["signal"]["acknowledge_unstable"]
        )
    notes = [str(w.message) for w in caught if issubclass(w.category, UnstableModelWarning)]
    column = bt["signal"]["column"]
    if column != "prob_high_risk" or (aligned["probability_type"] != "filtered").any():
        raise ValueError("C3 chỉ được dùng xác suất filtered của state rủi ro cao")
    if aligned["hindsight"].any():
        raise ValueError("Không dùng xác suất smoothed/hindsight cho backtest")
    if not (aligned["signal_date_utc"] < aligned["return_date_utc"]).all():
        raise AssertionError("Tín hiệu phải có trước kỳ lợi suất")
    if (aligned["as_of_utc"] > aligned["return_date_utc"]).any():
        raise AssertionError("Tín hiệu có sẵn muộn hơn thời điểm bắt đầu kỳ lợi suất")

    prices = pd.read_parquet(path_of(cfg, "processed") / "prices_daily.parquet")
    prices = prices.loc[prices["asset"] == asset, ["date_utc", "segment_id", "open", "close"]].copy()
    prices["date_utc"] = pd.to_datetime(prices["date_utc"], utc=True)
    o1 = prices.rename(columns={"date_utc": "return_date_utc", "open": "open_t1", "close": "close_t1"})
    o2 = prices[["date_utc", "segment_id", "open"]].rename(columns={"open": "open_t2"})
    o2["return_date_utc"] = o2.pop("date_utc") - pd.Timedelta(days=1)
    frame = aligned.merge(o1, on=["return_date_utc", "segment_id"], how="left", validate="one_to_one")
    frame = frame.merge(o2, on=["return_date_utc", "segment_id"], how="left", validate="one_to_one")
    frame["next_open_return"] = frame["open_t2"] / frame["open_t1"] - 1.0
    frame = frame.sort_values("signal_date_utc").reset_index(drop=True)
    gaps = frame["return_date_utc"].diff().dropna()
    if not (gaps == pd.Timedelta(days=1)).all():
        raise ValueError(f"{asset}: chuỗi ngày test không liên tục; cần xử lý theo segment")
    return frame, metadata, notes


def run_scenarios(frame, metadata, bt, asset, data_through):
    a, costs, m = bt["allocation"], bt["costs"], bt["metrics"]
    regime_w = regime_weights(frame["prob_high_risk"], a["risk_threshold"], a["weight_risk_on"], a["weight_risk_off"])
    weights = {"buy_and_hold": np.full(len(frame), float(a["weight_risk_on"])), "regime_aware": regime_w}
    nav_parts, rows = [], []
    for scenario, sc in bt["scenarios"].items():
        if sc["return_basis"] == "close_to_close":
            mask = np.ones(len(frame), dtype=bool)
            returns = frame["target_simple_return"].to_numpy(dtype=float)
        else:
            # Ngày test cuối chưa có open t+2: loại khỏi CẢ HAI chiến lược của kịch bản này.
            mask = frame["next_open_return"].notna().to_numpy()
            returns = frame.loc[mask, "next_open_return"].to_numpy(dtype=float)
        sub = frame.loc[mask].reset_index(drop=True)
        start, end = sub["return_date_utc"].iloc[0], sub["return_date_utc"].iloc[-1]
        calendar_days = (end - start).days + 1
        for strategy in STRATEGIES:
            w = weights[strategy][mask]
            sim = simulate(
                w,
                returns,
                fee_rate=costs["fee_rate"],
                turnover_method=sc["turnover_method"],
                initial_weight=a["initial_weight"],
                cash_return=a["cash_return_daily"],
            )
            nav = pd.DataFrame(
                {
                    "asset": asset,
                    "date_utc": sub["return_date_utc"].dt.strftime("%Y-%m-%d"),
                    "strategy": strategy,
                    "scenario": scenario,
                    "return_basis": sc["return_basis"],
                    "turnover_method": sc["turnover_method"],
                    "signal_date_utc": sub["signal_date_utc"].dt.strftime("%Y-%m-%d"),
                    "filtered_asof_utc": sub["as_of_utc"].astype(str),
                    "prob_high_risk": sub["prob_high_risk"].to_numpy(),
                    "signal_state_id": sub["state_id"].to_numpy(),
                    "signal_label": sub["label"].to_numpy(),
                }
            )
            nav = pd.concat([nav, sim], axis=1)
            nav_parts.append(nav)
            metrics = performance_metrics(
                sim["net"], sim["nav"], sim["turnover"], sim["fee"], calendar_days,
                m["annualization_days"], m["risk_free_daily"],
            )
            rows.append(
                {
                    "asset": asset,
                    "strategy": strategy,
                    "scenario": scenario,
                    "return_basis": sc["return_basis"],
                    "turnover_method": sc["turnover_method"],
                    "period": bt["period"],
                    "start": start.strftime("%Y-%m-%d"),
                    "end": end.strftime("%Y-%m-%d"),
                    "n_days": len(sub),
                    "calendar_days": calendar_days,
                    **metrics,
                    "avg_weight": float(np.mean(w)),
                    "share_days_risk_off": float(np.mean(w < a["weight_risk_on"])),
                    "n_rebalances": int((sim["turnover"] > 0).sum()),
                    "fee_rate": costs["fee_rate"],
                    "risk_threshold": a["risk_threshold"],
                    "weight_risk_on": a["weight_risk_on"],
                    "weight_risk_off": a["weight_risk_off"],
                    "annualization_days": m["annualization_days"],
                    "risk_free_daily": m["risk_free_daily"],
                    "sharpe_basis": "after_fees",
                    "units": "decimal (0.05 = 5%)",
                    "model_run_id": metadata["model_run_id"],
                    "K": metadata["K"],
                    "high_risk_state": metadata["high_risk_state"],
                    "seed_stability_status": metadata["seed_stability_status"],
                    "data_through_utc": data_through,
                    "label": "simulated research backtest; not investment advice",
                }
            )
    return pd.concat(nav_parts, ignore_index=True), pd.DataFrame(rows)


def export_h05(nav, figures, dpi, bt):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base = nav.loc[nav["scenario"] == "baseline_close_to_close"].copy()
    base["date"] = pd.to_datetime(base["date_utc"])
    assets = list(dict.fromkeys(base["asset"]))
    colors = {"buy_and_hold": "#6b7280", "regime_aware": "#2563eb"}
    a, fee = bt["allocation"], bt["costs"]["fee_rate"]
    caption = (
        f"Simulated test-period backtest, close-to-close, fee {fee:.2%} x turnover, start in cash. "
        f"Regime-aware: w=1 if filtered P(high-risk) < {a['risk_threshold']}, else {a['weight_risk_off']}. "
        "Not investment advice."
    )
    written = []

    def draw(ax_nav, ax_dd, sub, asset):
        for strategy in STRATEGIES:
            s = sub.loc[sub["strategy"] == strategy]
            ax_nav.plot(s["date"], s["nav"], color=colors[strategy], lw=1.4, label=STRATEGY_LABELS[strategy])
            ax_dd.plot(s["date"], s["drawdown"], color=colors[strategy], lw=1.0)
        ax_nav.axhline(1.0, color="#9ca3af", lw=0.6, ls=":")
        ax_nav.set_title(f"{asset}: cumulative NAV on test ({sub['date_utc'].min()} to {sub['date_utc'].max()})")
        ax_nav.set_ylabel("NAV (start = 1)")
        ax_dd.set_ylabel("Drawdown")
        ax_dd.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax_nav.legend(loc="upper left", frameon=False)
        locator = matplotlib.dates.MonthLocator(bymonth=(1, 4, 7, 10))
        ax_dd.xaxis.set_major_locator(locator)
        ax_dd.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y-%m"))
        for ax in (ax_nav, ax_dd):
            ax.grid(alpha=0.25)

    for asset in assets:
        sub = base.loc[base["asset"] == asset]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
        draw(ax1, ax2, sub, asset)
        fig.text(0.01, 0.005, caption, fontsize=7, color="#4b5563")
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        for ext in ("png", "svg"):
            p = figures / f"H05_cumulative_nav_{asset.lower()}.{ext}"
            fig.savefig(p, dpi=dpi)
            written.append(p)
        plt.close(fig)

    fig, axes = plt.subplots(
        2, len(assets), figsize=(7 * len(assets), 6), sharex="col", gridspec_kw={"height_ratios": [3, 1]},
        squeeze=False,
    )
    for j, asset in enumerate(assets):
        draw(axes[0, j], axes[1, j], base.loc[base["asset"] == asset], asset)
    fig.text(0.01, 0.005, caption, fontsize=7, color="#4b5563")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    for ext in ("png", "svg"):
        p = figures / f"H05_cumulative_nav.{ext}"
        fig.savefig(p, dpi=dpi)
        written.append(p)
    plt.close(fig)
    return written


def run(cfg, bt=None):
    bt = bt or load_backtest_config()
    folders = {k: path_of(cfg, k) for k in ("tables", "predictions", "figures", "metadata")}
    split_manifest = read_json(path_of(cfg, "metadata") / "split_manifest.json")
    data_through = split_manifest.get("data_through_utc")
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "running",
        "backtest_config": bt["_path"],
        "backtest_config_version": bt["version"],
        "backtest_config_hash": bt["_hash"],
        "config_version": cfg["version"],
        "python": platform.python_version(),
        "data_through_utc": data_through,
        "signal": "C2 filtered P(high-risk state | X_1:t) at end of UTC day t; smoothed never used",
        "assumptions": [
            "simulated historical research backtest; not live trading and not investment advice",
            "baseline is idealized end-of-day exposure: close t -> close t+1 (Methodology 3.4-3.5)",
            "next_open cross-check enters at open t+1 and holds to open t+2; no latency, spread or slippage",
            "fee 0.10% x turnover is a research assumption, not Binance's actual fee schedule",
            "cash earns 0; long/cash only, no leverage; both strategies start in cash and pay the opening fee",
            "annualization 365 days; Sharpe after fees with rf = 0",
            "threshold 0.5 and weights 1/0.25 fixed before test; not tuned on test results",
        ],
        "assets": {},
    }
    manifest_path = folders["metadata"] / "backtest_manifest.json"
    write_json(manifest, manifest_path)
    try:
        navs, tables = [], []
        for asset in cfg["assets"]:
            frame, metadata, notes = build_inputs(cfg, bt, asset)
            asset_through = data_through.get(asset) if isinstance(data_through, dict) else data_through
            nav, table = run_scenarios(frame, metadata, bt, asset, asset_through)
            navs.append(nav)
            tables.append(table)
            manifest["assets"][asset] = {
                "model_run_id": metadata["model_run_id"],
                "K": metadata["K"],
                "high_risk_state": metadata["high_risk_state"],
                "label_map": metadata["label_map"],
                "seed_stability_status": metadata["seed_stability_status"],
                "stability_warnings": notes,
                "n_signals": len(frame),
                "first_signal_date": frame["signal_date_utc"].min().strftime("%Y-%m-%d"),
                "first_return_date": frame["return_date_utc"].min().strftime("%Y-%m-%d"),
                "last_return_date": frame["return_date_utc"].max().strftime("%Y-%m-%d"),
            }
            base = table.loc[table["scenario"] == "baseline_close_to_close"].set_index("strategy")
            log.info(
                "%s: test %s..%s | B&H NAV %.4f MDD %.2f%% | regime-aware NAV %.4f MDD %.2f%% | risk-off days %.1f%%",
                asset, base.loc["regime_aware", "start"], base.loc["regime_aware", "end"],
                base.loc["buy_and_hold", "final_nav"], 100 * base.loc["buy_and_hold", "max_drawdown"],
                base.loc["regime_aware", "final_nav"], 100 * base.loc["regime_aware", "max_drawdown"],
                100 * base.loc["regime_aware", "share_days_risk_off"],
            )
        nav_all = pd.concat(navs, ignore_index=True)
        b08 = pd.concat(tables, ignore_index=True)
        outputs = []
        b08_path = folders["tables"] / "B08_portfolio.csv"
        nav_path = folders["predictions"] / "backtest_nav.csv"
        b08.to_csv(b08_path, index=False)
        nav_all.to_csv(nav_path, index=False)
        outputs += [b08_path, nav_path]
        outputs += export_h05(nav_all, folders["figures"], bt["figures"]["dpi"], bt)
        manifest["status"] = "complete"
        manifest["output_hashes"] = {p.relative_to(ROOT).as_posix(): file_hash(p) for p in outputs}
    except Exception as exc:
        manifest["status"], manifest["error"] = "failed", f"{type(exc).__name__}: {exc}"
        raise
    finally:
        write_json(manifest, manifest_path)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description="C3 regime-aware backtest (research simulation)")
    parser.add_argument("--config", default=None)
    parser.add_argument("--backtest-config", default=None)
    args = parser.parse_args(argv)
    run(load_config(args.config), load_backtest_config(args.backtest_config))


if __name__ == "__main__":
    main()
