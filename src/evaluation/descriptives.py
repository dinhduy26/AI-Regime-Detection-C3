"""C1-12 — Bảng B01 (Descriptives), B02 (Sample manifest), hình H01, data dictionary.

B01 (đặc tả C1 §1.9): với x_1..x_n hợp lệ
    mean = (1/n) Σx,   sd = sqrt(Σ(x-mean)^2 / (n-1))
    m_k  = (1/n) Σ(x-mean)^k,   skew = m3 / m2^1.5,   kurtosis = m4 / m2^2  (PEARSON, không trừ 3)
Xuất cho 2 giai đoạn, ghi rõ trong cột `period`:
    full_sample = toàn bộ quan sát đủ điều kiện (train+validation+test)
    train       = chỉ tập train
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.config import check_columns, get_logger, load_schema, path_of, read_json

log = get_logger("descriptives")
B01_VARS = ["log_return", "simple_return", "rv21", "log_dollar_volume"]
COLORS = {"BTC": "#2a78d6", "ETH": "#eb6834"}  # categorical slot 1, 2 (palette mặc định)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def describe(x: pd.Series) -> dict:
    x = x.dropna().to_numpy(dtype=float)
    n = len(x)
    m = x.mean()
    d = x - m
    m2, m3, m4 = (d ** 2).mean(), (d ** 3).mean(), (d ** 4).mean()
    return {"n": n, "mean": m, "median": float(np.median(x)), "sd": float(x.std(ddof=1)),
            "min": float(x.min()), "max": float(x.max()),
            "skew": float(m3 / m2 ** 1.5) if m2 > 0 else np.nan,
            "kurtosis": float(m4 / m2 ** 2) if m2 > 0 else np.nan}


def make_b01(feats: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for asset, f in feats.items():
        for period, part in (("full_sample", f), ("train", f[f["split"] == "train"])):
            for v in B01_VARS:
                rows.append({"asset": asset, "variable": v, "period": period, **describe(part[v]),
                             "kurtosis_type": "Pearson (không trừ 3)",
                             "start": part["date_utc"].min(), "end": part["date_utc"].max()})
    return pd.DataFrame(rows)


def make_b02(manifest: dict, prices: pd.DataFrame, gaps: pd.DataFrame, source_log: pd.DataFrame,
             quality: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for asset, info in manifest["assets"].items():
        p = prices[prices["asset"] == asset]
        sl = source_log[source_log["asset"] == asset]
        n_raw = quality[(quality["asset"] == asset) & (quality["check"] == "số dòng thô")]["n_flagged"]
        g = gaps[gaps["asset"] == asset]
        rows.append({
            "asset": asset, "symbol": p["symbol"].iloc[0],
            "source": " | ".join(p["source"].unique()),
            "start": p["date_utc"].min(), "end": p["date_utc"].max(),
            "n_raw": int(n_raw.iloc[0]) if len(n_raw) else None, "n_clean": len(p),
            "n_missing_days": int(g["gap_length"].sum()) if len(g) else 0, "n_gaps": len(g),
            "missing": (f"{int(g['gap_length'].sum())} ngày thiếu: " + "; ".join(
                f"{a}..{b}" for a, b in zip(g["missing_from"], g["missing_to"]))) if len(g) else "0",
            "n_warmup_dropped": info["n_warmup_dropped"], "n_eligible": info["n_eligible"],
            "n_train": info["train"]["n"], "n_val": info["validation"]["n"], "n_test": info["test"]["n"],
            "train_start": info["train"]["start"], "train_end": info["train"]["end"],
            "val_start": info["validation"]["start"], "val_end": info["validation"]["end"],
            "test_start": info["test"]["start"], "test_end": info["test"]["end"],
            "gap_handling_method": manifest["gap_handling_method"],
            "split_mode": info.get("split_mode", "ratio"),
            "share_train": info["train"].get("share"), "share_val": info["validation"].get("share"),
            "share_test": info["test"].get("share"),
            "data_through_utc": info.get("data_through_utc", info["test"]["end"]),
            "fetched_at_utc": sl["fetched_at_utc"].iloc[0] if len(sl) else "",
        })
    return pd.DataFrame(rows)


def make_h01(prices: pd.DataFrame, gaps: pd.DataFrame, manifest: dict, out_png, out_svg) -> None:
    """Giá đóng cửa BTC, ETH (thang log), 2 panel riêng — KHÔNG dùng 2 trục y trên 1 hình."""
    assets = list(manifest["assets"])
    fig, axes = plt.subplots(len(assets), 1, figsize=(10, 3.3 * len(assets)), sharex=True,
                             constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, asset in zip(axes, assets):
        p = prices[prices["asset"] == asset].copy()
        p["d"] = pd.to_datetime(p["date_utc"])
        # vẽ theo từng đoạn liên tục -> ngày thiếu hiện là chỗ đứt, không nối thẳng qua gap
        for _, seg in p.groupby("segment_id"):
            ax.plot(seg["d"], seg["close"], color=COLORS.get(asset, "#2a78d6"), lw=1.4)
        ax.set_yscale("log")
        ax.set_title(f"{asset}/USDT — giá đóng cửa ngày (UTC, thang log)", loc="left",
                     fontsize=11, color=INK)
        ax.set_ylabel("USDT", color=INK2)
        info = manifest["assets"][asset]
        # Vùng có giá nhưng KHÔNG vào mẫu mô hình (đoạn trước gap + warm-up RV21)
        model_start = pd.to_datetime(info["train"]["start"])
        if p["d"].min() < model_start:
            ax.axvspan(p["d"].min(), model_start, color="#e4e3df", alpha=0.7, lw=0, zorder=0)
            ax.annotate("ngoài mẫu\nmô hình", (p["d"].min(), 1), xycoords=("data", "axes fraction"),
                        xytext=(3, -4), textcoords="offset points", fontsize=8, color=INK2,
                        va="top")
        # Mốc chia mẫu; nhãn đặt ở đáy khung để không đè lên đường giá
        for name in ("train", "validation", "test"):
            x = pd.to_datetime(info[name]["start"])
            if name != "train":
                ax.axvline(x, color=INK2, lw=0.8, ls="--")
            ax.annotate(name, (x, 0), xycoords=("data", "axes fraction"), xytext=(4, 4),
                        textcoords="offset points", fontsize=8, color=INK2)
        for _, g in gaps[gaps["asset"] == asset].iterrows():
            ax.axvspan(pd.to_datetime(g["missing_from"]),
                       pd.to_datetime(g["missing_to"]) + pd.Timedelta(days=1),
                       color="#c3c2b7", alpha=0.6, lw=0)
        ax.grid(True, color=GRID, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=INK2, labelsize=9)
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    src = " | ".join(prices["source"].unique())
    fig.text(0.01, -0.04, f"Nguồn: {src}. Nến ngày UTC đã đóng.\nVùng nền xám nhạt = có giá nhưng không "
             "vào mẫu mô hình (đoạn trước ngày thiếu + warm-up RV21); đường đứt = mốc chia mẫu theo thời gian.",
             fontsize=8, color=INK2)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)


DICTIONARY = [
    # file, variable, definition, unit, source, role
    ("prices_daily", "asset", "Mã tài sản BTC hoặc ETH", "—", "C1", "khóa"),
    ("prices_daily", "symbol", "Cặp giao dịch Binance Spot (BTCUSDT/ETHUSDT)", "—", "Binance", "provenance"),
    ("prices_daily", "date_utc", "Ngày UTC của nến (mở 00:00:00, đóng 23:59:59.999 UTC)", "YYYY-MM-DD", "Binance", "khóa"),
    ("prices_daily", "open/high/low/close", "Giá mở/cao/thấp/đóng cửa của nến ngày", "USDT", "Binance /api/v3/klines", "dữ liệu gốc"),
    ("prices_daily", "base_volume", "Khối lượng tính bằng tài sản cơ sở (BTC/ETH) — KHÔNG dùng cho HMM", "BTC hoặc ETH", "Binance (cột 5)", "đối soát"),
    ("prices_daily", "quote_volume", "Giá trị giao dịch theo đồng định giá", "USDT", "Binance (cột 7)", "đầu vào log_dollar_volume"),
    ("prices_daily", "n_trades", "Số giao dịch trong ngày", "lệnh", "Binance (cột 8)", "mô tả"),
    ("prices_daily", "source", "Nguồn của từng dòng", "—", "C1", "provenance"),
    ("prices_daily", "available_at_utc", "Thời điểm thông tin của nến được coi là đã biết (= nến đóng)", "ISO-8601 UTC", "C1", "point-in-time"),
    ("prices_daily", "segment_id", "Mã đoạn liên tục; tăng 1 sau mỗi khoảng ngày thiếu", "số nguyên", "C1", "xử lý gap"),
    ("features", "log_return", "ln(C_t / C_{t-1}), C_{t-1} cùng đoạn liên tục", "thập phân/ngày", "C1", "HMM lõi"),
    ("features", "rv21", "Độ lệch chuẩn mẫu (ddof=1) của 21 log return t-20..t; không annualize", "thập phân/ngày", "C1", "HMM lõi"),
    ("features", "log_dollar_volume", "ln(1 + quote_volume)", "log(1+USDT)", "C1", "HMM lõi"),
    ("features", "simple_return", "C_t / C_{t-1} - 1", "thập phân/ngày", "C1", "NAV backtest (C3); không vào HMM"),
    ("features", "split", "train / validation / test theo thời gian 60/20/20", "—", "C1", "chia mẫu"),
    ("features", "rv7, rv14, rv30", "Như rv21 với cửa sổ 7/14/30", "thập phân/ngày", "C1", "tùy chọn, robustness"),
    ("features", "rv21_annualized", "rv21 × sqrt(365)", "thập phân/năm", "C1", "chỉ mô tả"),
    ("features", "amihud_illiq", "|simple_return| / quote_volume; để trống nếu quote_volume = 0", "1/USDT", "C1", "tùy chọn; proxy, không phải spread"),
    ("features", "btc_eth_corr21", "Tương quan trượt 21 ngày của log return BTC và ETH; cần đủ 21 ngày trùng", "[-1, 1]", "C1", "tùy chọn, mô tả"),
    ("scaled_features", "z_log_return, z_rv21, z_log_dollar_volume", "(x - mean_train) / std_train (ddof=0); scaler fit CHỈ trên train", "z-score", "C1", "đầu vào HMM/GMM của C2"),
    ("features_ext", "vix, spx, dxy", "Giá đóng cửa Mỹ ghép as-of theo available_at <= thời điểm nến đóng", "điểm chỉ số", "Yahoo/FRED", "mở rộng RQ3"),
    ("features_ext", "funding_sum / funding_mean / funding_n", "Tổng/trung bình/số lần funding thanh toán trong ngày UTC", "thập phân", "Binance Futures", "mở rộng RQ3"),
    ("features_ext", "us10y, us2y, us3m", "Lợi suất trái phiếu Mỹ 10 năm / 2 năm / 3 tháng, ghép as-of", "%/năm", "FRED DGS10/DGS2/DGS3MO", "mở rộng RQ3 (blueprint §3.3)"),
    ("features_ext", "term_spread_10y_2y", "us10y - us2y, chỉ khi hai kỳ hạn cùng ngày quan sát", "điểm %", "FRED", "mở rộng RQ3"),
    ("features_ext", "stablecoin_supply", "Tổng stablecoin lưu hành (mọi loại peg quy USD), ghép as-of; để trống trước crypto_market.reliable_from (mặc định 2020-01-01: DefiLlama chưa theo dõi đủ giai đoạn đầu)", "USD", "DefiLlama", "mở rộng RQ3 (blueprint §2.4, §3.5)"),
    ("features_ext", "stablecoin_supply_growth_7d", "ln(S_d / S_{d-7}) tính trên chuỗi gốc rồi ghép as-of; để trống nếu giá trị <= 0 hoặc thiếu", "log", "DefiLlama", "mở rộng RQ3"),
    ("features_ext", "defi_tvl, defi_tvl_growth_7d", "Tổng giá trị khóa DeFi và tăng trưởng 7 ngày (để trống nếu giá trị <= 0)", "USD / log", "DefiLlama", "robustness (blueprint §3.2)"),
    ("features_ext", "mcap_btc, mcap_eth", "Vốn hóa thị trường (CoinGecko miễn phí, lịch sử giới hạn)", "USD", "CoinGecko", "mô tả/robustness (blueprint §3.2)"),
    ("features_ext", "cmc200", "CMC Crypto 200 Index (nếu Yahoo có dữ liệu)", "điểm", "Yahoo ^CMC200", "mở rộng (blueprint §3.1)"),
    ("features_ext", "open_interest, open_interest_value_usdt", "Open interest ảnh chụp HỢP LỆ cuối ngày UTC t (bỏ dòng lặp và ảnh chụp OI <= 0)", "hợp đồng / USDT", "Binance Futures metrics (data.binance.vision)", "mở rộng RQ3 (blueprint §2.4)"),
    ("features_ext", "oi_growth_7d", "ln(OI_t / OI_{t-7}) theo giá trị USDT; để trống nếu giá trị <= 0 hoặc thiếu", "log", "Binance Futures", "mở rộng RQ3"),
    ("features_ext", "<biến>_obs_date, <biến>_available_at", "Ngày quan sát gốc và thời điểm có sẵn của giá trị đã ghép (kiểm chứng point-in-time)", "ngày / UTC", "C1", "kiểm toán"),
    ("extension/features_sol, features_bnb", "như features BTC/ETH", "SOL, BNB xử lý bằng đúng hàm và mốc chia mẫu của BTC/ETH", "như trên", "Binance Spot", "tài sản mở rộng (blueprint §3.1)"),
]


def make_dictionary() -> pd.DataFrame:
    return pd.DataFrame(DICTIONARY, columns=["file", "variable", "definition", "unit", "source", "role"])


def run(cfg: dict) -> None:
    proc, meta, tables, figs = (path_of(cfg, k) for k in ("processed", "metadata", "tables", "figures"))
    schema = load_schema()["files"]
    manifest = read_json(meta / "split_manifest.json")
    prices = pd.read_parquet(proc / "prices_daily.parquet")
    feats = {a: pd.read_parquet(proc / f"features_{a.lower()}.parquet") for a in manifest["assets"]}
    gaps = pd.read_csv(meta / "gap_report.csv", dtype=str)
    if len(gaps):
        gaps["gap_length"] = gaps["gap_length"].astype(int)
    source_log = pd.read_csv(meta / "source_log.csv", dtype=str) if (meta / "source_log.csv").exists() \
        else pd.DataFrame(columns=["asset", "fetched_at_utc"])
    quality = pd.read_csv(meta / "data_quality.csv")

    b01 = make_b01(feats)
    b01[schema["B01_descriptives"]["columns"]].to_csv(tables / "B01_descriptives.csv", index=False)
    b02 = make_b02(manifest, prices, gaps, source_log, quality)
    b02_cols = schema["B02_sample_manifest"]["columns"]
    check_columns(b02, b02_cols, "B02_sample_manifest")
    b02[b02_cols + [c for c in b02.columns if c not in b02_cols]].to_csv(
        tables / "B02_sample_manifest.csv", index=False)
    make_h01(prices, gaps, manifest, figs / "H01_price_history.png", figs / "H01_price_history.svg")
    make_dictionary().to_csv(meta / "data_dictionary.csv", index=False)
    log.info("Đã ghi B01_descriptives.csv, B02_sample_manifest.csv, H01_price_history.png/svg, "
             "data_dictionary.csv")
