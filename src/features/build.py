"""C1-06 / C1-07 / C1-10 — Tính feature, ghép ma trận, chia train/val/test theo thời gian.

Công thức (đặc tả C1 §1.3–1.6), t = ngày UTC, chỉ dùng thông tin đến hết ngày t:
    log_return_t        = ln(C_t / C_{t-1})                   (thập phân/ngày)
    simple_return_t     = C_t / C_{t-1} - 1                   (dùng cho NAV của C3)
    rv21_t              = std mẫu (ddof=1) của log_return_{t-20..t}   (KHÔNG annualize)
    log_dollar_volume_t = ln(1 + quote_volume_t)              (quote volume USDT)
Mọi phép tính lag/rolling làm RIÊNG TRONG TỪNG ĐOẠN LIÊN TỤC (segment_id),
nên không bao giờ có return/RV21 vắt qua ngày thiếu.

Chia mẫu theo thời gian, không shuffle (config split.mode):
    fixed_dates (mặc định): train <= train_end < validation <= validation_end < test.
        Mốc được chốt bằng 60/20/20 tại ngày anchored_on; dữ liệu cập nhật hằng ngày chỉ
        nối vào cuối tập test, nên train/validation/scaler/mô hình không đổi.
    ratio: n_tr = floor(0.6 N), n_val = floor(0.2 N), n_te = N - n_tr - n_val (tính lại mỗi lần).
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from src.utils.config import get_logger

log = get_logger("features")


def compute_asset_features(prices: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """prices: các dòng của MỘT tài sản từ prices_daily (đã sạch, có segment_id)."""
    fc = cfg["features"]
    df = prices.sort_values("date_utc").reset_index(drop=True).copy()
    g = df.groupby("segment_id", sort=False)

    prev_close = g["close"].shift(1)                     # chỉ lấy close ngày trước TRONG đoạn
    df["log_return"] = np.log(df["close"] / prev_close)
    df["simple_return"] = df["close"] / prev_close - 1.0

    w = int(fc["rv_window"])
    df["rv21"] = (df.groupby("segment_id", sort=False)["log_return"]
                    .transform(lambda s: s.rolling(w, min_periods=w).std(ddof=fc["rv_ddof"])))
    df["log_dollar_volume"] = np.log1p(df["quote_volume"])

    # ---- Tùy chọn / robustness (không vào HMM lõi) ----
    for k in fc.get("optional_rv_windows", []):
        df[f"rv{k}"] = (df.groupby("segment_id", sort=False)["log_return"]
                          .transform(lambda s, k=k: s.rolling(k, min_periods=k).std(ddof=fc["rv_ddof"])))
    df["rv21_annualized"] = df["rv21"] * math.sqrt(fc["annualization_days"])
    if fc.get("amihud", {}).get("enabled"):
        q = df["quote_volume"].where(df["quote_volume"] > 0)   # Q = 0 -> để trống, không cộng epsilon
        df["amihud_illiq"] = df["simple_return"].abs() / q
    return df


def add_btc_eth_corr(feats: dict[str, pd.DataFrame], cfg: dict) -> None:
    """Tương quan trượt 21 ngày giữa log return BTC và ETH (chỉ để mô tả).

    Hai chuỗi được đặt lên lịch UTC đầy đủ; cửa sổ nào có ngày thiếu ở bất kỳ chuỗi nào
    sẽ cho NaN (min_periods = window) -> chỉ tính khi đủ 21 ngày trùng nhau hợp lệ.
    """
    cc = cfg["features"].get("btc_eth_corr", {})
    if not cc.get("enabled") or not {"BTC", "ETH"} <= set(feats):
        return
    w = int(cc["window"])
    a = feats["BTC"].set_index("date_utc")["log_return"]
    b = feats["ETH"].set_index("date_utc")["log_return"]
    cal = pd.date_range(min(a.index.min(), b.index.min()), max(a.index.max(), b.index.max()),
                        freq="D").strftime("%Y-%m-%d")
    a, b = a.reindex(cal), b.reindex(cal)
    corr = a.rolling(w, min_periods=w).corr(b)
    for k in ("BTC", "ETH"):
        feats[k]["btc_eth_corr21"] = feats[k]["date_utc"].map(corr)


def select_and_split(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Bỏ warm-up, chọn đoạn theo gap_handling_method, rồi chia 60/20/20 theo thời gian."""
    core = cfg["features"]["core"]
    method = cfg["cleaning"]["gap_handling_method"]
    eligible = df[core].notna().all(axis=1)

    seg_info = (df.assign(eligible=eligible).groupby("segment_id")
                  .agg(start=("date_utc", "min"), end=("date_utc", "max"),
                       n_days=("date_utc", "size"), n_eligible=("eligible", "sum"))
                  .reset_index())
    if method == "longest_segment":
        keep = [int(seg_info.sort_values(["n_eligible", "end"]).iloc[-1]["segment_id"])]
    elif method == "multi_segment":
        keep = seg_info.loc[seg_info["n_eligible"] > 0, "segment_id"].astype(int).tolist()
    else:
        raise ValueError(f"gap_handling_method không hợp lệ: {method}")

    in_keep = df["segment_id"].isin(keep)
    n_warmup = int((in_keep & ~eligible).sum())          # đếm THẬT, không ghi cứng 21
    out = df[in_keep & eligible].sort_values("date_utc").reset_index(drop=True).copy()

    N = len(out)
    sp = cfg["split"]
    mode = sp.get("mode", "ratio")
    if mode == "ratio":
        n_tr = math.floor(sp["train"] * N)
        n_val = math.floor(sp["validation"] * N)
        out["split"] = ["train"] * n_tr + ["validation"] * n_val + ["test"] * (N - n_tr - n_val)
        rule = "n_tr=floor(0.6N), n_val=floor(0.2N), n_te=N-n_tr-n_val; theo thời gian, không shuffle"
    elif mode == "fixed_dates":
        # Mốc cố định: dữ liệu mới chỉ nối vào CUỐI tập test; train/validation không đổi
        tr_end, val_end = sp["train_end"], sp["validation_end"]
        out["split"] = np.select([out["date_utc"] <= tr_end, out["date_utc"] <= val_end],
                                 ["train", "validation"], default="test")
        rule = (f"mốc cố định: train <= {tr_end} < validation <= {val_end} < test (đến ngày mới nhất); "
                f"mốc được chốt bằng 60/20/20 trên dữ liệu đến {sp.get('anchored_on', '?')}; không shuffle")
    else:
        raise ValueError(f"split.mode không hợp lệ: {mode}")

    counts = out["split"].value_counts()
    if any(counts.get(k, 0) <= 0 for k in ("train", "validation", "test")):
        raise ValueError(f"Không đủ dữ liệu cho cả 3 tập (N={N}, {counts.to_dict()}). "
                         "Báo thiếu dữ liệu, không tự nội suy.")

    def rng(name):
        part = out.loc[out["split"] == name, "date_utc"]
        return {"start": part.min(), "end": part.max(), "n": len(part), "share": round(len(part) / N, 4)}

    latest_clean = df["date_utc"].max()
    info = {
        "n_rows_clean": len(df),
        "n_rows_dropped_other_segments": int((~in_keep).sum()),
        "n_warmup_dropped": n_warmup,
        "n_eligible": N,
        "segments": json.loads(seg_info.assign(selected=seg_info["segment_id"].isin(keep))
                                       .to_json(orient="records")),
        "selected_segments": keep,
        "split_mode": mode,
        "train": rng("train"), "validation": rng("validation"), "test": rng("test"),
        "split_rule": rule,
        # Ngày nến đã đóng gần nhất có trong mẫu — C3 hiển thị "dữ liệu đến hết ngày ..."
        "data_through_utc": out["date_utc"].max(),
        "latest_clean_date_utc": latest_clean,
        "latest_date_in_sample": bool(out["date_utc"].max() == latest_clean),
    }
    if not info["latest_date_in_sample"]:
        log.warning("Ngày mới nhất %s KHÔNG nằm trong đoạn được chọn (có gap mới sau mẫu) — "
                    "xem gap_report.csv", latest_clean)
    # Kiểm tra thứ tự bắt buộc
    assert info["train"]["end"] < info["validation"]["start"] < info["validation"]["end"] < info["test"]["start"]
    assert sum(counts) == N
    return out, info


# ======================================================================
# Hàm giao tiếp theo tài liệu chung §3.2 (C2/C3 gọi trực tiếp nếu cần):
#   build_features(prices) -> DataFrame
#   chronological_split(df) -> dict(train, val, test)
#   fit_scaler(train) -> scaler
# Các hàm này dùng ĐÚNG logic của pipeline (compute_asset_features, select_and_split, scaler
# ddof=0 chỉ trên train) — test_pipeline.py kiểm tra kết quả trùng với file đã lưu.
# ======================================================================
def build_features(prices: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """prices: bảng giá sạch (prices_daily, một hoặc nhiều tài sản).
    Trả về feature đủ điều kiện (đã bỏ warm-up, đã chọn đoạn) kèm cột split, nối các tài sản."""
    from src.utils.config import load_config

    cfg = cfg or load_config()
    full = {a: compute_asset_features(p, cfg) for a, p in prices.groupby("asset", sort=True)}
    add_btc_eth_corr(full, cfg)
    return pd.concat([select_and_split(df, cfg)[0] for df in full.values()], ignore_index=True)


def chronological_split(df: pd.DataFrame, cfg: dict | None = None) -> dict[str, pd.DataFrame]:
    """Tách theo thời gian thành {'train', 'val', 'test'} dựa trên cột split (đã tính theo config)."""
    if "split" not in df.columns:
        from src.utils.config import load_config
        df, _ = select_and_split(df, cfg or load_config())
    out = {"train": df[df["split"] == "train"], "val": df[df["split"] == "validation"],
           "test": df[df["split"] == "test"]}
    for k in ("train", "val"):
        nxt = "val" if k == "train" else "test"
        assert out[k]["date_utc"].max() < out[nxt]["date_utc"].min(), "các tập chồng lấn thời gian"
    return {k: v.reset_index(drop=True) for k, v in out.items()}


def fit_scaler(train: pd.DataFrame, cfg: dict | None = None):
    """StandardScaler (ddof=0) fit CHỈ trên train, theo thứ tự feature lõi trong config."""
    from sklearn.preprocessing import StandardScaler

    from src.utils.config import load_config

    core = (cfg or load_config())["features"]["core"]
    return StandardScaler().fit(train[core].to_numpy())