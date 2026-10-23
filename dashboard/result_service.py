"""C3-02 - Lớp đọc kết quả cho dashboard (result service).

Dashboard CHỈ đọc các file đã được pipeline xuất và version hóa (CSV/Parquet/JSON).
Không fit lại mô hình, không tự tính lại chỉ số hiệu suất quan trọng trong UI.
Thiếu file nào thì trả về None kèm thông báo; không sinh dữ liệu giả.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_config  # noqa: E402

CFG = load_config()
P = {k: ROOT / v for k, v in CFG["paths"].items()}

FILES = {
    "split_manifest": P["metadata"] / "split_manifest.json",
    "model_manifest": P["metadata"] / "model_manifest.json",
    "backtest_manifest": P["metadata"] / "backtest_manifest.json",
    "prices": P["processed"] / "prices_daily.parquet",
    "B03": P["tables"] / "B03_model_selection.csv",
    "B04": P["tables"] / "B04_regime_statistics.csv",
    "B05": P["tables"] / "B05_transition_matrix.csv",
    "B07": P["tables"] / "B07_hmm_vs_gmm.csv",
    "B08": P["tables"] / "B08_portfolio.csv",
    "stability": P["tables"] / "C2_stability_summary.csv",
    "window_sensitivity": P["tables"] / "C2_window_sensitivity.csv",
    "nav": P["predictions"] / "backtest_nav.csv",
}


def asset_files(asset):
    a = asset.lower()
    return {
        "features": P["processed"] / f"features_{a}.parquet",
        "filtered": P["predictions"] / f"regime_filtered_{a}.csv",
        "smoothed": P["predictions"] / f"regime_smoothed_{a}.csv",
        "gmm": P["predictions"] / f"regime_gmm_{a}.csv",
    }


REQUIRED = {
    "Dữ liệu C1": ["split_manifest", "prices"],
    "Mô hình C2": ["model_manifest", "B03", "B04", "B05", "B07"],
    "Backtest C3": ["backtest_manifest", "B08", "nav"],
}
FIX_HINT = {
    "Dữ liệu C1": "python run_pipeline.py --steps clean features reports",
    "Mô hình C2": "python scripts/complete_c2.py   (hoặc python run_pipeline.py --steps models)",
    "Backtest C3": "python run_pipeline.py --steps backtest",
}


def assets():
    return list(CFG["assets"])


def file_status():
    """Trạng thái từng nhóm đầu ra: danh sách file thiếu và lệnh khắc phục."""
    out = {}
    for group, keys in REQUIRED.items():
        missing = [FILES[k].relative_to(ROOT).as_posix() for k in keys if not FILES[k].exists()]
        if group == "Mô hình C2":
            for asset in assets():
                for kind in ("filtered", "smoothed"):
                    p = asset_files(asset)[kind]
                    if not p.exists():
                        missing.append(p.relative_to(ROOT).as_posix())
        out[group] = {"missing": missing, "fix": FIX_HINT[group]}
    return out


def mtime_key(path):
    """Khóa cache: đổi file (chạy lại pipeline) thì dashboard đọc lại."""
    path = Path(path)
    return path.stat().st_mtime_ns if path.exists() else None


def read_json(key):
    p = FILES[key]
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def read_table(key):
    p = FILES[key]
    return pd.read_csv(p) if p.exists() else None


def prices(asset):
    p = FILES["prices"]
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df = df.loc[df["asset"] == asset].copy()
    df["date"] = pd.to_datetime(df["date_utc"]).dt.tz_localize(None)
    return df.sort_values("date").reset_index(drop=True)


def features(asset):
    p = asset_files(asset)["features"]
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date_utc"]).dt.tz_localize(None)
    return df


def filtered(asset):
    """Xác suất FILTERED qua reader kiểm chứng của C2 (hash, schema, tổng xác suất, thời điểm).

    Trả về (frame, metadata, cảnh_báo, lỗi). Mô hình unstable được đọc ở chế độ nghiên cứu
    có xác nhận rõ, và cảnh báo được hiển thị trên giao diện.
    """
    if not asset_files(asset)["filtered"].exists() or not FILES["model_manifest"].exists():
        return None, None, [], "Chưa có xác suất filtered của C2"
    from src.models.handoff import UnstableModelWarning, load_filtered

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", UnstableModelWarning)
            frame, meta = load_filtered(CFG, asset, acknowledge_unstable=True)
    except (ValueError, KeyError) as exc:
        return None, None, [], f"Không đọc được xác suất filtered: {exc}"
    frame["date"] = pd.to_datetime(frame["date_utc"])
    notes = [str(w.message) for w in caught if issubclass(w.category, UnstableModelWarning)]
    return frame, meta, notes, None


def smoothed(asset):
    p = asset_files(asset)["smoothed"]
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if not df["hindsight"].astype(bool).all() or not df["probability_type"].eq("smoothed").all():
        raise ValueError("File smoothed không đúng hợp đồng (hindsight=True, probability_type=smoothed)")
    df["date"] = pd.to_datetime(df["date_utc"])
    return df


def gmm(asset):
    """Nhãn GMM đối chứng (posterior từng ngày, không có cấu trúc Markov). None nếu chưa có file."""
    p = asset_files(asset)["gmm"]
    if not p.exists():
        return None, None
    df = pd.read_csv(p)
    df["date"] = pd.to_datetime(df["date_utc"])
    label_map = json.loads(df["label_map"].iloc[0]) if "label_map" in df and len(df) else None
    return df, label_map
