"""C1-11 — Chuẩn hóa z-score, CHỈ fit trên train.

    mu_j  = trung bình biến j trên train
    s_j   = độ lệch chuẩn TỔNG THỂ (ddof=0) trên train   -> giống sklearn StandardScaler
    z_t,j = (x_t,j - mu_j) / s_j   cho mọi t thuộc train ∪ validation ∪ test
Nếu s_j = 0: dùng scale = 1 và cảnh báo (sklearn cũng làm như vậy).
Tuyệt đối không fit lại trên validation/test.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.utils.config import get_logger

log = get_logger("scaling")


def fit_transform(feat: pd.DataFrame, core: list[str], scaler_path) -> tuple[pd.DataFrame, dict]:
    train = feat.loc[feat["split"] == "train", core]
    scaler = StandardScaler(with_mean=True, with_std=True).fit(train.to_numpy())

    # Kiểm chứng: tham số scaler khớp công thức ddof=0
    assert np.allclose(scaler.mean_, train.mean().to_numpy())
    std0 = train.std(ddof=0).to_numpy()
    zero = std0 == 0
    if zero.any():
        log.warning("Biến không đổi trên train (std=0): %s -> scale=1",
                    [c for c, z in zip(core, zero) if z])
    assert np.allclose(scaler.scale_, np.where(zero, 1.0, std0))

    z = scaler.transform(feat[core].to_numpy())
    scaled = feat[["asset", "date_utc", "segment_id", "split"]].copy()
    for j, c in enumerate(core):
        scaled[f"z_{c}"] = z[:, j]

    joblib.dump({"scaler": scaler, "features": core, "fit_on": "train",
                 "std_ddof": 0, "train_start": feat.loc[feat["split"] == "train", "date_utc"].min(),
                 "train_end": feat.loc[feat["split"] == "train", "date_utc"].max()}, scaler_path)
    params = {"features": core, "mean": dict(zip(core, scaler.mean_.tolist())),
              "scale": dict(zip(core, scaler.scale_.tolist())), "std_ddof": 0,
              "zero_std_features": [c for c, zz in zip(core, zero) if zz],
              "fit_on": "train", "n_fit": len(train)}
    return scaled, params
