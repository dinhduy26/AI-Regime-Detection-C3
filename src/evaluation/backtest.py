"""C3 - Lõi tính toán backtest regime-aware (hàm thuần, không đọc/ghi file).

Quy ước (Research & Technical Methodology, phần C3, mục 3.3-3.7):

* Tín hiệu cuối ngày t là xác suất FILTERED p_risk,t = P(S_t = k* | X_1:t).
* Tỷ trọng w_t = 1 nếu p_risk,t < 0,5, ngược lại 0,25 (long/cash, không đòn bẩy).
* w_t áp dụng cho lợi suất kỳ kế tiếp R_{t+1}; phần tiền mặt có lợi suất 0.
* Gross_{t+1} = w_t R_{t+1} + (1 - w_t) R^cash;  Cost_{t+1} = f * TO_{t+1};
  Net_{t+1} = Gross_{t+1} - Cost_{t+1};  NAV_{t+1} = NAV_t (1 + Net_{t+1}),  NAV_0 = 1.
* Turnover:
    - target_change_approx (baseline): TO_{t+1} = |w_t - w_{t-1}|.
    - drift_adjusted (kiểm tra chéo): TO_{t+1} = |w_t - w_{t-}|, với w_{t-} là tỷ trọng thực tế
      trước khi tái cân bằng sau khi giá biến động:
      w_{t-} = w_{t-1}(1 + R_t) / (w_{t-1}(1 + R_t) + (1 - w_{t-1})(1 + R^cash)).
* Bắt đầu từ tiền mặt (w_{-1} = 0), nên lệnh mua đầu tiên chịu phí cho mọi chiến lược.

Mọi lợi suất là số thập phân trên ngày, không phải phần trăm.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

TURNOVER_METHODS = ("target_change_approx", "drift_adjusted")


def regime_weights(p_risk, threshold=0.5, weight_risk_on=1.0, weight_risk_off=0.25):
    """Quy tắc phân bổ chỉ định trước: w = weight_risk_off khi p_risk >= threshold."""
    p = np.asarray(p_risk, dtype=float)
    if p.size == 0 or not np.isfinite(p).all() or (p < 0).any() or (p > 1 + 1e-9).any():
        raise ValueError("p_risk phải là xác suất hữu hạn trong [0, 1]")
    for w in (weight_risk_on, weight_risk_off):
        if not 0 <= w <= 1:
            raise ValueError("Tỷ trọng phải nằm trong [0, 1] (long/cash, không đòn bẩy)")
    return np.where(p >= threshold, float(weight_risk_off), float(weight_risk_on))


def drift_weight(weight, asset_return, cash_return=0.0):
    """Tỷ trọng tài sản sau một kỳ biến động giá, khi chưa tái cân bằng."""
    risky = weight * (1.0 + asset_return)
    total = risky + (1.0 - weight) * (1.0 + cash_return)
    if total <= 0:
        raise ValueError("Giá trị danh mục không dương; không xác định được tỷ trọng drift")
    return risky / total


def simulate(
    weights,
    asset_returns,
    fee_rate=0.001,
    turnover_method="target_change_approx",
    initial_weight=0.0,
    cash_return=0.0,
):
    """Mô phỏng NAV. weights[i] là w_t quyết định cuối ngày t, asset_returns[i] là R_{t+1}.

    Trả về DataFrame cùng độ dài với các cột: pre_trade_weight, weight, asset_return,
    gross, turnover, fee, net, nav, drawdown.
    """
    if turnover_method not in TURNOVER_METHODS:
        raise ValueError(f"turnover_method phải là một trong {TURNOVER_METHODS}")
    w = np.asarray(weights, dtype=float)
    r = np.asarray(asset_returns, dtype=float)
    if w.shape != r.shape or w.ndim != 1 or w.size == 0:
        raise ValueError("weights và asset_returns phải là mảng 1 chiều cùng độ dài, không rỗng")
    if not (np.isfinite(w).all() and np.isfinite(r).all()):
        raise ValueError("weights/returns chứa giá trị không hữu hạn")
    if (w < 0).any() or (w > 1).any():
        raise ValueError("Tỷ trọng phải nằm trong [0, 1]")
    if fee_rate < 0:
        raise ValueError("fee_rate không được âm")

    n = w.size
    pre = np.empty(n)
    turnover = np.empty(n)
    gross = np.empty(n)
    fee = np.empty(n)
    net = np.empty(n)
    nav = np.empty(n)
    nav_prev = 1.0
    held = float(initial_weight)  # tỷ trọng ngay trước giao dịch của kỳ hiện tại
    previous_target = float(initial_weight)
    for i in range(n):
        pre[i] = previous_target if turnover_method == "target_change_approx" else held
        turnover[i] = abs(w[i] - pre[i])
        fee[i] = fee_rate * turnover[i]
        gross[i] = w[i] * r[i] + (1.0 - w[i]) * cash_return
        net[i] = gross[i] - fee[i]
        nav[i] = nav_prev * (1.0 + net[i])
        nav_prev = nav[i]
        previous_target = w[i]
        held = drift_weight(w[i], r[i], cash_return)
    peak = np.maximum.accumulate(np.concatenate([[1.0], nav]))[1:]
    return pd.DataFrame(
        {
            "pre_trade_weight": pre,
            "weight": w,
            "asset_return": r,
            "gross": gross,
            "turnover": turnover,
            "fee": fee,
            "net": net,
            "nav": nav,
            "drawdown": nav / peak - 1.0,
        }
    )


def performance_metrics(net, nav, turnover, fee, calendar_days, annualization_days=365, risk_free_daily=0.0):
    """Chỉ số bắt buộc (mục 3.7): CR, CAGR, volatility năm hóa, Sharpe (sau phí), MDD, turnover, phí."""
    g = np.asarray(net, dtype=float)
    v = np.asarray(nav, dtype=float)
    if g.size < 2 or g.size != v.size:
        raise ValueError("Cần ít nhất 2 lợi suất ngày và NAV cùng độ dài")
    if calendar_days <= 0:
        raise ValueError("Số ngày lịch phải dương")
    final = float(v[-1])
    cagr = final ** (annualization_days / calendar_days) - 1.0 if final > 0 else float("nan")
    sd = float(np.std(g, ddof=1))
    ann_vol = sd * math.sqrt(annualization_days)
    sharpe = (
        (float(np.mean(g)) - risk_free_daily) / sd * math.sqrt(annualization_days) if sd > 0 else float("nan")
    )
    path = np.concatenate([[1.0], v])
    mdd = float(np.min(path / np.maximum.accumulate(path) - 1.0))
    return {
        "final_nav": final,
        "cumulative_return": final - 1.0,
        "CAGR": cagr,
        "ann_volatility": ann_vol,
        "Sharpe": sharpe,
        "max_drawdown": mdd,
        "turnover": float(np.sum(turnover)),
        "total_cost": float(np.sum(fee)),
    }
