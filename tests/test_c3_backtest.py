"""Nghiệm thu C3 - backtest regime-aware (Methodology phần C3, mục 3.13; Excel C3-03..C3-07).

Phần 1 kiểm công thức bằng ví dụ toán học của tài liệu (không phải dữ liệu thị trường).
Phần 2 kiểm đầu ra thật trong outputs/ và bị skip nếu chưa chạy C2 + C3
(`python run_pipeline.py --steps models backtest`); không tạo dữ liệu giả.
"""

import math

import numpy as np
import pandas as pd
import pytest

from src.evaluation.backtest import drift_weight, performance_metrics, regime_weights, simulate
from src.evaluation.backtest_run import load_backtest_config
from src.utils.config import path_of, read_json

# ----------------------------------------------------------------------------- công thức


def test_weight_rule_thresholds():
    # 3.13 (3): p_risk = 0.65 -> 0.25; 0.49 -> 1. Ngưỡng 0.5 thuộc nhánh phòng thủ (>= 0.5).
    assert regime_weights([0.65, 0.49, 0.5, 0.0, 1.0]).tolist() == [0.25, 1.0, 0.25, 1.0, 0.25]


def test_weight_rule_rejects_invalid_probability():
    with pytest.raises(ValueError):
        regime_weights([0.2, np.nan])
    with pytest.raises(ValueError):
        regime_weights([1.5])


def test_hand_example_section_3_8():
    # w_{t-1} = 1, p_risk = 0.65 -> w_t = 0.25; close 100 -> 110 -> R = 0.10.
    sim = simulate([0.25], [0.10], fee_rate=0.001, initial_weight=1.0)
    row = sim.iloc[0]
    assert row["gross"] == pytest.approx(0.025)
    assert row["turnover"] == pytest.approx(0.75)
    assert row["fee"] == pytest.approx(0.00075)
    assert row["net"] == pytest.approx(0.02425)
    assert row["nav"] == pytest.approx(1.02425)


def test_drift_example_section_3_6():
    # 25% BTC, 75% cash, BTC +10% -> tỷ trọng trước tái cân bằng ~0.26829; về 0.25 cần bán ~1.829% NAV.
    assert drift_weight(0.25, 0.10) == pytest.approx(0.26829, abs=1e-5)
    sim = simulate([0.25, 0.25], [0.10, 0.0], fee_rate=0.001, turnover_method="drift_adjusted", initial_weight=0.25)
    assert sim["turnover"].iloc[0] == pytest.approx(0.0)
    assert sim["turnover"].iloc[1] == pytest.approx(0.01829, abs=1e-5)
    approx = simulate([0.25, 0.25], [0.10, 0.0], fee_rate=0.001, initial_weight=0.25)
    assert approx["turnover"].tolist() == [0.0, 0.0]  # xấp xỉ theo mục tiêu bỏ qua drift


def test_zero_turnover_means_zero_fee_and_fee_never_adds_return():
    r = [0.01, -0.02, 0.03]
    sim = simulate([1.0, 1.0, 1.0], r, fee_rate=0.001, initial_weight=1.0)
    assert sim["turnover"].eq(0).all() and sim["fee"].eq(0).all()
    costly = simulate([0.25, 1.0, 0.25], r, fee_rate=0.001)
    free = simulate([0.25, 1.0, 0.25], r, fee_rate=0.0)
    assert (costly["fee"] >= 0).all()
    assert (costly["net"] <= free["net"] + 1e-15).all()


def test_start_in_cash_pays_opening_fee():
    sim = simulate([1.0, 1.0], [0.0, 0.0], fee_rate=0.001, initial_weight=0.0)
    assert sim["turnover"].tolist() == [1.0, 0.0]
    assert sim["nav"].iloc[-1] == pytest.approx(0.999)


def test_signal_applies_to_next_period_only():
    # Trọng số ở vị trí i nhân với lợi suất ở vị trí i (đã là R_{t+1}); đổi lợi suất tương lai
    # không làm đổi NAV của các kỳ trước (không nhìn tương lai).
    w = [1.0, 0.25, 1.0]
    a = simulate(w, [0.05, 0.10, -0.20], fee_rate=0.0, initial_weight=1.0)
    b = simulate(w, [0.05, 0.10, 0.50], fee_rate=0.0, initial_weight=1.0)
    assert a["nav"].iloc[:2].tolist() == b["nav"].iloc[:2].tolist()
    assert a["gross"].iloc[1] == pytest.approx(0.25 * 0.10)


def test_nav_recurrence_and_drawdown():
    sim = simulate([1.0, 1.0, 1.0, 1.0], [0.1, -0.2, 0.05, 0.3], fee_rate=0.001)
    nav = np.cumprod(1 + sim["net"].to_numpy())
    assert np.allclose(sim["nav"], nav)
    assert (sim["drawdown"] <= 1e-15).all()


def test_metrics_formulas():
    net = np.array([0.01, -0.02, 0.015, 0.0])
    nav = np.cumprod(1 + net)
    m = performance_metrics(net, nav, [1, 0, 0, 0], [0.001, 0, 0, 0], calendar_days=4)
    sd = np.std(net, ddof=1)
    assert m["CAGR"] == pytest.approx(nav[-1] ** (365 / 4) - 1)
    assert m["ann_volatility"] == pytest.approx(sd * math.sqrt(365))
    assert m["Sharpe"] == pytest.approx(net.mean() / sd * math.sqrt(365))
    path = np.r_[1.0, nav]
    assert m["max_drawdown"] == pytest.approx((path / np.maximum.accumulate(path) - 1).min())
    assert m["max_drawdown"] <= 0
    flat = performance_metrics([0.0, 0.0], [1.0, 1.0], [0, 0], [0, 0], calendar_days=2)
    assert math.isnan(flat["Sharpe"])  # s_g = 0: Sharpe không xác định, không ghi 0


def test_backtest_config_is_prespecified():
    bt = load_backtest_config()
    assert bt["period"] == "test"
    assert bt["allocation"] == {
        "risk_threshold": 0.5,
        "weight_risk_on": 1.0,
        "weight_risk_off": 0.25,
        "cash_return_daily": 0.0,
        "initial_weight": 0.0,
    }
    assert bt["costs"]["fee_rate"] == 0.001
    assert bt["metrics"] == {"annualization_days": 365, "risk_free_daily": 0.0}
    assert bt["signal"]["column"] == "prob_high_risk"
    assert {s["turnover_method"] for s in bt["scenarios"].values()} == {"target_change_approx", "drift_adjusted"}
    assert {s["return_basis"] for s in bt["scenarios"].values()} == {"close_to_close", "next_open"}


# ----------------------------------------------------------------------------- đầu ra thật


@pytest.fixture(scope="module")
def outputs(cfg):
    b08 = path_of(cfg, "tables") / "B08_portfolio.csv"
    nav = path_of(cfg, "predictions") / "backtest_nav.csv"
    man = path_of(cfg, "metadata") / "backtest_manifest.json"
    for p in (b08, nav, man):
        if not p.exists():
            pytest.skip(f"Chưa có {p.name}: chạy `python run_pipeline.py --steps models backtest`")
    return pd.read_csv(b08), pd.read_csv(nav), read_json(man)


def test_manifest_complete_and_flags_unstable(outputs, cfg):
    _, _, man = outputs
    assert man["status"] == "complete"
    c2 = read_json(path_of(cfg, "metadata") / "model_manifest.json")
    for asset, info in man["assets"].items():
        assert info["model_run_id"] == c2["assets"][asset]["model_run_id"]
        assert info["seed_stability_status"] == c2["assets"][asset]["seed_stability_status"]
        if info["seed_stability_status"] != "stable":
            assert info["stability_warnings"], "Mô hình unstable phải được ghi nhận cảnh báo"


def test_b08_schema_and_scenarios(outputs, cfg):
    b08, _, _ = outputs
    required = {"asset", "strategy", "scenario", "turnover_method", "CAGR", "ann_volatility", "Sharpe",
                "max_drawdown", "turnover", "total_cost", "start", "end", "n_days"}
    assert required <= set(b08.columns)
    assert b08.groupby(["asset", "strategy", "scenario"]).size().eq(1).all()
    assert set(b08["asset"]) == set(cfg["assets"])
    assert set(b08["strategy"]) == {"buy_and_hold", "regime_aware"}
    assert (b08["max_drawdown"] <= 0).all() and (b08["total_cost"] >= 0).all()
    assert b08["period"].eq("test").all() and b08["sharpe_basis"].eq("after_fees").all()


def test_same_period_and_rules_for_both_strategies(outputs):
    b08, nav, _ = outputs
    keys = ["asset", "scenario"]
    for _, g in b08.groupby(keys):
        assert g["start"].nunique() == 1 and g["end"].nunique() == 1 and g["n_days"].nunique() == 1
        assert g["fee_rate"].nunique() == 1 and g["turnover_method"].nunique() == 1
    for _, g in nav.groupby(keys):
        dates = [tuple(s["date_utc"]) for _, s in g.groupby("strategy")]
        assert dates[0] == dates[1]


def test_nav_rows_use_filtered_signal_before_return(outputs, cfg):
    _, nav, _ = outputs
    signal = pd.to_datetime(nav["signal_date_utc"], utc=True)
    ret = pd.to_datetime(nav["date_utc"], utc=True)
    asof = pd.to_datetime(nav["filtered_asof_utc"], utc=True)
    assert (signal + pd.Timedelta(days=1) == ret).all()
    assert (asof <= ret).all()
    # Trọng số regime-aware đúng bằng quy tắc áp lên xác suất FILTERED của file C2.
    for asset in cfg["assets"]:
        filtered = pd.read_csv(path_of(cfg, "predictions") / f"regime_filtered_{asset.lower()}.csv")
        prob = filtered.set_index("date_utc")["prob_high_risk"]
        sub = nav[(nav["asset"] == asset) & (nav["strategy"] == "regime_aware")]
        assert np.allclose(sub["prob_high_risk"], prob.loc[sub["signal_date_utc"]].to_numpy())
        assert np.array_equal(sub["weight"], np.where(sub["prob_high_risk"] >= 0.5, 0.25, 1.0))


def test_nav_reproduces_metrics_and_returns(outputs, cfg):
    b08, nav, _ = outputs
    for (asset, strategy, scenario), g in nav.groupby(["asset", "strategy", "scenario"]):
        assert np.allclose(g["nav"], np.cumprod(1 + g["net"]))
        row = b08[(b08["asset"] == asset) & (b08["strategy"] == strategy) & (b08["scenario"] == scenario)]
        assert row["final_nav"].iloc[0] == pytest.approx(g["nav"].iloc[-1])
        assert row["total_cost"].iloc[0] == pytest.approx(g["fee"].sum())
    # Lợi suất close-to-close khớp simple_return của C1 tại ngày t+1.
    for asset in cfg["assets"]:
        f = pd.read_parquet(path_of(cfg, "processed") / f"features_{asset.lower()}.parquet")
        sr = f.assign(d=pd.to_datetime(f["date_utc"]).dt.strftime("%Y-%m-%d")).set_index("d")["simple_return"]
        sub = nav[(nav["asset"] == asset) & (nav["scenario"] == "baseline_close_to_close")]
        assert np.allclose(sub["asset_return"], sr.loc[sub["date_utc"]].to_numpy())


def test_buy_and_hold_matches_closed_form(outputs, cfg):
    b08, nav, _ = outputs
    bt = load_backtest_config()
    fee = bt["costs"]["fee_rate"]
    for asset in cfg["assets"]:
        g = nav[(nav["asset"] == asset) & (nav["strategy"] == "buy_and_hold")
                & (nav["scenario"] == "baseline_close_to_close")]
        r = g["asset_return"].to_numpy()
        expected = (1 + r[0] - fee) * np.prod(1 + r[1:])  # chỉ phí mở vị thế ngày đầu
        assert g["nav"].iloc[-1] == pytest.approx(expected)
