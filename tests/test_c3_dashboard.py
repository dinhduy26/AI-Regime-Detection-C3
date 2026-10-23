"""Nghiệm thu C3 - dashboard 4 tab chạy trên kết quả thật (Excel C3-08..C3-13; Methodology 3.13 mục 10-12).

Dùng streamlit.testing (chạy app không cần trình duyệt). Skip nếu chưa cài streamlit hoặc
chưa có đầu ra C2/C3; không dùng dữ liệu mock.
"""

import pandas as pd
import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")
from streamlit.testing.v1 import AppTest  # noqa: E402

from src.utils.config import ROOT, path_of, read_json  # noqa: E402

APP = str(ROOT / "dashboard" / "app.py")


@pytest.fixture(scope="module")
def ready(cfg):
    need = [path_of(cfg, "metadata") / "model_manifest.json", path_of(cfg, "metadata") / "backtest_manifest.json",
            path_of(cfg, "tables") / "B08_portfolio.csv", path_of(cfg, "predictions") / "backtest_nav.csv"]
    for p in need:
        if not p.exists():
            pytest.skip(f"Chưa có {p.name}: chạy `python run_pipeline.py --steps models backtest`")
    return cfg


def _run(asset=None):
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    if asset:
        [r for r in at.radio if r.label == "Tài sản"][0].set_value(asset).run()
    return at


@pytest.mark.parametrize("asset", ["BTC", "ETH"])
def test_app_renders_four_tabs_without_errors(ready, asset):
    at = _run(asset)
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    assert [t.label for t in at.tabs] == ["1 · Overview", "2 · Regime History", "3 · Model Comparison",
                                          "4 · Portfolio & Risk"]


@pytest.mark.parametrize("asset", ["BTC", "ETH"])
def test_ui_kpis_match_exported_b08(ready, asset):
    at = _run(asset)
    b08 = pd.read_csv(path_of(ready, "tables") / "B08_portfolio.csv")
    row = b08[(b08["asset"] == asset) & (b08["strategy"] == "regime_aware")
              & (b08["scenario"] == "baseline_close_to_close")].iloc[0]
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["NAV cuối (regime-aware)"] == f"{row['final_nav']:.4f}"
    assert metrics["Max drawdown"] == f"{row['max_drawdown'] * 100:.2f}%"
    assert metrics["Sharpe (sau phí)"] == f"{row['Sharpe']:.3f}"


def test_unstable_model_is_flagged_and_states_follow_k(ready):
    c2 = read_json(path_of(ready, "metadata") / "model_manifest.json")
    for asset in ready["assets"]:
        at = _run(asset)
        meta = c2["assets"][asset]
        texts = " ".join(w.value for w in at.warning)
        if meta["seed_stability_status"] != "stable":
            assert "unstable" in texts
        # Tab Overview chỉ hiện đúng K state của mô hình (không ép 4 nhãn).
        captions = " ".join(c.value for c in at.caption)
        assert f"K = {meta['K']}" in captions


def test_smoothed_view_is_labelled_hindsight(ready):
    at = _run()
    mode = [r for r in at.radio if r.label.startswith("Loại xác suất")][0]
    mode.set_value(mode.options[1]).run()
    assert not at.exception
    assert any("hindsight" in w.value for w in at.warning)


def test_scenarios_switch_and_show_simulated_label(ready):
    at = _run()
    box = at.selectbox[0]
    for option in box.options:
        box.set_value(option).run()
        assert not at.exception
        assert any("simulated" in w.value for w in at.warning)
