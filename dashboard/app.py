"""Crypto Market Regime Radar - dashboard Streamlit 4 trang (C3-01, C3-08..C3-11, C3-13).

Chạy từ thư mục gốc dự án:
    streamlit run dashboard/app.py

Dashboard chỉ đọc kết quả đã xuất (outputs/, data/processed/); không huấn luyện lại mô hình
theo thao tác trên giao diện. Các thống kê mô tả thêm (lợi suất sau khi vào regime, lợi suất
theo tháng, độ biến động trượt) được tính trực tiếp từ các file đó. Đây là công cụ mô phỏng
nghiên cứu, không phải khuyến nghị đầu tư.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent))
import result_service as rs  # noqa: E402
import ui  # noqa: E402
from ui import num, pct, state_name  # noqa: E402

st.set_page_config(page_title="Crypto Market Regime Radar", page_icon="📡", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown(ui.CSS, unsafe_allow_html=True)

PAGES = ["Overview", "Regime History", "Model Comparison", "Portfolio & Risk"]
PAGE_SUB = {
    "Overview": "Tổng quan tại ngày dữ liệu cuối",
    "Regime History": "Lịch sử và đặc điểm các regime",
    "Model Comparison": "Chọn mô hình, HMM so với GMM",
    "Portfolio & Risk": "Mô phỏng danh mục trên tập test",
}
STRATEGY_NAMES = {"buy_and_hold": "Buy-and-Hold", "regime_aware": "Regime-aware"}
SCENARIO_NAMES = {
    "baseline_close_to_close": "Baseline: close t -> close t+1 (kết quả chính)",
    "crosscheck_drift_adjusted": "Kiểm tra chéo: turnover drift-adjusted",
    "crosscheck_next_open": "Kiểm tra chéo: khớp open t+1 -> open t+2",
}
SCENARIO_SHORT = {
    "baseline_close_to_close": "Baseline",
    "crosscheck_drift_adjusted": "Drift-adjusted",
    "crosscheck_next_open": "Next-open",
}
PERIOD_NAMES = {"train": "Train", "validation": "Validation", "test": "Test"}
HORIZONS = (5, 10, 20)
NOBAR = {"displayModeBar": False}
H_BIG, H_MID, H_STRIP = 400, 320, 190  # ba bậc chiều cao biểu đồ dùng thống nhất


# ----------------------------------------------------------------------------- đọc dữ liệu (cache theo mtime)


@st.cache_data(show_spinner=False)
def _json(key, _mtime):
    return rs.read_json(key)


@st.cache_data(show_spinner=False)
def _table(key, _mtime):
    return rs.read_table(key)


@st.cache_data(show_spinner=False)
def _prices(asset, _mtime):
    return rs.prices(asset)


@st.cache_data(show_spinner=False)
def _features(asset, _mtime):
    return rs.features(asset)


@st.cache_data(show_spinner=False)
def _filtered(asset, _mtime):
    return rs.filtered(asset)


@st.cache_data(show_spinner=False)
def _smoothed(asset, _mtime):
    return rs.smoothed(asset)


@st.cache_data(show_spinner=False)
def _gmm(asset, _mtime):
    return rs.gmm(asset)


def jload(key):
    return _json(key, rs.mtime_key(rs.FILES[key]))


def tload(key):
    return _table(key, rs.mtime_key(rs.FILES[key]))


def filtered_of(a):
    return _filtered(a, rs.mtime_key(rs.asset_files(a)["filtered"]))


# ----------------------------------------------------------------------------- tiện ích hiển thị


def md(text):
    st.markdown(text, unsafe_allow_html=True)


def sec(title, sub="", info=""):
    md(ui.sec_html(title, sub, info))


def card(name):
    return st.container(border=True, key=f"card_{name}")


def show(fig):
    st.plotly_chart(fig, width="stretch", config=NOBAR)


def kpis(items):
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        col.markdown(ui.kpi_html(*item), unsafe_allow_html=True)


def dmy(x):
    return pd.Timestamp(x).strftime("%d/%m/%Y")


def colored(text, x, higher_is_better=True):
    return f"<span style='color:{ui.sign_color(x, higher_is_better)}'>{text}</span>"


def runs(dates, values):
    """Các đoạn ngày liên tiếp cùng giá trị: (ngày đầu, ngày sau ngày cuối, giá trị)."""
    out, start = [], 0
    s = np.asarray(values)
    for i in range(1, len(s) + 1):
        if i == len(s) or s[i] != s[start]:
            out.append((dates.iloc[start], dates.iloc[i - 1] + pd.Timedelta(days=1), s[start]))
            start = i
    return out


def current_run(states):
    s = np.asarray(states)
    n = 1
    while n < len(s) and s[-n - 1] == s[-1]:
        n += 1
    return n


def split_lines(fig, split, labels=True):
    for key, name in (("train", "hết train"), ("validation", "hết validation")):
        end = split.get(key, {}).get("end") if split else None
        if end:
            fig.add_vline(x=pd.Timestamp(end) + pd.Timedelta(hours=12), line_dash="dot", line_color=ui.MUTED)
            if labels:
                fig.add_annotation(x=pd.Timestamp(end), y=0.99, yref="paper", text=name, showarrow=False,
                                   font={"size": 10, "color": ui.MUTED}, xanchor="right", yanchor="top",
                                   bgcolor="rgba(255,252,246,.8)")


def regime_strips(rows, height=H_STRIP):
    """Nhiều dải regime chồng nhau. rows: list (tên dòng, ngày, state_id, label_map)."""
    palette, z, hover, ylab = [], [], [], []
    dates = sorted(set().union(*[set(r[1]) for r in rows]))
    for name, d, states, lm in rows:
        colors = ui.state_colors(lm)
        m = dict(zip(d, states))
        zr, hr = [], []
        for day in dates:
            s_ = m.get(day)
            if s_ is None or (isinstance(s_, float) and math.isnan(s_)):
                zr.append(None)
                hr.append("")
                continue
            c = colors[int(s_)]
            if c not in palette:
                palette.append(c)
            zr.append(palette.index(c))
            hr.append(f"{name}: {state_name(int(s_), lm)}")
        z.append(zr)
        hover.append(hr)
        ylab.append(name)
    n = max(len(palette), 1)
    scale = []
    for i, c in enumerate(palette or [ui.MUTED]):
        scale += [[i / n, c], [(i + 1) / n, c]]
    fig = go.Figure(go.Heatmap(z=z, x=dates, y=ylab, zmin=-0.5, zmax=n - 0.5, colorscale=scale, showscale=False,
                               customdata=hover, ygap=6,
                               hovertemplate="%{x|%d/%m/%Y} · %{customdata}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", tickfont={"size": 13, "color": ui.INK})
    return ui.chart_layout(fig, height, legend=False)


def regime_legend(label_map, k):
    colors = ui.state_colors(label_map)
    return " &nbsp; ".join(f"<span class='muted'>{ui.chip(colors[i])}{ui.esc(state_name(i, label_map))}</span>"
                           for i in range(k))


def run_hint(cmd):
    st.info(f"Chưa có kết quả cần thiết. Chạy: `{cmd}`")


# ----------------------------------------------------------------------------- kiểm tra file

status = rs.file_status()
missing_groups = {g: s for g, s in status.items() if s["missing"]}

split_manifest = jload("split_manifest") or {}
model_manifest = jload("model_manifest")
bt_manifest = jload("backtest_manifest")
data_through = split_manifest.get("data_through_utc")

# ----------------------------------------------------------------------------- thanh bên

with st.sidebar:
    md("<div class='wordmark'>📡 REGIME <span>RADAR</span></div>")
    md("<div class='sidecap'>TÀI SẢN</div>")
    asset = st.radio("Tài sản", rs.assets(), horizontal=True, key="asset", label_visibility="collapsed")
    md("<div class='sidecap'>ĐIỀU HƯỚNG</div>")
    page = st.radio("Trang", PAGES, key="nav", label_visibility="collapsed",
                    format_func=lambda p: f"{PAGES.index(p) + 1}.  {p}")
    asset_through = data_through.get(asset) if isinstance(data_through, dict) else data_through
    if asset_through:
        md("<div class='sidecap'>DỮ LIỆU</div>"
           f"<div class='asof'>Đến hết ngày<br><b>{pd.Timestamp(asset_through):%d/%m/%Y}</b> (UTC)<br>"
           "Nến ngày Binance, cập nhật mỗi ngày</div>")
    a_meta = (model_manifest or {}).get("assets", {}).get(asset, {})
    if a_meta:
        st.caption(f"Gaussian HMM · K = {a_meta.get('K')} · {a_meta.get('seed_stability_status')}")

if missing_groups:
    with st.expander("⚠️ Thiếu một số file kết quả, xem cách khắc phục", expanded="Dữ liệu C1" in missing_groups):
        for group, s in missing_groups.items():
            st.markdown(f"**{group}** thiếu: " + ", ".join(f"`{m}`" for m in s["missing"]))
            st.code(s["fix"], language="bash")
if "Dữ liệu C1" in missing_groups:
    st.error("Chưa có dữ liệu C1 nên dashboard chưa thể hiển thị. Chạy lệnh ở trên rồi tải lại trang.")
    st.stop()

px_df = _prices(asset, rs.mtime_key(rs.FILES["prices"]))
feat = _features(asset, rs.mtime_key(rs.asset_files(asset)["features"]))
filt, meta, notes, filt_error = filtered_of(asset)
b04, b05, b08 = tload("B04"), tload("B05"), tload("B08")

# ----------------------------------------------------------------------------- tiêu đề trang

last_px = px_df.iloc[-1]
last_feat = feat.iloc[-1] if feat is not None else None
chg = last_feat["simple_return"] if last_feat is not None else None
unstable = meta is not None and meta.get("seed_stability_status") != "stable"
tags = []
if page == "Portfolio & Risk":
    tags.append("<span class='tag' title='Mô phỏng lịch sử với phí giả định; không phải giao dịch thật'>"
                "kết quả simulated</span>")
if unstable:
    tags.append("<span class='tag' title='Nhãn regime nhạy với seed khởi tạo; chỉ dùng cho nghiên cứu'>"
                "mô hình unstable</span>")
md(f"<div class='hero'><p class='t'>{asset} <span>/ USDT · {PAGE_SUB[page]}</span></p>"
   f"<div>{''.join(tags)}</div></div>")


def rule_params():
    thr, w_on, w_off = 0.5, 1.0, 0.25  # mặc định theo Methodology; ưu tiên giá trị đã ghi trong B08
    if b08 is not None and not b08.empty:
        r0 = b08.iloc[0]
        thr, w_on, w_off = float(r0["risk_threshold"]), float(r0["weight_risk_on"]), float(r0["weight_risk_off"])
    return thr, w_on, w_off


def with_states(frame):
    """Gắn lợi suất ngày (C1) vào state filtered cùng ngày."""
    f = feat[["date", "log_return", "simple_return", "rv21", "segment_id"]].copy()
    f["date"] = pd.to_datetime(f["date"]).dt.normalize()
    s = frame[["date", "state_id"]].copy()
    s["date"] = pd.to_datetime(s["date"]).dt.normalize()
    return f.merge(s, on="date", how="inner").sort_values("date").reset_index(drop=True)


def forward_returns(df):
    """Lợi suất tích lũy h ngày SAU ngày t (t+1..t+h), không vượt qua khoảng trống dữ liệu (segment)."""
    out = df.copy()
    cum = out.groupby("segment_id")["log_return"].cumsum()
    for h in HORIZONS:
        fwd = cum.groupby(out["segment_id"]).shift(-h) - cum
        out[f"fwd_{h}"] = np.exp(fwd) - 1
    out["entry"] = out["state_id"].ne(out["state_id"].shift(1)) & out.index.to_series().gt(0)
    return out


# ============================================================================= OVERVIEW


def page_overview():
    if filt is None:
        st.info(f"{filt_error}. Chạy: `python scripts/complete_c2.py`")
        return
    label_map, k, risk = meta["label_map"], meta["K"], meta["high_risk_state"]
    colors = ui.state_colors(label_map)
    last = filt.iloc[-1]
    thr, w_on, w_off = rule_params()
    p_now = float(last["prob_high_risk"])
    w_rule = w_off if p_now >= thr else w_on
    cur = int(last["state_id"])
    days = current_run(filt["state_id"])
    rv_now = float(last_feat["rv21"] * math.sqrt(365)) if last_feat is not None else None
    dur = None
    if b05 is not None:
        d = b05[(b05["asset"] == asset) & (b05["from_state"] == cur) & (b05["to_state"] == cur)]
        dur = float(d["expected_duration"].iloc[0]) if not d.empty else None

    kpis([
        ("Giá đóng cửa", f"{num(last_px['close'], 0)}",
         f"{colored(pct(chg, 2, sign=True), chg)} · {last_px['date']:%d/%m}",
         ui.INK, "Giá đóng cửa nến ngày UTC gần nhất trên Binance Spot, đơn vị USDT"),
        ("Regime hiện tại", f"<span class='pill' style='background:{colors[cur]}'>{ui.esc(label_map[str(cur)])}</span>",
         f"độ tin cậy {pct(float(last[f'prob_state_{cur}']), 0)}", colors[cur],
         "State có xác suất filtered lớn nhất, chỉ dùng dữ liệu đến ngày hiện tại"),
        ("Thời gian", f"{days} ngày", f"kỳ vọng {num(dur, 0)} ngày" if dur else "", colors[cur],
         "Số ngày liên tiếp gần nhất cùng regime; kỳ vọng = 1/(1 - p_kk) theo ma trận chuyển"),
        ("P(rủi ro cao)", pct(p_now, 1), f"ngưỡng {pct(thr, 0)}", ui.HIGH if p_now >= thr else ui.POS,
         "Xác suất filtered của state rủi ro cao (state có volatility train lớn nhất)"),
        (f"Tỷ trọng {asset}", pct(w_rule, 0), "nắm giữ đủ" if w_rule == w_on else "phòng thủ",
         ui.NAVY if w_rule == w_on else ui.WARN,
         f"Quy tắc chỉ định trước: w = {num(w_on, 0)} nếu P(rủi ro cao) < {num(thr, 1)}, ngược lại {num(w_off, 2)}. "
         "Không phải khuyến nghị giao dịch"),
        ("Biến động RV21", pct(rv_now, 1), "năm hóa", ui.NAVY, "Độ lệch chuẩn 21 log return gần nhất × √365"),
    ])

    left, right = st.columns([8, 4])
    with left, card("price12"):
        sec("Giá 12 tháng và regime", "nền màu = regime filtered",
            "Nền tô theo state có xác suất filtered lớn nhất mỗi ngày; dải dưới là P(rủi ro cao) và ngưỡng phòng thủ")
        start = filt["date"].iloc[-1] - pd.DateOffset(months=12)
        v = filt[filt["date"] > start].reset_index(drop=True)
        pr = px_df[px_df["date"] > start]
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.76, 0.24], vertical_spacing=0.04)
        fig.add_trace(go.Scatter(x=pr["date"], y=pr["close"], mode="lines", line={"color": ui.INK, "width": 1.6},
                                 name="Giá", hovertemplate="%{x|%d/%m/%Y}: %{y:,.0f} USDT<extra></extra>"),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=v["date"], y=v["prob_high_risk"], mode="lines", fill="tozeroy",
                                 line={"color": ui.HIGH, "width": 1.2}, fillcolor=ui.rgba(ui.HIGH, .25),
                                 name="P(rủi ro cao)", hovertemplate="%{x|%d/%m/%Y}: %{y:.0%}<extra></extra>"),
                      row=2, col=1)
        fig.add_hline(y=thr, line_dash="dot", line_color=ui.MUTED, row=2, col=1)
        for x0, x1, st_ in runs(v["date"], v["state_id"]):  # tô nền sau khi có trace (plotly bỏ qua subplot rỗng)
            fig.add_vrect(x0=x0, x1=x1, fillcolor=colors[int(st_)], opacity=0.2, line_width=0, layer="below",
                          row=1, col=1)
        for i in range(k):
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=state_name(i, label_map),
                                     marker={"size": 11, "color": colors[i], "symbol": "square"}), row=1, col=1)
        fig.update_yaxes(title="USDT", row=1, col=1)
        fig.update_yaxes(range=[0, 1], tickvals=[0, thr, 1], tickformat=".0%", row=2, col=1)
        show(ui.chart_layout(fig, H_BIG, hovermode="x unified"))

    with right, card("gauge"):
        sec("Thước đo biến động", "RV21 so với các regime",
            "Kim là RV21 năm hóa hiện tại; vùng màu chia theo volatility trung bình của từng state trên train")
        t = b04[(b04["asset"] == asset) & (b04["period"] == "train")].sort_values("annualized_volatility")
        vols = list(t["annualized_volatility"])
        ids = [int(x) for x in t["state_id"]]
        top = max(vols[-1] * 1.25, (rv_now or 0) * 1.1)
        cuts = [0] + [(a + b) / 2 for a, b in zip(vols[:-1], vols[1:])] + [top]
        steps = [{"range": [cuts[i] * 100, cuts[i + 1] * 100], "color": ui.rgba(colors[ids[i]], .6)}
                 for i in range(len(ids))]
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=(rv_now or 0) * 100,
            number={"suffix": "%", "valueformat": ".1f", "font": {"size": 40, "color": ui.INK}},
            gauge={"axis": {"range": [0, top * 100], "ticksuffix": "%", "tickcolor": ui.MUTED, "nticks": 6},
                   "bar": {"color": "rgba(0,0,0,0)", "thickness": 0}, "steps": steps, "borderwidth": 0,
                   "threshold": {"line": {"color": ui.INK, "width": 5}, "thickness": 1, "value": (rv_now or 0) * 100}},
            domain={"x": [0.06, 0.94], "y": [0, 1]}))
        ui.chart_layout(fig, H_BIG - 24 * len(ids) - 6, legend=False)
        fig.update_layout(margin={"l": 30, "r": 30, "t": 40, "b": 10})
        show(fig)
        md("".join(f"<div class='muted'>{ui.chip(colors[i])}{ui.esc(state_name(i, label_map))}: TB "
                   f"<b>{pct(vv, 0)}</b></div>" for i, vv in zip(ids, vols)))

    with card("strip"):
        others = [a for a in rs.assets() if a != asset]
        rows = [(asset, list(filt["date"]), list(filt["state_id"]), label_map)]
        both = None
        for o in others:
            of, om, _, _ = filtered_of(o)
            if of is not None:
                rows.append((o, list(of["date"]), list(of["state_id"]), om["label_map"]))
                a_hr = filt.set_index("date")["state_id"].eq(risk)
                o_hr = of.set_index("date")["state_id"].eq(om["high_risk_state"])
                j = pd.concat([a_hr, o_hr], axis=1, join="inner")
                union = (j.iloc[:, 0] | j.iloc[:, 1]).sum()
                both = (j.iloc[:, 0] & j.iloc[:, 1]).sum() / union if union else None
        sub = " và ".join(r[0] for r in rows) + f" · {dmy(filt['date'].iloc[0])} - {dmy(filt['date'].iloc[-1])}"
        sec("Dải regime toàn lịch sử", sub,
            "Mỗi vạch là một ngày, tô theo regime filtered của từng đồng; đường đứt là mốc hết train và hết "
            "validation. Hai đồng có bộ state riêng, cùng màu nghĩa là cùng nhãn mô tả (Low-Vol, High-Vol)")
        fig = regime_strips(rows, H_STRIP if len(rows) > 1 else 130)
        split_lines(fig, split_manifest.get("assets", {}).get(asset))
        show(fig)
        legend = "<br>".join(f"<span class='muted'><b>{r[0]}</b></span> &nbsp; "
                             + regime_legend(r[3], len(r[3])) for r in rows)
        if both is not None:
            legend += (f"<br><span class='muted'>Đồng pha rủi ro: trong các ngày có ít nhất một đồng ở regime rủi ro "
                       f"cao, <b>{pct(both, 0)}</b> là ngày cả hai cùng ở regime rủi ro cao</span>")
        md(legend)

    left, right = st.columns(2)
    with left, card("share"):
        sec("Thời gian ở mỗi regime", "toàn lịch sử", "Tỷ lệ số ngày có hard label filtered thuộc từng state")
        cnt = filt["state_id"].value_counts().reindex(range(k), fill_value=0)
        fig = go.Figure(go.Pie(labels=[state_name(i, label_map) for i in range(k)], values=list(cnt), hole=0.62,
                               marker={"colors": [colors[i] for i in range(k)], "line": {"color": "#FFFCF6", "width": 2}},
                               sort=False, textinfo="percent", texttemplate="%{percent:.0%}",
                               hovertemplate="%{label}: %{value} ngày (%{percent:.1%})<extra></extra>"))
        fig.add_annotation(text=f"<b>{num(len(filt), 0)}</b><br>ngày", showarrow=False,
                           font={"size": 16, "color": ui.INK})
        ui.chart_layout(fig, H_MID)
        fig.update_layout(legend={"orientation": "v", "x": 1.0, "y": 0.5, "xanchor": "left", "yanchor": "middle"})
        show(fig)
    with right, card("profile"):
        sec("Hồ sơ từng regime", "thống kê trên train", "Ước lượng trên tập train; nhãn giữ nguyên cho validation/test")
        t = b04[(b04["asset"] == asset) & (b04["period"] == "train")].sort_values("state_id")
        html = []
        for r in t.itertuples():
            c = colors[int(r.state_id)]
            html.append(
                f"<div class='prof' style='--c:{c}'><div class='n'>{ui.esc(state_name(r.state_id, label_map))}"
                + (" · rủi ro cao" if int(r.state_id) == risk else "") + "</div><div class='g'>"
                f"<div>Volatility/năm<b>{pct(r.annualized_volatility, 0)}</b></div>"
                f"<div>Lợi suất/ngày<b>{pct(r.mean_return, 2, sign=True)}</b></div>"
                f"<div>Thời lượng KV<b>{num(r.expected_duration, 0)} ngày</b></div>"
                f"<div>Tỷ trọng train<b>{pct(r.share, 0)}</b></div></div></div>")
        md(f"<div class='profgrid{' two' if k > 3 else ''}'>{''.join(html)}</div>")


# ============================================================================= REGIME HISTORY


def transition_diagram(mat, label_map, colors, height=H_MID):
    """Sơ đồ chuyển trạng thái: vòng tròn là state (ghi xác suất ở lại), mũi tên dày theo xác suất chuyển."""
    ids = list(mat.index)
    k = len(ids)
    layouts = {2: [(-1.5, 0), (1.5, 0)], 3: [(0, 0.72), (1.65, -0.62), (-1.65, -0.62)],
               4: [(-1.55, 0.7), (1.55, 0.7), (1.55, -0.7), (-1.55, -0.7)]}
    coords = layouts.get(k) or [(1.6 * math.cos(2 * math.pi * i / k), 0.8 * math.sin(2 * math.pi * i / k))
                                for i in range(k)]
    pos = dict(zip(ids, coords))
    off = [mat.loc[i, j] for i in ids for j in ids if i != j]
    pmax = max(off) if off else 1
    fig = go.Figure()
    r = 0.47  # bán kính vòng tròn theo đơn vị trục
    for i in ids:
        for j in ids:
            p = float(mat.loc[i, j])
            if i == j or p < 0.001:
                continue
            (x0, y0), (x1, y1) = pos[i], pos[j]
            dx, dy = x1 - x0, y1 - y0
            ln = math.hypot(dx, dy)
            ux, uy = dx / ln, dy / ln
            nx, ny = -uy * 0.09, ux * 0.09  # tách hai chiều i->j và j->i
            sx, sy, ex, ey = x0 + ux * r + nx, y0 + uy * r + ny, x1 - ux * r + nx, y1 - uy * r + ny
            fig.add_annotation(x=ex, y=ey, ax=sx, ay=sy, xref="x", yref="y", axref="x", ayref="y", text="",
                               showarrow=True, arrowhead=3, arrowsize=1, arrowwidth=1.2 + 5 * p / pmax,
                               arrowcolor=ui.rgba(colors[int(i)], .8))
            fig.add_annotation(x=sx + (ex - sx) * 0.3 + nx * 2.2, y=sy + (ey - sy) * 0.3 + ny * 2.2, text=pct(p, 1),
                               showarrow=False, font={"size": 10, "color": ui.INK}, bgcolor="rgba(255,252,246,.9)")
    stays = [float(mat.loc[s, s]) for s in ids]
    fig.add_trace(go.Scatter(
        x=[pos[s][0] for s in ids], y=[pos[s][1] for s in ids], mode="markers+text",
        text=[f"<b>{ui.esc(label_map[str(int(s))])}</b><br>ở lại {pct(v, 1)}" for s, v in zip(ids, stays)],
        textfont={"color": "white", "size": 11},
        marker={"size": 92, "color": [colors[int(s)] for s in ids], "line": {"width": 0}},
        customdata=[num(1 / (1 - v), 0) for v in stays],
        hovertemplate="%{text}<br>thời lượng kỳ vọng ~%{customdata} ngày<extra></extra>"))
    fig.update_xaxes(visible=False, range=[-2.4, 2.4])
    fig.update_yaxes(visible=False, range=[-1.3, 1.3], scaleanchor="x")
    return ui.chart_layout(fig, height, legend=False)


def page_history():
    if filt is None:
        st.info(f"{filt_error}. Chạy: `python scripts/complete_c2.py`")
        return
    label_map, k = meta["label_map"], meta["K"]
    colors = ui.state_colors(label_map)
    with card("controls"):
        c1, c2, c3 = st.columns([3.6, 3.8, 4.6])
        mode = c1.radio(
            "Loại xác suất để tô màu",
            ["Filtered (nhân quả, dùng cho backtest)", "Smoothed (hindsight, chỉ mô tả lịch sử)"],
        )
        period = c2.segmented_control("Giai đoạn", ["Toàn bộ", "Train", "Validation", "Test"], default="Toàn bộ")
        use_smoothed = mode.startswith("Smoothed")
        source = _smoothed(asset, rs.mtime_key(rs.asset_files(asset)["smoothed"])) if use_smoothed else filt
        if source is None:
            st.info("Chưa có file smoothed của C2.")
            source, use_smoothed = filt, False
        if period and period != "Toàn bộ":
            source = source[source["split"] == period.lower()]
        rng = c3.slider("Khoảng ngày", min_value=source["date"].min().date(), max_value=source["date"].max().date(),
                        value=(source["date"].min().date(), source["date"].max().date()), format="DD/MM/YYYY")
    if use_smoothed:
        st.warning("Smoothed dùng thông tin của cả các ngày sau t (hindsight). "
                   "Chỉ để đọc lịch sử, không được dùng làm tín hiệu giao dịch.", icon="🕰️")
    view = source[(source["date"].dt.date >= rng[0]) & (source["date"].dt.date <= rng[1])].reset_index(drop=True)
    price = px_df[(px_df["date"] >= pd.Timestamp(rng[0])) & (px_df["date"] <= pd.Timestamp(rng[1]))]
    kind = "smoothed" if use_smoothed else "filtered"
    dfv = forward_returns(with_states(view))

    with card("price"):
        sec(f"Giá {asset} và regime {kind}", f"{dmy(rng[0])} - {dmy(rng[1])} · thang log",
            "Nền tô theo state có xác suất lớn nhất mỗi ngày; đường đứt là mốc hết train và hết validation")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=price["date"], y=price["close"], mode="lines", line={"color": ui.INK, "width": 1.3},
                                 name="Giá đóng cửa", hovertemplate="%{x|%d/%m/%Y}: %{y:,.2f} USDT<extra></extra>"))
        for x0, x1, s in runs(view["date"], view["state_id"]):
            fig.add_vrect(x0=x0, x1=x1, fillcolor=colors[int(s)], opacity=0.26, line_width=0, layer="below")
        for i in range(k):
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=state_name(i, label_map),
                                     marker={"size": 11, "color": colors[i], "symbol": "square"}))
        lo, hi = float(price["close"].min()), float(price["close"].max())
        ticks = [m * 10 ** e for e in range(int(math.floor(math.log10(lo))), int(math.ceil(math.log10(hi))) + 1)
                 for m in (1, 2, 5) if lo * 0.8 <= m * 10 ** e <= hi * 1.25]
        fig.update_yaxes(type="log", title="USDT", tickvals=ticks, ticktext=[num(v, 0) for v in ticks])
        split_lines(fig, split_manifest.get("assets", {}).get(asset))
        show(ui.chart_layout(fig, H_BIG))

    left, right = st.columns(2)
    with left, card("fwd"):
        sec("Sau khi vào regime, giá đi đâu?", "lợi suất trung bình sau ngày vào regime",
            "Mỗi lần mô hình chuyển sang một regime (ngày đầu của đợt), đo lợi suất tích lũy của giá trong 5, 10, 20 "
            "ngày tiếp theo rồi lấy trung bình theo regime. Thanh xám là trung bình của mọi ngày, để so sánh. "
            "Thống kê mô tả trên khoảng ngày đang chọn, không phải dự báo")
        fig = go.Figure()
        xs = [f"{h} ngày" for h in HORIZONS]
        base = [dfv[f"fwd_{h}"].mean() for h in HORIZONS]
        fig.add_trace(go.Bar(x=xs, y=base, name="Mọi ngày", marker_color=ui.rgba(ui.MUTED, .45), marker_line_width=0,
                             text=[pct(v, 1) for v in base], textposition="outside", cliponaxis=False,
                             hovertemplate="Mọi ngày, %{x}: %{y:.2%}<extra></extra>"))
        ent = dfv[dfv["entry"]]
        for i in range(k):
            e = ent[ent["state_id"] == i]
            vals = [e[f"fwd_{h}"].mean() for h in HORIZONS]
            fig.add_trace(go.Bar(x=xs, y=vals, name=f"{state_name(i, label_map)} (n = {len(e)})",
                                 marker_color=colors[i], marker_line_width=0, text=[pct(v, 1) for v in vals],
                                 textposition="outside", cliponaxis=False,
                                 hovertemplate=state_name(i, label_map) + ", %{x}: %{y:.2%}<extra></extra>"))
        fig.add_hline(y=0, line_color=ui.LINE)
        fig.update_yaxes(tickformat=".0%")
        show(ui.chart_layout(fig, H_MID, barmode="group", bargap=0.25, bargroupgap=0.05))
    with right, card("dist"):
        sec("Phân phối lợi suất ngày theo regime", "khoảng ngày đang chọn",
            "Mỗi hình là phân phối lợi suất ngày (simple return) của các ngày thuộc regime đó; hình càng rộng "
            "nghĩa là biến động càng lớn. Đường ngang trong hộp là trung vị")
        fig = go.Figure()
        lo_, hi_ = dfv["simple_return"].quantile([0.005, 0.995])
        for i in range(k):
            y = dfv.loc[dfv["state_id"] == i, "simple_return"]
            fig.add_trace(go.Violin(y=y, name=state_name(i, label_map), line_color=colors[i],
                                    fillcolor=ui.rgba(colors[i], .35), box_visible=True, meanline_visible=False,
                                    points=False, spanmode="hard", hoverinfo="skip"))
        fig.update_yaxes(tickformat=".0%", range=[lo_ * 1.1, hi_ * 1.1], zeroline=True, zerolinecolor=ui.LINE)
        show(ui.chart_layout(fig, H_MID, legend=False))

    left, right = st.columns(2)
    if b05 is not None:
        t = b05[b05["asset"] == asset]
        mat = t.pivot(index="from_state", columns="to_state", values="probability").sort_index()
        with left, card("trans"):
            sec("Sơ đồ chuyển trạng thái", "ước lượng trên train",
                "Mũi tên là xác suất chuyển từ regime này sang regime khác sau một ngày; mũi tên càng dày xác suất "
                "càng lớn. Dưới mỗi vòng là xác suất ở lại và thời lượng kỳ vọng 1/(1 - p_kk)")
            show(transition_diagram(mat, label_map, colors))
    with right, card("runs"):
        sec("Mỗi đợt regime kéo dài bao lâu", "số ngày liên tiếp, thang log",
            "Mỗi chấm là một đợt liên tiếp cùng regime (hard label) trong khoảng ngày đang chọn; hộp là phân vị "
            "25-75%. Các đợt ở hai đầu khoảng chọn có thể bị cắt ngắn")
        rl = pd.DataFrame([(int(s), (x1 - x0).days) for x0, x1, s in runs(view["date"], view["state_id"])],
                          columns=["state", "days"])
        fig = go.Figure()
        for i in range(k):
            y = rl.loc[rl["state"] == i, "days"]
            fig.add_trace(go.Box(y=y, name=f"{state_name(i, label_map)}<br>{len(y)} đợt", marker_color=colors[i],
                                 line_color=colors[i], fillcolor=ui.rgba(colors[i], .25), boxpoints="all",
                                 jitter=0.45, pointpos=0, marker={"size": 5, "opacity": .7},
                                 hovertemplate="%{y} ngày<extra></extra>"))
        tv = [v for v in (1, 2, 5, 10, 20, 50, 100, 200, 500) if v <= max(rl["days"].max() * 1.5, 2)]
        fig.update_yaxes(type="log", title="ngày", tickvals=tv, ticktext=[str(v) for v in tv])
        show(ui.chart_layout(fig, H_MID, legend=False))

    with st.expander("Bảng số: ma trận chuyển, persistence, độ dài đoạn quan sát"):
        if b05 is not None:
            names = [state_name(i, label_map) for i in mat.index]
            rows = [[ui.esc(names[r])] + [num(mat.iloc[r, c], 3) for c in range(len(names))] for r in range(len(names))]
            md(ui.html_table(["Từ \\ sang"] + [ui.esc(n) for n in names], rows))
        if b04 is not None:
            o = b04[b04["asset"] == asset].pivot(index="state_id", columns="period", values="mean_observed_run_days")
            rows = [[ui.esc(state_name(s, label_map))] + [num(o.loc[s].get(p_), 1) for p_ in ("train", "validation", "test")]
                    for s in o.index]
            md("<br>" + ui.html_table(["State", "Đoạn TB train (ngày)", "Validation", "Test"], rows))


# ============================================================================= MODEL COMPARISON


def page_models():
    b03, b07, stab = tload("B03"), tload("B07"), tload("stability")
    if b03 is None or filt is None:
        run_hint("python scripts/complete_c2.py")
        return
    label_map, k = meta["label_map"], meta["K"]
    t = b03[b03["asset"] == asset].copy()
    sel = t[t["selected"]].iloc[0]
    t7 = b07[b07["asset"] == asset].set_index(["method", "period"]) if b07 is not None else None
    st_row = None
    if stab is not None:
        s = stab[(stab["asset"] == asset) & (stab["K"] == sel["K"])]
        st_row = s.iloc[0] if not s.empty else None

    def g(method, period, col):
        try:
            return float(t7.loc[(method, period), col])
        except (KeyError, TypeError):
            return None

    hmm_t, gmm_t = g("HMM", "test", "avg_log_score"), g("GMM", "test", "avg_log_score")
    hmm_f, gmm_f = g("HMM", "test", "flips_per_100d"), g("GMM", "test", "flips_per_100d")
    stab_status = st_row["status"] if st_row is not None else "-"
    kpis([
        ("K được chọn", str(int(sel["K"])), f"thử K = {', '.join(str(x) for x in sorted(t['K'].unique()))}", ui.NAVY,
         "Chọn theo validation log-score lớn nhất trong các fit hợp lệ; test không tham gia chọn"),
        ("Log-score validation", num(sel["val_avg_predictive_ll"], 3), "HMM, cao hơn tốt hơn", ui.NAVY,
         "Log-score dự báo một bước trung bình mỗi quan sát trên validation; không phải độ chính xác phân loại"),
        ("Log-score test", num(hmm_t, 3), f"GMM {num(gmm_t, 3)} · " + colored(num((hmm_t or 0) - (gmm_t or 0), 3,
                                                                                    sign=True), (hmm_t or 0) - (gmm_t or 0)),
         ui.POS if (hmm_t or 0) > (gmm_t or 0) else ui.NEG, "HMM so với GMM trên tập test, cùng dữ liệu và chuẩn hóa"),
        ("Đổi state / 100 ngày", num(hmm_f, 1), f"GMM {num(gmm_f, 1)} trên test", ui.POS if (hmm_f or 0) < (gmm_f or 0)
         else ui.NEG, "Số lần đổi hard label trên 100 ngày test; thấp hơn nghĩa là regime bền hơn"),
        ("Độ ổn định seed", f"<span class='pill' style='background:{ui.POS if stab_status == 'stable' else ui.WARN}'>"
         f"{ui.esc(stab_status)}</span>",
         f"ARI median {num(st_row['ari_median'], 2)}" if st_row is not None else "", ui.WARN,
         "ARI giữa các seed trên validation; ngưỡng vận hành median ≥ 0,8 và min ≥ 0,5. Chẩn đoán mô tả"),
    ])

    with card("hmmgmm"):
        sec("HMM và GMM trên cùng dải thời gian", "nhãn từng ngày, toàn lịch sử",
            "Dòng trên là regime filtered của HMM, dòng dưới là nhãn GMM (không có cấu trúc Markov). GMM đổi nhãn "
            "liên tục hơn vì không dùng thông tin hôm trước; HMM tạo các đợt regime liền mạch")
        g_df, g_lm = _gmm(asset, rs.mtime_key(rs.asset_files(asset)["gmm"]))
        rows = [("HMM", list(filt["date"]), list(filt["state_id"]), label_map)]
        if g_df is not None:
            rows.append(("GMM", list(g_df["date"]), list(g_df["state_id"]), g_lm))
        fig = regime_strips(rows)
        split_lines(fig, split_manifest.get("assets", {}).get(asset))
        show(fig)
        md(regime_legend(label_map, k))

    if t7 is not None:
        b7 = b07[b07["asset"] == asset]
        left, right = st.columns(2)
        for col, metric, title, sub, digits, info in (
            (left, "avg_log_score", "Log-score trung bình", "cao hơn tốt hơn", 3,
             "HMM: log p(X_t | X_1:t-1); GMM: mật độ biên log p(X_t). Hai mức điều kiện thông tin khác nhau"),
            (right, "flips_per_100d", "Số lần đổi regime / 100 ngày", "thấp hơn bền hơn", 1,
             "Số lần hard label đổi giữa hai ngày liên tiếp, quy về 100 ngày"),
        ):
            with col, card(f"b07_{metric}"):
                sec(title, sub, info)
                fig = go.Figure()
                for method in ("HMM", "GMM"):
                    s = b7[b7["method"] == method]
                    fig.add_trace(go.Bar(x=[PERIOD_NAMES[p] for p in s["period"]], y=s[metric], name=method,
                                         marker_color=ui.METHOD_COLORS[method], marker_line_width=0,
                                         text=[num(v, digits) for v in s[metric]], textposition="outside",
                                         cliponaxis=False))
                show(ui.chart_layout(fig, H_MID, barmode="group", bargap=0.35))

    left, right = st.columns(2)
    with left, card("ksel"):
        sec("Chọn số state K", "validation log-score theo K",
            "Mỗi chấm là một seed hợp lệ; vòng vàng là mô hình được chọn. Quy tắc: log-score validation lớn nhất, "
            "hòa thì BIC train thấp hơn, rồi K nhỏ hơn, rồi seed nhỏ hơn")
        ok = t[t["eligible"]]
        best = ok.groupby("K")["val_avg_predictive_ll"].max()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ok["K"], y=ok["val_avg_predictive_ll"], mode="markers", name="Seed hợp lệ",
                                 marker={"size": 9, "color": ui.rgba(ui.NAVY, .45)}, text=ok["seed"],
                                 hovertemplate="K=%{x}, seed %{text}: %{y:.4f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=best.index, y=best.values, mode="lines+markers+text", name="Tốt nhất mỗi K",
                                 line={"color": ui.NAVY, "width": 2}, marker={"size": 8, "color": ui.NAVY},
                                 text=[num(v, 3) for v in best.values], textposition="top center",
                                 hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=[sel["K"]], y=[sel["val_avg_predictive_ll"]], mode="markers", name="Được chọn",
                                 marker={"size": 22, "color": "rgba(0,0,0,0)", "line": {"color": ui.ACCENT, "width": 3}},
                                 hoverinfo="skip"))
        lo = float(ok["val_avg_predictive_ll"].min())
        hi = float(ok["val_avg_predictive_ll"].max())
        pad = (hi - lo) * 0.3 + 0.05
        for kk in sorted(set(t["K"]) - set(best.index)):
            fig.add_annotation(x=kk, y=(lo + hi) / 2, text="không có<br>fit hợp lệ", showarrow=False,
                               font={"size": 11, "color": ui.MUTED})
        fig.update_xaxes(tickvals=sorted(t["K"].unique()), range=[t["K"].min() - 0.5, t["K"].max() + 0.5],
                         title="K")
        fig.update_yaxes(range=[lo - pad, hi + pad])
        show(ui.chart_layout(fig, H_MID))
    with right, card("stab"):
        sec("Độ ổn định giữa các seed", "chẩn đoán mô tả",
            "ARI đo mức trùng nhãn giữa các seed (1 là trùng hoàn toàn, quanh 0 là ngẫu nhiên), bất biến theo hoán "
            "vị ID state. Không dùng để chọn mô hình")
        if stab is not None:
            s = stab[stab["asset"] == asset].sort_values("K")
            rows, cls = [], []
            for r in s.itertuples():
                c = ui.POS if r.status == "stable" else ui.WARN
                rows.append([f"K = {int(r.K)}" + (" (chọn)" if int(r.K) == int(r.selected_K) else ""),
                             str(int(r.n_eligible_seeds)), num(r.ari_min, 2), num(r.ari_median, 2),
                             f"<b style='color:{c}'>{ui.esc(r.status)}</b>"])
                cls.append("sel" if int(r.K) == int(r.selected_K) else "")
            md(ui.html_table(["K", "Seed hợp lệ", "ARI min", "ARI median", "Trạng thái"], rows, cls))
            md("<div class='note' style='--c:#B07A2A;margin-top:12px'>Mô hình được chọn có ARI giữa các seed thấp: "
               "nhãn regime phụ thuộc khởi tạo, nên kết quả cần đọc như thống kê mô tả cho nghiên cứu.</div>")

    with st.expander("Bảng chi tiết: toàn bộ fit (B03) và HMM so với GMM (B07)"):
        cols = ["K", "seed", "converged", "eligible", "selected", "val_avg_predictive_ll", "train_ll", "AIC", "BIC",
                "min_state_share", "rejection_reason"]
        st.dataframe(t[cols].round(4), width="stretch", hide_index=True, height=300)
        if b07 is not None:
            st.dataframe(b07[b07["asset"] == asset][["method", "period", "start", "end", "n", "avg_log_score", "flips",
                                                     "flips_per_100d", "persistence"]].round(4),
                         width="stretch", hide_index=True)


# ============================================================================= PORTFOLIO & RISK


def monthly_returns(d):
    m = d.set_index("date")["net"].add(1).groupby(pd.Grouper(freq="ME")).prod().sub(1)
    return pd.DataFrame({"year": m.index.year, "month": m.index.month, "ret": m.values})


def page_portfolio():
    nav = tload("nav")
    if b08 is None or nav is None or not bt_manifest or bt_manifest.get("status") != "complete":
        run_hint("python run_pipeline.py --steps backtest")
        return
    by_name = {v: k for k, v in SCENARIO_NAMES.items()}
    c1, c2 = st.columns([3, 5])
    scenario = by_name[c1.selectbox("Kịch bản", list(by_name))]
    n = nav[(nav["asset"] == asset) & (nav["scenario"] == scenario)].copy()
    n["date"] = pd.to_datetime(n["date_utc"])
    b = b08[(b08["asset"] == asset) & (b08["scenario"] == scenario)].set_index("strategy")
    ra, bh = b.loc["regime_aware"], b.loc["buy_and_hold"]
    c2.markdown(f"<div class='muted' style='padding-top:34px'>Tập test {dmy(ra['start'])} - {dmy(ra['end'])} · "
                f"{int(ra['n_days'])} ngày · phí 0,10% × turnover · tín hiệu ngày t áp dụng cho ngày t+1</div>",
                unsafe_allow_html=True)

    def vs(diff_text, x, better_high=True):
        return f"{colored(diff_text, x, better_high)} vs B&H"

    off_days = int(round(ra["share_days_risk_off"] * ra["n_days"]))
    kpis([
        ("NAV cuối (regime-aware)", num(ra["final_nav"], 4),
         vs(num(ra["final_nav"] - bh["final_nav"], 4, sign=True), ra["final_nav"] - bh["final_nav"]),
         ui.sign_color(ra["final_nav"] - 1), f"Buy-and-hold: {num(bh['final_nav'], 4)}; NAV bắt đầu = 1"),
        ("CAGR", pct(ra["CAGR"]), vs(ui.pp(ra["CAGR"] - bh["CAGR"]), ra["CAGR"] - bh["CAGR"]),
         ui.sign_color(ra["CAGR"]), f"Buy-and-hold: {pct(bh['CAGR'])}; năm hóa 365 ngày"),
        ("Sharpe (sau phí)", num(ra["Sharpe"], 3),
         vs(num(ra["Sharpe"] - bh["Sharpe"], 3, sign=True), ra["Sharpe"] - bh["Sharpe"]),
         ui.sign_color(ra["Sharpe"]), f"Buy-and-hold: {num(bh['Sharpe'], 3)}; rf = 0, năm hóa 365 ngày"),
        ("Max drawdown", pct(ra["max_drawdown"]),
         vs(ui.pp(ra["max_drawdown"] - bh["max_drawdown"]), ra["max_drawdown"] - bh["max_drawdown"]), ui.NEG,
         f"Buy-and-hold: {pct(bh['max_drawdown'])}; mức sụt giảm lớn nhất từ đỉnh"),
        ("Ngày phòng thủ", f"{off_days} ngày", f"{pct(ra['share_days_risk_off'], 1)} số ngày test", ui.WARN,
         "Số ngày quy tắc giảm tỷ trọng xuống mức phòng thủ (w < 1)"),
        ("Tổng phí", pct(ra["total_cost"]), f"B&H {pct(bh['total_cost'])}", ui.MUTED,
         "Tổng phí giao dịch 0,10% × turnover trên cả giai đoạn"),
    ])

    ra_nav = n[n["strategy"] == "regime_aware"].reset_index(drop=True)
    left, right = st.columns([8, 4])
    with left, card("nav"):
        sec("NAV tích lũy", "vùng màu đồng là ngày chiến lược ở chế độ phòng thủ",
            "NAV bắt đầu bằng 1 ở đầu tập test; cả hai chiến lược trả phí mở vị thế ngày đầu")
        fig = go.Figure()
        for s in ("buy_and_hold", "regime_aware"):
            d = n[n["strategy"] == s]
            fig.add_trace(go.Scatter(x=d["date"], y=d["nav"], mode="lines", name=STRATEGY_NAMES[s],
                                     line={"color": ui.STRATEGY_COLORS[s], "width": 2.2 if s == "regime_aware" else 1.6},
                                     hovertemplate="%{y:.4f}"))
        for x0, x1, v in runs(ra_nav["date"], ra_nav["weight"] < 1):
            if v:
                fig.add_vrect(x0=x0, x1=x1, fillcolor=ui.rgba(ui.ACCENT, .35), line_width=0, layer="below")
        fig.add_hline(y=1, line_dash="dot", line_color=ui.MUTED)
        show(ui.chart_layout(fig, H_BIG, hovermode="x unified"))
    with right, card("bars"):
        sec("So sánh nhanh", SCENARIO_SHORT[scenario], "Các chỉ số trên toàn tập test của kịch bản đang chọn")
        mets = [("Lợi nhuận tích lũy", "cumulative_return"), ("CAGR", "CAGR"), ("Volatility năm hóa", "ann_volatility"),
                ("Max drawdown", "max_drawdown")]
        fig = go.Figure()
        for s, row in (("buy_and_hold", bh), ("regime_aware", ra)):
            vals = [float(row[c]) for _, c in mets]
            fig.add_trace(go.Bar(y=[m for m, _ in mets], x=vals, orientation="h", name=STRATEGY_NAMES[s],
                                 marker_color=ui.STRATEGY_COLORS[s], marker_line_width=0,
                                 text=[pct(v, 1) for v in vals], textposition="outside", cliponaxis=False,
                                 hovertemplate="%{y}: %{x:.2%}<extra>" + STRATEGY_NAMES[s] + "</extra>"))
        allv = [float(row[c]) for row in (bh, ra) for _, c in mets]
        span = max(allv) - min(min(allv), 0)
        fig.update_xaxes(tickformat=".0%", zeroline=True, zerolinecolor=ui.LINE,
                         range=[min(min(allv), 0) - span * 0.32, max(max(allv), 0) + span * 0.32])
        fig.update_yaxes(autorange="reversed", ticklabelstandoff=8)
        ui.chart_layout(fig, H_BIG, barmode="group", bargap=0.3)
        fig.update_layout(margin={"l": 10, "r": 10, "t": 36, "b": 10})
        show(fig)

    left, right = st.columns(2)
    with left, card("dd"):
        sec("Drawdown", "mức sụt giảm so với đỉnh gần nhất")
        fig = go.Figure()
        for s in ("buy_and_hold", "regime_aware"):
            d = n[n["strategy"] == s]
            fig.add_trace(go.Scatter(x=d["date"], y=d["drawdown"], mode="lines", name=STRATEGY_NAMES[s],
                                     fill="tozeroy" if s == "buy_and_hold" else None,
                                     fillcolor=ui.rgba(ui.STRATEGY_COLORS[s], .25),
                                     line={"color": ui.STRATEGY_COLORS[s], "width": 1.5}, hovertemplate="%{y:.1%}"))
        fig.update_yaxes(tickformat=".0%")
        show(ui.chart_layout(fig, H_MID, hovermode="x unified"))
    with right, card("heat"):
        sec("Lợi suất theo tháng", "tháng × năm",
            "Lợi suất tích lũy sau phí của từng tháng; tháng đầu và tháng cuối có thể không đủ ngày")
        view = st.segmented_control("Hiển thị", ["Regime-aware", "Buy-and-Hold", "Chênh lệch"],
                                    default="Regime-aware", key="heat_view", label_visibility="collapsed")
        mr = monthly_returns(ra_nav)
        mb = monthly_returns(n[n["strategy"] == "buy_and_hold"].reset_index(drop=True))
        mm = mr.merge(mb, on=["year", "month"], suffixes=("_ra", "_bh"))
        mm["ret_diff"] = mm["ret_ra"] - mm["ret_bh"]
        col = {"Regime-aware": "ret_ra", "Buy-and-Hold": "ret_bh", "Chênh lệch": "ret_diff"}[view or "Regime-aware"]
        piv = mm.pivot(index="year", columns="month", values=col).reindex(columns=range(1, 13))
        zmax = float(np.nanmax(np.abs(piv.values))) if piv.notna().any().any() else 0.1
        scale = [[0, ui.HIGH], [0.5, "#F6F1E7"], [1, ui.LOW]]
        text = [[pct(v, 0) if pd.notna(v) else "" for v in r] for r in piv.values]
        fig = go.Figure(go.Heatmap(z=piv.values, x=[f"T{m}" for m in piv.columns], y=[str(y) for y in piv.index],
                                   colorscale=scale, zmid=0, zmin=-zmax, zmax=zmax, text=text,
                                   texttemplate="%{text}", textfont={"size": 10}, xgap=3, ygap=3, showscale=False,
                                   hovertemplate="%{y} %{x}: %{z:.2%}<extra></extra>"))
        fig.update_yaxes(autorange="reversed", type="category")
        show(ui.chart_layout(fig, H_MID - 46, legend=False))

    left, right = st.columns(2)
    with left, card("rollvol"):
        sec("Độ biến động trượt 60 ngày", "năm hóa",
            "Độ lệch chuẩn lợi suất ngày sau phí trong 60 ngày gần nhất × √365; thấp hơn nghĩa là danh mục êm hơn")
        fig = go.Figure()
        for s in ("buy_and_hold", "regime_aware"):
            d = n[n["strategy"] == s]
            rv = d["net"].rolling(60).std() * math.sqrt(365)
            fig.add_trace(go.Scatter(x=d["date"], y=rv, mode="lines", name=STRATEGY_NAMES[s],
                                     line={"color": ui.STRATEGY_COLORS[s], "width": 1.8}, hovertemplate="%{y:.1%}"))
        fig.update_yaxes(tickformat=".0%")
        show(ui.chart_layout(fig, H_MID, hovermode="x unified"))
    with right, card("retdist"):
        sec("Phân phối lợi suất ngày", "sau phí", "Histogram lợi suất ngày của hai chiến lược trên tập test")
        fig = go.Figure()
        lo_, hi_ = n["net"].quantile([0.005, 0.995])
        for s in ("buy_and_hold", "regime_aware"):
            d = n[n["strategy"] == s]
            fig.add_trace(go.Histogram(x=d["net"], name=STRATEGY_NAMES[s], marker_color=ui.STRATEGY_COLORS[s],
                                       opacity=0.6, xbins={"start": lo_, "end": hi_, "size": (hi_ - lo_) / 50},
                                       hovertemplate="%{x:.1%}: %{y} ngày<extra></extra>"))
        fig.update_xaxes(tickformat=".0%")
        fig.update_yaxes(title="số ngày")
        show(ui.chart_layout(fig, H_MID, barmode="overlay"))

    with st.expander("Bảng hiệu suất B08 (mọi kịch bản), tải dữ liệu, giả định"):
        view_ = b08[b08["asset"] == asset]
        rows, cls = [], []
        for r in view_.itertuples():
            rows.append([f"{SCENARIO_SHORT.get(r.scenario, r.scenario)} · {STRATEGY_NAMES[r.strategy]}",
                         num(r.final_nav, 4), pct(r.cumulative_return), pct(r.CAGR), pct(r.ann_volatility),
                         num(r.Sharpe, 3), pct(r.max_drawdown), num(r.turnover, 3), pct(r.total_cost),
                         pct(r.share_days_risk_off, 1)])
            cls.append("sel" if r.scenario == scenario else "")
        md(ui.html_table(["Kịch bản · chiến lược", "NAV cuối", "Tích lũy", "CAGR", "Volatility", "Sharpe",
                          "Max DD", "Turnover", "Chi phí", "Ngày phòng thủ"], rows, cls))
        dl = st.columns([1, 1, 3])
        dl[0].download_button("Tải B08_portfolio.csv", rs.FILES["B08"].read_bytes(), "B08_portfolio.csv", "text/csv",
                              width="stretch")
        dl[1].download_button("Tải backtest_nav.csv", rs.FILES["nav"].read_bytes(), "backtest_nav.csv", "text/csv",
                              width="stretch")
        for a in bt_manifest.get("assumptions", []):
            st.markdown(f"- {a}")
        st.code(json.dumps(bt_manifest.get("assets", {}).get(asset, {}), ensure_ascii=False, indent=2), language="json")


# ============================================================================= điều hướng

{"Overview": page_overview, "Regime History": page_history, "Model Comparison": page_models,
 "Portfolio & Risk": page_portfolio}[page]()

md("<div class='foot'>Crypto Market Regime Radar · Binance Spot, nến ngày UTC · mô phỏng nghiên cứu, "
   "không phải khuyến nghị đầu tư</div>")
