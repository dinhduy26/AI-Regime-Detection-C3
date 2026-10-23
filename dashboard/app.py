"""Crypto Market Regime Radar - dashboard Streamlit 4 trang (C3-01, C3-08..C3-11, C3-13).

Chạy từ thư mục gốc dự án:
    streamlit run dashboard/app.py

Dashboard chỉ đọc kết quả đã xuất (outputs/, data/processed/); không huấn luyện lại mô hình
theo thao tác trên giao diện. Đây là công cụ mô phỏng nghiên cứu, không phải khuyến nghị đầu tư.
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
    "Regime History": "Lịch sử giá và trạng thái thị trường",
    "Model Comparison": "Chọn mô hình, HMM so với GMM, độ ổn định",
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
NOBAR = {"displayModeBar": False}


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


def jload(key):
    return _json(key, rs.mtime_key(rs.FILES[key]))


def tload(key):
    return _table(key, rs.mtime_key(rs.FILES[key]))


# ----------------------------------------------------------------------------- tiện ích hiển thị


def md(text):
    st.markdown(text, unsafe_allow_html=True)


def sec(title, sub=""):
    md(ui.sec_html(title, sub))


def card(name):
    return st.container(border=True, key=f"card_{name}")


def show(fig):
    st.plotly_chart(fig, width="stretch", config=NOBAR)


def kpis(items):
    cols = st.columns(len(items))
    for col, (label, value, sub, color) in zip(cols, items):
        col.markdown(ui.kpi_html(label, value, sub, color), unsafe_allow_html=True)


def dmy(x):
    return pd.Timestamp(x).strftime("%d/%m/%Y")


def runs(dates, values):
    """Các đoạn ngày liên tiếp cùng giá trị, dùng để tô nền biểu đồ."""
    out, start = [], 0
    s = np.asarray(values)
    for i in range(1, len(s) + 1):
        if i == len(s) or s[i] != s[start]:
            out.append((dates.iloc[start], dates.iloc[i - 1] + pd.Timedelta(days=1), s[start]))
            start = i
    return out


def split_lines(fig, split):
    for key, name in (("train", "hết train"), ("validation", "hết validation")):
        end = split.get(key, {}).get("end") if split else None
        if end:
            fig.add_vline(x=pd.Timestamp(end) + pd.Timedelta(hours=12), line_dash="dot", line_color="#94A3B8")
            fig.add_annotation(x=pd.Timestamp(end), y=0.99, yref="paper", text=name, showarrow=False,
                               font={"size": 10, "color": ui.MUTED}, xanchor="right", yanchor="top",
                               bgcolor="rgba(255,255,255,.7)")


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
    md("<div class='wordmark'>📡 REGIME RADAR</div>"
       "<div class='wordsub'>Nhận diện trạng thái thị trường BTC/ETH bằng Gaussian HMM và mô phỏng phân bổ "
       "phòng thủ theo regime</div>")
    md("<div class='sidecap'>TÀI SẢN</div>")
    asset = st.radio("Tài sản", rs.assets(), horizontal=True, key="asset", label_visibility="collapsed")
    md("<div class='sidecap'>ĐIỀU HƯỚNG</div>")
    page = st.radio("Trang", PAGES, key="nav", label_visibility="collapsed",
                    format_func=lambda p: f"{PAGES.index(p) + 1}.  {p}")
    asset_through = data_through.get(asset) if isinstance(data_through, dict) else data_through
    md("<div class='sidecap'>DỮ LIỆU</div>"
       f"<div class='sideval'>Binance Spot, nến ngày UTC<br>Đến hết ngày: <code>{asset_through}</code> (UTC)<br>"
       "Cập nhật theo ngày, <b>không phải realtime</b></div>")
    if model_manifest:
        a_meta = model_manifest.get("assets", {}).get(asset, {})
        stable = a_meta.get("seed_stability_status") == "stable"
        md("<div class='sidecap'>MÔ HÌNH C2</div>"
           f"<div class='sideval'><span class='dot' style='background:{'#22C55E' if stable else '#F59E0B'}'></span>"
           f"Gaussian HMM, K = {a_meta.get('K', '?')}, seed {a_meta.get('seed', '?')}<br>"
           f"Độ ổn định seed: <code>{a_meta.get('seed_stability_status', '?')}</code><br>"
           f"Config v{model_manifest.get('config_version')} · <code>{str(model_manifest.get('config_hash'))[:10]}</code>"
           "</div>")
    if bt_manifest:
        ok = bt_manifest.get("status") == "complete"
        md("<div class='sidecap'>BACKTEST C3</div>"
           f"<div class='sideval'><span class='dot' style='background:{'#22C55E' if ok else '#F59E0B'}'></span>"
           f"Trạng thái: <code>{bt_manifest.get('status')}</code><br>"
           f"Tạo lúc {str(bt_manifest.get('created_at_utc', ''))[:16].replace('T', ' ')} UTC</div>")

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
filt, meta, notes, filt_error = _filtered(asset, rs.mtime_key(rs.asset_files(asset)["filtered"]))
b04, b05, b08 = tload("B04"), tload("B05"), tload("B08")

# ----------------------------------------------------------------------------- thẻ đầu trang

last_px = px_df.iloc[-1]
last_feat = feat.iloc[-1] if feat is not None else None
chg = last_feat["simple_return"] if last_feat is not None else None
with card("hero"):
    left, mid, right = st.columns([5, 3, 2])
    left.markdown(
        f"<div class='hero'><div><p class='t'>{asset} <span>/ USDT · {PAGE_SUB[page]}</span></p>"
        f"<div class='muted'>Binance Spot · nến ngày UTC · dữ liệu đến hết {dmy(last_px['date'])}</div></div></div>",
        unsafe_allow_html=True)
    mid.markdown(
        f"<div class='px'>{num(last_px['close'], 2)} USDT "
        f"<span style='font-size:.9rem;color:{ui.sign_color(chg)}'>{pct(chg, 2, sign=True)}</span></div>"
        f"<div class='muted' style='text-align:right'>Giá đóng cửa ngày {dmy(last_px['date'])}</div>",
        unsafe_allow_html=True)
    if filt is not None:
        lm = meta["label_map"]
        cur = int(filt.iloc[-1]["state_id"])
        color = ui.state_colors(lm)[cur]
        right.markdown(
            "<div style='text-align:right'><div class='muted'>Regime hiện tại (filtered)</div>"
            f"<span class='badge' style='background:{color}'>{ui.esc(lm[str(cur)])}</span></div>",
            unsafe_allow_html=True)

if meta is not None and meta.get("seed_stability_status") != "stable":
    st.warning(
        f"**{asset}: mô hình C2 được đánh dấu `{meta.get('seed_stability_status')}`** theo chẩn đoán độ ổn định "
        "giữa các seed (nhãn regime phụ thuộc khởi tạo). Kết quả chỉ dùng cho nghiên cứu, cần đọc kèm giới hạn này.",
        icon="⚠️",
    )


def rule_params():
    thr, w_on, w_off = 0.5, 1.0, 0.25  # mặc định theo Methodology; ưu tiên giá trị đã ghi trong B08
    if b08 is not None and not b08.empty:
        r0 = b08.iloc[0]
        thr, w_on, w_off = float(r0["risk_threshold"]), float(r0["weight_risk_on"]), float(r0["weight_risk_off"])
    return thr, w_on, w_off


# ============================================================================= OVERVIEW


def page_overview():
    if filt is None:
        st.info(f"{filt_error}. Chạy: `python scripts/complete_c2.py`")
        return
    label_map, k, risk = meta["label_map"], meta["K"], meta["high_risk_state"]
    colors = ui.state_colors(label_map)
    last = filt.iloc[-1]
    thr, w_on, w_off = rule_params()
    p_now, p_next = float(last["prob_high_risk"]), float(last["prob_high_risk_next_day"])
    w_rule = w_off if p_now >= thr else w_on
    cur = int(last["state_id"])
    dur = None
    if b05 is not None:
        d = b05[(b05["asset"] == asset) & (b05["from_state"] == cur) & (b05["to_state"] == cur)]
        dur = float(d["expected_duration"].iloc[0]) if not d.empty else None

    kpis([
        ("P(high-risk) filtered", pct(p_now, 1), f"ngày {dmy(last['date'])}, ngưỡng {num(thr, 2)}",
         ui.HIGH if p_now >= thr else ui.POS),
        ("P(high-risk) ngày kế tiếp", pct(p_next, 1), "theo ma trận chuyển, không phải dự báo giá",
         ui.HIGH if p_next >= thr else ui.POS),
        ("Tỷ trọng mô phỏng kỳ tới", num(w_rule, 2), (f"w = {num(w_on, 0)} nếu P < {num(thr, 1)}, ngược lại "
         f"{num(w_off, 2)}"), ui.NAVY if w_rule == w_on else ui.WARN),
        ("RV21 năm hóa", pct(last_feat["rv21"] * math.sqrt(365), 1) if last_feat is not None else "-",
         "độ lệch chuẩn 21 log return × √365", ui.NAVY),
        ("Thời lượng kỳ vọng", f"{num(dur, 1)} ngày" if dur is not None else "-",
         f"của regime {ui.esc(label_map[str(cur)])}, E[D] = 1/(1 - p_kk)", colors[cur]),
    ])
    st.caption(
        f"Mô hình `{meta['model_run_id']}` · K = {k} · seed {meta['seed']} · "
        f"state rủi ro cao = {state_name(risk, label_map)} (state có volatility train lớn nhất). "
        "Nhãn chỉ là mô tả theo thống kê train, không phải ground truth."
    )

    left, right = st.columns([5, 7])
    with left, card("probs"):
        sec("Xác suất filtered từng state", f"ngày {dmy(last['date'])}")
        probs = [float(last[f"prob_state_{i}"]) for i in range(k)]
        names = [state_name(i, label_map) for i in range(k)]
        fig = go.Figure(go.Bar(
            y=names, x=probs, orientation="h", marker_color=[colors[i] for i in range(k)], marker_line_width=0,
            text=[pct(p, 1) for p in probs], textposition="outside", cliponaxis=False,
            hovertemplate="%{y}: %{x:.1%}<extra></extra>",
        ))
        fig.update_xaxes(range=[0, 1.22], tickvals=[0, .25, .5, .75, 1], tickformat=".0%", title="P(S_t = k | X_1:t)", showgrid=True,
                         gridcolor=ui.GRID)
        fig.update_yaxes(autorange="reversed", showgrid=False)
        show(ui.chart_layout(fig, 300, legend=False))
        md(f"<div class='muted'>Quy tắc nghiên cứu chỉ định trước, áp dụng cho kỳ kế tiếp:</div>"
           f"<div class='rule'>"
           f"<div class='{'on' if w_rule == w_on else ''}' style='--c:{ui.POS};--bg:{ui.rgba(ui.POS, .06)}'>"
           f"<span class='muted'>P(high-risk) &lt; {num(thr, 1)}</span><b style='color:{ui.POS}'>w = {num(w_on, 2)}</b></div>"
           f"<div class='{'on' if w_rule == w_off else ''}' style='--c:{ui.HIGH};--bg:{ui.rgba(ui.HIGH, .06)}'>"
           f"<span class='muted'>P(high-risk) ≥ {num(thr, 1)}</span><b style='color:{ui.HIGH}'>w = {num(w_off, 2)}</b></div>"
           "</div><div class='muted'>Đầu ra của quy tắc mô phỏng, không phải khuyến nghị giao dịch.</div>")
    with right, card("p180"):
        sec("P(high-risk) filtered", "180 ngày gần nhất")
        recent = filt.tail(180)
        fig = go.Figure()
        fig.add_hrect(y0=thr, y1=1.02, fillcolor=ui.rgba(ui.HIGH, .06), line_width=0, layer="below")
        fig.add_trace(go.Scatter(x=recent["date"], y=recent["prob_high_risk"], mode="lines", fill="tozeroy",
                                 fillcolor=ui.rgba(ui.HIGH, .12), line={"color": ui.HIGH, "width": 1.6},
                                 name="P(high-risk) filtered", hovertemplate="%{x|%d/%m/%Y}: %{y:.1%}<extra></extra>"))
        fig.add_hline(y=thr, line_dash="dot", line_color=ui.MUTED,
                      annotation_text=f"ngưỡng {num(thr, 1)}", annotation_position="top left",
                      annotation_font_color=ui.MUTED)
        fig.update_yaxes(range=[0, 1.02], tickformat=".0%")
        show(ui.chart_layout(fig, 420, legend=False))

    if b04 is not None:
        with card("states"):
            sec("Đặc điểm từng state trên train", "dùng để gán nhãn, giữ nguyên cho validation và test")
            t = b04[(b04["asset"] == asset) & (b04["period"] == "train")].sort_values("state_id")
            rows = [[ui.chip(colors[int(r.state_id)]) + ui.esc(state_name(r.state_id, label_map))
                     + (" <b style='color:#B91C1C'>· rủi ro cao</b>" if int(r.state_id) == risk else ""),
                     pct(r.mean_return, 3), pct(r.annualized_volatility, 1), pct(r.share, 1),
                     num(r.expected_duration, 1), num(r.mean_quote_volume_usdt / 1e9, 2)]
                    for r in t.itertuples()]
            md(ui.html_table(["State", "Mean log return/ngày", "Volatility năm hóa", "Tỷ trọng ngày (hard)",
                              "Thời lượng kỳ vọng (ngày)", "Mean quote volume (tỷ USDT)"], rows))


# ============================================================================= REGIME HISTORY


def page_history():
    if filt is None:
        st.info(f"{filt_error}. Chạy: `python scripts/complete_c2.py`")
        return
    label_map, k = meta["label_map"], meta["K"]
    colors = ui.state_colors(label_map)
    with card("controls"):
        c1, c2 = st.columns([3, 2])
        mode = c1.radio(
            "Loại xác suất để tô màu",
            ["Filtered (nhân quả, dùng cho backtest)", "Smoothed (hindsight, chỉ mô tả lịch sử)"],
            horizontal=True,
        )
        period = c2.segmented_control("Giai đoạn", ["Toàn bộ", "Train", "Validation", "Test"], default="Toàn bộ")
        use_smoothed = mode.startswith("Smoothed")
        source = _smoothed(asset, rs.mtime_key(rs.asset_files(asset)["smoothed"])) if use_smoothed else filt
        if source is None:
            st.info("Chưa có file smoothed của C2.")
            source, use_smoothed = filt, False
        if period and period != "Toàn bộ":
            source = source[source["split"] == period.lower()]
        rng = st.slider("Khoảng ngày", min_value=source["date"].min().date(), max_value=source["date"].max().date(),
                        value=(source["date"].min().date(), source["date"].max().date()), format="DD/MM/YYYY")
    if use_smoothed:
        st.warning("Smoothed dùng thông tin của cả các ngày sau t (hindsight). "
                   "Chỉ để đọc lịch sử, không được dùng làm tín hiệu giao dịch.", icon="🕰️")
    view = source[(source["date"].dt.date >= rng[0]) & (source["date"].dt.date <= rng[1])].reset_index(drop=True)
    price = px_df[(px_df["date"] >= pd.Timestamp(rng[0])) & (px_df["date"] <= pd.Timestamp(rng[1]))]
    kind = "smoothed (hindsight)" if use_smoothed else "filtered (nhân quả)"

    with card("price"):
        sec(f"Giá {asset} tô nền theo regime", f"state có xác suất {kind} lớn nhất, thang log")
        fig = go.Figure()
        for x0, x1, s in runs(view["date"], view["state_id"]):
            fig.add_vrect(x0=x0, x1=x1, fillcolor=colors[int(s)], opacity=0.16, line_width=0, layer="below")
        fig.add_trace(go.Scatter(x=price["date"], y=price["close"], mode="lines", line={"color": ui.INK, "width": 1.3},
                                 name="Giá đóng cửa", hovertemplate="%{x|%d/%m/%Y}: %{y:,.2f} USDT<extra></extra>"))
        for i in range(k):  # mục chú giải cho màu nền
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker={"size": 12, "color": colors[i],
                                     "symbol": "square"}, name=state_name(i, label_map)))
        lo, hi = float(price["close"].min()), float(price["close"].max())
        ticks = [m * 10 ** e for e in range(math.floor(math.log10(lo)), math.ceil(math.log10(hi)) + 1)
                 for m in (1, 2, 5) if lo * 0.8 <= m * 10 ** e <= hi * 1.25]
        fig.update_yaxes(type="log", title="USDT", tickvals=ticks, ticktext=[num(v, 0) for v in ticks])
        split_lines(fig, split_manifest.get("assets", {}).get(asset))
        show(ui.chart_layout(fig, 440))

    with card("stack"):
        sec(f"Xác suất {kind} từng state")
        fig = go.Figure()
        for i in range(k):
            fig.add_trace(go.Scatter(x=view["date"], y=view[f"prob_state_{i}"], stackgroup="p", mode="lines",
                                     line={"width": 0.4, "color": colors[i]}, fillcolor=ui.rgba(colors[i], .75),
                                     name=state_name(i, label_map),
                                     hovertemplate="%{x|%d/%m/%Y}: %{y:.1%}<extra>" + state_name(i, label_map)
                                     + "</extra>"))
        fig.update_yaxes(range=[0, 1], tickformat=".0%")
        ui.chart_layout(fig, 280)
        fig.update_layout(legend_traceorder="normal")
        show(fig)

    if b05 is not None:
        t = b05[b05["asset"] == asset]
        mat = t.pivot(index="from_state", columns="to_state", values="probability").sort_index()
        names = [state_name(i, label_map) for i in mat.index]
        left, right = st.columns([5, 4])
        with left, card("trans"):
            sec("Ma trận chuyển trạng thái P", "ước lượng trên train")
            fig = go.Figure(go.Heatmap(
                z=mat.values, x=names, y=names, colorscale=[[0, "#F8FAFC"], [1, ui.NAVY]], zmin=0, zmax=1,
                text=np.vectorize(lambda v: num(v, 3))(mat.values), texttemplate="%{text}", showscale=False,
                hovertemplate="từ %{y} sang %{x}: %{z:.4f}<extra></extra>"))
            fig.update_yaxes(autorange="reversed", title="Từ state (t)")
            fig.update_xaxes(title="Sang state (t+1)")
            show(ui.chart_layout(fig, 340, legend=False))
        with right, card("persist"):
            sec("Persistence và thời lượng")
            diag = t[t["from_state"] == t["to_state"]].sort_values("from_state")
            run_len = {}
            if b04 is not None:
                o = b04[b04["asset"] == asset].pivot(index="state_id", columns="period", values="mean_observed_run_days")
                run_len = {int(s): o.loc[s] for s in o.index}
            rows = []
            for r in diag.itertuples():
                obs = run_len.get(int(r.from_state))
                rows.append([ui.chip(colors[int(r.from_state)]) + ui.esc(state_name(r.from_state, label_map)),
                             num(r.persistence_model, 3), num(r.expected_duration, 1),
                             num(obs.get("train") if obs is not None else None, 1),
                             num(obs.get("test") if obs is not None else None, 1)])
            md(ui.html_table(["State", "p_kk", "E[D] (ngày)", "Đoạn TB train", "Đoạn TB test"], rows))
            md("<div class='muted' style='margin-top:8px'>E[D] = 1/(1 - p_kk) là kỳ vọng dưới giả định Markov. "
               "Đoạn TB là độ dài trung bình các đoạn liên tiếp cùng hard label filtered quan sát được; "
               "không bảo đảm regime thực tế kéo dài đúng chừng đó ngày.</div>")


# ============================================================================= MODEL COMPARISON


def page_models():
    md("<div class='note' style='--c:#1E3A8A'>Không có nhãn regime thật, nên các chỉ số dưới đây là log-score dự báo "
       "và thống kê mô tả, <b>không phải độ chính xác phân loại</b>.</div>")
    b03 = tload("B03")
    if b03 is None:
        run_hint("python scripts/complete_c2.py")
        return
    t = b03[b03["asset"] == asset].copy()
    sel = t[t["selected"]]
    left, right = st.columns([3, 2])
    with left, card("ksel"):
        sec("Chọn K theo validation", "test không tham gia")
        ok, bad = t[t["eligible"]], t[~t["eligible"]]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ok["K"], y=ok["val_avg_predictive_ll"], mode="markers", name="Fit hợp lệ",
                                 marker={"size": 10, "color": ui.rgba(ui.NAVY, .55), "line": {"width": 0}},
                                 text=ok["seed"], hovertemplate="K=%{x}, seed %{text}: %{y:.4f}<extra></extra>"))
        if not bad.empty:
            lo = ok["val_avg_predictive_ll"].min() if not ok.empty else 0
            fig.add_trace(go.Scatter(x=bad["K"], y=[lo] * len(bad), mode="markers", name="Fit bị loại (vẽ ở đáy)",
                                     marker={"size": 9, "symbol": "x", "color": "#94A3B8"}, text=bad["rejection_reason"],
                                     hovertemplate="K=%{x}: %{text}<extra></extra>"))
        fig.add_trace(go.Scatter(x=sel["K"], y=sel["val_avg_predictive_ll"], mode="markers", name="Được chọn",
                                 marker={"size": 20, "color": "rgba(0,0,0,0)", "line": {"color": ui.ACCENT, "width": 3}}))
        fig.update_xaxes(dtick=1, title="Số state K")
        fig.update_yaxes(title="Validation avg one-step log-score")
        show(ui.chart_layout(fig, 360))
        md("<div class='muted'>Quy tắc chọn: validation log-score lớn nhất trong các fit hợp lệ (hội tụ, mọi state "
           "≥ 2% train); hòa thì BIC train thấp hơn, rồi K nhỏ hơn, rồi seed nhỏ hơn.</div>")
    with right, card("chosen"):
        sec("Mô hình được chọn")
        if not sel.empty:
            r = sel.iloc[0]
            stab = tload("stability")
            st_row = None
            if stab is not None:
                s = stab[(stab["asset"] == asset) & (stab["K"] == r["K"])]
                st_row = s.iloc[0] if not s.empty else None
            rows = [["Số state K", str(int(r["K"]))], ["Seed", str(int(r["seed"]))],
                    ["Validation log-score", num(r["val_avg_predictive_ll"], 4)],
                    ["Train log-likelihood", num(r["train_ll"], 1)], ["BIC (train)", num(r["BIC"], 1)],
                    ["Tỷ trọng state nhỏ nhất (train)", pct(r["min_state_share"], 1)],
                    ["Số fit hợp lệ / tổng", f"{int(t['eligible'].sum())} / {len(t)}"]]
            if st_row is not None:
                color = ui.POS if st_row["status"] == "stable" else ui.WARN
                rows += [["ARI giữa seed (min / median)",
                          f"{num(st_row['ari_min'], 2)} / {num(st_row['ari_median'], 2)}"],
                         ["Độ ổn định seed", f"<b style='color:{color}'>{ui.esc(st_row['status'])}</b>"]]
            md(ui.html_table(["Chỉ tiêu", "Giá trị"], rows))
        with st.expander("Toàn bộ fit (bảng B03)"):
            cols = ["K", "seed", "converged", "eligible", "selected", "val_avg_predictive_ll", "train_ll", "AIC", "BIC",
                    "min_state_share", "rejection_reason"]
            st.dataframe(t[cols].round(4), width="stretch", hide_index=True, height=300)

    b07 = tload("B07")
    if b07 is not None:
        with card("hmmgmm"):
            sec("HMM so với GMM", "cùng dữ liệu, cùng chuẩn hóa")
            t7 = b07[b07["asset"] == asset].copy()
            c1, c2 = st.columns(2)
            for col, metric, title, fmt in (
                (c1, "avg_log_score", "Log-score trung bình mỗi quan sát (cao hơn tốt hơn)", 3),
                (c2, "flips_per_100d", "Số lần đổi state trên 100 ngày (thấp hơn bền hơn)", 1),
            ):
                fig = go.Figure()
                for method in ("HMM", "GMM"):
                    s = t7[t7["method"] == method]
                    fig.add_trace(go.Bar(x=[PERIOD_NAMES[p] for p in s["period"]], y=s[metric], name=method,
                                         marker_color=ui.METHOD_COLORS[method], marker_line_width=0,
                                         text=[num(v, fmt) for v in s[metric]], textposition="outside",
                                         cliponaxis=False))
                with col:
                    show(ui.chart_layout(fig, 320, title=title, barmode="group", bargap=0.35))
            rows = [[f"<b>{m}</b> · {PERIOD_NAMES[p]}", f"{dmy(a)} - {dmy(b)}", num(n, 0), num(ls, 4), num(fl, 0),
                     num(f100, 1), num(pe, 3)]
                    for m, p, a, b, n, ls, fl, f100, pe in t7[["method", "period", "start", "end", "n", "avg_log_score",
                                                               "flips", "flips_per_100d", "persistence"]].itertuples(
                                                                   index=False)]
            md(ui.html_table(["Phương pháp · giai đoạn", "Khoảng ngày", "Số ngày", "Avg log-score", "Flips",
                              "Flips/100 ngày", "Persistence"], rows))
            md("<div class='muted' style='margin-top:8px'>HMM: log p(X_t | X_1:t-1) theo lịch sử; GMM: mật độ biên "
               "log p(X_t). Hai mức điều kiện thông tin khác nhau; flips và persistence chỉ mang tính mô tả.</div>")

    left, right = st.columns(2)
    stab = tload("stability")
    if stab is not None:
        with left, card("stab"):
            sec("Độ ổn định giữa các seed", "chẩn đoán mô tả, không dùng để chọn mô hình")
            s = stab[stab["asset"] == asset].sort_values("K")
            rows, cls = [], []
            for r in s.itertuples():
                color = ui.POS if r.status == "stable" else ui.WARN
                rows.append([f"K = {int(r.K)}" + (" (chọn)" if int(r.K) == int(r.selected_K) else ""),
                             str(int(r.n_eligible_seeds)), num(r.ari_min, 2), num(r.ari_median, 2),
                             num(r.high_risk_disagreement_on_union_mean, 2),
                             f"<b style='color:{color}'>{ui.esc(r.status)}</b>"])
                cls.append("sel" if int(r.K) == int(r.selected_K) else "")
            md(ui.html_table(["K", "Seed hợp lệ", "ARI min", "ARI median", "Bất đồng high-risk", "Trạng thái"],
                             rows, cls))
            md("<div class='muted' style='margin-top:8px'>ARI giữa các seed trên validation, bất biến theo hoán vị ID "
               "state. Ngưỡng cảnh báo: median ARI ≥ 0,8 và min ARI ≥ 0,5 (quy ước vận hành của C2, không phải "
               "kiểm định thống kê).</div>")
    if b04 is not None and meta is not None:
        with right, card("share"):
            sec("Tỷ trọng state theo giai đoạn", "hard filtered, mapping khóa từ train")
            o = b04[b04["asset"] == asset].copy()
            piv = o.pivot(index="state_id", columns="period", values="share")[["train", "validation", "test"]]
            piv = piv.sort_index()
            piv.index = [state_name(sid, meta["label_map"]) for sid in piv.index]
            fig = go.Figure()
            for p in piv.columns:
                fig.add_trace(go.Bar(x=piv.index, y=piv[p], name=PERIOD_NAMES[p], marker_color=ui.PERIOD_COLORS[p],
                                     marker_line_width=0, text=[pct(v, 0) for v in piv[p]], textposition="outside",
                                     cliponaxis=False))
            fig.update_yaxes(tickformat=".0%")
            show(ui.chart_layout(fig, 330, barmode="group", bargap=0.3))

    win = tload("window_sensitivity")
    if win is not None and "asset" in win:
        with card("window"):
            sec("Độ nhạy theo cửa sổ RV", "RV14 / RV21 / RV30")
            st.dataframe(win[win["asset"] == asset], width="stretch", hide_index=True)


# ============================================================================= PORTFOLIO & RISK


def page_portfolio():
    nav = tload("nav")
    if b08 is None or nav is None or not bt_manifest or bt_manifest.get("status") != "complete":
        run_hint("python run_pipeline.py --steps backtest")
        return
    st.warning("**Kết quả mô phỏng (simulated).** Phí giả định 0,10% × turnover, không gồm slippage, spread, thuế; "
               "lợi suất tiền mặt 0; Sharpe sau phí với rf = 0; năm hóa 365 ngày. Không phải khuyến nghị đầu tư, "
               "không suy ra quan hệ nhân quả từ một lần backtest.", icon="🧪")
    by_name = {v: k for k, v in SCENARIO_NAMES.items()}
    c1, c2 = st.columns([2, 3])
    scenario = by_name[c1.selectbox("Kịch bản", list(by_name))]
    n = nav[(nav["asset"] == asset) & (nav["scenario"] == scenario)].copy()
    n["date"] = pd.to_datetime(n["date_utc"])
    b = b08[(b08["asset"] == asset) & (b08["scenario"] == scenario)].set_index("strategy")
    ra, bh = b.loc["regime_aware"], b.loc["buy_and_hold"]
    c2.markdown(
        f"<div class='muted' style='padding-top:30px'>Test {dmy(ra['start'])} - {dmy(ra['end'])} "
        f"({int(ra['n_days'])} ngày) · turnover <code>{ra['turnover_method']}</code> · lợi suất "
        f"<code>{ra['return_basis']}</code> · mô hình <code>{ra['model_run_id']}</code></div>", unsafe_allow_html=True)

    def vs(diff_text, x, better_high=True):
        return f"<span style='color:{ui.sign_color(x, better_high)}'>{diff_text}</span> so với B&H"

    kpis([
        ("NAV cuối (regime-aware)", num(ra["final_nav"], 4),
         f"B&H {num(bh['final_nav'], 4)} · " + vs(num(ra["final_nav"] - bh["final_nav"], 4, sign=True),
                                                 ra["final_nav"] - bh["final_nav"]),
         ui.sign_color(ra["final_nav"] - 1)),
        ("CAGR", pct(ra["CAGR"]), vs(ui.pp(ra["CAGR"] - bh["CAGR"]), ra["CAGR"] - bh["CAGR"]),
         ui.sign_color(ra["CAGR"])),
        ("Sharpe (sau phí)", num(ra["Sharpe"], 3), vs(num(ra["Sharpe"] - bh["Sharpe"], 3, sign=True),
                                                      ra["Sharpe"] - bh["Sharpe"]),
         ui.sign_color(ra["Sharpe"])),
        ("Max drawdown", pct(ra["max_drawdown"]),
         vs(ui.pp(ra["max_drawdown"] - bh["max_drawdown"]), ra["max_drawdown"] - bh["max_drawdown"]), ui.NEG),
        ("Ngày phòng thủ (w < 1)", pct(ra["share_days_risk_off"], 1),
         f"{round(ra['share_days_risk_off'] * ra['n_days'])} / {int(ra['n_days'])} ngày test", ui.WARN),
    ])
    st.write("")

    ra_nav = n[n["strategy"] == "regime_aware"].reset_index(drop=True)
    off = [(x0, x1) for x0, x1, v in runs(ra_nav["date"], ra_nav["weight"] < 1) if v]
    with card("nav"):
        sec("NAV tích lũy", "bắt đầu = 1, vùng vàng là ngày quy tắc ở chế độ phòng thủ")
        fig = go.Figure()
        for x0, x1 in off:
            fig.add_vrect(x0=x0, x1=x1, fillcolor=ui.rgba(ui.ACCENT, .28), line_width=0, layer="below")
        for s in ("buy_and_hold", "regime_aware"):
            d = n[n["strategy"] == s]
            fig.add_trace(go.Scatter(x=d["date"], y=d["nav"], mode="lines", name=STRATEGY_NAMES[s],
                                     line={"color": ui.STRATEGY_COLORS[s], "width": 2 if s == "regime_aware" else 1.6},
                                     hovertemplate="%{x|%d/%m/%Y}: %{y:.4f}<extra>" + STRATEGY_NAMES[s] + "</extra>"))
        fig.add_hline(y=1, line_dash="dot", line_color="#94A3B8")
        fig.update_yaxes(title="NAV")
        show(ui.chart_layout(fig, 400, hovermode="x unified"))

    left, right = st.columns(2)
    with left, card("dd"):
        sec("Drawdown")
        fig = go.Figure()
        for s in ("buy_and_hold", "regime_aware"):
            d = n[n["strategy"] == s]
            fig.add_trace(go.Scatter(x=d["date"], y=d["drawdown"], mode="lines", name=STRATEGY_NAMES[s],
                                     fill="tozeroy" if s == "buy_and_hold" else None,
                                     fillcolor=ui.rgba("#94A3B8", .18),
                                     line={"color": ui.STRATEGY_COLORS[s], "width": 1.4},
                                     hovertemplate="%{x|%d/%m/%Y}: %{y:.1%}<extra>" + STRATEGY_NAMES[s] + "</extra>"))
        fig.update_yaxes(tickformat=".0%")
        show(ui.chart_layout(fig, 300, hovermode="x unified"))
    with right, card("signal"):
        sec("Tín hiệu và tỷ trọng regime-aware")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ra_nav["date"], y=ra_nav["prob_high_risk"], mode="lines",
                                 name="P(high-risk) filtered ngày t", line={"color": ui.HIGH, "width": 1},
                                 fill="tozeroy", fillcolor=ui.rgba(ui.HIGH, .10)))
        fig.add_trace(go.Scatter(x=ra_nav["date"], y=ra_nav["weight"], mode="lines", name="Tỷ trọng w_t",
                                 line_shape="hv", line={"color": ui.NAVY, "width": 1.8}))
        fig.update_yaxes(range=[0, 1.05], tickformat=".0%")
        show(ui.chart_layout(fig, 300, hovermode="x unified"))

    with card("cmp"):
        sec("So sánh hai chiến lược", SCENARIO_SHORT[scenario])

        def row(label, f, key, better_high=True, diff="num", d_digits=4):
            a, b_ = ra[key], bh[key]
            d = a - b_
            dt = ui.pp(d) if diff == "pp" else num(d, d_digits, sign=True)
            return [label, f(b_), f(a), f"<span style='color:{ui.sign_color(d, better_high)}'>{dt}</span>"]

        rows = [["Lợi nhuận"], row("NAV cuối", lambda v: num(v, 4), "final_nav"),
                row("Lợi nhuận tích lũy", pct, "cumulative_return", diff="pp"),
                row("CAGR", pct, "CAGR", diff="pp"),
                ["Rủi ro"], row("Volatility năm hóa", pct, "ann_volatility", better_high=False, diff="pp"),
                row("Max drawdown", pct, "max_drawdown", diff="pp"),
                row("Sharpe (sau phí, rf = 0)", lambda v: num(v, 3), "Sharpe", d_digits=3),
                ["Giao dịch"], row("Turnover tích lũy", lambda v: num(v, 3), "turnover", better_high=False, d_digits=3),
                row("Tổng chi phí giao dịch", pct, "total_cost", better_high=False, diff="pp"),
                row("Tỷ trọng trung bình", lambda v: num(v, 3), "avg_weight", d_digits=3),
                row("Số lần tái cân bằng", lambda v: num(v, 0), "n_rebalances", better_high=False, d_digits=0)]
        cls = ["grp" if len(r) == 1 else ("key" if r[0] in ("NAV cuối", "Max drawdown") else "") for r in rows]
        md(ui.html_table(["Chỉ tiêu", "Buy-and-Hold", "Regime-aware", "Chênh lệch"], rows, cls))

    with card("b08"):
        sec("Bảng hiệu suất B08", "mọi kịch bản, dòng tô vàng là kịch bản đang xem")
        view = b08[b08["asset"] == asset]
        rows, cls = [], []
        for r in view.itertuples():
            rows.append([f"{SCENARIO_SHORT.get(r.scenario, r.scenario)} · {STRATEGY_NAMES[r.strategy]}",
                         num(r.final_nav, 4), pct(r.cumulative_return), pct(r.CAGR), pct(r.ann_volatility),
                         num(r.Sharpe, 3), pct(r.max_drawdown), num(r.turnover, 3), pct(r.total_cost),
                         pct(r.share_days_risk_off, 1)])
            cls.append("sel" if r.scenario == scenario else "")
        md(ui.html_table(["Kịch bản · chiến lược", "NAV cuối", "Tích lũy", "CAGR", "Volatility", "Sharpe",
                          "Max DD", "Turnover", "Chi phí", "Ngày phòng thủ"], rows, cls))
        md("<div class='muted' style='margin-top:8px'>Baseline close-to-close là kết quả chính theo Methodology. "
           "Hai kiểm tra chéo báo riêng, không thay baseline và không được gọi là giao dịch thực.</div>")
        dl = st.columns([1, 1, 3])
        dl[0].download_button("Tải B08_portfolio.csv", rs.FILES["B08"].read_bytes(), "B08_portfolio.csv", "text/csv",
                              width="stretch")
        dl[1].download_button("Tải backtest_nav.csv", rs.FILES["nav"].read_bytes(), "backtest_nav.csv", "text/csv",
                              width="stretch")
    with st.expander("Giả định và provenance của backtest"):
        for a in bt_manifest.get("assumptions", []):
            st.markdown(f"- {a}")
        st.code(json.dumps(bt_manifest.get("assets", {}).get(asset, {}), ensure_ascii=False, indent=2), language="json")


# ============================================================================= điều hướng

{"Overview": page_overview, "Regime History": page_history, "Model Comparison": page_models,
 "Portfolio & Risk": page_portfolio}[page]()

md("<div class='foot'>Crypto Market Regime Radar · giá: Binance Spot, nến ngày UTC · mô hình: Gaussian HMM "
   "(hmmlearn), GMM (scikit-learn) · dashboard chỉ đọc file kết quả đã xuất, không huấn luyện lại · "
   "mô phỏng nghiên cứu, không phải khuyến nghị đầu tư</div>")
