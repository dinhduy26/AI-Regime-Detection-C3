"""Crypto Market Regime Radar - dashboard Streamlit 4 tab (C3-01, C3-08..C3-11, C3-13).

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

st.set_page_config(page_title="Crypto Market Regime Radar", page_icon="📡", layout="wide")

HIGH, LOW = "#dc2626", "#16a34a"
OTHER = ["#f59e0b", "#8b5cf6", "#0ea5e9", "#64748b"]
STRATEGY_NAMES = {"buy_and_hold": "Buy-and-Hold", "regime_aware": "Regime-aware"}
STRATEGY_COLORS = {"buy_and_hold": "#6b7280", "regime_aware": "#2563eb"}
SCENARIO_NAMES = {
    "baseline_close_to_close": "Baseline: close t -> close t+1 (bắt buộc)",
    "crosscheck_drift_adjusted": "Kiểm tra chéo: turnover drift-adjusted",
    "crosscheck_next_open": "Kiểm tra chéo: khớp open t+1 -> open t+2",
}
PERIOD_NAMES = {"train": "Train", "validation": "Validation", "test": "Test"}


# ----------------------------------------------------------------------------- đọc dữ liệu (cache)


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


def pct(x, digits=2):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x * 100:.{digits}f}%"


def num(x, digits=3):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{digits}f}"


def state_colors(label_map):
    colors, i = {}, 0
    for sid in sorted(label_map, key=int):
        label = label_map[sid]
        if label == "High-Vol":
            colors[int(sid)] = HIGH
        elif label == "Low-Vol":
            colors[int(sid)] = LOW
        else:
            colors[int(sid)] = OTHER[i % len(OTHER)]
            i += 1
    return colors


def state_name(sid, label_map):
    return f"{label_map[str(sid)]} (state {sid})"


def runs(dates, states):
    """Các đoạn ngày liên tiếp cùng state, dùng để tô nền biểu đồ giá."""
    out, start = [], 0
    s = np.asarray(states)
    for i in range(1, len(s) + 1):
        if i == len(s) or s[i] != s[start]:
            out.append((dates.iloc[start], dates.iloc[i - 1] + pd.Timedelta(days=1), int(s[start])))
            start = i
    return out


def layout(fig, height=380, title=None, **kw):
    # Tiêu đề ở mép trên khung, chú giải nằm ngay trên vùng vẽ: không đè lên nhau.
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=80 if title else 40, b=10), hovermode="x unified",
        title=dict(text=title, y=0.98, yanchor="top", x=0, xanchor="left") if title else None,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0), **kw,
    )
    return fig


def show(fig):
    st.plotly_chart(fig, width="stretch")


def table(df, **kw):
    st.dataframe(df, width="stretch", hide_index=True, **kw)


def split_lines(fig, split):
    for key, name in (("train", "hết train"), ("validation", "hết validation")):
        end = split.get(key, {}).get("end") if split else None
        if end:
            fig.add_vline(x=pd.Timestamp(end) + pd.Timedelta(hours=12), line_dash="dot", line_color="#9ca3af")
            fig.add_annotation(x=pd.Timestamp(end), y=1, yref="paper", text=name, showarrow=False,
                               font=dict(size=10, color="#6b7280"), xanchor="right", yanchor="bottom")


# ----------------------------------------------------------------------------- khung trang

st.title("📡 Crypto Market Regime Radar")
st.caption(
    "Nhận diện trạng thái thị trường BTC/ETH bằng Gaussian HMM (C2), đối chứng GMM, "
    "và mô phỏng phân bổ phòng thủ theo regime (C3). Mô phỏng nghiên cứu, **không phải khuyến nghị đầu tư**."
)

status = rs.file_status()
missing_groups = {g: s for g, s in status.items() if s["missing"]}
if missing_groups:
    with st.expander("⚠️ Thiếu một số file kết quả, xem cách khắc phục", expanded="Dữ liệu C1" in missing_groups):
        for group, s in missing_groups.items():
            st.markdown(f"**{group}** thiếu: " + ", ".join(f"`{m}`" for m in s["missing"]))
            st.code(s["fix"], language="bash")
if "Dữ liệu C1" in missing_groups:
    st.error("Chưa có dữ liệu C1 nên dashboard chưa thể hiển thị. Chạy lệnh ở trên rồi tải lại trang.")
    st.stop()

split_manifest = jload("split_manifest") or {}
model_manifest = jload("model_manifest")
bt_manifest = jload("backtest_manifest")
data_through = split_manifest.get("data_through_utc")

with st.sidebar:
    st.header("Bộ lọc")
    asset = st.radio("Tài sản", rs.assets(), horizontal=True)
    asset_through = data_through.get(asset) if isinstance(data_through, dict) else data_through
    st.markdown(f"**Dữ liệu {asset} đến hết ngày:** `{asset_through}` (UTC)")
    st.caption(
        "Last available daily observation: nến ngày UTC đã đóng gần nhất. "
        "Cập nhật theo ngày, **không phải realtime**."
    )
    if model_manifest:
        st.markdown("**Mô hình C2**")
        st.caption(
            f"Trạng thái: `{model_manifest.get('status')}` · chất lượng: `{model_manifest.get('quality_status')}`  \n"
            f"Config: v{model_manifest.get('config_version')} · hash `{model_manifest.get('config_hash')}`"
        )
    if bt_manifest:
        st.markdown("**Backtest C3**")
        st.caption(
            f"Trạng thái: `{bt_manifest.get('status')}` · backtest config v{bt_manifest.get('backtest_config_version')}  \n"
            f"Tạo lúc {str(bt_manifest.get('created_at_utc', ''))[:19]} UTC"
        )

px_df = _prices(asset, rs.mtime_key(rs.FILES["prices"]))
feat = _features(asset, rs.mtime_key(rs.asset_files(asset)["features"]))
filt, meta, notes, filt_error = _filtered(asset, rs.mtime_key(rs.asset_files(asset)["filtered"]))

if meta is not None and meta.get("seed_stability_status") != "stable":
    st.warning(
        f"**{asset}: mô hình C2 được đánh dấu `{meta.get('seed_stability_status')}`** theo chẩn đoán độ ổn định "
        "giữa các seed (nhãn regime phụ thuộc khởi tạo). Kết quả chỉ dùng cho nghiên cứu, cần đọc kèm giới hạn này.",
        icon="⚠️",
    )

tab1, tab2, tab3, tab4 = st.tabs(["1 · Overview", "2 · Regime History", "3 · Model Comparison", "4 · Portfolio & Risk"])

# ============================================================================= TAB 1 OVERVIEW
with tab1:
    st.subheader(f"{asset}: tổng quan tại ngày dữ liệu cuối")
    last_px = px_df.iloc[-1]
    last_feat = feat.iloc[-1] if feat is not None else None
    c = st.columns(4)
    c[0].metric("Giá đóng cửa (USDT)", f"{last_px['close']:,.2f}", help=f"Nến ngày UTC {last_px['date']:%Y-%m-%d}")
    if last_feat is not None:
        c[1].metric("Lợi suất ngày cuối", pct(last_feat["simple_return"]), help="Simple return C_t/C_{t-1} - 1")
        c[2].metric("RV21 năm hóa", pct(last_feat["rv21"] * math.sqrt(365), 1),
                    help="Độ lệch chuẩn 21 log return gần nhất x sqrt(365); chỉ để mô tả, HMM dùng bản chưa năm hóa")
    if filt is None:
        st.info(f"{filt_error}. Chạy: `python scripts/complete_c2.py`")
    else:
        label_map, k, risk = meta["label_map"], meta["K"], meta["high_risk_state"]
        colors = state_colors(label_map)
        last = filt.iloc[-1]
        c[3].metric("Regime hiện tại (filtered)", label_map[str(int(last["state_id"]))],
                    help="State có xác suất filtered lớn nhất tại ngày cuối")
        st.caption(
            f"Mô hình `{meta['model_run_id']}` · K = {k} · seed {meta['seed']} · "
            f"state rủi ro cao = {state_name(risk, label_map)} (state có volatility train lớn nhất). "
            "Nhãn chỉ là mô tả theo thống kê train, không phải ground truth."
        )
        left, right = st.columns([1, 1])
        with left:
            probs = [float(last[f"prob_state_{i}"]) for i in range(k)]
            fig = go.Figure(go.Bar(
                x=[state_name(i, label_map) for i in range(k)], y=probs,
                marker_color=[colors[i] for i in range(k)], text=[pct(p, 1) for p in probs], textposition="outside",
            ))
            fig.update_yaxes(range=[0, 1.1], tickformat=".0%", title="P(S_t = k | X_1:t)")
            show(layout(fig, 340, title=f"Xác suất filtered từng state, ngày {last['date']:%Y-%m-%d}"))
        with right:
            b08 = tload("B08")  # quy tắc lấy từ kết quả backtest đã xuất, mặc định theo Methodology
            thr, w_on, w_off = 0.5, 1.0, 0.25
            if b08 is not None and not b08.empty:
                r0 = b08.iloc[0]
                thr, w_on, w_off = r0["risk_threshold"], r0["weight_risk_on"], r0["weight_risk_off"]
            p_now, p_next = float(last["prob_high_risk"]), float(last["prob_high_risk_next_day"])
            cc = st.columns(2)
            cc[0].metric("P(high-risk) filtered", pct(p_now, 1))
            cc[1].metric("P(high-risk) ngày kế tiếp", pct(p_next, 1), help="alpha_t x P: phân phối state ngày t+1 "
                         "theo ma trận chuyển; không phải dự báo giá")
            w_rule = w_off if p_now >= thr else w_on
            st.markdown(
                f"Theo **quy tắc nghiên cứu chỉ định trước** (w = {w_on:g} nếu P(high-risk) < {thr:g}, "
                f"ngược lại {w_off:g}), tỷ trọng mô phỏng cho kỳ kế tiếp là **{w_rule:g}**. "
                "Đây là đầu ra của quy tắc mô phỏng, không phải khuyến nghị giao dịch."
            )
            recent = filt.tail(180)
            fig = go.Figure(go.Scatter(x=recent["date"], y=recent["prob_high_risk"], mode="lines",
                                       line=dict(color=HIGH), name="P(high-risk) filtered"))
            fig.add_hline(y=thr, line_dash="dot", line_color="#6b7280", annotation_text=f"ngưỡng {thr:g}")
            fig.update_yaxes(range=[0, 1.02], tickformat=".0%")
            show(layout(fig, 250, title="P(high-risk) filtered, 180 ngày gần nhất"))
        b04 = tload("B04")
        if b04 is not None:
            st.markdown("**Đặc điểm từng state trên train** (dùng để gán nhãn, giữ nguyên cho validation/test)")
            t = b04[(b04["asset"] == asset) & (b04["period"] == "train")].copy()
            t["state"] = [state_name(s, label_map) for s in t["state_id"]]
            table(pd.DataFrame({
                "State": t["state"],
                "Mean log return/ngày": t["mean_return"].map(lambda v: pct(v, 3)),
                "Volatility năm hóa": t["annualized_volatility"].map(lambda v: pct(v, 1)),
                "Tỷ trọng ngày (hard)": t["share"].map(lambda v: pct(v, 1)),
                "Thời lượng kỳ vọng (ngày)": t["expected_duration"].map(lambda v: num(v, 1)),
                "Mean quote volume (tỷ USDT)": (t["mean_quote_volume_usdt"] / 1e9).map(lambda v: num(v, 2)),
            }))

# ============================================================================= TAB 2 REGIME HISTORY
with tab2:
    st.subheader(f"{asset}: lịch sử giá và regime")
    if filt is None:
        st.info(f"{filt_error}. Chạy: `python scripts/complete_c2.py`")
    else:
        label_map, k = meta["label_map"], meta["K"]
        colors = state_colors(label_map)
        mode = st.radio(
            "Loại xác suất để tô màu",
            ["Filtered (nhân quả, dùng cho backtest)", "Smoothed (hindsight, chỉ mô tả lịch sử)"],
            horizontal=True,
        )
        use_smoothed = mode.startswith("Smoothed")
        source = _smoothed(asset, rs.mtime_key(rs.asset_files(asset)["smoothed"])) if use_smoothed else filt
        if source is None:
            st.info("Chưa có file smoothed của C2.")
            source = filt
            use_smoothed = False
        if use_smoothed:
            st.warning("Smoothed dùng thông tin của cả các ngày sau t (hindsight). "
                       "Chỉ để đọc lịch sử, không được dùng làm tín hiệu giao dịch.", icon="🕰️")
        period =st.segmented_control("Giai đoạn", ["Toàn bộ", "Train", "Validation", "Test"], default="Toàn bộ")
        if period and period != "Toàn bộ":
            source = source[source["split"] == period.lower()]
        rng = st.slider("Khoảng ngày", min_value=source["date"].min().date(), max_value=source["date"].max().date(),
                        value=(source["date"].min().date(), source["date"].max().date()), format="YYYY-MM-DD")
        view = source[(source["date"].dt.date >= rng[0]) & (source["date"].dt.date <= rng[1])].reset_index(drop=True)
        price = px_df[(px_df["date"] >= pd.Timestamp(rng[0])) & (px_df["date"] <= pd.Timestamp(rng[1]))]

        fig = go.Figure()
        for x0, x1, s in runs(view["date"], view["state_id"]):
            fig.add_vrect(x0=x0, x1=x1, fillcolor=colors[s], opacity=0.18, line_width=0, layer="below")
        fig.add_trace(go.Scatter(x=price["date"], y=price["close"], mode="lines", line=dict(color="#111827", width=1.3),
                                 name="Giá đóng cửa", hovertemplate="%{y:,.2f} USDT"))
        for i in range(k):  # mục chú giải cho màu nền
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(size=12, color=colors[i]),
                                     name=state_name(i, label_map)))
        fig.update_yaxes(type="log", title="USDT (thang log)")
        split_lines(fig, split_manifest.get("assets", {}).get(asset))
        kind = "smoothed (hindsight)" if use_smoothed else "filtered (nhân quả)"
        show(layout(fig, 430, title=f"Giá {asset} tô nền theo state có xác suất {kind} lớn nhất"))

        fig = go.Figure()
        for i in range(k):
            fig.add_trace(go.Scatter(x=view["date"], y=view[f"prob_state_{i}"], stackgroup="p", mode="lines",
                                     line=dict(width=0.5, color=colors[i]), name=state_name(i, label_map)))
        fig.update_yaxes(range=[0, 1], tickformat=".0%")
        show(layout(fig, 300, title=f"Xác suất {kind} từng state"))

        b05 = tload("B05")
        if b05 is not None:
            t = b05[b05["asset"] == asset]
            mat = t.pivot(index="from_state", columns="to_state", values="probability").sort_index()
            names = [state_name(i, label_map) for i in mat.index]
            left, right = st.columns([3, 2])
            with left:
                fig = go.Figure(go.Heatmap(z=mat.values, x=names, y=names, colorscale="Blues", zmin=0, zmax=1,
                                           text=np.vectorize(lambda v: f"{v:.3f}")(mat.values), texttemplate="%{text}",
                                           hovertemplate="từ %{y} sang %{x}: %{z:.4f}<extra></extra>"))
                fig.update_yaxes(autorange="reversed", title="Từ state (t)")
                fig.update_xaxes(title="Sang state (t+1)")
                show(layout(fig, 360, title="Ma trận chuyển trạng thái P (ước lượng trên train)"))
            with right:
                diag = t[t["from_state"] == t["to_state"]]
                st.markdown("**Persistence và thời lượng kỳ vọng**")
                table(pd.DataFrame({
                    "State": [state_name(s, label_map) for s in diag["from_state"]],
                    "p_kk": diag["persistence_model"].map(lambda v: num(v, 4)),
                    "E[D] = 1/(1-p_kk), ngày": diag["expected_duration"].map(lambda v: num(v, 1)),
                }))
                st.caption("Thời lượng là kỳ vọng dưới giả định Markov, không bảo đảm regime thực tế kéo dài đúng "
                           "chừng đó ngày.")
                b04 = tload("B04")
                if b04 is not None:
                    o = b04[b04["asset"] == asset]
                    st.markdown("**Độ dài đoạn quan sát trung bình (ngày, hard label filtered)**")
                    piv = o.pivot(index="state_id", columns="period", values="mean_observed_run_days")
                    piv = piv[[c for c in ("train", "validation", "test") if c in piv]]
                    piv.index = [state_name(s, label_map) for s in piv.index]
                    st.dataframe(piv.round(1), width="stretch")

# ============================================================================= TAB 3 MODEL COMPARISON
with tab3:
    st.subheader(f"{asset}: lựa chọn mô hình, HMM so với GMM, độ ổn định")
    st.caption("Không có nhãn regime thật, nên các chỉ số dưới đây là log-score dự báo và thống kê mô tả, "
               "**không phải độ chính xác phân loại**.")
    b03 = tload("B03")
    if b03 is None:
        st.info("Chưa có B03. Chạy: `python scripts/complete_c2.py`")
    else:
        t = b03[b03["asset"] == asset].copy()
        left, right = st.columns([2, 3])
        with left:
            fig = go.Figure()
            ok = t[t["eligible"]]
            bad = t[~t["eligible"]]
            fig.add_trace(go.Scatter(x=ok["K"], y=ok["val_avg_predictive_ll"], mode="markers", name="Fit hợp lệ",
                                     marker=dict(size=10, color="#2563eb"), text=ok["seed"],
                                     hovertemplate="K=%{x}, seed %{text}: %{y:.4f}<extra></extra>"))
            sel = t[t["selected"]]
            fig.add_trace(go.Scatter(x=sel["K"], y=sel["val_avg_predictive_ll"], mode="markers", name="Được chọn",
                                     marker=dict(size=18, color="rgba(0,0,0,0)", line=dict(color="#dc2626", width=2.5))))
            if not bad.empty:
                fig.add_annotation(x=bad["K"].iloc[0], y=ok["val_avg_predictive_ll"].min(), showarrow=False,
                                   text=f"K={','.join(map(str, sorted(bad['K'].unique())))}: có fit bị loại",
                                   font=dict(size=10, color="#6b7280"), yanchor="top")
            fig.update_xaxes(dtick=1, title="Số state K")
            fig.update_yaxes(title="Validation avg one-step log-score")
            show(layout(fig, 360, title="Chọn K theo validation (test không tham gia)"))
        with right:
            cols = ["K", "seed", "converged", "eligible", "selected", "val_avg_predictive_ll", "train_ll", "AIC", "BIC",
                    "min_state_share", "rejection_reason"]
            table(t[cols].round(4), height=360)
        st.caption("Quy tắc chọn: validation log-score lớn nhất trong các fit hợp lệ (hội tụ, mọi state >= 2% train); "
                   "hòa thì BIC train thấp hơn, rồi K nhỏ hơn, rồi seed nhỏ hơn.")

    b07 = tload("B07")
    if b07 is not None:
        st.markdown("#### HMM so với GMM trên cùng dữ liệu và cùng chuẩn hóa")
        t = b07[b07["asset"] == asset].copy()
        left, right = st.columns(2)
        with left:
            fig = go.Figure()
            for method, color in (("HMM", "#2563eb"), ("GMM", "#f59e0b")):
                s = t[t["method"] == method]
                fig.add_trace(go.Bar(x=[PERIOD_NAMES[p] for p in s["period"]], y=s["avg_log_score"], name=method,
                                     marker_color=color, text=s["avg_log_score"].round(3), textposition="outside"))
            fig.update_yaxes(title="Avg log-score (cao hơn tốt hơn)")
            show(layout(fig, 340, title="Log-score trung bình mỗi quan sát", barmode="group"))
        with right:
            fig = go.Figure()
            for method, color in (("HMM", "#2563eb"), ("GMM", "#f59e0b")):
                s = t[t["method"] == method]
                fig.add_trace(go.Bar(x=[PERIOD_NAMES[p] for p in s["period"]], y=s["flips_per_100d"], name=method,
                                     marker_color=color, text=s["flips_per_100d"].round(1), textposition="outside"))
            fig.update_yaxes(title="Số lần đổi state / 100 ngày")
            show(layout(fig, 340, title="Flips trên 100 ngày (thấp hơn = bền hơn)", barmode="group"))
        st.caption("HMM: log p(X_t | X_1:t-1) theo lịch sử; GMM: mật độ biên log p(X_t). Hai mức điều kiện thông tin "
                   "khác nhau; flips/persistence chỉ mang tính mô tả.")
        table(t[["method", "period", "start", "end", "n", "avg_log_score", "flips", "flips_per_100d",
                 "persistence"]].round(4))

    stab = tload("stability")
    if stab is not None:
        st.markdown("#### Độ ổn định giữa các seed (chẩn đoán mô tả, không dùng để chọn mô hình)")
        s = stab[stab["asset"] == asset]
        table(s[["K", "selected_K", "n_eligible_seeds", "ari_min", "ari_median", "ari_mean",
                 "high_risk_disagreement_on_union_mean", "status", "reason"]].round(4))
        st.caption("ARI giữa các seed trên validation, bất biến theo hoán vị ID state. Ngưỡng cảnh báo: median ARI "
                   ">= 0.8 và min ARI >= 0.5 (quy ước vận hành của C2, không phải kiểm định thống kê).")
    win = tload("window_sensitivity")
    if win is not None and "asset" in win:
        st.markdown("#### Độ nhạy theo cửa sổ RV (RV14 / RV21 / RV30)")
        table(win[win["asset"] == asset])
    else:
        st.caption("Chưa có bảng độ nhạy cửa sổ RV. Chạy `python -m src.models.diagnostics` (C2) nếu cần.")

    b04 = tload("B04")
    if b04 is not None and meta is not None:
        st.markdown("#### Tỷ trọng state theo giai đoạn (mô tả ngoài mẫu với mapping khóa từ train)")
        o = b04[b04["asset"] == asset].copy()
        o["State"] = [state_name(sid, meta["label_map"]) for sid in o["state_id"]]
        piv = o.pivot(index="State", columns="period", values="share")[["train", "validation", "test"]]
        fig = go.Figure()
        period_colors = {"train": "#1e3a8a", "validation": "#60a5fa", "test": "#9ca3af"}
        for p in piv.columns:
            fig.add_trace(go.Bar(x=piv.index, y=piv[p], name=PERIOD_NAMES[p], text=[pct(v, 1) for v in piv[p]],
                                 textposition="outside", marker_color=period_colors[p]))
        fig.update_yaxes(tickformat=".0%", title="Tỷ trọng ngày (hard filtered)")
        show(layout(fig, 340, barmode="group"))

# ============================================================================= TAB 4 PORTFOLIO & RISK
with tab4:
    st.subheader(f"{asset}: mô phỏng danh mục trên tập test")
    b08, nav = tload("B08"), tload("nav")
    if b08 is None or nav is None or not bt_manifest or bt_manifest.get("status") != "complete":
        st.info("Chưa có kết quả backtest C3 hoàn chỉnh. Chạy: `python run_pipeline.py --steps backtest`")
    else:
        st.warning("**Kết quả mô phỏng (simulated).** Phí giả định 0,10% x turnover, không gồm slippage, spread, thuế; "
                   "lợi suất tiền mặt 0; Sharpe sau phí với rf = 0; năm hóa 365 ngày. Không phải khuyến nghị đầu tư, "
                   "không suy ra quan hệ nhân quả từ một lần backtest.", icon="🧪")
        by_name = {v: k for k, v in SCENARIO_NAMES.items()}
        scenario = by_name[st.selectbox("Kịch bản", list(by_name))]
        n = nav[(nav["asset"] == asset) & (nav["scenario"] == scenario)].copy()
        n["date"] = pd.to_datetime(n["date_utc"])
        b = b08[(b08["asset"] == asset) & (b08["scenario"] == scenario)].set_index("strategy")
        ra, bh = b.loc["regime_aware"], b.loc["buy_and_hold"]
        st.caption(f"Giai đoạn test {ra['start']} đến {ra['end']} ({int(ra['n_days'])} ngày) · turnover: "
                   f"`{ra['turnover_method']}` · cơ sở lợi suất: `{ra['return_basis']}` · mô hình `{ra['model_run_id']}`")
        c = st.columns(5)
        c[0].metric("NAV cuối (regime-aware)", num(ra["final_nav"], 4), delta=f"{(ra['final_nav'] - bh['final_nav']):+.4f} so với B&H")
        c[1].metric("CAGR", pct(ra["CAGR"]), delta=f"{(ra['CAGR'] - bh['CAGR']) * 100:+.2f} điểm %")
        c[2].metric("Sharpe (sau phí)", num(ra["Sharpe"], 3), delta=f"{ra['Sharpe'] - bh['Sharpe']:+.3f}")
        c[3].metric("Max drawdown", pct(ra["max_drawdown"]),
                    delta=f"{(ra['max_drawdown'] - bh['max_drawdown']) * 100:+.2f} điểm %")
        c[4].metric("Ngày phòng thủ (w < 1)", pct(ra["share_days_risk_off"], 1),
                    help="Tỷ lệ ngày test mà quy tắc giảm tỷ trọng xuống mức phòng thủ")

        fig = go.Figure()
        for s in ("buy_and_hold", "regime_aware"):
            d = n[n["strategy"] == s]
            fig.add_trace(go.Scatter(x=d["date"], y=d["nav"], mode="lines", name=STRATEGY_NAMES[s],
                                     line=dict(color=STRATEGY_COLORS[s], width=1.6)))
        fig.add_hline(y=1, line_dash="dot", line_color="#9ca3af")
        fig.update_yaxes(title="NAV (bắt đầu = 1)")
        show(layout(fig, 400, title="NAV tích lũy (simulated)"))

        left, right = st.columns(2)
        with left:
            fig = go.Figure()
            for s in ("buy_and_hold", "regime_aware"):
                d = n[n["strategy"] == s]
                fig.add_trace(go.Scatter(x=d["date"], y=d["drawdown"], mode="lines", name=STRATEGY_NAMES[s],
                                         line=dict(color=STRATEGY_COLORS[s], width=1.2)))
            fig.update_yaxes(tickformat=".0%", title="Drawdown")
            show(layout(fig, 300, title="Drawdown"))
        with right:
            d = n[n["strategy"] == "regime_aware"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=d["date"], y=d["prob_high_risk"], mode="lines", name="P(high-risk) filtered ngày t",
                                     line=dict(color=HIGH, width=1)))
            fig.add_trace(go.Scatter(x=d["date"], y=d["weight"], mode="lines", name="Tỷ trọng w_t", line_shape="hv",
                                     line=dict(color="#2563eb", width=1.6)))
            fig.update_yaxes(range=[0, 1.05], tickformat=".0%")
            show(layout(fig, 300, title="Tín hiệu và tỷ trọng regime-aware"))

        st.markdown("#### Bảng hiệu suất B08 (mọi kịch bản)")
        view = b08[b08["asset"] == asset].copy()
        view["strategy"] = view["strategy"].map(STRATEGY_NAMES)
        view["scenario"] = view["scenario"].map(lambda s: s.replace("_", " "))
        fmt = view[["scenario", "strategy", "final_nav", "cumulative_return", "CAGR", "ann_volatility", "Sharpe",
                    "max_drawdown", "turnover", "total_cost", "n_rebalances", "share_days_risk_off"]].copy()
        for col in ("cumulative_return", "CAGR", "ann_volatility", "max_drawdown", "total_cost", "share_days_risk_off"):
            fmt[col] = fmt[col].map(lambda v: pct(v, 2))
        for col in ("final_nav", "Sharpe", "turnover"):
            fmt[col] = fmt[col].map(lambda v: num(v, 4))
        table(fmt)
        st.caption("Baseline close-to-close là kết quả chính theo Methodology. Hai kiểm tra chéo báo riêng, không thay "
                   "baseline và không được gọi là giao dịch thực.")
        dl = st.columns(2)
        dl[0].download_button("Tải B08_portfolio.csv", rs.FILES["B08"].read_bytes(), "B08_portfolio.csv", "text/csv")
        dl[1].download_button("Tải backtest_nav.csv", rs.FILES["nav"].read_bytes(), "backtest_nav.csv", "text/csv")
        with st.expander("Giả định và provenance của backtest"):
            for a in bt_manifest.get("assumptions", []):
                st.markdown(f"- {a}")
            st.code(json.dumps(bt_manifest.get("assets", {}).get(asset, {}), ensure_ascii=False, indent=2), language="json")

st.divider()
st.caption(
    "Crypto Market Regime Radar · nguồn giá: Binance Spot, nến ngày UTC · mô hình: Gaussian HMM (hmmlearn), GMM "
    "(scikit-learn) · dashboard chỉ đọc file kết quả đã xuất, không huấn luyện lại. Mô phỏng nghiên cứu, không phải "
    "khuyến nghị đầu tư."
)
