"""Thành phần giao diện dùng chung cho dashboard: bảng màu, định dạng số, CSS, thẻ KPI, bố cục biểu đồ.

Tách khỏi app.py để test có thể dùng đúng hàm định dạng mà giao diện hiển thị.
Không phụ thuộc Streamlit ở mức import (trừ các hàm render nhận module st làm tham số ngầm).
"""

from __future__ import annotations

import html
import math

# ----------------------------------------------------------------------------- bảng màu (phương án B: đất ấm)

BRAND = "#3E4A3D"  # nâu ô liu đậm: thanh bên, tiêu đề mục
ACCENT = "#C08A3E"  # đồng: mục đang chọn, điểm nhấn
NAVY = "#36584A"  # xanh rêu đậm: màu chính (regime-aware, HMM)
POS, NEG, MUTED = "#4F7F5C", "#A9472F", "#7A7368"
INK, GRID, LINE = "#2E2A24", "#ECE6DC", "#D6CEC1"
WARN = "#B07A2A"
BG = "#E7E1D8"

HIGH, LOW = "#B5523B", "#5B8C6A"  # state rủi ro cao / thấp
OTHER = ["#C9A227", "#7A6C9B", "#5E8A9A", "#8C8273"]  # các state trung gian
STRATEGY_COLORS = {"buy_and_hold": "#A79C8C", "regime_aware": NAVY}
METHOD_COLORS = {"HMM": NAVY, "GMM": ACCENT}
PERIOD_COLORS = {"train": NAVY, "validation": "#8FB39A", "test": "#D6CEC1"}


# ----------------------------------------------------------------------------- định dạng số kiểu Việt Nam


def _bad(x) -> bool:
    try:
        return x is None or not math.isfinite(float(x))
    except (TypeError, ValueError):
        return True


def _vn(s: str) -> str:
    return s.replace(",", "§").replace(".", ",").replace("§", ".")


def num(x, d: int = 2, sign: bool = False) -> str:
    """Số thập phân kiểu Việt Nam: 1.234,56."""
    return "-" if _bad(x) else _vn(f"{float(x):{'+' if sign else ''},.{d}f}")


def pct(x, d: int = 2, sign: bool = False) -> str:
    """Tỷ lệ (0,05 -> 5,00%)."""
    return "-" if _bad(x) else _vn(f"{float(x) * 100:{'+' if sign else ''},.{d}f}") + "%"


def pp(x, d: int = 2) -> str:
    """Chênh lệch theo điểm phần trăm."""
    return "-" if _bad(x) else _vn(f"{float(x) * 100:+,.{d}f}") + " điểm %"


def sign_color(x, higher_is_better: bool = True) -> str:
    if _bad(x) or float(x) == 0:
        return MUTED
    good = float(x) > 0 if higher_is_better else float(x) < 0
    return POS if good else NEG


def esc(x) -> str:
    return html.escape(str(x))


# ----------------------------------------------------------------------------- nhãn state


def state_name(sid, label_map) -> str:
    """'High-Vol (state 0)'; state chưa có nhãn mô tả thì chỉ ghi 'State 1'."""
    label = label_map[str(int(sid))]
    return label if label == f"State {int(sid)}" else f"{label} (state {int(sid)})"


def state_colors(label_map) -> dict:
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


def rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


# ----------------------------------------------------------------------------- nền chủ đề và CSS


def _background_svg() -> str:
    """Họa tiết nền tự vẽ: lưới mờ, chuỗi nến giá và hai đường dữ liệu chạy ngang (không dùng ảnh bản quyền)."""
    import math as _m
    import random

    rnd = random.Random(7)
    w, h = 1600, 900
    parts = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{w}' height='{h}' viewBox='0 0 {w} {h}'>"]
    for x in range(0, w, 80):
        parts.append(f"<line x1='{x}' y1='0' x2='{x}' y2='{h}' stroke='#3E4A3D' stroke-opacity='.05'/>")
    for y in range(0, h, 80):
        parts.append(f"<line x1='0' y1='{y}' x2='{w}' y2='{y}' stroke='#3E4A3D' stroke-opacity='.05'/>")
    price = 560.0
    for i in range(64):
        x = 20 + i * 25
        o = price
        price += rnd.gauss(-1.2, 16)
        c = price
        hi, lo = max(o, c) + rnd.uniform(4, 18), min(o, c) - rnd.uniform(4, 18)
        col = "#5B8C6A" if c < o else "#B5523B"
        parts.append(f"<line x1='{x + 6}' y1='{lo:.0f}' x2='{x + 6}' y2='{hi:.0f}' stroke='{col}' stroke-opacity='.16'/>")
        parts.append(f"<rect x='{x}' y='{min(o, c):.0f}' width='12' height='{max(abs(c - o), 2):.0f}' "
                     f"fill='{col}' fill-opacity='.13'/>")
    for k, (amp, base, col) in enumerate(((60, 300, "#C08A3E"), (45, 760, "#36584A"))):
        pts = " ".join(f"{x},{base + amp * _m.sin(x / 140 + k) + 18 * _m.sin(x / 37):.0f}" for x in range(0, w + 1, 20))
        parts.append(f"<polyline points='{pts}' fill='none' stroke='{col}' stroke-opacity='.18' stroke-width='2'/>")
    parts.append("</svg>")
    return "".join(parts)


def _bg_uri() -> str:
    import base64

    return "data:image/svg+xml;base64," + base64.b64encode(_background_svg().encode()).decode()


CSS = f"""
<style>
.stApp {{background-color:{BG}; background-image:url("{_bg_uri()}"); background-size:cover;
    background-attachment:fixed; background-position:center;}}
header[data-testid="stHeader"] {{background:transparent;}}
.block-container {{padding-top:1.6rem; padding-bottom:1.6rem; max-width:1440px;}}
[data-testid="stHorizontalBlock"] {{gap:14px;}}

section[data-testid="stSidebar"] {{background:{BRAND};}}
section[data-testid="stSidebar"] * {{color:#E9E3D7;}}
section[data-testid="stSidebar"] code {{background:rgba(255,255,255,.10); color:#F0D9A8 !important;}}
/* menu điều hướng */
.st-key-nav, .st-key-nav [data-testid="stRadio"], .st-key-nav [role="radiogroup"],
.st-key-nav [role="radiogroup"] > div {{width:100% !important;}}
.st-key-nav [role="radiogroup"] {{gap:2px;}}
[class*="st-key-nav"] [data-testid="stRadioOption"] > div > div:first-child,
[class*="st-key-asset"] [data-testid="stRadioOption"] > div > div:first-child {{display:none;}}
.st-key-nav [data-testid="stRadioOption"] {{padding:10px 12px; border-radius:6px; width:100%; margin:0;
    border-left:3px solid transparent; cursor:pointer;}}
.st-key-nav [data-testid="stRadioOption"]:hover {{background:rgba(255,255,255,.07);}}
.st-key-nav [data-testid="stRadioOption"][data-selected="true"] {{background:rgba(255,255,255,.13);
    border-left:3px solid {ACCENT};}}
.st-key-nav [data-testid="stRadioOption"] p {{font-size:.95rem; font-weight:500;}}
.st-key-nav [data-testid="stRadioOption"][data-selected="true"] p {{color:#fff !important; font-weight:700;}}
.st-key-asset [role="radiogroup"] {{gap:6px;}}
.st-key-asset [data-testid="stRadioOption"] {{background:rgba(255,255,255,.07); padding:6px 20px; border-radius:6px;
    border:1px solid rgba(255,255,255,.18); margin:0; cursor:pointer;}}
.st-key-asset [data-testid="stRadioOption"][data-selected="true"] {{background:{ACCENT}; border-color:{ACCENT};}}
.st-key-asset [data-testid="stRadioOption"][data-selected="true"] p {{color:#2E2A24 !important; font-weight:800;}}
.wordmark {{font-weight:800; letter-spacing:.08em; font-size:1.1rem; color:#fff !important; margin:4px 0 22px 0;}}
.wordmark span {{color:{ACCENT} !important;}}
.sidecap {{font-size:.68rem; letter-spacing:.1em; color:#B9B09F !important; margin:18px 0 6px 0; font-weight:700;}}
.asof {{background:rgba(255,255,255,.07); border-radius:6px; padding:10px 12px; font-size:.82rem; line-height:1.5;}}
.asof b {{font-size:1.05rem; color:#fff !important;}}

div[class*="st-key-card"] {{background:rgba(255,252,246,.86); backdrop-filter:blur(6px); border-radius:12px !important;
    border:1px solid rgba(62,74,61,.10) !important; box-shadow:0 2px 10px rgba(62,74,61,.07);}}
.sec {{font-size:.8rem; font-weight:700; letter-spacing:.06em; color:{BRAND}; text-transform:uppercase;
    margin:2px 0 6px 0; display:flex; justify-content:space-between; align-items:center;}}
.sec span {{text-transform:none; letter-spacing:0; font-weight:500; color:{MUTED};}}
.info {{display:inline-block; width:16px; height:16px; border-radius:50%; border:1px solid {MUTED}; color:{MUTED};
    font-size:.68rem; line-height:14px; text-align:center; cursor:help; font-weight:600; text-transform:none;}}

.kpi {{background:rgba(255,252,246,.88); backdrop-filter:blur(6px); border:1px solid rgba(62,74,61,.10);
    border-top:4px solid var(--c); border-radius:12px; padding:12px 16px; height:112px; margin-bottom:14px; overflow:hidden;
    box-shadow:0 2px 10px rgba(62,74,61,.07);}}
.kpi .l {{color:{MUTED}; font-size:.7rem; white-space:nowrap; text-transform:uppercase; letter-spacing:.06em; font-weight:600;
    display:flex; justify-content:space-between;}}
.kpi .v {{font-size:1.7rem; font-weight:800; color:var(--c); line-height:1.25; margin-top:6px; white-space:nowrap;}}
.kpi .s {{color:{MUTED}; font-size:.8rem; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}}
.pill {{display:inline-block; padding:2px 10px; border-radius:999px; color:#fff; font-weight:700; font-size:1.25rem;}}

.hero {{display:flex; justify-content:space-between; align-items:flex-end; margin:0 0 12px 2px;}}
.hero .t {{font-size:1.75rem; font-weight:800; color:{INK}; margin:0; line-height:1.1;}}
.hero .t span {{color:{MUTED}; font-weight:500; font-size:1rem;}}
.tag {{display:inline-block; padding:3px 10px; border-radius:999px; font-size:.75rem; font-weight:600;
    background:rgba(176,122,42,.14); color:{WARN}; border:1px solid rgba(176,122,42,.35); margin-left:6px;}}
.muted {{color:{MUTED}; font-size:.82rem; line-height:1.5;}}
.chip {{display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:6px; vertical-align:middle;}}
.profgrid {{display:grid; grid-template-columns:1fr; gap:8px;}}
.profgrid.two {{grid-template-columns:1fr 1fr;}}
.profgrid .prof {{margin-bottom:0;}}
.profgrid > div {{min-width:0;}}
.profgrid.two .prof .g {{grid-template-columns:1fr 1fr; row-gap:8px;}}
.prof {{border-radius:10px; padding:10px 12px; background:rgba(255,255,255,.6); border-left:5px solid var(--c);
    margin-bottom:8px;}}
.prof .n {{font-weight:800; color:var(--c); font-size:.95rem;}}
.prof .g {{display:grid; grid-template-columns:repeat(4, 1fr); gap:6px; margin-top:4px;}}
.prof .g div {{font-size:.7rem; color:{MUTED}; text-transform:uppercase; letter-spacing:.03em;}}
.prof .g b {{display:block; font-size:1.02rem; color:{INK}; text-transform:none; letter-spacing:0;}}
.note {{border-left:3px solid var(--c); background:rgba(255,255,255,.55); padding:8px 12px; margin:6px 0;
    font-size:.88rem; color:#4A443B; border-radius:4px;}}
.rule {{display:flex; gap:10px; margin:8px 0 4px 0;}}
.rule div {{flex:1; border:1px solid {LINE}; border-radius:6px; padding:8px 10px; text-align:center;}}
.rule div.on {{border:2px solid var(--c); background:var(--bg);}}
.rule b {{display:block; font-size:1.25rem;}}

table.cmp {{width:100%; border-collapse:collapse; font-size:.88rem; margin:0;}}
table.cmp th, table.cmp td {{border:none !important;}}
table.cmp th {{background:{BRAND}; color:#fff; padding:8px 10px; text-align:right; font-weight:700;}}
table.cmp th:first-child {{text-align:left;}}
table.cmp td {{padding:6px 10px; border-bottom:1px solid {GRID} !important; text-align:right; color:{INK};}}
table.cmp td:first-child {{text-align:left; color:#4A443B; white-space:nowrap;}}
table.cmp tr.grp td {{background:rgba(231,225,216,.6); font-weight:700; color:{BRAND}; font-size:.76rem;
    text-transform:uppercase; letter-spacing:.04em;}}
table.cmp tr.key td {{font-weight:700;}}
table.cmp tr.sel td {{background:rgba(192,138,62,.12);}}
[data-testid="stAlert"] {{padding:6px 12px;}}
.foot {{color:{MUTED}; font-size:.75rem; text-align:center; margin-top:18px;}}
</style>
"""


# ----------------------------------------------------------------------------- HTML nhỏ


def sec_html(title: str, sub: str = "", info: str = "") -> str:
    right = info_html(info) if info else ""
    return f"<div class='sec'><div>{title}{f' <span>· {sub}</span>' if sub else ''}</div>{right}</div>"


def kpi_html(label: str, value: str, sub: str = "", color: str = NAVY, info: str = "") -> str:
    tip = f"<span class='info' title='{esc(info)}'>i</span>" if info else ""
    return (f"<div class='kpi' style='--c:{color}'><div class='l'><span>{label}</span>{tip}</div>"
            f"<div class='v'>{value}</div><div class='s'>{sub}</div></div>")


def info_html(text: str) -> str:
    return f"<span class='info' title='{esc(text)}'>i</span>"


def chip(color: str) -> str:
    return f"<span class='chip' style='background:{color}'></span>"


def html_table(header: list, rows: list, row_classes: list | None = None) -> str:
    """Bảng HTML gọn; cột đầu canh trái, các cột còn lại canh phải. rows là list các list chuỗi (đã escape)."""
    head = "".join(f"<th>{h}</th>" for h in header)
    body = []
    for i, r in enumerate(rows):
        cls = row_classes[i] if row_classes else ""
        if cls == "grp":
            body.append(f"<tr class='grp'><td colspan='{len(header)}'>{r[0]}</td></tr>")
        else:
            body.append(f"<tr class='{cls}'>" + "".join(f"<td>{v}</td>" for v in r) + "</tr>")
    return f"<table class='cmp'><tr>{head}</tr>{''.join(body)}</table>"


# ----------------------------------------------------------------------------- Plotly


def chart_layout(fig, height: int = 360, title: str | None = None, legend: bool = True, **kw):
    """Nền trắng, lưới nhạt, chú giải nằm ngang phía trên vùng vẽ, dấu thập phân kiểu Việt Nam."""
    fig.update_layout(
        height=height, margin={"l": 10, "r": 10, "t": 72 if title else 36, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", separators=",.",
        font={"family": "Inter, Segoe UI, sans-serif", "size": 12, "color": INK},
        title={"text": title, "x": 0, "xanchor": "left", "y": 0.98, "yanchor": "top", "font": {"size": 14}} if title else None,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.01, "x": 0, "bgcolor": "rgba(0,0,0,0)"} if legend else None,
        showlegend=legend, hoverlabel={"bgcolor": "white", "font_size": 12}, **kw,
    )
    fig.update_xaxes(showgrid=False, linecolor=LINE)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig
