"""Thành phần giao diện dùng chung cho dashboard: bảng màu, định dạng số, CSS, thẻ KPI, bố cục biểu đồ.

Tách khỏi app.py để test có thể dùng đúng hàm định dạng mà giao diện hiển thị.
Không phụ thuộc Streamlit ở mức import (trừ các hàm render nhận module st làm tham số ngầm).
"""

from __future__ import annotations

import html
import math

# ----------------------------------------------------------------------------- bảng màu

BRAND = "#0F2A4A"  # navy thanh bên, tiêu đề mục
ACCENT = "#D4A017"  # vàng đánh dấu mục đang chọn
NAVY = "#1E3A8A"  # màu chính (regime-aware, HMM)
POS, NEG, MUTED = "#15803D", "#B91C1C", "#64748B"
INK, GRID, LINE = "#0F172A", "#EEF2F7", "#CBD5E1"
WARN = "#B45309"

HIGH, LOW = "#B91C1C", "#15803D"  # state rủi ro cao / thấp
OTHER = ["#B45309", "#6D28D9", "#0E7490", "#64748B"]  # các state trung gian
STRATEGY_COLORS = {"buy_and_hold": "#94A3B8", "regime_aware": NAVY}
METHOD_COLORS = {"HMM": NAVY, "GMM": WARN}
PERIOD_COLORS = {"train": NAVY, "validation": "#60A5FA", "test": "#CBD5E1"}


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


# ----------------------------------------------------------------------------- CSS

CSS = f"""
<style>
.stApp {{background:#F4F6F9;}}
header[data-testid="stHeader"] {{background:transparent;}}
.block-container {{padding-top:2.2rem; padding-bottom:2rem; max-width:1400px;}}

section[data-testid="stSidebar"] {{background:{BRAND};}}
section[data-testid="stSidebar"] * {{color:#E2E8F0;}}
section[data-testid="stSidebar"] code {{background:rgba(255,255,255,.10); color:#FDE68A !important;}}
section[data-testid="stSidebar"] hr {{border-color:rgba(255,255,255,.15);}}
/* menu điều hướng: radio dọc, ẩn nút tròn, mục đang chọn có vạch vàng */
.st-key-nav, .st-key-nav [data-testid="stRadio"], .st-key-nav [role="radiogroup"],
.st-key-nav [role="radiogroup"] > div {{width:100% !important;}}
.st-key-nav [role="radiogroup"] {{gap:2px;}}
[class*="st-key-nav"] [data-testid="stRadioOption"] > div > div:first-child,
[class*="st-key-asset"] [data-testid="stRadioOption"] > div > div:first-child {{display:none;}}
.st-key-nav [data-testid="stRadioOption"] {{padding:9px 12px; border-radius:4px; width:100%; margin:0;
    border-left:3px solid transparent; cursor:pointer;}}
.st-key-nav [data-testid="stRadioOption"]:hover {{background:rgba(255,255,255,.08);}}
.st-key-nav [data-testid="stRadioOption"][data-selected="true"] {{background:rgba(255,255,255,.14);
    border-left:3px solid {ACCENT};}}
.st-key-nav [data-testid="stRadioOption"] p {{font-size:.93rem; font-weight:500;}}
.st-key-nav [data-testid="stRadioOption"][data-selected="true"] p {{color:#fff !important; font-weight:700;}}
.st-key-asset [role="radiogroup"] {{gap:6px;}}
.st-key-asset [data-testid="stRadioOption"] {{background:rgba(255,255,255,.08); padding:5px 18px; border-radius:4px;
    border:1px solid rgba(255,255,255,.2); margin:0; cursor:pointer;}}
.st-key-asset [data-testid="stRadioOption"][data-selected="true"] {{background:{ACCENT}; border-color:{ACCENT};}}
.st-key-asset [data-testid="stRadioOption"][data-selected="true"] p {{color:{BRAND} !important; font-weight:800;}}
.wordmark {{font-weight:800; letter-spacing:.06em; font-size:1.05rem; color:#fff !important; margin-top:4px;}}
.wordsub {{font-size:.74rem; color:#94A3B8 !important; margin-bottom:18px; line-height:1.45;}}
.sidecap {{font-size:.68rem; letter-spacing:.09em; color:#94A3B8 !important; margin:16px 0 4px 0; font-weight:700;}}
.sideval {{font-size:.8rem; line-height:1.55;}}
.dot {{display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:6px;}}

div[data-testid="stVerticalBlockBorderWrapper"], .st-key-card, div[class*="st-key-card"] {{background:#fff;
    border-radius:8px !important; border:1px solid #E2E8F0 !important;}}
.sec {{font-size:.78rem; font-weight:700; letter-spacing:.06em; color:{BRAND}; text-transform:uppercase;
    border-bottom:2px solid {BRAND}; padding-bottom:4px; margin:4px 0 10px 0;}}
.sec span {{text-transform:none; letter-spacing:0; font-weight:500; color:{MUTED};}}

.kpi {{background:#fff; border:1px solid #E2E8F0; border-left:4px solid var(--c); border-radius:6px;
    padding:10px 14px; height:100%; min-height:96px;}}
.kpi .l {{color:{MUTED}; font-size:.72rem; text-transform:uppercase; letter-spacing:.04em;}}
.kpi .v {{font-size:1.38rem; font-weight:750; color:var(--c); line-height:1.35;}}
.kpi .s {{color:{MUTED}; font-size:.74rem;}}

.hero {{display:flex; justify-content:space-between; align-items:center; gap:18px; flex-wrap:wrap;}}
.hero .t {{font-size:1.55rem; font-weight:800; color:{INK}; margin:0;}}
.hero .t span {{color:{MUTED}; font-weight:500; font-size:1rem;}}
.px {{font-size:1.45rem; font-weight:800; color:{INK}; text-align:right;}}
.badge {{display:inline-block; padding:5px 16px; border-radius:4px; color:#fff; font-weight:700; letter-spacing:.04em;}}
.chip {{display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:6px; vertical-align:middle;}}
.muted {{color:{MUTED}; font-size:.82rem; line-height:1.5;}}
.note {{border-left:3px solid var(--c); background:#FBFCFE; padding:8px 12px; margin:6px 0; font-size:.88rem;
    color:#334155;}}
.rule {{display:flex; gap:10px; margin:8px 0 4px 0;}}
.rule div {{flex:1; border:1px solid #E2E8F0; border-radius:6px; padding:8px 10px; text-align:center;}}
.rule div.on {{border:2px solid var(--c); background:var(--bg);}}
.rule b {{display:block; font-size:1.25rem;}}

table.cmp {{width:100%; border-collapse:collapse; font-size:.88rem; margin:0;}}
table.cmp th, table.cmp td {{border:none !important;}}
table.cmp th {{background:{BRAND}; color:#fff; padding:8px 10px; text-align:right; font-weight:700;}}
table.cmp th:first-child {{text-align:left;}}
table.cmp td {{padding:6px 10px; border-bottom:1px solid #EEF2F7 !important; text-align:right; color:{INK};}}
table.cmp td:first-child {{text-align:left; color:#334155; white-space:nowrap;}}
table.cmp tr.grp td {{background:#F8FAFC; font-weight:700; color:{BRAND}; font-size:.76rem; text-transform:uppercase;
    letter-spacing:.04em;}}
table.cmp tr.key td {{font-weight:700;}}
table.cmp tr.sel td {{background:#FFFBEB;}}
.foot {{color:{MUTED}; font-size:.78rem; text-align:center; margin-top:22px;}}
</style>
"""


# ----------------------------------------------------------------------------- HTML nhỏ


def sec_html(title: str, sub: str = "") -> str:
    return f"<div class='sec'>{title}{f' <span>| {sub}</span>' if sub else ''}</div>"


def kpi_html(label: str, value: str, sub: str = "", color: str = NAVY) -> str:
    return (f"<div class='kpi' style='--c:{color}'><div class='l'>{label}</div><div class='v'>{value}</div>"
            f"<div class='s'>{sub}</div></div>")


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
        paper_bgcolor="white", plot_bgcolor="white", separators=",.",
        font={"family": "Inter, Segoe UI, sans-serif", "size": 12, "color": INK},
        title={"text": title, "x": 0, "xanchor": "left", "y": 0.98, "yanchor": "top", "font": {"size": 14}} if title else None,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.01, "x": 0, "bgcolor": "rgba(0,0,0,0)"} if legend else None,
        showlegend=legend, hoverlabel={"bgcolor": "white", "font_size": 12}, **kw,
    )
    fig.update_xaxes(showgrid=False, linecolor=LINE)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig
