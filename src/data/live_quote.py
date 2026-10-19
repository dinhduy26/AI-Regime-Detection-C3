"""Giá THAM KHẢO trực tiếp cho dashboard (C3) — CHỈ ĐỂ HIỂN THỊ.

- Gọi API công khai Binance Spot GET /api/v3/ticker/24hr (không cần API key).
- KHÔNG ghi vào data/, KHÔNG dùng để tính feature, regime hay backtest.
  Regime luôn dựa trên nến ngày ĐÃ ĐÓNG (đặc tả C1 §1.2; Excel C3-13: không quảng cáo realtime).
- Lỗi mạng -> trả về dict có "error", dashboard vẫn chạy bằng dữ liệu đã có.

Cách C3 dùng trong Streamlit:
    import streamlit as st
    from src.data.live_quote import get_live_quote, LIVE_QUOTE_NOTE

    @st.cache_data(ttl=60)              # gọi Binance tối đa 1 lần/phút
    def quote(asset): return get_live_quote(asset)

    q = quote("BTC")
    if q["error"]:
        st.warning("Không lấy được giá tham khảo: " + q["error"])
    else:
        st.metric("BTC — giá tham khảo", f"{q['price']:,.0f} USDT", f"{q['change_pct_24h']:+.2f}% (24h)")
    st.caption(LIVE_QUOTE_NOTE)
"""
from __future__ import annotations

from datetime import datetime, timezone

import requests

from src.utils.config import load_config

LIVE_QUOTE_NOTE = ("Giá trực tiếp từ Binance Spot, chỉ để tham khảo — KHÔNG dùng để tính regime. "
                   "Regime được xác định trên nến ngày UTC đã đóng.")


def get_live_quote(asset: str, cfg: dict | None = None, timeout: float = 10) -> dict:
    cfg = cfg or load_config()
    symbol = cfg["assets"][asset]["symbol"]
    fetched = datetime.now(timezone.utc).isoformat(timespec="seconds")
    last_err = ""
    for base in cfg["collection"]["base_urls"]:
        try:
            r = requests.get(f"{base}/api/v3/ticker/24hr", params={"symbol": symbol}, timeout=timeout)
            if r.status_code != 200:
                last_err = f"HTTP {r.status_code}"
                continue
            d = r.json()
            return {"asset": asset, "symbol": symbol, "price": float(d["lastPrice"]),
                    "change_pct_24h": float(d["priceChangePercent"]),
                    "quote_volume_24h": float(d["quoteVolume"]),
                    "fetched_at_utc": fetched, "source": f"{base}/api/v3/ticker/24hr",
                    "use": "display_only", "error": None}
        except (requests.RequestException, ValueError, KeyError) as e:
            last_err = repr(e)
    return {"asset": asset, "symbol": symbol, "price": None, "change_pct_24h": None,
            "quote_volume_24h": None, "fetched_at_utc": fetched, "source": None,
            "use": "display_only", "error": last_err or "không truy cập được Binance"}


if __name__ == "__main__":  # python -m src.data.live_quote
    for a in load_config()["assets"]:
        print(get_live_quote(a))
