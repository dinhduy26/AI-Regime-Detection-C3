"""C3-12/C3-13 - Chụp ảnh dashboard thật làm hình H06 (tùy chọn, cần Playwright).

Cách dùng (từ thư mục gốc dự án):
    streamlit run dashboard/app.py            # cửa sổ 1, để server chạy
    pip install playwright && python -m playwright install chromium   # chỉ lần đầu
    python scripts/capture_dashboard.py       # cửa sổ 2

Ghi outputs/figures/H06_dashboard.png (trang Overview) và H06_dashboard_tab{1..4}_{asset}.png (4 trang),
kèm outputs/metadata/H06_capture.json (thời điểm chụp, URL, ngày dữ liệu). Ảnh là screenshot
của ứng dụng chạy trên dữ liệu thật, không phải mock. Nếu không cài được Playwright, có thể
chụp màn hình thủ công và đặt tên theo đúng quy ước trên.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config, path_of, read_json, write_json  # noqa: E402

PAGES = ["Overview", "Regime History", "Model Comparison", "Portfolio & Risk"]  # menu điều hướng ở thanh bên


def main(argv=None):
    ap = argparse.ArgumentParser(description="Chụp dashboard thật làm H06")
    ap.add_argument("--url", default="http://localhost:8501")
    ap.add_argument("--assets", nargs="*", default=None)
    ap.add_argument("--executable", default=None, help="đường dẫn Chromium nếu không dùng bản Playwright tải về")
    args = ap.parse_args(argv)

    from playwright.sync_api import sync_playwright

    cfg = load_config()
    figures, meta = path_of(cfg, "figures"), path_of(cfg, "metadata")
    assets = args.assets or list(cfg["assets"])
    written = []
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=args.executable) if args.executable else p.chromium.launch()
        # Streamlit cuộn trong khung riêng nên full_page không mở rộng; dùng viewport cao để thấy đủ trang.
        page = browser.new_page(viewport={"width": 1500, "height": 3000}, device_scale_factor=1.5)
        page.goto(args.url, wait_until="networkidle")
        page.get_by_text("REGIME RADAR").first.wait_for(timeout=60_000)
        sidebar = page.get_by_test_id("stSidebar")
        for asset in assets:
            sidebar.get_by_text(asset, exact=True).click()
            page.wait_for_timeout(2500)
            for i, name in enumerate(PAGES, start=1):
                sidebar.get_by_text(f"{i}.  {name}").click()
                page.wait_for_timeout(2500)
                out = figures / f"H06_dashboard_tab{i}_{asset.lower()}.png"
                page.screenshot(path=str(out), full_page=True)
                written.append(out)
        sidebar.get_by_text(assets[0], exact=True).click()
        page.wait_for_timeout(1500)
        sidebar.get_by_text(f"1.  {PAGES[0]}").click()
        page.set_viewport_size({"width": 1500, "height": 1000})
        page.wait_for_timeout(2500)
        main_shot = figures / "H06_dashboard.png"
        page.screenshot(path=str(main_shot), full_page=False)
        written.append(main_shot)
        browser.close()

    split = read_json(meta / "split_manifest.json")
    write_json(
        {
            "captured_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "url": args.url,
            "data_through_utc": split.get("data_through_utc"),
            "assets": assets,
            "files": [str(p.relative_to(ROOT).as_posix()) for p in written],
            "note": "Screenshot of the running Streamlit app on real pipeline outputs; not a mock.",
        },
        meta / "H06_capture.json",
    )
    print(json.dumps([p.name for p in written], ensure_ascii=False))


if __name__ == "__main__":
    main()
