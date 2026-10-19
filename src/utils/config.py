"""Đọc cấu hình và các tiện ích đường dẫn dùng chung cho toàn dự án."""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import yaml

# Thư mục gốc của repo = 2 cấp trên file này (src/utils/config.py)
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config" / "config.yaml"
SCHEMA_FILE = ROOT / "config" / "schema.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Đọc config.yaml và trả về dict."""
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(path)
    cfg["_config_hash"] = file_hash(path)
    return cfg


def load_schema() -> dict[str, Any]:
    with open(SCHEMA_FILE, encoding="utf-8") as f:
        return yaml.safe_load(f)


def file_hash(path: str | Path) -> str:
    """SHA-256 rút gọn của file — dùng để ghi phiên bản cấu hình/dữ liệu vào manifest."""
    h = hashlib.sha256(Path(path).read_bytes())
    return h.hexdigest()[:16]


def path_of(cfg: dict[str, Any], key: str) -> Path:
    """Trả về đường dẫn tuyệt đối của một thư mục khai báo trong cfg['paths'] và tạo nếu chưa có."""
    p = ROOT / cfg["paths"][key]
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(obj: Any, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def read_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(name)


def check_columns(df, required: list[str], name: str) -> None:
    """Dừng chương trình nếu thiếu cột so với hợp đồng schema."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"[{name}] thiếu cột theo schema: {missing}")
