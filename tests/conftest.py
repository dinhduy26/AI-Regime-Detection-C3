"""Fixture dùng chung. TẤT CẢ test chạy trên dữ liệu THẬT đã tải về.
Không có dữ liệu thật -> test được bỏ qua (skip), tuyệt đối không tạo dữ liệu giả."""
import pandas as pd
import pytest

from src.utils.config import load_config, path_of, read_json


@pytest.fixture(scope="session")
def cfg():
    return load_config()


def _need(path):
    if not path.exists():
        pytest.skip(f"Chưa có {path.name} — chạy `python run_pipeline.py` với dữ liệu thật trước")
    return path


@pytest.fixture(scope="session")
def raw(cfg):
    d = path_of(cfg, "raw")
    return {a: pd.read_csv(_need(d / f"{a.lower()}_ohlcv.csv"), dtype={"date_utc": str})
            for a in cfg["assets"]}


@pytest.fixture(scope="session")
def prices(cfg):
    return pd.read_parquet(_need(path_of(cfg, "processed") / "prices_daily.parquet"))


@pytest.fixture(scope="session")
def feats(cfg):
    d = path_of(cfg, "processed")
    return {a: pd.read_parquet(_need(d / f"features_{a.lower()}.parquet")) for a in cfg["assets"]}


@pytest.fixture(scope="session")
def scaled(cfg):
    d = path_of(cfg, "processed")
    return {a: pd.read_parquet(_need(d / f"scaled_features_{a.lower()}.parquet")) for a in cfg["assets"]}


@pytest.fixture(scope="session")
def manifest(cfg):
    return read_json(_need(path_of(cfg, "metadata") / "split_manifest.json"))
