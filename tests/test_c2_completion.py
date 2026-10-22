"""Regression coverage for final artifact consistency and the C3 consumer boundary."""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import pytest

from src.models.artifacts import canonical_metadata, save_checked_bundle
from src.models.data import ZCOLS, load_asset
from src.models.diagnostics import window_inputs
from src.models.handoff import UnstableModelWarning, load_filtered, prepare_next_day_research_inputs
from src.models.integration import check_handoff
from src.utils.config import file_hash, path_of, read_json


@pytest.mark.parametrize("asset", ["BTC", "ETH"])
def test_final_metadata_matches_every_artifact_and_csv(cfg, asset):
    manifest = read_json(path_of(cfg, "metadata") / "model_manifest.json")
    metadata = read_json(path_of(cfg, "metadata") / f"model_{asset.lower()}.json")
    assert metadata == manifest["assets"][asset]
    assert metadata["test_evaluated"] is True
    assert set(metadata["n_by_period"]) == {"train", "validation", "test"}
    lock = read_json(path_of(cfg, "metadata") / f"model_{asset.lower()}_selection.json")
    assert lock["stage"] == "selection_locked" and lock["test_evaluated"] is False
    assert pd.Timestamp(lock["selection_locked_at_utc"]) <= pd.Timestamp(metadata["finalized_at_utc"])
    for key in ("K", "seed", "label_map", "high_risk_state", "model_run_id"):
        assert lock[key] == metadata[key]
    for kind in ("hmm", "gmm"):
        bundle = joblib.load(path_of(cfg, "models") / f"{kind}_{asset.lower()}.joblib")
        assert bundle["metadata"] == metadata  # full equality, not only matching run IDs
        assert bundle["model"].n_components == metadata["K"]
    assert all(
        r["metadata_equal"] and r["reload_score_equal"] for r in manifest["artifact_reload_checks"][asset]
    )
    for kind in ("filtered", "smoothed", "gmm"):
        frame = pd.read_csv(path_of(cfg, "predictions") / f"regime_{kind}_{asset.lower()}.csv")
        assert frame["model_run_id"].eq(metadata["model_run_id"]).all()
        assert frame["split"].value_counts().to_dict() == metadata["n_by_period"]
        expected_mapping = metadata["gmm_label_map"] if kind == "gmm" else metadata["label_map"]
        assert frame["label_map"].map(json.loads).map(lambda m: m == expected_mapping).all()


def test_actual_model_reload_and_failed_save_preserves_previous_file(cfg, tmp_path):
    model = joblib.load(path_of(cfg, "models") / "hmm_btc.joblib")["model"]
    frame, _, _ = load_asset(cfg, "BTC")
    x = frame.loc[frame["split"] == "train", ZCOLS].iloc[:40].to_numpy()
    destination = tmp_path / "model.joblib"
    metadata = {"label_map": {0: "A", 1: "B"}, "test_evaluated": True, "n": np.int64(40)}
    result = save_checked_bundle(model, metadata, destination, x, kind="HMM", lengths=[len(x)])
    assert result["reload_score_equal"]
    assert joblib.load(destination)["metadata"] == canonical_metadata(metadata)
    before = destination.read_bytes()
    with pytest.raises(ValueError, match="Unknown model"):
        save_checked_bundle(model, metadata, destination, x, kind="invalid")
    assert destination.read_bytes() == before


@pytest.mark.parametrize("asset", ["BTC", "ETH"])
def test_research_requires_explicit_acknowledgement_and_stable_gate_is_strict(cfg, asset):
    with pytest.raises(ValueError, match="acknowledge_unstable"):
        load_filtered(cfg, asset)
    with pytest.raises(ValueError, match="seed stability"):
        load_filtered(cfg, asset, require_stable=True, acknowledge_unstable=True)
    with pytest.warns(UnstableModelWarning):
        data, _ = load_filtered(cfg, asset, acknowledge_unstable=True)
    assert data.attrs["c2_use_policy"]["acknowledged_unstable"] is True


@pytest.mark.parametrize("asset", ["BTC", "ETH"])
def test_real_consumer_uses_previous_day_signal_for_next_day_return(cfg, asset):
    with pytest.warns(UnstableModelWarning):
        data, metadata = prepare_next_day_research_inputs(cfg, asset, acknowledge_unstable=True)
    assert len(data) == metadata["n_by_period"]["test"]
    assert (data["return_date_utc"] - data["signal_date_utc"]).eq(pd.Timedelta(days=1)).all()
    assert (data["as_of_utc"] <= data["return_date_utc"]).all()
    assert data.iloc[0]["signal_split"] == "validation"
    assert data["return_split"].eq("test").all()
    features = pd.read_parquet(path_of(cfg, "processed") / f"features_{asset.lower()}.parquet")
    expected = features.loc[features["split"] == "test", "simple_return"].to_numpy()
    assert np.allclose(data["target_simple_return"], expected)
    assert np.isfinite((1 - data["prob_high_risk"]) * data["target_simple_return"]).all()


def test_real_c3_integration_smoke_report(cfg):
    report = check_handoff(cfg)
    assert report["status"] == "passed"
    for asset in cfg["assets"]:
        info = report["assets"][asset]
        assert info["stable_gate_rejected"] and info["research_acknowledgement"]["acknowledged_unstable"]
        assert info["finite_consumer_calculation"] and info["as_of_before_return_interval"]


def test_late_signal_cannot_be_used_for_an_earlier_return_interval(cfg, monkeypatch):
    from src.models import handoff

    with pytest.warns(UnstableModelWarning):
        signals, metadata = load_filtered(cfg, "BTC", acknowledge_unstable=True)
    signals.loc[signals["split"] == "test", "as_of_utc"] += pd.Timedelta(hours=1)
    monkeypatch.setattr(handoff, "load_filtered", lambda *args, **kwargs: (signals, metadata))
    with pytest.raises(ValueError, match="too late"):
        prepare_next_day_research_inputs(cfg, "BTC", acknowledge_unstable=True)


@pytest.fixture
def modified_handoff(cfg, tmp_path, monkeypatch):
    from src.models import handoff

    manifest = read_json(path_of(cfg, "metadata") / "model_manifest.json")
    original = pd.read_csv(path_of(cfg, "predictions") / "regime_filtered_btc.csv")
    folders = {key: tmp_path / cfg["paths"][key] for key in ("metadata", "predictions")}
    for folder in folders.values():
        folder.mkdir(parents=True)
    monkeypatch.setattr(handoff, "ROOT", tmp_path)
    monkeypatch.setattr(handoff, "path_of", lambda config, key: folders[key])

    def save(frame):
        source = folders["predictions"] / "regime_filtered_btc.csv"
        frame.to_csv(source, index=False)
        manifest["output_hashes"][source.relative_to(tmp_path).as_posix()] = file_hash(source)
        (folders["metadata"] / "model_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    return original, save


@pytest.mark.parametrize(
    "problem",
    [
        "smoothed",
        "hindsight",
        "missing_asof",
        "early_asof",
        "wrong_mapping",
        "wrong_state",
        "wrong_risk",
        "duplicate_date",
        "extra_state",
        "nonfinite",
    ],
)
def test_invalid_filtered_contract_rejected_even_with_matching_file_hash(cfg, modified_handoff, problem):
    frame, save = modified_handoff
    if problem == "smoothed":
        frame["probability_type"] = "smoothed"
    elif problem == "hindsight":
        frame["hindsight"] = True
    elif problem == "missing_asof":
        frame = frame.drop(columns="as_of_utc")
    elif problem == "early_asof":
        frame.loc[0, "as_of_utc"] = frame.loc[0, "date_utc"]
    elif problem == "wrong_mapping":
        frame.loc[0, "label_map"] = "{}"
    elif problem == "wrong_state":
        frame.loc[0, "state_id"] = 99
    elif problem == "wrong_risk":
        frame.loc[0, "prob_high_risk"] = 2
    elif problem == "duplicate_date":
        frame.loc[1, "date_utc"] = frame.loc[0, "date_utc"]
    elif problem == "extra_state":
        frame["prob_state_99"] = 0
    else:
        frame.loc[0, "prob_state_0"] = np.nan
    save(frame)
    with pytest.raises(ValueError):
        load_filtered(cfg, "BTC", acknowledge_unstable=True)


def test_window_diagnostics_cannot_use_future_rows_or_refit_scaler_on_validation(cfg):
    frame, _, _ = load_asset(cfg, "BTC")
    history, scaler = window_inputs(frame, 14)
    assert not history["split"].eq("test").any()
    changed = frame.copy()
    changed.loc[changed["split"] == "test", ["rv14", "rv21", "rv30"]] = np.nan
    repeated, same_scaler = window_inputs(changed, 14)
    pd.testing.assert_frame_equal(history, repeated)
    assert scaler == same_scaler
    changed = frame.copy()
    changed.loc[changed["split"] == "validation", "rv14"] *= 10
    _, same_train_scaler = window_inputs(changed, 14)
    assert scaler["mean"] == same_train_scaler["mean"]
    assert scaler["scale"] == same_train_scaler["scale"]
