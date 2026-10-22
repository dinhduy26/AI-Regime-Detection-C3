"""Small C2 -> C3 reader for the causal probability contract."""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd

from src.models.inference import sequence_starts
from src.utils.config import ROOT, file_hash, path_of, read_json


class UnstableModelWarning(UserWarning):
    """Research use explicitly acknowledges the model's seed sensitivity."""


def load_filtered(cfg, asset, period=None, *, require_stable=False, acknowledge_unstable=False):
    """Read one asset, enforcing a complete run, correct identity and causal timing.

    C3 must apply a signal from candle t only to the following contiguous day,
    after checking segment_id and as_of_utc. This reader does not trade or backtest.
    """
    manifest = read_json(path_of(cfg, "metadata") / "model_manifest.json")
    if manifest["status"] != "complete":
        raise ValueError("C2 model run is incomplete; do not use partial outputs")
    metadata = manifest["assets"][asset]
    if require_stable and metadata.get("seed_stability_status") != "stable":
        raise ValueError(
            f"{asset}: seed stability is {metadata.get('seed_stability_status', 'not assessed')}"
        )
    unstable = metadata.get("seed_stability_status") != "stable"
    if unstable and not acknowledge_unstable:
        raise ValueError(
            f"{asset}: unstable research model; explicitly pass acknowledge_unstable=True for research use"
        )
    source = path_of(cfg, "predictions") / f"regime_filtered_{asset.lower()}.csv"
    if manifest["output_hashes"].get(source.relative_to(ROOT).as_posix()) != file_hash(source):
        raise ValueError("Filtered output differs from the completed model manifest")
    frame = pd.read_csv(source)
    k = metadata["K"]
    required = [
        "date_utc",
        "asset",
        "segment_id",
        "split",
        "as_of_utc",
        "model_run_id",
        "K",
        "state_id",
        "label",
        "label_map",
        "high_risk_state",
        "prob_high_risk",
        "probability_type",
        "hindsight",
    ]
    cols = [f"prob_state_{i}" for i in range(k)]
    missing = set(required + cols) - set(frame.columns)
    if missing:
        raise ValueError(f"C2 filtered schema is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Empty filtered output")
    sequence_starts(frame)
    if {c for c in frame if c.startswith("prob_state_")} != set(cols):
        raise ValueError("State probability columns disagree with K")
    probabilities = frame[cols].to_numpy(dtype=float)
    if (
        not frame["probability_type"].eq("filtered").all()
        or frame["hindsight"].dtype != bool
        or frame["hindsight"].any()
        or not frame["model_run_id"].eq(metadata["model_run_id"]).all()
        or not frame["asset"].eq(asset).all()
        or not frame["K"].eq(k).all()
    ):
        raise ValueError("C3 requires matching causal filtered probabilities")
    if (
        not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or (probabilities > 1).any()
        or not np.allclose(probabilities.sum(axis=1), 1, atol=cfg["models"]["probability_tolerance"], rtol=0)
    ):
        raise ValueError("Invalid C3 state probability vectors")
    risk = metadata["high_risk_state"]
    if (
        not 0 <= risk < k
        or not frame["high_risk_state"].eq(risk).all()
        or not np.array_equal(frame["state_id"], probabilities.argmax(axis=1))
        or not np.allclose(frame["prob_high_risk"], probabilities[:, risk], atol=1e-10)
    ):
        raise ValueError("State IDs or high-risk probabilities disagree with model metadata")
    mapping = metadata["label_map"]
    if set(mapping) != {str(i) for i in range(k)}:
        raise ValueError("Invalid state mapping in metadata")
    if not frame["label"].equals(frame["state_id"].astype(str).map(mapping)):
        raise ValueError("CSV labels disagree with model metadata")
    try:
        if not frame["label_map"].map(lambda value: json.loads(value) == mapping).all():
            raise ValueError("CSV state mapping disagrees with metadata")
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid CSV label_map JSON") from exc
    dates = pd.to_datetime(frame["date_utc"], utc=True)
    if not dates.equals(dates.dt.normalize()):
        raise ValueError("date_utc must identify the UTC daily candle")
    frame["as_of_utc"] = pd.to_datetime(frame["as_of_utc"], utc=True)
    if frame["as_of_utc"].isna().any() or (frame["as_of_utc"] < dates + pd.Timedelta(days=1)).any():
        raise ValueError("Signal precedes completed candle availability")
    if frame["split"].value_counts().to_dict() != metadata["n_by_period"]:
        raise ValueError("Filtered split counts disagree with metadata")
    if period is not None:
        if period not in ("train", "validation", "test"):
            raise ValueError("Unknown period")
        frame = frame.loc[frame["split"] == period].reset_index(drop=True)
    frame.attrs["c2_use_policy"] = {
        "research_only": True,
        "acknowledged_unstable": bool(unstable),
        "seed_stability_status": metadata["seed_stability_status"],
        "model_run_id": metadata["model_run_id"],
    }
    if unstable:
        warnings.warn(
            f"{asset}: research use acknowledged; seed stability is {metadata['seed_stability_status']}",
            UnstableModelWarning,
            stacklevel=2,
        )
    return frame, metadata


def prepare_next_day_research_inputs(
    cfg, asset, period="test", *, require_stable=False, acknowledge_unstable=False
):
    """C2 consumer example: align candle-t signal with actual contiguous day-t+1 returns.

    This exercises the agreed close-to-close research contract. It does not
    implement the C3 portfolio, execution latency, fees, NAV or performance claims.
    """
    if period not in ("train", "validation", "test"):
        raise ValueError("Unknown period")
    signals, metadata = load_filtered(
        cfg, asset, require_stable=require_stable, acknowledge_unstable=acknowledge_unstable
    )
    feature_path = path_of(cfg, "processed") / f"features_{asset.lower()}.parquet"
    if file_hash(feature_path) != metadata["c1_provenance"]["features_sha256"]:
        raise ValueError("C1 feature file changed since model run; regenerate C2 before integration")
    returns = pd.read_parquet(feature_path)[
        ["asset", "date_utc", "segment_id", "split", "simple_return"]
    ].copy()
    returns["return_date_utc"] = pd.to_datetime(returns.pop("date_utc"), utc=True)
    returns = returns.rename(columns={"split": "return_split", "simple_return": "target_simple_return"})
    left = signals.rename(columns={"date_utc": "signal_date_utc", "split": "signal_split"}).copy()
    left["signal_date_utc"] = pd.to_datetime(left["signal_date_utc"], utc=True)
    left["return_date_utc"] = left["signal_date_utc"] + pd.Timedelta(days=1)
    aligned = left.merge(
        returns, on=["asset", "segment_id", "return_date_utc"], how="inner", validate="one_to_one"
    )
    if (aligned["as_of_utc"] > aligned["return_date_utc"]).any():
        raise ValueError("Signal is available too late for the assumed next-day close-to-close return")
    result = aligned.loc[aligned["return_split"] == period].reset_index(drop=True)
    if result.empty or not np.isfinite(result["target_simple_return"]).all():
        raise ValueError("No valid finite next-day research returns")
    result.attrs["c2_use_policy"] = signals.attrs["c2_use_policy"]
    result.attrs["integration_counts"] = {
        "signals": len(signals),
        "aligned_all_periods": len(aligned),
        "unmatched_next_day": len(signals) - len(aligned),
        "returned_period": len(result),
    }
    return result, metadata
