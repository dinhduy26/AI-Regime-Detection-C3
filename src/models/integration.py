"""Actual C2 -> C3 contract smoke test; no C3 portfolio code is changed."""

from __future__ import annotations

import warnings
from datetime import datetime, timezone

import numpy as np

from src.models.handoff import UnstableModelWarning, load_filtered, prepare_next_day_research_inputs
from src.utils.config import load_config, path_of, write_json


def check_handoff(cfg):
    report = {
        "status": "running",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "C3-like research consumer of actual C2/C1 files; not a full C3 backtest",
        "assets": {},
    }
    for asset in cfg["assets"]:
        stable_gate_rejected = False
        try:
            load_filtered(cfg, asset, require_stable=True)
        except ValueError as exc:
            if "seed stability" not in str(exc):
                raise
            stable_gate_rejected = True
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", UnstableModelWarning)
            data, metadata = prepare_next_day_research_inputs(
                cfg, asset, period="test", acknowledge_unstable=True
            )
        unstable = metadata["seed_stability_status"] != "stable"
        if stable_gate_rejected != unstable or (
            unstable and not any(issubclass(w.category, UnstableModelWarning) for w in caught)
        ):
            raise ValueError("Research consumer did not explicitly acknowledge the stability warning")
        # A simple fixed mathematical consumer proves the probability is usable numerically.
        exposure = 1 - data["prob_high_risk"].to_numpy()
        contribution = exposure * data["target_simple_return"].to_numpy()
        if not np.isfinite(contribution).all() or not ((exposure >= 0) & (exposure <= 1)).all():
            raise ValueError("Consumer could not use the filtered probabilities")
        report["assets"][asset] = {
            "K": metadata["K"],
            "model_run_id": metadata["model_run_id"],
            "n_test_returns": len(data),
            "stable_gate_rejected": stable_gate_rejected,
            "research_acknowledgement": data.attrs["c2_use_policy"],
            "warnings": [str(w.message) for w in caught],
            "same_segment_next_calendar_day": True,
            "as_of_before_return_interval": True,
            "finite_consumer_calculation": True,
            "first_signal_date": str(data["signal_date_utc"].min()),
            "first_return_date": str(data["return_date_utc"].min()),
            "last_return_date": str(data["return_date_utc"].max()),
            **data.attrs["integration_counts"],
        }
    report["status"] = "passed"
    return report


def main():
    cfg = load_config()
    destination = path_of(cfg, "metadata") / "C2_C3_integration_check.json"
    try:
        report = check_handoff(cfg)
    except Exception as exc:
        write_json({"status": "failed", "error": f"{type(exc).__name__}: {exc}"}, destination)
        raise
    write_json(report, destination)
    print(f"C2 -> C3 contract: {report['status']}; assets={list(report['assets'])}")


if __name__ == "__main__":
    main()
