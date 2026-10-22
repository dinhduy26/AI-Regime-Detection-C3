"""Optional post-hoc RV14/RV21/RV30 diagnostics on train/validation only.

No production model, C1 file, scaler or selection is changed. All windows use
the same common rows. Scores across different RV variables are not comparable.
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from src.models.data import ZCOLS, load_asset
from src.models.inference import infer
from src.models.reports import train_mapping
from src.models.selection import fit_candidates
from src.models.stability import summarize_seed_stability
from src.utils.config import file_hash, get_logger, load_config, path_of, read_json, write_json

log = get_logger("C2-diagnostics")


def window_inputs(frame, window):
    # Slice BEFORE checking windows, scaling or fitting: test never affects common sample.
    history = frame.loc[frame["split"].isin(["train", "validation"])].copy()
    required = ["log_return", "rv14", "rv21", "rv30", "log_dollar_volume"]
    if set(required) - set(history):
        raise ValueError("C1 optional RV14/RV30 columns are required for window diagnostics")
    history = history.loc[np.isfinite(history[required]).all(axis=1)].reset_index(drop=True)
    columns = ["log_return", f"rv{window}", "log_dollar_volume"]
    train = history.loc[history["split"] == "train", columns].to_numpy()
    if not len(train) or not history["split"].eq("validation").any():
        raise ValueError("Insufficient common train/validation rows")
    mean, std = train.mean(axis=0), train.std(axis=0, ddof=0)
    scale = np.where(std == 0, 1, std)
    history[ZCOLS] = (history[columns].to_numpy() - mean) / scale
    return history, {
        "features": columns,
        "fit_on": "train",
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "n_train": len(train),
        "n_validation": int(history["split"].eq("validation").sum()),
        "common_sample_start": str(history["date_utc"].iloc[0]),
        "std_ddof": 0,
    }


def run(cfg):
    tables, models, meta = (path_of(cfg, k) for k in ("tables", "models", "metadata"))
    core = read_json(meta / "model_manifest.json")
    if core["status"] != "complete":
        raise ValueError("Run the core pipeline before diagnostics")
    core_hash = file_hash(meta / "model_manifest.json")
    report = {
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "post-hoc train/validation sensitivity only; no test evaluation or production model selection",
        "core_manifest_hash": core_hash,
        "test_used": False,
        "production_models_changed": False,
        "score_warning": "RV definitions differ; do not rank windows by likelihood",
        "scalers": {},
        "assets": {},
    }
    rows, all_fits = [], []
    write_json(report, meta / "C2_window_diagnostics.json")
    try:
        for asset in cfg["assets"]:
            frame, _, _ = load_asset(cfg, asset)
            bundle = joblib.load(models / f"hmm_{asset.lower()}.joblib")
            k = bundle["metadata"]["K"]
            model_hash = file_hash(models / f"hmm_{asset.lower()}.joblib")
            info = {
                "K_fixed_to_core": k,
                "model_run_id": bundle["metadata"]["model_run_id"],
                "core_model_hash": model_hash,
            }
            for window in (14, 21, 30):
                log.info(
                    "%s RV%d: common train/validation sample; production K fixed to %d", asset, window, k
                )
                history, scaler = window_inputs(frame, window)
                report["scalers"][f"{asset}_RV{window}"] = scaler
                settings = copy.deepcopy(cfg["models"])
                settings["states"] = [k]
                candidate, fits, pairs = fit_candidates(history, settings)
                summary = summarize_seed_stability(fits, pairs, settings["stability"]).iloc[0].to_dict()
                fits = fits.rename(columns={"selected": "diagnostic_best_validation"})
                fits.insert(0, "asset", asset)
                fits.insert(1, "rv_window", window)
                all_fits.append(fits)
                result = {
                    "asset": asset,
                    "rv_window": window,
                    "core_model_run_id": info["model_run_id"],
                    "K": k,
                    "n_train": scaler["n_train"],
                    "n_validation": scaler["n_validation"],
                    "n_eligible_seeds": summary["n_eligible_seeds"],
                    "seed_ari_median": summary["ari_median"],
                    "seed_ari_min": summary["ari_min"],
                    "seed_stability_status": summary["status"],
                    "used_for_production_selection": False,
                    "test_used": False,
                    "score_comparable_across_windows": False,
                }
                if candidate is not None:
                    tr = history.loc[history["split"] == "train"].reset_index(drop=True)
                    train_alpha, _ = infer(candidate, tr[ZCOLS], tr)
                    _, risk = train_mapping(tr, train_alpha)
                    alpha, score = infer(candidate, history[ZCOLS], history)
                    # Core model uses its own frozen RV21 scale and all prior history.
                    ref_history = frame.loc[frame["split"] != "test"].reset_index(drop=True)
                    reference, _ = infer(bundle["model"], ref_history[ZCOLS], ref_history)
                    val_dates = history.loc[history["split"] == "validation", "date_utc"]
                    ref_mask = ref_history["date_utc"].isin(val_dates).to_numpy()
                    val_mask = history["split"].eq("validation").to_numpy()
                    ref_prob, prob = reference[ref_mask], alpha[val_mask]
                    ref_risk = bundle["metadata"]["high_risk_state"]
                    result.update(
                        validation_ari_vs_frozen_core=adjusted_rand_score(
                            ref_prob.argmax(axis=1), prob.argmax(axis=1)
                        ),
                        high_risk_probability_mae_vs_core=float(
                            np.abs(ref_prob[:, ref_risk] - prob[:, risk]).mean()
                        ),
                        high_risk_disagreement_vs_core=float(
                            np.mean((ref_prob.argmax(axis=1) == ref_risk) != (prob.argmax(axis=1) == risk))
                        ),
                        own_validation_log_score=float(score[val_mask].mean()),
                        diagnostic_seed=int(fits.loc[fits["diagnostic_best_validation"], "seed"].iloc[0]),
                    )
                rows.append(result)
            if file_hash(models / f"hmm_{asset.lower()}.joblib") != model_hash:
                raise ValueError("Diagnostic changed a core model")
            report["assets"][asset] = info
        if file_hash(meta / "model_manifest.json") != core_hash:
            raise ValueError("Diagnostic changed core manifest")
        pd.DataFrame(rows).to_csv(tables / "C2_window_sensitivity.csv", index=False)
        pd.concat(all_fits, ignore_index=True).to_csv(tables / "C2_window_candidates.csv", index=False)
        selection = pd.read_csv(tables / "B03_model_selection.csv")
        seed_summary = selection.groupby(["asset", "K"]).agg(
            n_fits=("seed", "count"),
            n_converged=("converged", "sum"),
            n_eligible=("eligible", "sum"),
            train_ll_min=("train_ll", "min"),
            train_ll_max=("train_ll", "max"),
            min_train_state_share=("min_state_share", "min"),
            validation_score_min=("val_avg_predictive_ll", "min"),
            validation_score_max=("val_avg_predictive_ll", "max"),
        )
        seed_summary["train_ll_range"] = seed_summary["train_ll_max"] - seed_summary["train_ll_min"]
        seed_summary.reset_index().to_csv(tables / "C2_seed_fit_diagnostics.csv", index=False)
        coverage = pd.read_csv(tables / "B04_regime_statistics.csv")
        coverage = coverage[
            ["asset", "period", "state_id", "label", "n_hard", "n_period", "share", "soft_share"]
        ].copy()
        coverage["posterior_mass_days"] = coverage["n_period"] * coverage["soft_share"]
        coverage["sparse_hard_occupancy"] = coverage["share"] < cfg["models"]["min_state_share"]
        coverage["used_for_selection"] = False
        coverage.to_csv(tables / "C2_state_coverage.csv", index=False)
        report["status"] = "complete"
        report["output_hashes"] = {
            name: file_hash(tables / name)
            for name in (
                "C2_window_sensitivity.csv",
                "C2_window_candidates.csv",
                "C2_seed_fit_diagnostics.csv",
                "C2_state_coverage.csv",
            )
        }
    except Exception as exc:
        report["status"], report["error"] = "failed", f"{type(exc).__name__}: {exc}"
        raise
    finally:
        write_json(report, meta / "C2_window_diagnostics.json")
    return report


if __name__ == "__main__":
    run(load_config())
