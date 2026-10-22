"""C2 entry point: python -m src.models.run [--config config/config.yaml]."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.artifacts import canonical_metadata, save_checked_bundle
from src.models.data import CORE, ZCOLS, load_asset
from src.models.figures import export_figures
from src.models.inference import (
    diagonal_variances,
    emission_log_density,
    forward_filter,
    sequence_lengths,
    sequence_starts,
    smooth_probabilities,
)
from src.models.reports import (
    comparison_table,
    probability_frame,
    regime_statistics,
    train_mapping,
    transition_table,
)
from src.models.selection import fit_candidates, fit_gmm
from src.models.stability import summarize_seed_stability
from src.utils.config import ROOT, file_hash, get_logger, load_config, path_of, write_json

log = get_logger("C2-models")


def _settings(cfg):
    settings = cfg["models"]
    if settings["covariance_type"] != "diag" or not {2, 3, 4}.issubset(settings["states"]):
        raise ValueError("C2 requires diagonal covariance and candidates K=2,3,4")
    if set(settings["states"]) - {2, 3, 4, 5} or len(set(settings["states"])) != len(settings["states"]):
        raise ValueError("Only distinct candidates K=2,3,4 and optional 5 are supported")
    if len(settings["seeds"]) < 5 or len(set(settings["seeds"])) != len(settings["seeds"]):
        raise ValueError("At least five distinct random seeds are required")
    if settings["variance_floor"] <= 0 or not 0 < settings["min_state_share"] < 1:
        raise ValueError("Invalid covariance/state-share quality thresholds")
    if settings["tol"] <= 0 or settings["max_iter"] < 2 or settings["score_tie_tolerance"] < 0:
        raise ValueError("Invalid convergence/selection settings")
    criteria = settings["stability"]
    if (
        not 2 <= criteria["min_eligible_seeds"] <= len(settings["seeds"])
        or not -1 <= criteria["min_ari_min"] <= criteria["median_ari_min"] <= 1
    ):
        raise ValueError("Invalid descriptive stability thresholds")
    return settings


def _assert_probabilities(prob, tolerance):
    if (
        not np.isfinite(prob).all()
        or (prob < 0).any()
        or (prob > 1 + tolerance).any()
        or not np.allclose(prob.sum(axis=1), 1, atol=tolerance, rtol=0)
    ):
        raise ValueError("Invalid state probabilities")


def run(cfg):
    settings = _settings(cfg)
    folders = {key: path_of(cfg, key) for key in ("models", "metadata", "predictions", "tables", "figures")}
    packages = {
        p: importlib.metadata.version(p)
        for p in ("numpy", "pandas", "scipy", "scikit-learn", "hmmlearn", "pyarrow", "joblib", "matplotlib")
    }
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "config_version": cfg["version"],
        "config_hash": cfg["_config_hash"],
        "python": platform.python_version(),
        "packages": packages,
        "settings": settings,
        "core_features": CORE,
        "assets": {},
        "test_used_for_selection": False,
        "artifact_reload_checks": {},
        "segment_first_observation_in_log_score": True,
        "selection_rule": "max validation avg predictive LL; ties within configured tolerance: min train BIC, K, seed",
        "limitations": [
            "RV21 overlap makes conditional independence approximate",
            "return and volatility differences on train are not independent evidence",
            "quote volume is not market depth or a full liquidity test",
            "regime IDs are latent model states, not ground truth classes",
            "RQ3/NHHMM/K-means not implemented or tested",
            "A-02 meeting and C3 acceptance require team confirmation",
        ],
    }
    all_selection, all_stats, all_trans, all_comparison = [], [], [], []
    all_filtered, all_smoothed, all_gmm, all_stability, all_gmm_fits = [], [], [], [], []
    all_stability_summaries = []
    outputs = []

    def csv(frame, folder, name):
        destination = folders[folder] / name
        frame.to_csv(destination, index=False)
        outputs.append(destination)

    manifest_path = folders["metadata"] / "model_manifest.json"
    write_json(manifest, manifest_path)
    try:
        for asset in cfg["assets"]:
            frame, c1_info, provenance = load_asset(cfg, asset)
            log.info("%s: fit train only; selecting on validation", asset)
            model, selection, sensitivity = fit_candidates(frame, settings)
            selection.insert(0, "asset", asset)
            all_selection.append(selection)
            stability_summary = summarize_seed_stability(selection, sensitivity, settings["stability"])
            stability_summary.insert(0, "asset", asset)
            all_stability_summaries.append(stability_summary)
            csv(pd.concat(all_stability_summaries, ignore_index=True), "tables", "C2_stability_summary.csv")
            # Persist all candidates even when no eligible model exists.
            csv(pd.concat(all_selection, ignore_index=True), "tables", "B03_model_selection.csv")
            if model is None:
                raise ValueError(f"{asset}: no eligible HMM; see B03_model_selection.csv")
            chosen = selection.loc[selection["selected"]].iloc[0]
            k, seed = int(chosen["K"]), int(chosen["seed"])
            selected_row = stability_summary.loc[stability_summary["selected_K"]].iloc[0]
            selected_stability = selected_row.where(pd.notna(selected_row), None).to_dict()
            if selected_stability["status"] != "stable":
                log.warning(
                    "%s K=%d: seed stability %s (median ARI %.6f, min ARI %.6f); selection unchanged",
                    asset,
                    k,
                    selected_stability["status"],
                    selected_stability["ari_median"]
                    if selected_stability["ari_median"] is not None
                    else np.nan,
                    selected_stability["ari_min"] if selected_stability["ari_min"] is not None else np.nan,
                )
            tr_mask = frame["split"].eq("train").to_numpy()
            train = frame.loc[tr_mask].reset_index(drop=True)
            # Mapping uses train probabilities only, before test evaluation.
            train_b = emission_log_density(train[ZCOLS], model.means_, diagonal_variances(model))
            train_alpha, _ = forward_filter(
                train_b, model.startprob_, model.transmat_, sequence_starts(train)
            )
            mapping, high_risk = train_mapping(train, train_alpha)
            gmm, gmm_fits = fit_gmm(train, k, settings)
            gmm_fits.insert(0, "asset", asset)
            all_gmm_fits.append(gmm_fits)
            csv(pd.concat(all_gmm_fits, ignore_index=True), "tables", "GMM_initializations.csv")
            if gmm is None:
                raise ValueError(f"{asset}: no valid GMM benchmark initialization")
            gmm_map, gmm_risk = train_mapping(train, gmm.predict_proba(train[ZCOLS].to_numpy()))
            frozen_identity = {
                "asset": asset,
                "K": k,
                "seed": seed,
                "settings": settings,
                "train_validation_values_sha256": hashlib.sha256(
                    pd.util.hash_pandas_object(
                        frame.loc[
                            frame["split"] != "test", ["date_utc", "segment_id", "split"] + CORE + ZCOLS
                        ],
                        index=False,
                    ).values.tobytes()
                ).hexdigest(),
                "scaler": c1_info["scaler"],
                "packages": packages,
                "source_sha256": {
                    f: file_hash(Path(__file__).with_name(f))
                    for f in (
                        "run.py",
                        "selection.py",
                        "inference.py",
                        "data.py",
                        "reports.py",
                        "stability.py",
                        "artifacts.py",
                        "handoff.py",
                    )
                },
            }
            run_id = (
                asset.lower()
                + "-"
                + hashlib.sha256(
                    json.dumps(frozen_identity, sort_keys=True, default=str).encode()
                ).hexdigest()[:16]
            )
            frozen = {
                "model_run_id": run_id,
                **frozen_identity,
                "label_map": mapping,
                "high_risk_state": high_risk,
                "gmm_label_map": gmm_map,
                "gmm_high_risk_cluster": gmm_risk,
                "gmm_seed": int(gmm_fits.loc[gmm_fits["selected"], "seed"].iloc[0]),
                "selection_locked_at_utc": datetime.now(timezone.utc).isoformat(),
                "seed_stability": selected_stability,
                "seed_stability_status": selected_stability["status"],
                "test_evaluated": False,
                "transition_matrix": model.transmat_.tolist(),
                "startprob": model.startprob_.tolist(),
                "c1_provenance": provenance,
            }
            frozen_path = folders["metadata"] / f"model_{asset.lower()}.json"
            lock_path = folders["metadata"] / f"model_{asset.lower()}_selection.json"
            write_json(canonical_metadata(frozen | {"stage": "selection_locked"}), lock_path)
            model_path = folders["models"] / f"hmm_{asset.lower()}.joblib"
            gmm_path = folders["models"] / f"gmm_{asset.lower()}.joblib"
            outputs.extend([model_path, gmm_path, frozen_path, lock_path])

            # Selection and train mapping are now persisted. First evaluate frozen model on test.
            x = frame[ZCOLS].to_numpy()
            b = emission_log_density(x, model.means_, diagonal_variances(model))
            starts = sequence_starts(frame)
            alpha, scores = forward_filter(b, model.startprob_, model.transmat_, starts)
            gamma = smooth_probabilities(b, model.startprob_, model.transmat_, starts)
            gmm_prob, gmm_scores = gmm.predict_proba(x), gmm.score_samples(x)
            for probabilities in (alpha, gamma, gmm_prob):
                _assert_probabilities(probabilities, settings["probability_tolerance"])
            prefix_n = int((frame["split"] != "test").sum())
            prefix, _ = forward_filter(b[:prefix_n], model.startprob_, model.transmat_, starts[:prefix_n])
            if not np.allclose(prefix, alpha[:prefix_n], atol=settings["probability_tolerance"], rtol=0):
                raise ValueError("Causal prefix consistency failed")
            # Independent check against hmmlearn's whole-sequence likelihood.
            if not np.isclose(
                scores.sum(), model.score(x, lengths=sequence_lengths(frame)), rtol=1e-9, atol=1e-6
            ):
                raise ValueError("Forward likelihood disagrees with fitted library")
            filtered = probability_frame(asset, frame, alpha, mapping, high_risk, run_id, "filtered")
            smoothed = probability_frame(asset, frame, gamma, mapping, high_risk, run_id, "smoothed")
            gmm_out = probability_frame(asset, frame, gmm_prob, gmm_map, gmm_risk, run_id, "gmm_posterior")
            filtered["prob_high_risk_next_day"] = (alpha @ model.transmat_)[:, high_risk]
            comparison = comparison_table(asset, frame, alpha, scores, gmm_prob, gmm_scores)
            stats = regime_statistics(
                asset,
                frame,
                alpha,
                mapping,
                model.transmat_,
                cfg["features"]["annualization_days"],
                settings["low_confidence_threshold"],
            )
            trans = transition_table(asset, model.transmat_, mapping)
            sensitivity.insert(0, "asset", asset)
            all_stability.append(sensitivity)
            all_stats.append(stats)
            all_trans.append(trans)
            all_comparison.append(comparison)
            all_filtered.append(filtered)
            all_smoothed.append(smoothed)
            all_gmm.append(gmm_out)
            for kind, out in (("filtered", filtered), ("smoothed", smoothed), ("gmm", gmm_out)):
                csv(out, "predictions", f"regime_{kind}_{asset.lower()}.csv")
            if settings["export_figures"]:
                export_figures(
                    asset,
                    frame,
                    filtered,
                    smoothed,
                    gmm_out,
                    comparison,
                    folders["figures"],
                    settings["figure_dpi"],
                )
                outputs.extend(
                    p
                    for p in folders["figures"].glob(f"H0[234]_*_{asset.lower()}.*")
                    if p.suffix in (".svg", ".png")
                )
            frozen["test_evaluated"] = True
            frozen["n_by_period"] = {
                period: int((frame["split"] == period).sum()) for period in ("train", "validation", "test")
            }
            frozen["n_segments"] = len(sequence_lengths(frame))
            frozen["seed_stability_pairs"] = int(selected_stability["n_pairs"])
            frozen["seed_stability_pairs_all_K"] = len(sensitivity)
            frozen["probability_schema_version"] = "1.1.0"
            frozen["finalized_at_utc"] = datetime.now(timezone.utc).isoformat()
            frozen = canonical_metadata(frozen)
            # Persist both models only after ALL final metadata fields exist.
            probe = train[ZCOLS].to_numpy()
            manifest["artifact_reload_checks"][asset] = [
                save_checked_bundle(
                    model, frozen, model_path, probe, kind="HMM", lengths=sequence_lengths(train)
                ),
                save_checked_bundle(gmm, frozen, gmm_path, probe, kind="GMM"),
            ]
            write_json(frozen, frozen_path)
            manifest["assets"][asset] = frozen
            log.info(
                "%s: selected K=%d seed=%d; test HMM log-score %.6f",
                asset,
                k,
                seed,
                comparison.loc[
                    (comparison["method"] == "HMM") & (comparison["period"] == "test"), "avg_log_score"
                ].iloc[0],
            )

        for frames, name in (
            (all_stats, "B04_regime_statistics.csv"),
            (all_trans, "B05_transition_matrix.csv"),
            (all_comparison, "B07_hmm_vs_gmm.csv"),
            (all_stability, "C2_seed_stability.csv"),
        ):
            csv(pd.concat(frames, ignore_index=True), "tables", name)
        # When K differs, absent state columns remain empty for that asset, never synthetic zeros.
        for frames, name in (
            (all_filtered, "regime_filtered.csv"),
            (all_smoothed, "regime_smoothed.csv"),
            (all_gmm, "regime_gmm.csv"),
        ):
            csv(pd.concat(frames, ignore_index=True), "predictions", name)
        manifest["status"] = "complete"
        manifest["quality_status"] = (
            "stable_under_seed_diagnostics"
            if all(info["seed_stability_status"] == "stable" for info in manifest["assets"].values())
            else "warning_seed_sensitivity"
        )
        manifest["output_hashes"] = {
            p.relative_to(ROOT).as_posix(): file_hash(p) for p in sorted(set(outputs))
        }
    except Exception as exc:
        manifest["status"], manifest["error"] = "failed", f"{type(exc).__name__}: {exc}"
        raise
    finally:
        write_json(manifest, manifest_path)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description="C2 HMM/GMM core models and C3 handoff")
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    run(load_config(args.config))


if __name__ == "__main__":
    main()
