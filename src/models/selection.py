"""Train-only EM fits; validation selection; no test-dependent decisions."""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

from src.models.data import ZCOLS
from src.models.inference import diagonal_variances, infer, sequence_lengths
from src.models.reports import train_mapping


def parameter_count(k: int, d: int = 3) -> int:
    return k * k - 1 + 2 * k * d


def choose_candidate(table, tolerance):
    valid = table.loc[table["eligible"]].copy()
    if valid.empty:
        raise ValueError("No eligible HMM fit; inspect B03 rejection_reason; do not relax rules using test")
    best = valid["val_avg_predictive_ll"].max()
    tied = valid.loc[valid["val_avg_predictive_ll"] >= best - tolerance]
    return tied.sort_values(["BIC", "K", "seed"], kind="stable").index[0]


def fit_candidates(frame, settings):
    # Slice before any candidate inference so test cannot enter model selection.
    history = frame.loc[frame["split"].isin(["train", "validation"])].reset_index(drop=True)
    train = history.loc[history["split"] == "train"].reset_index(drop=True)
    x, history_x = train[ZCOLS].to_numpy(), history[ZCOLS].to_numpy()
    lengths = sequence_lengths(train)
    rows, models, validation_probabilities, high_risk_states = [], {}, {}, {}
    for k in settings["states"]:
        for seed in settings["seeds"]:
            row = {
                "K": k,
                "seed": seed,
                "converged": False,
                "eligible": False,
                "selected": False,
                "n_train": len(x),
                "n_validation": int((history["split"] == "validation").sum()),
                "q": parameter_count(k, x.shape[1]),
                "train_ll": np.nan,
                "AIC": np.nan,
                "BIC": np.nan,
                "val_avg_predictive_ll": np.nan,
                "min_state_share": np.nan,
                "min_variance": np.nan,
                "iterations": 0,
                "last_em_gain": np.nan,
                "rejection_reason": "",
                "warnings": "",
            }
            caught = []
            try:
                model = GaussianHMM(
                    n_components=k,
                    covariance_type="diag",
                    n_iter=settings["max_iter"],
                    tol=settings["tol"],
                    min_covar=settings["variance_floor"],
                    random_state=seed,
                    implementation="log",
                )
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    model.fit(x, lengths=lengths)
                h = list(model.monitor_.history)
                gain = h[-1] - h[-2] if len(h) >= 2 else np.inf
                # hmmlearn's monitor also calls reaching max_iter 'converged'.
                # Require an actual small EM gain, and reject material decreases.
                converged = bool(
                    np.isfinite(gain) and -settings["em_negative_tolerance"] <= gain < settings["tol"]
                )
                cov = diagonal_variances(model)
                covariance_ok = bool(np.isfinite(cov).all() and (cov >= settings["variance_floor"]).all())
                alpha, _ = infer(model, x, train)
                shares = np.bincount(alpha.argmax(axis=1), minlength=k) / len(x)
                row["train_hard_state_shares"] = json.dumps(shares.tolist())
                row["train_soft_state_shares"] = json.dumps(alpha.mean(axis=0).tolist())
                row["train_state_means_scaled"] = json.dumps(model.means_.tolist())
                ll = model.score(x, lengths=lengths)
                reasons = []
                if not converged:
                    reasons.append("EM did not converge with an acceptable likelihood gain")
                if not covariance_ok:
                    reasons.append("diagonal variance below floor or nonfinite")
                if shares.min() < settings["min_state_share"]:
                    reasons.append("hard-filtered train state share below threshold")
                if not np.isfinite(ll):
                    reasons.append("nonfinite train likelihood")
                row.update(
                    converged=converged,
                    min_variance=float(cov.min()),
                    iterations=model.monitor_.iter,
                    last_em_gain=float(gain),
                    train_ll=float(ll),
                    AIC=-2 * ll + 2 * row["q"],
                    BIC=-2 * ll + row["q"] * np.log(len(x)),
                    min_state_share=float(shares.min()),
                )
                if not reasons:
                    history_alpha, score = infer(model, history_x, history)
                    mask = history["split"].eq("validation").to_numpy()
                    val_score = float(score[mask].mean())
                    if not np.isfinite(val_score):
                        reasons.append("nonfinite validation predictive log-score")
                    else:
                        row["val_avg_predictive_ll"] = val_score
                        _, risk_state = train_mapping(train, alpha)
                        row["high_risk_state"] = risk_state
                        val_labels = history_alpha[mask].argmax(axis=1)
                        row["validation_hard_state_shares"] = json.dumps(
                            (np.bincount(val_labels, minlength=k) / len(val_labels)).tolist()
                        )
                        row["validation_high_risk_days"] = int((val_labels == risk_state).sum())
                        models[(k, seed)] = model
                        validation_probabilities[(k, seed)] = history_alpha[mask]
                        high_risk_states[(k, seed)] = risk_state
                row["eligible"] = not reasons
                row["rejection_reason"] = "; ".join(reasons)
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                row["rejection_reason"] = f"{type(exc).__name__}: {exc}"
            row["warnings"] = "; ".join(str(w.message) for w in caught)
            rows.append(row)
    table = pd.DataFrame(rows)
    stability = []
    for k in settings["states"]:
        keys = sorted(key for key in validation_probabilities if key[0] == k)
        for i, left in enumerate(keys):
            for right in keys[i + 1 :]:
                left_prob, right_prob = validation_probabilities[left], validation_probabilities[right]
                left_risk, right_risk = high_risk_states[left], high_risk_states[right]
                left_labels, right_labels = left_prob.argmax(axis=1), right_prob.argmax(axis=1)
                left_event, right_event = left_labels == left_risk, right_labels == right_risk
                union = left_event | right_event
                stability.append(
                    {
                        "K": k,
                        "seed_a": left[1],
                        "seed_b": right[1],
                        "period": "validation",
                        "n_observations": len(left_labels),
                        "adjusted_rand_index": adjusted_rand_score(left_labels, right_labels),
                        "high_risk_disagreement_rate": float(
                            np.mean((left_labels == left_risk) != (right_labels == right_risk))
                        ),
                        "high_risk_probability_mae": float(
                            np.abs(left_prob[:, left_risk] - right_prob[:, right_risk]).mean()
                        ),
                        "high_risk_union_days": int(union.sum()),
                        "high_risk_event_jaccard": float((left_event & right_event).sum() / union.sum())
                        if union.any()
                        else np.nan,
                        "high_risk_disagreement_on_union": float(
                            (left_event != right_event).sum() / union.sum()
                        )
                        if union.any()
                        else np.nan,
                        "method": "permutation-invariant filtered labels; risk states mapped on train; no refit on validation",
                    }
                )
    stability_cols = [
        "K",
        "seed_a",
        "seed_b",
        "period",
        "n_observations",
        "adjusted_rand_index",
        "high_risk_disagreement_rate",
        "high_risk_probability_mae",
        "high_risk_union_days",
        "high_risk_event_jaccard",
        "high_risk_disagreement_on_union",
        "method",
    ]
    try:
        winner = choose_candidate(table, settings["score_tie_tolerance"])
    except ValueError:
        return None, table, pd.DataFrame(stability, columns=stability_cols)
    table.loc[winner, "selected"] = True
    chosen = table.loc[winner]
    return (
        models[(int(chosen["K"]), int(chosen["seed"]))],
        table,
        pd.DataFrame(stability, columns=stability_cols),
    )


def fit_gmm(train, k, settings):
    """Five train-only initializations, best valid train likelihood; independent cluster IDs."""
    x = train[ZCOLS].to_numpy()
    rows, candidates = [], []
    for seed in settings["seeds"]:
        row = {
            "K": k,
            "seed": seed,
            "converged": False,
            "eligible": False,
            "train_ll": np.nan,
            "selected": False,
            "rejection_reason": "",
            "warnings": "",
        }
        caught = []
        try:
            model = GaussianMixture(
                n_components=k,
                covariance_type="diag",
                n_init=1,
                max_iter=settings["max_iter"],
                tol=settings["tol"],
                reg_covar=settings["variance_floor"],
                random_state=seed,
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model.fit(x)
            ll = float(model.score_samples(x).sum())
            eligible = bool(
                model.converged_
                and np.isfinite(ll)
                and np.isfinite(model.covariances_).all()
                and (model.covariances_ >= settings["variance_floor"]).all()
            )
            row.update(
                converged=bool(model.converged_),
                eligible=eligible,
                train_ll=ll,
                rejection_reason="" if eligible else "nonconvergent or degenerate GMM",
            )
            if eligible:
                candidates.append((ll, -seed, len(rows), model))
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            row["rejection_reason"] = f"{type(exc).__name__}: {exc}"
        row["warnings"] = "; ".join(str(w.message) for w in caught)
        rows.append(row)
    table = pd.DataFrame(rows)
    if not candidates:
        return None, table
    _, _, index, model = max(candidates, key=lambda item: item[:2])
    table.loc[index, "selected"] = True
    return model, table
