"""Train-frozen labels and descriptive statistics; no supervised accuracy claims."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.models.data import PERIODS
from src.models.inference import sequence_starts


def stability_metrics(frame, labels):
    labels = np.asarray(labels)
    pairs = ~sequence_starts(frame)
    n_pairs = int(pairs.sum())
    changed = np.r_[False, labels[1:] != labels[:-1]]
    flips = int((pairs & changed).sum())
    return {
        "n_pairs": n_pairs,
        "flips": flips,
        "flips_per_100d": 100 * flips / n_pairs if n_pairs else np.nan,
        "persistence": 1 - flips / n_pairs if n_pairs else np.nan,
    }


def expected_durations(transmat):
    p = np.diag(transmat)
    with np.errstate(divide="ignore"):
        return 1 / (1 - p)


def weighted_state_stats(frame, probabilities):
    r = frame["log_return"].to_numpy(dtype=float)
    volume = np.expm1(frame["log_dollar_volume"].to_numpy(dtype=float))
    rows = []
    for state in range(probabilities.shape[1]):
        weight = probabilities[:, state]
        mass = float(weight.sum())
        mean = float(weight @ r / mass) if mass else np.nan
        variance = float(weight @ ((r - mean) ** 2) / mass) if mass else np.nan
        rows.append(
            {
                "state_id": state,
                "mean_return": mean,
                "volatility": np.sqrt(variance),
                "soft_share": mass / len(r),
                "mean_quote_volume_usdt": float(weight @ volume / mass) if mass else np.nan,
            }
        )
    return rows


def train_mapping(train, probabilities):
    stats = weighted_state_stats(train, probabilities)
    if not all(np.isfinite(row["volatility"]) for row in stats):
        raise ValueError("Cannot assign train mapping to a state with zero posterior mass")
    # Stable, conservative labels describe measured volatility rather than invent Bull/Bear classes.
    order = sorted(stats, key=lambda row: (row["volatility"], row["state_id"]))
    mapping = {row["state_id"]: f"State {row['state_id']}" for row in stats}
    mapping[order[0]["state_id"]] = "Low-Vol"
    mapping[order[-1]["state_id"]] = "High-Vol"
    return mapping, int(order[-1]["state_id"])


def _run_lengths(frame, labels, state):
    starts = sequence_starts(frame)
    lengths, current = [], 0
    for i, label in enumerate(labels):
        if starts[i] and current:
            lengths.append(current)
            current = 0
        if label == state:
            current += 1
        elif current:
            lengths.append(current)
            current = 0
    if current:
        lengths.append(current)
    return lengths


def regime_statistics(asset, frame, probabilities, mapping, transmat, annual_days, confidence):
    rows = []
    durations = expected_durations(transmat)
    for period in PERIODS:
        mask = frame["split"].eq(period).to_numpy()
        part = frame.loc[mask].reset_index(drop=True)
        prob = probabilities[mask]
        hard = prob.argmax(axis=1)
        for row in weighted_state_stats(part, prob):
            state = row["state_id"]
            runs = _run_lengths(part, hard, state)
            rows.append(
                {
                    "asset": asset,
                    "period": period,
                    **row,
                    "label": mapping[state],
                    "share": float((hard == state).mean()),
                    "n_hard": int((hard == state).sum()),
                    "n_period": len(part),
                    "annualized_volatility": row["volatility"] * np.sqrt(annual_days),
                    "expected_duration": durations[state],
                    "duration_unit": "days",
                    "mean_observed_run_days": float(np.mean(runs)) if runs else np.nan,
                    "n_observed_runs": len(runs),
                    "runs_censored_at_period_and_segment_edges": True,
                    "n_low_confidence": int((prob.max(axis=1) < confidence).sum()),
                    "method": "soft filtered population moments; share=hard filtered occupancy",
                    "return_unit": "daily log return decimal",
                    "volatility_unit": "daily log return standard deviation",
                    "scope": "train description and frozen mapping"
                    if period == "train"
                    else "OOS description; frozen model and mapping",
                }
            )
    return pd.DataFrame(rows)


def transition_table(asset, transmat, mapping):
    durations = expected_durations(transmat)
    return pd.DataFrame(
        [
            {
                "asset": asset,
                "from_state": i,
                "to_state": j,
                "probability": float(p),
                "label": mapping[i],
                "persistence_model": float(transmat[i, i]),
                "expected_duration": durations[i],
                "duration_unit": "days",
            }
            for i, row in enumerate(transmat)
            for j, p in enumerate(row)
        ]
    )


def comparison_table(asset, frame, hmm_alpha, hmm_scores, gmm_alpha, gmm_scores):
    rows = []
    for period in PERIODS:
        mask = frame["split"].eq(period).to_numpy()
        part = frame.loc[mask].reset_index(drop=True)
        for method, alpha, scores, definition in (
            ("HMM", hmm_alpha, hmm_scores, "one-step log p(X_t | X_1:t-1); start pi at segment boundary"),
            ("GMM", gmm_alpha, gmm_scores, "marginal log p(X_t); no historical conditioning"),
        ):
            metrics = stability_metrics(part, alpha[mask].argmax(axis=1))
            rows.append(
                {
                    "asset": asset,
                    "method": method,
                    "period": period,
                    "n": len(part),
                    "start": str(part["date_utc"].iloc[0]),
                    "end": str(part["date_utc"].iloc[-1]),
                    "avg_log_score": float(scores[mask].mean()),
                    "score_definition": definition,
                    **metrics,
                    "notes": "same C1 rows and train scaling; first segment observation included; no ground-truth accuracy",
                }
            )
    return pd.DataFrame(rows)


def probability_frame(asset, frame, probabilities, mapping, high_risk, run_id, kind):
    k = probabilities.shape[1]
    out = frame[["date_utc", "segment_id", "split"]].copy()
    out["asset"], out["model_run_id"], out["K"] = asset, run_id, k
    out["state_id"] = probabilities.argmax(axis=1)
    out["label"] = out["state_id"].map(mapping)
    out["label_map"] = json.dumps(mapping, sort_keys=True)
    out["high_risk_state"] = high_risk
    out["prob_high_risk"] = probabilities[:, high_risk]
    out["probability_type"], out["hindsight"] = kind, kind == "smoothed"
    if kind == "filtered":
        out["as_of_utc"] = pd.to_datetime(frame["available_at_utc"], utc=True)
    for state in range(k):
        out[f"prob_state_{state}"] = probabilities[:, state]
    return out
