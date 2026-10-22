"""Descriptive seed sensitivity diagnostics, separate from model selection and code correctness."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd


def summarize_seed_stability(selection, pairs, criteria):
    """Include every K, even when no eligible seeds or complete pair comparisons exist.

    These operational thresholds flag sensitivity; they are not a statistical
    significance test and never change the validation-based model selection.
    """
    rows = []
    for k, fits in selection.groupby("K", sort=True):
        eligible = sorted(fits.loc[fits["eligible"], "seed"].astype(int).tolist())
        comparisons = pairs.loc[pairs["K"].eq(k)]
        expected = set(itertools.combinations(eligible, 2))
        actual = [
            tuple(sorted((int(a), int(b))))
            for a, b in comparisons[["seed_a", "seed_b"]].itertuples(index=False, name=None)
        ]
        scores = comparisons["adjusted_rand_index"].to_numpy(dtype=float)
        complete = (
            len(actual) == len(expected)
            and set(actual) == expected
            and np.isfinite(scores).all()
            and (scores >= -1).all()
            and (scores <= 1).all()
        )
        chosen = fits.loc[fits["selected"]]
        selected_seed = int(chosen["seed"].iloc[0]) if len(chosen) else None
        reference = (
            comparisons.loc[comparisons["seed_a"].eq(selected_seed) | comparisons["seed_b"].eq(selected_seed)]
            if selected_seed is not None
            else comparisons.iloc[:0]
        )
        median = float(np.median(scores)) if len(scores) else np.nan
        minimum = float(np.min(scores)) if len(scores) else np.nan
        event_pairs = (
            comparisons.loc[comparisons["high_risk_union_days"].gt(0)]
            if "high_risk_union_days" in comparisons
            else comparisons.iloc[:0]
        )
        if not eligible:
            status, reason = "no_eligible_fits", "All fits failed the existing train fit quality rules"
        elif len(eligible) < criteria["min_eligible_seeds"]:
            status, reason = "insufficient_seeds", "Too few eligible seeds to assess seed stability"
        elif not complete:
            status, reason = "incomplete_pairs", "Missing, duplicate, nonfinite or invalid pair comparisons"
        elif median < criteria["median_ari_min"] or minimum < criteria["min_ari_min"]:
            status, reason = (
                "unstable",
                "Validation labels depend on initialization under the configured descriptive thresholds",
            )
        else:
            status, reason = "stable", "All configured descriptive seed stability criteria met"
        rows.append(
            {
                "K": int(k),
                "period": "validation",
                "selected_K": bool(len(chosen)),
                "selected_seed": selected_seed,
                "n_requested_seeds": len(fits),
                "n_eligible_seeds": len(eligible),
                "eligible_fraction": len(eligible) / len(fits),
                "n_expected_pairs": len(expected),
                "n_pairs": len(comparisons),
                "pairs_complete": complete,
                "ari_min": minimum,
                "ari_median": median,
                "ari_mean": float(scores.mean()) if len(scores) else np.nan,
                "ari_max": float(scores.max()) if len(scores) else np.nan,
                "selected_seed_ari_mean": float(reference["adjusted_rand_index"].mean())
                if len(reference)
                else np.nan,
                "high_risk_disagreement_mean": float(comparisons["high_risk_disagreement_rate"].mean())
                if len(comparisons)
                else np.nan,
                "high_risk_probability_mae_mean": float(comparisons["high_risk_probability_mae"].mean())
                if len(comparisons)
                else np.nan,
                "n_high_risk_event_pairs": len(event_pairs),
                "high_risk_event_jaccard_mean": float(event_pairs["high_risk_event_jaccard"].mean())
                if len(event_pairs)
                else np.nan,
                "high_risk_disagreement_on_union_mean": float(
                    event_pairs["high_risk_disagreement_on_union"].mean()
                )
                if len(event_pairs)
                else np.nan,
                "status": status,
                "reason": reason,
                **criteria,
                "used_for_model_selection": False,
                "interpretation": "descriptive seed sensitivity; stable does not establish economic validity or trading performance",
            }
        )
    return pd.DataFrame(rows)
