"""A passing implementation test must not turn unstable scientific evidence into 'stable'."""

from __future__ import annotations

import copy
import itertools

import numpy as np
import pandas as pd
import pytest

from src.models.data import load_asset
from src.models.handoff import load_filtered
from src.models.selection import fit_candidates
from src.models.stability import summarize_seed_stability
from src.utils.config import path_of, read_json

CRITERIA = {"min_eligible_seeds": 3, "median_ari_min": 0.8, "min_ari_min": 0.5}


def diagnostic_tables(scores, seeds=(0, 1, 2)):
    """Controlled diagnostic records, not generated market data or research results."""
    fits = pd.DataFrame(
        {
            "K": [3] * len(seeds),
            "seed": list(seeds),
            "eligible": [True] * len(seeds),
            "selected": [True] + [False] * (len(seeds) - 1),
        }
    )
    rows = [
        {
            "K": 3,
            "seed_a": a,
            "seed_b": b,
            "adjusted_rand_index": score,
            "high_risk_disagreement_rate": 0.1,
            "high_risk_probability_mae": 0.1,
        }
        for (a, b), score in zip(itertools.combinations(seeds, 2), scores)
    ]
    columns = [
        "K",
        "seed_a",
        "seed_b",
        "adjusted_rand_index",
        "high_risk_disagreement_rate",
        "high_risk_probability_mae",
    ]
    return fits, pd.DataFrame(rows, columns=columns)


def test_low_agreement_cannot_be_reported_as_stable():
    fits, pairs = diagnostic_tables([-0.04, 0.03, 0.99])
    result = summarize_seed_stability(fits, pairs, CRITERIA).iloc[0]
    assert result["status"] == "unstable"
    assert result["pairs_complete"]
    assert result["ari_median"] == 0.03
    assert not result["used_for_model_selection"]


def test_all_pair_minimum_prevents_majority_masking_an_unstable_seed():
    fits, pairs = diagnostic_tables([0.2, 0.95, 0.96])
    assert summarize_seed_stability(fits, pairs, CRITERIA).iloc[0]["status"] == "unstable"
    pairs["adjusted_rand_index"] = [0.9, 0.95, 0.96]
    assert summarize_seed_stability(fits, pairs, CRITERIA).iloc[0]["status"] == "stable"


@pytest.mark.parametrize("problem", ["missing", "duplicate", "nan"])
def test_incomplete_pairs_never_imply_stability(problem):
    fits, pairs = diagnostic_tables([0.99, 0.99, 0.99])
    if problem == "missing":
        pairs = pairs.iloc[:2]
    elif problem == "duplicate":
        pairs = pd.concat([pairs, pairs.iloc[:1]], ignore_index=True)
    else:
        pairs.loc[0, "adjusted_rand_index"] = np.nan
    assert summarize_seed_stability(fits, pairs, CRITERIA).iloc[0]["status"] == "incomplete_pairs"


def test_too_few_seeds_and_all_rejected_states_remain_visible():
    fits, pairs = diagnostic_tables([1.0], seeds=(0, 1))
    assert summarize_seed_stability(fits, pairs, CRITERIA).iloc[0]["status"] == "insufficient_seeds"
    fits["eligible"], fits["selected"] = False, False
    result = summarize_seed_stability(fits, pairs.iloc[:0], CRITERIA).iloc[0]
    assert result["status"] == "no_eligible_fits"
    assert result["n_eligible_seeds"] == 0
    assert pd.isna(result["ari_median"])


def test_recomputed_summary_matches_raw_evidence_and_manifest(cfg):
    fits = pd.read_csv(path_of(cfg, "tables") / "B03_model_selection.csv")
    pairs = pd.read_csv(path_of(cfg, "tables") / "C2_seed_stability.csv")
    stored = pd.read_csv(path_of(cfg, "tables") / "C2_stability_summary.csv")
    manifest = read_json(path_of(cfg, "metadata") / "model_manifest.json")
    for asset in cfg["assets"]:
        actual = summarize_seed_stability(
            fits.loc[fits["asset"] == asset], pairs.loc[pairs["asset"] == asset], cfg["models"]["stability"]
        )
        expected = stored.loc[stored["asset"] == asset].drop(columns="asset").reset_index(drop=True)
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
        selected = actual.loc[actual["selected_K"]].iloc[0]
        info = manifest["assets"][asset]
        assert info["seed_stability_status"] == selected["status"]
        assert info["seed_stability_pairs"] == selected["n_pairs"]
        assert np.isclose(info["seed_stability"]["ari_median"], selected["ari_median"])
        if selected["status"] != "stable":
            with pytest.raises(ValueError, match="seed stability"):
                load_filtered(cfg, asset, require_stable=True)
        else:
            load_filtered(cfg, asset, require_stable=True)
    flagged = stored.loc[stored["selected_K"], "status"].ne("stable").any()
    assert (manifest["quality_status"] == "warning_seed_sensitivity") == flagged


def test_stability_flags_do_not_choose_the_model_or_use_test(cfg):
    # Reuse real C1 data. Changing diagnostic thresholds or held-out rows cannot change B03.
    frame, _, _ = load_asset(cfg, "BTC")
    settings = copy.deepcopy(cfg["models"])
    settings.update(states=[2], seeds=[0])
    _, before, _ = fit_candidates(frame, settings)
    settings["stability"] = {"min_eligible_seeds": 99, "median_ari_min": 1, "min_ari_min": 1}
    changed = frame.copy()
    mask = changed["split"].eq("test")
    changed.loc[mask, "z_log_return"] = changed.loc[mask, "z_log_return"].to_numpy()[::-1]
    _, after, _ = fit_candidates(changed, settings)
    pd.testing.assert_frame_equal(before, after)
