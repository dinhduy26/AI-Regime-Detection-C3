"""C2 acceptance on actual C1 data, plus the specification's analytic probability example."""

from __future__ import annotations

import copy
import itertools
import json

import joblib
import numpy as np
import pandas as pd
import pytest

from src.models.data import ZCOLS, load_asset
from src.models.handoff import UnstableModelWarning, load_filtered
from src.models.inference import (
    diagonal_variances,
    emission_log_density,
    forward_filter,
    infer,
    sequence_lengths,
    sequence_starts,
    smooth_probabilities,
)
from src.models.reports import expected_durations, stability_metrics, train_mapping
from src.models.selection import choose_candidate, fit_candidates, parameter_count
from src.utils.config import path_of, read_json


def test_forward_specification_hand_example():
    # alpha[t-1]=(0.8,0.2), q[t]=(0.76,0.24), emissions=(0.2,0.5).
    alpha, scores = forward_filter(np.log([[1, 1], [0.2, 0.5]]), [0.8, 0.2], [[0.9, 0.1], [0.2, 0.8]])
    assert np.allclose(alpha[-1], [0.5588235294117647, 0.4411764705882353])
    assert np.isclose(scores[-1], np.log(0.272))


def test_forward_and_smoothing_against_exact_path_enumeration():
    # Analytic finite-state calculation, not fabricated market observations or results.
    pi, p = np.array([0.7, 0.3]), np.array([[0.9, 0.1], [0.2, 0.8]])
    emissions = np.array([[0.2, 0.5], [0.9, 0.1], [0.3, 0.8]])
    alpha, score = forward_filter(np.log(emissions), pi, p)
    smooth = smooth_probabilities(np.log(emissions), pi, p)
    joint = np.zeros_like(emissions)
    likelihood = 0
    for path in itertools.product(range(2), repeat=3):
        weight = pi[path[0]] * emissions[0, path[0]]
        for t in (1, 2):
            weight *= p[path[t - 1], path[t]] * emissions[t, path[t]]
        likelihood += weight
        for t, state in enumerate(path):
            joint[t, state] += weight
    assert np.isclose(score.sum(), np.log(likelihood))
    assert np.allclose(smooth, joint / likelihood)
    assert np.allclose(smooth[-1], alpha[-1])
    assert not np.allclose(smooth[0], alpha[0])


def test_parameter_count_and_absorbing_state_duration():
    assert parameter_count(3, 3) == 26
    assert np.allclose(expected_durations(np.array([[0.9, 0.1], [0, 1]])), [10, np.inf])


def test_smoothing_retains_state_mass_below_float_exponential_range():
    # A future observation can restore tiny posterior mass; log(exp(alpha)) would lose it.
    b = np.array([[0.0, -1000.0], [-1001.0, 0.0]])
    probabilities = smooth_probabilities(b, [0.5, 0.5], np.eye(2))
    expected = np.array([1 / (1 + np.e), np.e / (1 + np.e)])
    assert np.allclose(probabilities, [expected, expected])


def test_selection_tie_rule_and_invalid_fits():
    table = pd.DataFrame(
        {
            "eligible": [True, True, True, False],
            "val_avg_predictive_ll": [1, 1, 1, 99],
            "BIC": [20, 10, 10, 0],
            "K": [2, 4, 3, 2],
            "seed": [0, 0, 1, 1],
        }
    )
    assert choose_candidate(table, 1e-8) == 2
    table["eligible"] = False
    with pytest.raises(ValueError, match="No eligible"):
        choose_candidate(table, 1e-8)


@pytest.fixture(scope="module")
def actual(cfg):
    return {asset: load_asset(cfg, asset)[0] for asset in cfg["assets"]}


@pytest.fixture(scope="module")
def artifacts(cfg):
    manifest_path = path_of(cfg, "metadata") / "model_manifest.json"
    if not manifest_path.exists():
        pytest.fail("Run python -m src.models.run before C2 artifact acceptance tests")
    manifest = read_json(manifest_path)
    assert manifest["status"] == "complete"
    return {a: joblib.load(path_of(cfg, "models") / f"hmm_{a.lower()}.joblib") for a in cfg["assets"]}


@pytest.mark.parametrize("asset", ["BTC", "ETH"])
def test_hmm_parameters_and_likelihood(cfg, actual, artifacts, asset):
    frame, model = actual[asset], artifacts[asset]["model"]
    assert np.all(model.transmat_ >= 0)
    assert np.allclose(model.transmat_.sum(axis=1), 1)
    assert (diagonal_variances(model) > 0).all()
    assert model.covariance_type == "diag"
    _, scores = infer(model, frame[ZCOLS], frame)
    assert np.isclose(scores.sum(), model.score(frame[ZCOLS].to_numpy(), lengths=sequence_lengths(frame)))


@pytest.mark.parametrize("asset", ["BTC", "ETH"])
def test_causal_prefix_and_library_smoothing(actual, artifacts, asset):
    frame, model = actual[asset], artifacts[asset]["model"]
    x = frame[ZCOLS].to_numpy()
    alpha, scores = infer(model, x, frame)
    for period in ("train", "validation"):
        n = int(frame["split"].isin(["train"] if period == "train" else ["train", "validation"]).sum())
        prefix, prefix_scores = infer(model, x[:n], frame.iloc[:n])
        assert np.allclose(prefix, alpha[:n], atol=1e-12)
        assert np.allclose(prefix_scores, scores[:n], atol=1e-12)
    b = emission_log_density(x, model.means_, diagonal_variances(model))
    smooth = smooth_probabilities(b, model.startprob_, model.transmat_, sequence_starts(frame))
    assert np.allclose(smooth, model.predict_proba(x, lengths=sequence_lengths(frame)), atol=1e-8)


def test_gap_resets_and_flips_use_real_calendar(actual, artifacts):
    # Remove an actual day from a short real-data slice to check a discontinuity.
    frame = actual["BTC"].iloc[:40].drop(actual["BTC"].index[20]).reset_index(drop=True)
    starts = sequence_starts(frame)
    assert starts[20]
    assert sequence_lengths(frame) == [20, 19]
    model = artifacts["BTC"]["model"]
    alpha, _ = infer(model, frame[ZCOLS], frame)
    suffix, _ = infer(model, frame[ZCOLS].iloc[20:], frame.iloc[20:])
    assert np.allclose(alpha[20:], suffix)
    labels = np.r_[np.zeros(20, dtype=int), np.ones(19, dtype=int)]
    metrics = stability_metrics(frame, labels)
    assert metrics["flips"] == 0
    assert metrics["n_pairs"] == 37
    assert metrics["persistence"] == 1


def test_candidate_fitting_never_receives_test(cfg, actual, monkeypatch):
    from src.models import selection

    original_fit, original_infer = selection.GaussianHMM.fit, selection.infer
    frame = actual["BTC"]
    n_train = int(frame["split"].eq("train").sum())
    calls = []

    def checked_fit(self, x, lengths=None):
        assert len(x) == n_train
        assert sum(lengths) == n_train
        return original_fit(self, x, lengths=lengths)

    def checked_infer(model, x, history):
        assert not history["split"].eq("test").any()
        calls.append(set(history["split"]))
        return original_infer(model, x, history)

    monkeypatch.setattr(selection.GaussianHMM, "fit", checked_fit)
    monkeypatch.setattr(selection, "infer", checked_infer)
    settings = copy.deepcopy(cfg["models"])
    settings.update(states=[2], seeds=[0])
    _, before, _ = fit_candidates(frame, settings)
    changed = frame.copy()
    # Reverse only actual held-out observations; candidate results must be identical.
    mask = changed["split"].eq("test")
    changed.loc[mask, ZCOLS] = changed.loc[mask, ZCOLS].to_numpy()[::-1]
    _, after, _ = fit_candidates(changed, settings)
    pd.testing.assert_frame_equal(before, after)
    assert calls


@pytest.mark.parametrize("asset", ["BTC", "ETH"])
def test_handoff_mapping_asof_and_probability_contract(cfg, actual, artifacts, asset):
    frame = actual[asset]
    model, metadata = artifacts[asset]["model"], artifacts[asset]["metadata"]
    out = pd.read_csv(path_of(cfg, "predictions") / f"regime_filtered_{asset.lower()}.csv")
    hindsight = pd.read_csv(path_of(cfg, "predictions") / f"regime_smoothed_{asset.lower()}.csv")
    k = model.n_components
    cols = [f"prob_state_{i}" for i in range(k)]
    assert len(out) == len(frame) == len(hindsight)
    assert not out["hindsight"].any() and hindsight["hindsight"].all()
    assert "as_of_utc" not in hindsight
    assert (out[cols] >= 0).all().all() and (out[cols] <= 1).all().all()
    assert np.allclose(out[cols].sum(axis=1), 1, atol=1e-6)
    assert np.array_equal(out["state_id"], out[cols].to_numpy().argmax(axis=1))
    assert (
        pd.to_datetime(out["as_of_utc"], utc=True)
        >= pd.to_datetime(out["date_utc"], utc=True) + pd.Timedelta(days=1)
    ).all()
    train = frame.loc[frame["split"] == "train"].reset_index(drop=True)
    alpha, _ = infer(model, train[ZCOLS], train)
    mapping, risk = train_mapping(train, alpha)
    assert {str(key): value for key, value in mapping.items()} == metadata["label_map"]
    assert risk == metadata["high_risk_state"]
    assert out["model_run_id"].eq(metadata["model_run_id"]).all()
    assert json.loads(out["label_map"].iloc[0]) == {str(key): value for key, value in mapping.items()}


def test_selection_stability_and_comparison_tables(cfg):
    table = pd.read_csv(path_of(cfg, "tables") / "B03_model_selection.csv")
    comparison = pd.read_csv(path_of(cfg, "tables") / "B07_hmm_vs_gmm.csv")
    manifest = read_json(path_of(cfg, "metadata") / "model_manifest.json")
    assert not manifest["test_used_for_selection"]
    for asset in cfg["assets"]:
        part = table.loc[table["asset"] == asset]
        assert set(part["K"]) == {2, 3, 4}
        assert len(part) == 15
        assert part["selected"].sum() == 1
        assert part.loc[part["selected"], "eligible"].all()
        for period in ("train", "validation", "test"):
            sample = comparison.loc[(comparison["asset"] == asset) & (comparison["period"] == period)]
            assert len(sample) == 2 and sample["n"].nunique() == 1
            assert sample["start"].nunique() == sample["end"].nunique() == 1
        info = manifest["assets"][asset]
        assert info["test_evaluated"]
        assert info["seed_stability_pairs"] > 0
    assert not (path_of(cfg, "tables") / "B06_drivers.csv").exists()


def test_btc_eth_models_independent(artifacts):
    btc, eth = artifacts["BTC"]["model"], artifacts["ETH"]["model"]
    assert btc is not eth
    assert artifacts["BTC"]["metadata"]["asset"] == "BTC"
    assert artifacts["ETH"]["metadata"]["asset"] == "ETH"
    if btc.means_.shape == eth.means_.shape:
        assert not np.allclose(btc.means_, eth.means_)


@pytest.mark.parametrize("asset", ["BTC", "ETH"])
def test_c3_reader_returns_matching_causal_test_signals(cfg, asset):
    with pytest.warns(UnstableModelWarning):
        signals, metadata = load_filtered(cfg, asset, period="test", acknowledge_unstable=True)
    assert len(signals) == metadata["n_by_period"]["test"]
    assert signals["split"].eq("test").all()
    assert signals["high_risk_state"].eq(metadata["high_risk_state"]).all()
