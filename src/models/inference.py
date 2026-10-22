"""Causal Gaussian HMM filtering and explicitly separate hindsight smoothing."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import logsumexp


def sequence_starts(frame: pd.DataFrame) -> np.ndarray:
    """Reset on a declared segment boundary OR a missing calendar day, never on split."""
    dates = pd.to_datetime(frame["date_utc"], utc=True)
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("Dates must be unique, valid and strictly increasing within one asset")
    if "asset" in frame and frame["asset"].nunique() != 1:
        raise ValueError("Never concatenate assets into one HMM sequence")
    starts = dates.diff().ne(pd.Timedelta(days=1)).to_numpy()
    if "segment_id" in frame:
        if frame["segment_id"].isna().any():
            raise ValueError("Missing segment_id")
        starts |= frame["segment_id"].ne(frame["segment_id"].shift()).to_numpy()
    return starts


def sequence_lengths(frame: pd.DataFrame) -> list[int]:
    starts = np.flatnonzero(sequence_starts(frame))
    return np.diff(np.r_[starts, len(frame)]).tolist()


def diagonal_variances(model) -> np.ndarray:
    cov = np.asarray(model.covars_, dtype=float)
    return np.diagonal(cov, axis1=1, axis2=2).copy() if cov.ndim == 3 else cov.copy()


def emission_log_density(x, means, variances) -> np.ndarray:
    x, means, variances = (np.asarray(a, dtype=float) for a in (x, means, variances))
    if x.ndim != 2 or means.ndim != 2 or variances.shape != means.shape or x.shape[1] != means.shape[1]:
        raise ValueError("Incompatible Gaussian dimensions")
    if not all(np.isfinite(a).all() for a in (x, means, variances)) or (variances <= 0).any():
        raise ValueError("Features and means must be finite; diagonal variances must be positive")
    return -0.5 * (
        x.shape[1] * np.log(2 * np.pi)
        + np.log(variances).sum(axis=1)
        + ((x[:, None, :] - means) ** 2 / variances).sum(axis=2)
    )


def _log_probability(a):
    with np.errstate(divide="ignore"):
        return np.log(a)


def _forward_log(log_emissions, startprob, transmat, starts=None):
    """Keep normalized log probabilities internally, including near-zero state mass."""
    b = np.asarray(log_emissions, dtype=float)
    pi, p = np.asarray(startprob, dtype=float), np.asarray(transmat, dtype=float)
    if b.ndim != 2 or not len(b) or not np.isfinite(b).all():
        raise ValueError("Expected a nonempty finite emission matrix")
    if pi.shape != (b.shape[1],) or p.shape != (len(pi), len(pi)):
        raise ValueError("Invalid HMM dimensions")
    if (
        not np.isfinite(pi).all()
        or not np.isfinite(p).all()
        or (pi < 0).any()
        or (p < 0).any()
        or not np.isclose(pi.sum(), 1)
        or not np.allclose(p.sum(axis=1), 1)
    ):
        raise ValueError("Invalid HMM probability parameters")
    starts = np.zeros(len(b), dtype=bool) if starts is None else np.asarray(starts, dtype=bool).copy()
    if starts.shape != (len(b),):
        raise ValueError("Invalid sequence boundary mask")
    starts[0] = True
    log_pi, log_p = _log_probability(pi), _log_probability(p)
    log_alpha, scores = np.empty_like(b), np.empty(len(b))
    for t in range(len(b)):
        prior = log_pi if starts[t] else logsumexp(log_alpha[t - 1, :, None] + log_p, axis=0)
        update = prior + b[t]
        scores[t] = logsumexp(update)
        if not np.isfinite(scores[t]):
            raise ValueError("Observation has zero probability under every state")
        log_alpha[t] = update - scores[t]
    return log_alpha, scores


def forward_filter(log_emissions, startprob, transmat, starts=None):
    """Return P(S_t|X_1:t) and each one-step log predictive density.

    The first observation of every segment IS included in log-score reporting.
    Split boundaries do not reset the chain. Parameters remain frozen.
    """
    log_alpha, scores = _forward_log(log_emissions, startprob, transmat, starts)
    return np.exp(log_alpha), scores


def smooth_probabilities(log_emissions, startprob, transmat, starts=None):
    """P(S_t|X_1:T): uses future observations; never a C3 trading signal."""
    b = np.asarray(log_emissions, dtype=float)
    log_alpha, scores = _forward_log(b, startprob, transmat, starts)
    resets = np.zeros(len(b), dtype=bool) if starts is None else np.asarray(starts, dtype=bool)
    log_p = _log_probability(np.asarray(transmat))
    beta = np.zeros_like(b)
    for t in range(len(b) - 2, -1, -1):
        if not resets[t + 1]:
            beta[t] = logsumexp(log_p + b[t + 1] + beta[t + 1], axis=1) - scores[t + 1]
    joint = log_alpha + beta
    return np.exp(joint - logsumexp(joint, axis=1, keepdims=True))


def infer(model, x, frame):
    b = emission_log_density(x, model.means_, diagonal_variances(model))
    return forward_filter(b, model.startprob_, model.transmat_, sequence_starts(frame))
