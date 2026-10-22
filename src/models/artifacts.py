"""Canonical final metadata and atomic model persistence with a real reload check."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import joblib
import numpy as np


def canonical_metadata(metadata):
    def encode(value):
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"Unsupported metadata value: {type(value).__name__}")

    # JSON state-map keys are strings. Use that representation in every artifact.
    return json.loads(json.dumps(metadata, sort_keys=True, allow_nan=False, default=encode))


def save_checked_bundle(model, metadata, destination, x, *, kind, lengths=None):
    """Save only final metadata, reload from disk, compare metadata and actual scores."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata = canonical_metadata(metadata)
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".joblib", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        joblib.dump({"model": model, "metadata": metadata}, temporary)
        loaded = joblib.load(temporary)
        if loaded["metadata"] != metadata:
            raise ValueError("Reloaded model metadata differs from final metadata")
        if kind == "HMM":
            before, after = model.score(x, lengths=lengths), loaded["model"].score(x, lengths=lengths)
        elif kind == "GMM":
            before, after = model.score_samples(x), loaded["model"].score_samples(x)
        else:
            raise ValueError("Unknown model artifact kind")
        if not np.isfinite(after).all() or not np.allclose(before, after, atol=1e-10, rtol=1e-10):
            raise ValueError("Reloaded model scores differ from the model being saved")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {"kind": kind, "metadata_equal": True, "reload_score_equal": True, "probe_n": len(x)}
