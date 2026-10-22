"""H02-H04 from exported probabilities and scores, with explicit information timing."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def export_figures(asset, frame, filtered, smoothed, gmm, comparison, folder, dpi):
    dates = pd.to_datetime(frame["date_utc"], utc=True)
    k = int(filtered["K"].iloc[0])
    source = "Source: C1 Binance Spot daily UTC; frozen train model/scaler."

    def save(fig, name):
        fig.text(0.01, 0.01, source, fontsize=8)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        for ext in ("png", "svg"):
            fig.savefig(folder / f"{name}_{asset.lower()}.{ext}", dpi=dpi)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    for state in range(k):
        mask = filtered["state_id"].eq(state).to_numpy()
        label = filtered.loc[mask, "label"].iloc[0] if mask.any() else f"State {state}"
        ax.scatter(dates[mask], frame.loc[mask, "close"], s=5, label=f"{state}: {label}")
    ax.set(
        yscale="log", ylabel="Close (USDT, log scale)", title=f"{asset} H02 Price and filtered causal states"
    )
    ax.legend(ncol=k, fontsize=8)
    save(fig, "H02_price_regimes")

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for state in range(k):
        axes[0].plot(dates, filtered[f"prob_state_{state}"], lw=0.7, label=f"State {state}")
        axes[1].plot(dates, smoothed[f"prob_state_{state}"], lw=0.7, label=f"State {state}")
    axes[0].set(title=f"{asset} H03 Filtered causal probabilities", ylabel="Probability", ylim=(0, 1))
    axes[1].set(
        title="Smoothed hindsight probabilities (future data; analysis only)",
        ylabel="Probability",
        ylim=(0, 1),
    )
    axes[0].legend(ncol=k, fontsize=8)
    save(fig, "H03_filtered_smoothed")

    test = frame["split"].eq("test").to_numpy()
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [1, 1, 1.3]})
    for state in range(k):
        axes[0].plot(dates[test], filtered.loc[test, f"prob_state_{state}"], lw=0.8, label=f"State {state}")
        axes[1].plot(dates[test], gmm.loc[test, f"prob_state_{state}"], lw=0.8, label=f"Cluster {state}")
    axes[0].set(title=f"{asset} H04 Test HMM filtered probabilities", ylim=(0, 1), ylabel="Probability")
    axes[1].set(
        title="Test GMM posterior (cluster IDs independent of HMM)", ylim=(0, 1), ylabel="Probability"
    )
    for ax in axes[:2]:
        ax.legend(ncol=k, fontsize=8)
    positions = np.arange(3)
    for method, offset in (("HMM", -0.18), ("GMM", 0.18)):
        values = comparison.loc[comparison["method"] == method, "avg_log_score"]
        axes[2].bar(positions + offset, values, width=0.36, label=method)
    axes[2].set_xticks(positions, ["train", "validation", "test"])
    axes[2].set(
        ylabel="Average log-score",
        title="HMM conditional density versus GMM marginal density; same rows and scale",
    )
    axes[2].legend()
    save(fig, "H04_hmm_gmm")
