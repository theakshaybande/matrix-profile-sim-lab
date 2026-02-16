"""Visualization helpers for simulation and Matrix Profile outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save_figure(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_series_overview(ts: np.ndarray, metadata: dict[str, Any], output_path: Path) -> None:
    """Plot synthetic time series with motif/anomaly/regime overlays."""
    ts = np.asarray(ts, dtype=float)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(ts, color="#2f5d80", linewidth=1.0, label="Time series")

    motif_positions = metadata.get("motif_positions", [])
    motif_len = int(metadata.get("motif_len", 1))
    for idx, pos in enumerate(motif_positions):
        ax.axvline(
            pos,
            color="#2a9d8f",
            linestyle="--",
            linewidth=1.0,
            alpha=0.8,
            label="Motif start" if idx == 0 else None,
        )
        ax.axvspan(pos, pos + motif_len, color="#2a9d8f", alpha=0.08)

    anomaly_positions = [int(i) for i in metadata.get("anomaly_positions", [])]
    if anomaly_positions:
        ax.scatter(
            anomaly_positions,
            ts[anomaly_positions],
            color="#d62828",
            s=24,
            label="Anomaly",
            zorder=3,
        )

    for idx, boundary in enumerate(metadata.get("regime_boundaries", [])):
        ax.axvline(
            boundary,
            color="#111111",
            linestyle=":",
            linewidth=1.2,
            alpha=0.9,
            label="Regime boundary" if idx == 0 else None,
        )

    ax.set_title("Synthetic Time Series with Embedded Structure")
    ax.set_xlabel("Time Index")
    ax.set_ylabel("Value")
    ax.legend(loc="upper right", fontsize=9)
    _save_figure(fig, Path(output_path))


def plot_matrix_profile(mp_dict: dict[str, Any], n: int, output_path: Path) -> None:
    """Plot univariate matrix profile curve."""
    mp = np.asarray(mp_dict["mp"], dtype=float)
    inferred_m = n - mp.shape[0] + 1
    centers = np.arange(mp.shape[0]) + inferred_m // 2

    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.plot(centers, mp, color="#264653", linewidth=1.2, label="Matrix profile")
    ax.set_title("Univariate Matrix Profile (stump)")
    ax.set_xlabel("Time Index")
    ax.set_ylabel("Distance")
    ax.legend(loc="upper right")
    _save_figure(fig, Path(output_path))


def plot_multidimensional_profile(mstump_dict: dict[str, Any], n: int, output_path: Path) -> None:
    """Plot multidimensional matrix profile curves from `mstump`."""
    mp = np.asarray(mstump_dict["mp"], dtype=float)
    inferred_m = n - mp.shape[1] + 1
    centers = np.arange(mp.shape[1]) + inferred_m // 2

    fig, ax = plt.subplots(figsize=(12, 4))
    for dim_idx in range(mp.shape[0]):
        ax.plot(centers, mp[dim_idx], linewidth=1.0, label=f"Dim {dim_idx + 1}")
    ax.set_title("Multidimensional Matrix Profile (mstump)")
    ax.set_xlabel("Time Index")
    ax.set_ylabel("Distance")
    ax.legend(loc="upper right", ncol=min(4, mp.shape[0]), fontsize=9)
    _save_figure(fig, Path(output_path))


def plot_clusters_over_time(df: pd.DataFrame, output_path: Path) -> None:
    """Plot discovered cluster labels over time alongside the time series."""
    if "cluster" not in df.columns:
        raise ValueError("df must include a 'cluster' column")
    if "ts" not in df.columns:
        raise ValueError("df must include a 'ts' column")

    x = df.index.to_numpy()
    y = df["ts"].to_numpy(dtype=float)
    labels = df["cluster"].to_numpy(dtype=int)

    fig, axes = plt.subplots(
        2, 1, figsize=(12, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    axes[0].plot(x, y, color="#b0b0b0", linewidth=1.0, label="Time series")
    scatter = axes[0].scatter(x, y, c=labels, cmap="tab10", s=10, alpha=0.8, label="Cluster")
    axes[0].set_ylabel("Value")
    axes[0].set_title("Clusters Over Time")
    axes[0].legend(loc="upper right", fontsize=9)

    axes[1].step(x, labels, where="mid", color="#1d3557", linewidth=1.0)
    axes[1].set_xlabel("Time Index")
    axes[1].set_ylabel("Cluster")
    axes[1].set_yticks(sorted(np.unique(labels)))
    fig.colorbar(scatter, ax=axes[0], orientation="vertical", label="Cluster ID")

    _save_figure(fig, Path(output_path))


def plot_pca_scatter(df: pd.DataFrame, output_path: Path) -> None:
    """Scatter plot of PC1 vs PC2 colored by cluster."""
    required = {"pc1", "pc2", "cluster"}
    if not required.issubset(df.columns):
        raise ValueError("df must include columns: pc1, pc2, cluster")

    fig, ax = plt.subplots(figsize=(7, 5.5))
    scatter = ax.scatter(
        df["pc1"].to_numpy(dtype=float),
        df["pc2"].to_numpy(dtype=float),
        c=df["cluster"].to_numpy(dtype=int),
        cmap="tab10",
        s=14,
        alpha=0.85,
    )
    ax.set_title("PCA Latent Space Colored by Cluster")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.colorbar(scatter, ax=ax, label="Cluster ID")
    _save_figure(fig, Path(output_path))
