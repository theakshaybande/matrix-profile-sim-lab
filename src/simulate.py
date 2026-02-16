"""Synthetic time-series generation utilities."""

from __future__ import annotations

from typing import Any

import numpy as np


def _sample_non_overlapping_positions(
    rng: np.random.Generator, max_start: int, motif_len: int, n_motifs: int
) -> list[int]:
    """Sample approximately uniform non-overlapping start positions."""
    if n_motifs <= 0 or max_start <= 0:
        return []

    candidates = np.arange(0, max_start + 1, dtype=int)
    rng.shuffle(candidates)

    positions: list[int] = []
    for pos in candidates:
        if all(abs(pos - existing) >= motif_len for existing in positions):
            positions.append(int(pos))
            if len(positions) == n_motifs:
                break

    if len(positions) < n_motifs:
        spacing = max(motif_len, max_start // max(1, n_motifs))
        fallback = list(range(0, max_start + 1, spacing))
        for pos in fallback:
            if all(abs(pos - existing) >= motif_len for existing in positions):
                positions.append(int(pos))
                if len(positions) == n_motifs:
                    break

    positions.sort()
    return positions


def simulate_univariate_series(
    n: int,
    seed: int,
    motif_len: int,
    n_motifs: int,
    noise_std: float,
    regime_shift: bool = True,
    anomalies: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Generate a synthetic univariate series with motif repeats and anomalies.

    Parameters
    ----------
    n:
        Number of time steps in the generated series.
    seed:
        Random seed for reproducibility.
    motif_len:
        Length of the repeated motif template.
    n_motifs:
        Number of motif insertions.
    noise_std:
        Base standard deviation of white noise.
    regime_shift:
        If True, create piecewise regimes with different mean/variance.
    anomalies:
        If True, inject spikes and one short level-shift anomaly.

    Returns
    -------
    tuple[np.ndarray, dict[str, Any]]
        The generated signal and metadata:
        `motif_positions`, `anomaly_positions`, `regime_boundaries`, `motif_len`.
    """
    if n <= 50:
        raise ValueError("n must be greater than 50")
    if motif_len < 8 or motif_len >= n // 2:
        raise ValueError("motif_len must be at least 8 and smaller than n/2")
    if noise_std <= 0:
        raise ValueError("noise_std must be > 0")

    rng = np.random.default_rng(seed)
    signal = np.zeros(n, dtype=float)
    regime_boundaries: list[int] = []

    if regime_shift:
        regime_boundaries = [n // 3, (2 * n) // 3]
        means = rng.uniform(-1.25, 1.25, size=3)
        scales = noise_std * rng.uniform(0.7, 1.7, size=3)
        starts = [0, regime_boundaries[0], regime_boundaries[1]]
        stops = [regime_boundaries[0], regime_boundaries[1], n]

        for start, stop, mean, scale in zip(starts, stops, means, scales):
            signal[start:stop] = rng.normal(mean, scale, size=stop - start)
    else:
        signal = rng.normal(0.0, noise_std, size=n)

    t = np.linspace(0.0, 8.0 * np.pi, n)
    signal += 0.25 * np.sin(t) + 0.15 * np.cos(0.45 * t)

    phase = np.linspace(0.0, 2.0 * np.pi, motif_len)
    motif_template = 1.8 * np.sin(phase) + 0.8 * np.hanning(motif_len)
    max_start = n - motif_len - 1
    motif_positions = _sample_non_overlapping_positions(rng, max_start, motif_len, n_motifs)

    for pos in motif_positions:
        local_noise = rng.normal(0.0, noise_std * 0.12, size=motif_len)
        signal[pos : pos + motif_len] += motif_template + local_noise

    anomaly_positions: list[int] = []
    if anomalies:
        n_spikes = max(3, n // 1200)
        spike_positions = rng.integers(0, n, size=n_spikes)
        for pos in spike_positions:
            pos = int(pos)
            signal[pos] += float(rng.choice([-1.0, 1.0]) * noise_std * rng.uniform(6.0, 10.0))
            anomaly_positions.append(pos)

        if n > 240:
            level_start = int(rng.integers(0, max(1, n - motif_len)))
            level_len = int(rng.integers(max(20, motif_len // 3), max(30, motif_len)))
            level_end = min(n, level_start + level_len)
            magnitude = float(rng.choice([-1.0, 1.0]) * noise_std * rng.uniform(2.5, 4.5))
            signal[level_start:level_end] += magnitude
            anomaly_positions.append(level_start)

    metadata = {
        "motif_positions": sorted(motif_positions),
        "anomaly_positions": sorted(set(anomaly_positions)),
        "regime_boundaries": regime_boundaries,
        "motif_len": motif_len,
    }
    return signal.astype(float), metadata
