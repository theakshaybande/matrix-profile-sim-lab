"""Preprocessing helpers for time-series analysis."""

from __future__ import annotations

import numpy as np


def z_normalize_series(ts: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Z-normalize a full time series."""
    ts = np.asarray(ts, dtype=float)
    mean = float(np.mean(ts))
    std = float(np.std(ts))
    if std < eps:
        return np.zeros_like(ts, dtype=float)
    return (ts - mean) / std


def z_normalize_windows(ts: np.ndarray, m: int, eps: float = 1e-8) -> np.ndarray:
    """Create z-normalized sliding windows from a series.

    Notes
    -----
    STUMPY computes normalized distance internally for `stump`, so this utility
    is primarily for custom feature engineering or diagnostics.
    """
    ts = np.asarray(ts, dtype=float)
    if ts.ndim != 1:
        raise ValueError("ts must be a 1D array")
    if m < 2 or m > ts.shape[0]:
        raise ValueError("m must satisfy 2 <= m <= len(ts)")

    windows = np.lib.stride_tricks.sliding_window_view(ts, window_shape=m)
    means = windows.mean(axis=1, keepdims=True)
    stds = windows.std(axis=1, keepdims=True)
    stds = np.where(stds < eps, 1.0, stds)
    return (windows - means) / stds


def clip_outliers(ts: np.ndarray, z_thresh: float = 5.0) -> np.ndarray:
    """Clip extreme values using mean +/- z_thresh * std."""
    ts = np.asarray(ts, dtype=float)
    if z_thresh <= 0:
        raise ValueError("z_thresh must be > 0")

    mean = float(ts.mean())
    std = float(ts.std())
    if std < 1e-12:
        return ts.copy()
    lower = mean - z_thresh * std
    upper = mean + z_thresh * std
    return np.clip(ts, lower, upper)
