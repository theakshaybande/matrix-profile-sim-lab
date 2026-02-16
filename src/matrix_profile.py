"""Matrix Profile wrappers around STUMPY."""

from __future__ import annotations

from typing import Any

import numpy as np
import stumpy


def _validate_univariate_inputs(ts: np.ndarray, m: int) -> np.ndarray:
    ts = np.asarray(ts, dtype=float)
    if ts.ndim != 1:
        raise ValueError("ts must be a 1D array")
    if ts.size < 8:
        raise ValueError("ts must have at least 8 points")
    if m < 3 or m >= ts.size:
        raise ValueError("m must satisfy 3 <= m < len(ts)")
    return ts


def _z_norm_euclidean(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    """Compute z-normalized Euclidean distance between two windows."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a_mean, b_mean = float(np.mean(a)), float(np.mean(b))
    a_std, b_std = float(np.std(a)), float(np.std(b))

    if a_std < eps and b_std < eps:
        return 0.0
    if a_std < eps or b_std < eps:
        return float(np.sqrt(a.size))

    za = (a - a_mean) / a_std
    zb = (b - b_mean) / b_std
    return float(np.linalg.norm(za - zb))


def _profile_from_neighbor_indices(ts: np.ndarray, m: int, indices: np.ndarray) -> np.ndarray:
    """Convert neighbor indices into profile distances when possible."""
    profile = np.full(indices.shape[0], np.nan, dtype=float)
    for i, j in enumerate(indices):
        if j < 0:
            continue
        j_int = int(j)
        profile[i] = _z_norm_euclidean(ts[i : i + m], ts[j_int : j_int + m])
    return profile


def compute_univariate_mp(ts: np.ndarray, m: int) -> dict[str, Any]:
    """Compute univariate Matrix Profile using `stumpy.stump`.

    Parameters
    ----------
    ts:
        1D time series.
    m:
        Subsequence length.

    Returns
    -------
    dict[str, Any]
        Dictionary containing matrix profile arrays and raw STUMPY output.
    """
    ts = _validate_univariate_inputs(ts, m)
    raw = stumpy.stump(ts, m)

    mp = np.asarray(raw[:, 0], dtype=float)
    mpi = np.asarray(raw[:, 1], dtype=int)
    left_mpi = np.asarray(raw[:, 2], dtype=int)
    right_mpi = np.asarray(raw[:, 3], dtype=int)

    left_mp = _profile_from_neighbor_indices(ts, m, left_mpi)
    right_mp = _profile_from_neighbor_indices(ts, m, right_mpi)

    return {
        "mp": mp,
        "mpi": mpi,
        "left_mp": left_mp,
        "right_mp": right_mp,
        "left_mpi": left_mpi,
        "right_mpi": right_mpi,
        "m": m,
        "raw_stump": raw,
    }


def compute_multidimensional_mp(X: np.ndarray, m: int) -> dict[str, Any]:
    """Compute multidimensional Matrix Profile using `stumpy.mstump`.

    Parameters
    ----------
    X:
        Multivariate array with shape `(d, n)`, where `d` is dimensions.
    m:
        Subsequence length.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array with shape (d, n)")
    d, n = X.shape
    if d < 2:
        raise ValueError("X must contain at least two dimensions for mstump")
    if m < 3 or m >= n:
        raise ValueError("m must satisfy 3 <= m < n")

    mp, mpi = stumpy.mstump(X, m)
    return {
        "mp": np.asarray(mp, dtype=float),
        "mpi": np.asarray(mpi, dtype=int),
        "m": m,
        "raw_mstump": (mp, mpi),
    }
