"""Feature engineering from Matrix Profile outputs."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _align_to_series_length(values: np.ndarray, n: int, center_indices: np.ndarray) -> np.ndarray:
    aligned = np.full(n, np.nan, dtype=float)
    aligned[center_indices] = values
    return aligned


def build_feature_frame(
    ts: np.ndarray, mp_dict: dict[str, Any], rolling_window: int = 25
) -> pd.DataFrame:
    """Build a feature DataFrame aligned to original time indices.

    Features include:
    - Matrix profile value at each aligned index
    - First difference of profile
    - Rolling mean and rolling std of profile
    - Optional left/right profile distances when present
    """
    ts = np.asarray(ts, dtype=float)
    if ts.ndim != 1:
        raise ValueError("ts must be a 1D array")
    if rolling_window < 3:
        raise ValueError("rolling_window must be >= 3")
    if "mp" not in mp_dict:
        raise ValueError("mp_dict must include key 'mp'")

    mp = np.asarray(mp_dict["mp"], dtype=float)
    n = ts.shape[0]
    profile_len = mp.shape[0]
    inferred_m = n - profile_len + 1
    if inferred_m <= 0:
        raise ValueError("Could not infer subsequence length from ts and mp lengths")

    centers = np.arange(profile_len, dtype=int) + inferred_m // 2
    valid = centers < n
    centers = centers[valid]
    mp = mp[valid]

    frame = pd.DataFrame(index=np.arange(n))
    frame["ts"] = ts
    frame["mp"] = _align_to_series_length(mp, n, centers)

    mp_diff = np.diff(mp, prepend=mp[0])
    frame["mp_diff"] = _align_to_series_length(mp_diff, n, centers)

    mp_series = pd.Series(frame["mp"], index=frame.index)
    frame["mp_roll_mean"] = mp_series.rolling(rolling_window, center=True, min_periods=1).mean()
    frame["mp_roll_std"] = mp_series.rolling(rolling_window, center=True, min_periods=1).std(ddof=0)

    for optional_key in ("left_mp", "right_mp"):
        if optional_key in mp_dict and mp_dict[optional_key] is not None:
            optional_values = np.asarray(mp_dict[optional_key], dtype=float)[valid]
            frame[optional_key] = _align_to_series_length(optional_values, n, centers)

    return frame
