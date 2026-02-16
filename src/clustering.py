"""Clustering helpers for feature representations."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def cluster_features(
    df: pd.DataFrame, n_components: int = 5, k: int = 3, random_state: int = 42
) -> tuple[pd.DataFrame, PCA, KMeans]:
    """Cluster feature vectors using PCA projection followed by KMeans."""
    if df.empty:
        raise ValueError("df must not be empty")
    if n_components < 1:
        raise ValueError("n_components must be >= 1")
    if k < 2:
        raise ValueError("k must be >= 2")

    numeric = df.select_dtypes(include=[np.number]).copy()
    if numeric.empty:
        raise ValueError("df must contain numeric feature columns")
    numeric = numeric.drop(columns=["cluster"], errors="ignore")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    numeric = numeric.fillna(numeric.median(numeric_only=True))

    scaler = StandardScaler()
    X = scaler.fit_transform(numeric.to_numpy(dtype=float))

    max_components = min(n_components, X.shape[0], X.shape[1])
    pca = PCA(n_components=max_components, random_state=random_state)
    X_pca = pca.fit_transform(X)

    n_clusters = min(k, X_pca.shape[0])
    kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=random_state)
    labels = kmeans.fit_predict(X_pca)

    out = df.copy()
    for idx in range(X_pca.shape[1]):
        out[f"pc{idx + 1}"] = X_pca[:, idx]
    out["cluster"] = labels

    return out, pca, kmeans
