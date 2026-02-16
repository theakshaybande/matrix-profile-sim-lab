"""End-to-end CLI pipeline for simulation, Matrix Profile, and clustering."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .clustering import cluster_features
from .config import Config
from .features import build_feature_frame
from .matrix_profile import compute_multidimensional_mp, compute_univariate_mp
from .simulate import simulate_univariate_series
from .viz import (
    plot_clusters_over_time,
    plot_matrix_profile,
    plot_multidimensional_profile,
    plot_pca_scatter,
    plot_series_overview,
)


def run_pipeline(config: Config | None = None) -> dict[str, Path]:
    """Run the complete Matrix Profile simulation workflow."""
    cfg = config or Config()
    cfg.ensure_dirs()

    ts, metadata = simulate_univariate_series(
        n=cfg.n,
        seed=cfg.random_seed,
        motif_len=cfg.motif_len,
        n_motifs=cfg.n_motifs,
        noise_std=cfg.noise_std,
        regime_shift=cfg.regime_shift,
        anomalies=cfg.anomalies,
    )

    univariate_mp = compute_univariate_mp(ts, cfg.m)
    X_multi = np.vstack([ts, np.gradient(ts)])
    multidim_mp = compute_multidimensional_mp(X_multi, cfg.m)

    features = build_feature_frame(ts, univariate_mp, rolling_window=cfg.feature_rolling_window)
    clustered, _, _ = cluster_features(
        features, n_components=cfg.pca_components, k=cfg.n_clusters, random_state=cfg.random_seed
    )

    output_paths: dict[str, Path] = {}

    features_path = cfg.output_dir / "features.csv"
    clustered.to_csv(features_path, index_label="t")
    output_paths["features_csv"] = features_path

    centers = np.arange(univariate_mp["mp"].shape[0]) + cfg.m // 2
    mp_frame = pd.DataFrame(
        {
            "t_center": centers,
            "mp": univariate_mp["mp"],
            "mpi": univariate_mp["mpi"],
            "left_mpi": univariate_mp["left_mpi"],
            "right_mpi": univariate_mp["right_mpi"],
        }
    )
    mp_path = cfg.output_dir / "matrix_profile.csv"
    mp_frame.to_csv(mp_path, index=False)
    output_paths["matrix_profile_csv"] = mp_path

    md_centers = np.arange(multidim_mp["mp"].shape[1]) + cfg.m // 2
    md_columns = [f"mp_dim_{i + 1}" for i in range(multidim_mp["mp"].shape[0])]
    md_frame = pd.DataFrame(multidim_mp["mp"].T, columns=md_columns)
    md_frame.insert(0, "t_center", md_centers)
    md_path = cfg.output_dir / "multidim_matrix_profile.csv"
    md_frame.to_csv(md_path, index=False)
    output_paths["multidim_matrix_profile_csv"] = md_path

    series_plot = cfg.output_dir / "series_overview.png"
    plot_series_overview(ts, metadata, series_plot)
    output_paths["series_plot"] = series_plot

    mp_plot = cfg.output_dir / "matrix_profile.png"
    plot_matrix_profile(univariate_mp, n=cfg.n, output_path=mp_plot)
    output_paths["matrix_profile_plot"] = mp_plot

    md_plot = cfg.output_dir / "multidim_matrix_profile.png"
    plot_multidimensional_profile(multidim_mp, n=cfg.n, output_path=md_plot)
    output_paths["multidim_matrix_profile_plot"] = md_plot

    cluster_plot = cfg.output_dir / "clusters_over_time.png"
    plot_clusters_over_time(clustered, cluster_plot)
    output_paths["clusters_plot"] = cluster_plot

    pca_plot = cfg.output_dir / "pca_clusters.png"
    plot_pca_scatter(clustered, pca_plot)
    output_paths["pca_plot"] = pca_plot

    print("Pipeline complete. Saved outputs:")
    for key, path in output_paths.items():
        print(f"- {key}: {path}")

    return output_paths


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for ad-hoc pipeline overrides."""
    parser = argparse.ArgumentParser(description="Run Matrix Profile simulation pipeline.")
    parser.add_argument("--n", type=int, default=3000, help="Time series length")
    parser.add_argument("--m", type=int, default=100, help="Subsequence window length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--motif-len", type=int, default=80, help="Motif length")
    parser.add_argument("--n-motifs", type=int, default=4, help="Number of motif insertions")
    parser.add_argument("--noise-std", type=float, default=0.8, help="Base noise standard deviation")
    parser.add_argument("--no-regime-shift", action="store_true", help="Disable regime shifts")
    parser.add_argument("--no-anomalies", action="store_true", help="Disable anomalies")
    parser.add_argument("--output-dir", type=str, default=None, help="Optional output directory override")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    cfg = Config(
        random_seed=args.seed,
        n=args.n,
        m=args.m,
        motif_len=args.motif_len,
        n_motifs=args.n_motifs,
        noise_std=args.noise_std,
        regime_shift=not args.no_regime_shift,
        anomalies=not args.no_anomalies,
    )
    if args.output_dir:
        cfg.output_dir = Path(args.output_dir).resolve()
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
