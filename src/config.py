"""Central configuration for the Matrix Profile simulation lab."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class Config:
    """Configuration container used by the end-to-end pipeline."""

    random_seed: int = 42
    n: int = 3000
    m: int = 100
    motif_len: int = 80
    n_motifs: int = 4
    noise_std: float = 0.8
    regime_shift: bool = True
    anomalies: bool = True

    feature_rolling_window: int = 25
    n_clusters: int = 3
    pca_components: int = 5

    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])
    data_dir: Path = field(init=False)
    output_dir: Path = field(init=False)
    features_csv: Path = field(init=False)

    def __post_init__(self) -> None:
        self.data_dir = self.project_root / "data"
        self.output_dir = self.project_root / "outputs"
        self.features_csv = self.output_dir / "features.csv"

    def ensure_dirs(self) -> None:
        """Create directories required by the pipeline."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
