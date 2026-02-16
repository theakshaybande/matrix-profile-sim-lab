# matrix-profile-sim-lab

Small, modular Python lab for simulating time series data and computing Matrix Profile representations with [STUMPY](https://stumpy.readthedocs.io/).

The project focuses on:
- Synthetic univariate signals with motifs, regime shifts, and anomalies
- Matrix Profile computation (`stump` and `mstump`)
- Feature extraction from profile outputs
- PCA + KMeans clustering for regime exploration
- Reproducible plots and a short notebook walkthrough

## Setup

```bash
pip install -r requirements.txt
```

## Run the Pipeline

From the repository root:

```bash
python -m src.run_pipeline
```

Optional arguments:

```bash
python -m src.run_pipeline --n 3000 --m 100 --seed 42
python -m src.run_pipeline --output-dir outputs
```

The pipeline writes:
- `outputs/features.csv`
- `outputs/matrix_profile.csv`
- `outputs/multidim_matrix_profile.csv`
- `outputs/series_overview.png`
- `outputs/matrix_profile.png`
- `outputs/multidim_matrix_profile.png`
- `outputs/clusters_over_time.png`
- `outputs/pca_clusters.png`

## Notebook

Start Jupyter:

```bash
jupyter lab
```

Open:
- `notebooks/01_matrix_profile_simulation.ipynb`

The notebook demonstrates simulation, Matrix Profile analysis, cluster visualization, and how changing window length `m` changes the profile shape.
