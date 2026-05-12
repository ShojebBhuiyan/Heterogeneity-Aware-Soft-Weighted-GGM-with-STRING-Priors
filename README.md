# Heterogeneity-Aware Soft-Weighted GGM with STRING Priors

Research-grade pipeline for gene interaction inference from single-cell RNA-seq using soft-weighted Gaussian graphical models (GGM), Gaussian mixture heterogeneity modeling, and uncertain knowledge-graph priors from [STRING](https://string-db.org/) (API only).

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -e ".[eda,test,dev]"
```

## PBMC data

Place or keep the PBMC 3k 10x matrix under `datasets/filtered_gene_bc_matrices/hg19/` (`matrix.mtx`, `genes.tsv`, `barcodes.tsv`). Paths are configurable in [`configs/default.yaml`](configs/default.yaml).

## Smoke mode

Fast wiring checks use `configs/smoke.yaml`. From Python:

```python
from pgm.utils.notebook import is_smoke
from pgm.config.loader import load_config
cfg = load_config()  # reads SMOKE env or merges smoke yaml
```

Or set `SMOKE=1` before loading config.

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `notebooks/01_eda.ipynb` | EDA |
| `notebooks/02_preprocessing.ipynb` | QC, normalization, HVG |
| `notebooks/03_clustering.ipynb` | Leiden + GMM soft |
| `notebooks/04_global_ggm.ipynb` | Baseline graphical lasso |
| `notebooks/05_soft_weighted_ggm.ipynb` | Per-state soft-weighted GGM |
| `notebooks/06_kg_integration.ipynb` | STRING priors |
| `notebooks/07_evaluation.ipynb` | Stability / overlap |
| `notebooks/final_pipeline.ipynb` | End-to-end |

Run tests: `pytest tests/`.

## Outputs

Intermediate artifacts live under `data/`; figures and reports under `reports/`; logs under `logs/`.
