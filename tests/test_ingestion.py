"""Test PBMC ingestion when local 10x data is present."""

from __future__ import annotations

import pytest

from pgm.config.loader import load_config
from pgm.data.loader import load_pbmc_mtx
from pgm.data.summary import dataset_summary
from pgm.data.validate import validate_anndata_basic
from pgm.utils.paths import tenx_mtx_path


def _mtx_exists(cfg) -> bool:
    p = tenx_mtx_path(cfg)
    return p.is_dir() and (p / "matrix.mtx").is_file()


@pytest.mark.smoke
def test_pbmc_loading_smoke():
    """Load full PBMC when ``datasets/filtered_gene_bc_matrices/hg19`` exists."""
    cfg = load_config(smoke=True)
    if not _mtx_exists(cfg):
        pytest.skip("PBMC mtx directory missing")
    adata = load_pbmc_mtx(cfg)
    validate_anndata_basic(adata)
    assert adata.n_obs > 2000
    assert adata.n_vars > 10_000
    summ = dataset_summary(adata)
    assert summ["sparsity"] > 0.9
