"""End-to-end preprocessing on AnnData."""

from __future__ import annotations

import logging
import re

import numpy as np
import scanpy as sc
from scipy import sparse

from pgm.config.schemas import ProjectConfig
from pgm.utils.notebook import is_smoke

logger = logging.getLogger("pgm.preprocessing")


def preprocess_adata(
    adata: sc.AnnData,
    cfg: ProjectConfig,
    *,
    inplace: bool = True,
) -> sc.AnnData:
    """
    Filter cells/genes, normalize, log1p, HVG selection, scale zero-mean, PCA.

    Stores highly variable mask in ``adata.var['highly_variable']``.
    """
    if not inplace:
        adata = adata.copy()

    if cfg.preprocessing.smoke_max_cells is not None and is_smoke(cfg):
        ncap = min(cfg.preprocessing.smoke_max_cells, adata.n_obs)
        rng = np.random.default_rng(cfg.run.random_seed)
        idx = rng.choice(adata.n_obs, ncap, replace=False)
        adata = adata[idx].copy()
        logger.info("Smoke cap: using %d cells", ncap)

    pat = cfg.preprocessing.mitochondrial_prefix
    adata.var["mt"] = adata.var_names.str.match(pat, case=False, flags=re.IGNORECASE)
    sc.pp.calculate_qc_metrics(adata, qc_vars=("mt",), inplace=True)
    pct_mito_col = None
    for cand in ["pct_counts_mt", "pct_counts_in_top_50_mt"]:
        if cand in adata.obs.columns:
            pct_mito_col = cand
            break
    if pct_mito_col is None:
        for col in adata.obs.columns:
            if "pct" in col.lower() and col.lower().endswith("mt"):
                pct_mito_col = col
                break
    if pct_mito_col is None:
        raise KeyError(f"No mitochondrial % QC column among {list(adata.obs.columns)}")

    sc.pp.filter_cells(adata, min_genes=cfg.preprocessing.min_genes)
    sc.pp.filter_genes(adata, min_cells=cfg.preprocessing.min_cells)
    adata = adata[
        adata.obs[pct_mito_col] < cfg.preprocessing.max_pct_counts_mito
    ].copy()

    sc.pp.normalize_total(adata, target_sum=cfg.preprocessing.target_sum)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=min(cfg.preprocessing.n_top_hvg, adata.n_vars),
        subset=True,
        flavor="seurat",
    )
    if sparse.issparse(adata.X):
        adata.X = np.asarray(adata.X.todense(), dtype=np.float32)
    sc.pp.scale(adata, max_value=10, zero_center=True)
    Xs = np.asarray(adata.X)
    np.nan_to_num(Xs, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    with np.errstate(invalid="ignore"):
        col_std = Xs.std(axis=0, ddof=1)
    bad_gene = ~np.isfinite(col_std) | (col_std <= 1e-12)
    n_bad = int(bad_gene.sum())
    if n_bad:
        logger.warning(
            "After scale, dropping %d zero-variance / invalid genes (required for stable PCA/GLasso)",
            n_bad,
        )
        adata = adata[:, ~bad_gene].copy()
        Xs = np.asarray(adata.X)
    nhv = adata.n_vars
    if nhv < 3:
        raise ValueError(f"Too few genes after QC and variance filter: {nhv}")
    n_pcs = max(5, min(cfg.preprocessing.n_pcs, nhv - 1))
    if is_smoke(cfg):
        n_pcs = min(n_pcs, 20)

    logger.info("Running PCA n_comps=%d (nhvg=%d)", n_pcs, nhv)
    sc.tl.pca(adata, n_comps=n_pcs)
    logger.info(
        "Preprocessed shape=%s; HVG count=%d",
        adata.shape,
        int(adata.var["highly_variable"].sum()),
    )
    return adata
