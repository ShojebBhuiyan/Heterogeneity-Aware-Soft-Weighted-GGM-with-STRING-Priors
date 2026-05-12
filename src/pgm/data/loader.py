"""Load 10x MTX-formatted single-cell data."""

from __future__ import annotations

import logging
from pathlib import Path

import scanpy as sc

from pgm.config.schemas import ProjectConfig
from pgm.utils.paths import interim_dir, tenx_mtx_path

logger = logging.getLogger("pgm.data")


def load_pbmc_mtx(cfg: ProjectConfig, *, cache_raw: Path | None = None) -> sc.AnnData:
    """
    Read 10x ``matrix.mtx`` + ``genes.tsv`` + ``barcodes.tsv`` directory.

    Parameters
    ----------
    cfg
        Project configuration containing ``data.tenx_mtx_dir``.
    cache_raw
        If set, write a copy here (e.g. under ``data/raw`` for provenance).
    """
    mtx_dir = tenx_mtx_path(cfg)
    if not mtx_dir.is_dir():
        raise FileNotFoundError(f"10x directory not found: {mtx_dir}")
    logger.info("Loading 10x from %s", mtx_dir)
    adata = sc.read_10x_mtx(mtx_dir, var_names="gene_symbols", cache=True)
    adata.var_names_make_unique()
    if cache_raw is not None:
        cache_raw.parent.mkdir(parents=True, exist_ok=True)
        adata.write_h5ad(cache_raw)
        logger.info("Wrote raw copy to %s", cache_raw)
    return adata


def interim_h5ad_path(cfg: ProjectConfig) -> Path:
    return interim_dir(cfg) / cfg.ingestion.write_interim_filename
