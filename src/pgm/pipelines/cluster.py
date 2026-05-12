"""Clustering + GMM pipeline."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import scanpy as sc

from pgm.clustering.heterogeneity import (
    neighbors_leiden_umap,
    fit_gmm_soft_labels,
    plot_cluster_visuals,
)
from pgm.config.schemas import ProjectConfig
from pgm.utils.checkpoint import checkpoint_exists, save_checkpoint
from pgm.utils.logging_setup import configure_logging
from pgm.utils.paths import ensure_parents, processed_dir
from pgm.utils.seeds import set_global_seed

logger = logging.getLogger("pgm.pipelines.cluster")


def run_clustering_pipeline(
    cfg: ProjectConfig,
    processed_h5ad: Path,
    *,
    force: bool = False,
    checkpoint_name: str = "cluster_done",
    output_filename: str | None = None,
) -> Path:
    """
    Loads processed AnnData, adds neighbors / Leiden / GMM soft labels, saves.

    Output defaults to sibling ``*_clustered.h5ad`` under ``processed/``.
    """
    configure_logging("pgm", cfg)
    set_global_seed(cfg)
    t0 = time.perf_counter()
    out_fn = (
        output_filename
        or cfg.preprocessing.output_filename.replace(".h5ad", "_clustered.h5ad")
    )
    outp = processed_dir(cfg) / out_fn
    processed_dir(cfg).mkdir(parents=True, exist_ok=True)

    if not force and checkpoint_exists(cfg, checkpoint_name) and outp.exists():
        logger.info("Cluster pipeline skip %.2fs", time.perf_counter() - t0)
        return outp

    adata = sc.read_h5ad(processed_h5ad)
    neighbors_leiden_umap(adata, cfg)
    fit_gmm_soft_labels(adata, cfg)
    plot_cluster_visuals(adata, cfg)
    ensure_parents(outp)
    adata.write_h5ad(outp)
    save_checkpoint(cfg, checkpoint_name, adata)
    logger.info(
        "Clustering done %.2fs → %s (n_obs=%s)", time.perf_counter() - t0, outp, adata.n_obs
    )
    return outp
