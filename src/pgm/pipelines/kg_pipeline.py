"""STRING prior + KG-biased GGM."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

from pgm.config.schemas import ProjectConfig
from pgm.kg.prior_matrix import edges_to_sparse_prior
from pgm.kg.string_api import fetch_string_for_genes
from pgm.models.kg_ggm import fit_kg_soft_graphs
from pgm.utils.logging_setup import configure_logging
from pgm.utils.paths import (
    checkpoints_dir,
    ensure_parents,
    figures_dir,
    results_reports_dir,
)
from pgm.utils.seeds import set_global_seed
from pgm.visualization.theme import apply_publication_theme, save_dual_format

logger = logging.getLogger("pgm.pipelines.kg")


def run_kg_pipeline(
    cfg: ProjectConfig,
    clustered_h5ad: Path,
    *,
    force: bool = False,
    bundle_name: str = "kg_soft_ggm.joblib",
) -> Path:
    """Fetch STRING edges, build prior ``P``, fit KG-GGM per mixture state."""
    configure_logging("pgm", cfg)
    set_global_seed(cfg)
    t0 = time.perf_counter()
    outp = checkpoints_dir(cfg) / bundle_name
    ensure_parents(outp)
    if not force and outp.exists():
        logger.info("KG pipeline skip (checkpoint exists) %.2fs → %s", time.perf_counter() - t0, outp)
        return outp

    logger.info("KG pipeline loading AnnData %s", clustered_h5ad.resolve())
    t_read = time.perf_counter()
    adata = sc.read_h5ad(clustered_h5ad)
    logger.info("KG AnnData read %.3fs shape=%s", time.perf_counter() - t_read, adata.shape)
    genes = list(adata.var_names.astype(str))

    t_str = time.perf_counter()
    edges = fetch_string_for_genes(genes, cfg)
    logger.info(
        "KG STRING edges %.3fs rows=%d cols=%s",
        time.perf_counter() - t_str,
        len(edges),
        list(edges.columns) if not edges.empty else [],
    )

    t_pr = time.perf_counter()
    prior = edges_to_sparse_prior(
        edges, genes, cfg.kg.confidence_threshold
    )
    logger.info(
        "KG prior matrix %.3fs shape=%s offdiag_nnz=%d",
        time.perf_counter() - t_pr,
        prior.shape,
        int(np.count_nonzero(np.triu(prior, k=1))),
    )

    t_fit = time.perf_counter()
    kg_res = fit_kg_soft_graphs(adata, prior, cfg)
    logger.info("KG fit_kg_soft_graphs %.3fs", time.perf_counter() - t_fit)

    fig_dir = figures_dir(cfg) / "kg"
    fig_dir.mkdir(parents=True, exist_ok=True)
    if not edges.empty and "score" in edges.columns:
        apply_publication_theme()
        fig_h, axh = plt.subplots(figsize=(5, 3.5))
        axh.hist(edges["score"], bins=40)
        axh.set_xlabel("STRING score (norm)")
        save_dual_format(fig_h, fig_dir / "string_score_hist")
        plt.close(fig_h)

    t_dump = time.perf_counter()
    joblib.dump({"edges": edges, "prior": prior, "kg_results": kg_res}, outp, compress=3)
    logger.info("KG joblib.dump → %s %.3fs", outp, time.perf_counter() - t_dump)
    pq = results_reports_dir(cfg) / "string_edges.parquet"
    ensure_parents(pq)
    t_pq = time.perf_counter()
    edges.to_parquet(pq)
    logger.info("KG edges.to_parquet %.3fs → %s", time.perf_counter() - t_pq, pq)
    logger.info("KG pipeline total %.3fs", time.perf_counter() - t0)
    return outp

