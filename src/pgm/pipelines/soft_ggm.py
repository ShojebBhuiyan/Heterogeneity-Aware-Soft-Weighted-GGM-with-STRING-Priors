"""Soft-weighted GGM persistence."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc

from pgm.config.schemas import ProjectConfig
from pgm.models.soft_weighted_ggm import fit_soft_component_graphs
from pgm.utils.logging_setup import configure_logging
from pgm.utils.paths import checkpoints_dir, ensure_parents, figures_dir
from pgm.utils.seeds import set_global_seed
from pgm.visualization.theme import apply_publication_theme, save_dual_format

logger = logging.getLogger("pgm.pipelines.soft_ggm")


def run_soft_weighted_ggm_pipeline(
    cfg: ProjectConfig,
    clustered_h5ad: Path,
    *,
    force: bool = False,
    bundle_name: str = "soft_weighted_ggm.joblib",
) -> Path:
    """Fit per-component graphical models."""
    configure_logging("pgm", cfg)
    set_global_seed(cfg)
    t0 = time.perf_counter()
    outp = checkpoints_dir(cfg) / bundle_name
    ensure_parents(outp)
    if not force and outp.exists():
        logger.info("Soft GGM skip %.2fs", time.perf_counter() - t0)
        return outp

    adata = sc.read_h5ad(clustered_h5ad)
    comps = fit_soft_component_graphs(adata, cfg)
    fig_dir = figures_dir(cfg) / "ggm"
    fig_dir.mkdir(parents=True, exist_ok=True)
    apply_publication_theme()
    fig_o, ax = plt.subplots(figsize=(6, 4))
    ol = []
    labels = []
    for c in comps:
        tri = c["adjacency"][np.triu_indices_from(c["adjacency"], k=1)]
        ol.append(np.mean(tri))
        labels.append(f"k={c['k']}")
    ax.bar(labels, ol)
    ax.set_ylabel("edge density (triu)")
    ax.set_title("Soft-weighted GGM edge density")
    fig_o.tight_layout()
    save_dual_format(fig_o, fig_dir / "soft_density")
    plt.close(fig_o)

    joblib.dump(comps, outp, compress=3)
    logger.info("Soft weighted GGM done %.2fs", time.perf_counter() - t0)
    return outp

