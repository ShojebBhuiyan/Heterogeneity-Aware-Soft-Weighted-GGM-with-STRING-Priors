"""Serialize global GGM results."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import joblib
import scanpy as sc

from pgm.config.schemas import ProjectConfig
from pgm.models.global_ggm import fit_global_graphical_lasso
from pgm.utils.logging_setup import configure_logging
from pgm.utils.paths import checkpoints_dir, ensure_parents
from pgm.utils.seeds import set_global_seed

logger = logging.getLogger("pgm.pipelines.global_ggm")


def run_global_ggm_pipeline(
    cfg: ProjectConfig,
    clustered_h5ad: Path,
    *,
    force: bool = False,
    bundle_name: str = "global_ggm.joblib",
) -> Path:
    """Fit global graphical lasso and persist under ``data/checkpoints``."""
    configure_logging("pgm", cfg)
    set_global_seed(cfg)
    t0 = time.perf_counter()
    outp = checkpoints_dir(cfg) / bundle_name
    ensure_parents(outp)
    if not force and outp.exists():
        logger.info("Global GGM skip (exists) %.2fs", time.perf_counter() - t0)
        return outp

    adata = sc.read_h5ad(clustered_h5ad)
    bundle = fit_global_graphical_lasso(adata, cfg)
    joblib.dump(
        {"precision": bundle["precision"], "adjacency": bundle["adjacency"]},
        outp,
        compress=3,
    )
    logger.info("Global GGM saved %s", outp)
    return outp

