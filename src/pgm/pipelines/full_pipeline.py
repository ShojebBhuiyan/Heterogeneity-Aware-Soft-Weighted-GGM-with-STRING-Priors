"""Run all stages in order with checkpoints (smoke-friendly defaults)."""

from __future__ import annotations

import logging
import time
from typing import Any

from pgm.config.schemas import ProjectConfig
from pgm.pipelines.cluster import run_clustering_pipeline
from pgm.pipelines.evaluation import run_evaluation_pipeline
from pgm.pipelines.global_ggm import run_global_ggm_pipeline
from pgm.pipelines.ingest import ingest_pbmc_pipeline
from pgm.pipelines.kg_pipeline import run_kg_pipeline
from pgm.pipelines.preprocess import run_preprocess_pipeline
from pgm.pipelines.soft_ggm import run_soft_weighted_ggm_pipeline
from pgm.utils.env import snapshot_run_environment
from pgm.utils.logging_setup import configure_logging

logger = logging.getLogger("pgm.pipelines.full")


def run_full_pipeline(cfg: ProjectConfig, *, force: bool = False) -> dict[str, Any]:
    """Ingest → preprocess → cluster → global/soft KG GGM → metrics + env snapshot."""
    configure_logging("pgm", cfg)
    t_wall = time.perf_counter()
    raw = ingest_pbmc_pipeline(cfg, force=force)
    proc = run_preprocess_pipeline(cfg, raw, force=force)
    cl_p = run_clustering_pipeline(cfg, proc, force=force)

    gpath = run_global_ggm_pipeline(cfg, cl_p, force=force)
    spath = run_soft_weighted_ggm_pipeline(cfg, cl_p, force=force)
    kpath = run_kg_pipeline(cfg, cl_p, force=force)
    mpath = run_evaluation_pipeline(
        cfg,
        cl_p,
        gpath,
        spath,
        kpath,
        force=force,
    )
    snap = snapshot_run_environment(cfg, tag="full_pipeline")

    logger.info("Full pipeline finished wall=%.2fs", time.perf_counter() - t_wall)
    return {
        "interim_raw_h5ad": raw,
        "processed_h5ad": proc,
        "clustered_h5ad": cl_p,
        "global_ggm_joblib": gpath,
        "soft_ggm_joblib": spath,
        "kg_joblib": kpath,
        "metrics_csv": mpath,
        "env_json": snap,
    }

