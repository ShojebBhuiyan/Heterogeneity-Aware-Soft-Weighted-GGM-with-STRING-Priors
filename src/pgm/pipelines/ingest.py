"""Stage: load raw PBMC 10x, validate, summarize, interim checkpoint."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from pgm.config.schemas import ProjectConfig
from pgm.data.loader import interim_h5ad_path, load_pbmc_mtx
from pgm.data.summary import dataset_summary
from pgm.data.validate import validate_anndata_basic
from pgm.utils.checkpoint import checkpoint_exists, load_checkpoint, save_checkpoint
from pgm.utils.logging_setup import configure_logging
from pgm.utils.paths import ensure_parents, interim_dir, results_reports_dir
from pgm.utils.seeds import set_global_seed

logger = logging.getLogger("pgm.pipelines.ingest")


def ingest_pbmc_pipeline(
    cfg: ProjectConfig,
    *,
    force: bool = False,
    checkpoint_name: str = "ingestion_raw",
) -> Path:
    set_global_seed(cfg)
    t0 = time.perf_counter()
    out_h5ad = interim_h5ad_path(cfg)
    interim_dir(cfg).mkdir(parents=True, exist_ok=True)

    if not force and checkpoint_exists(cfg, checkpoint_name):
        adata = load_checkpoint(cfg, checkpoint_name)
        if not out_h5ad.exists():
            ensure_parents(out_h5ad)
            adata.write_h5ad(out_h5ad)
        logger.info("Ingest skip (checkpoint) in %.2fs", time.perf_counter() - t0)
        return out_h5ad

    adata = load_pbmc_mtx(cfg)
    validate_anndata_basic(adata)
    summary = dataset_summary(adata)

    summ_path = results_reports_dir(cfg) / "ingestion_summary.json"
    ensure_parents(summ_path)
    with summ_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Wrote %s", summ_path)

    ensure_parents(out_h5ad)
    adata.write_h5ad(out_h5ad)
    save_checkpoint(cfg, checkpoint_name, adata)
    logger.info(
        "Ingestion done in %.2fs, shape=%s, wrote %s",
        time.perf_counter() - t0,
        adata.shape,
        out_h5ad,
    )
    return out_h5ad
