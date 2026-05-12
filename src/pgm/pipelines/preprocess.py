"""Preprocessing stage: interim raw → processed AnnData."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import scanpy as sc

from pgm.config.schemas import ProjectConfig
from pgm.preprocessing.pipeline import preprocess_adata
from pgm.utils.checkpoint import checkpoint_exists, save_checkpoint
from pgm.utils.logging_setup import configure_logging
from pgm.utils.paths import ensure_parents, processed_dir, processed_h5ad_path
from pgm.utils.seeds import set_global_seed

logger = logging.getLogger("pgm.pipelines.preprocess")


def run_preprocess_pipeline(
    cfg: ProjectConfig,
    interim_h5ad: Path,
    *,
    force: bool = False,
    checkpoint_name: str = "preprocess_done",
) -> Path:
    """QC, normalize, HVG, scale, PCA; write ``data/processed/*.h5ad``."""
    configure_logging("pgm", cfg)
    set_global_seed(cfg)
    t0 = time.perf_counter()
    out = processed_h5ad_path(cfg)
    processed_dir(cfg).mkdir(parents=True, exist_ok=True)

    if not force and checkpoint_exists(cfg, checkpoint_name) and out.exists():
        try:
            chk = sc.read_h5ad(out)
        except OSError:
            chk = None
        if chk is not None and "X_pca" in chk.obsm:
            logger.info(
                "Preprocess skip %.2fs (checkpoint)", time.perf_counter() - t0
            )
            return out
        logger.warning("Preprocess checkpoint invalid — recomputing")

    adata = sc.read_h5ad(interim_h5ad)
    adata = preprocess_adata(adata, cfg)
    ensure_parents(out)
    adata.write_h5ad(out)
    save_checkpoint(cfg, checkpoint_name, adata)
    logger.info(
        "Preprocessed %.2fs → %s shape=%s",
        time.perf_counter() - t0,
        out,
        adata.shape,
    )
    return out

