"""Evaluation stage: stability, overlap with STRING, cross-state similarity."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import joblib
import pandas as pd
import scanpy as sc

from pgm.config.schemas import ProjectConfig
from pgm.evaluation.overlap import adjacency_to_pairs, precision_recall, string_to_pairs
from pgm.evaluation.similarity import frobenius_diff, jaccard_triu
from pgm.evaluation.sparsity import degree_stats
from pgm.evaluation.stability import bootstrap_global_adjacency
from pgm.utils.logging_setup import configure_logging
from pgm.utils.paths import ensure_parents, results_reports_dir
from pgm.utils.seeds import set_global_seed

logger = logging.getLogger("pgm.pipelines.eval")


def run_evaluation_pipeline(
    cfg: ProjectConfig,
    clustered_h5ad: Path,
    global_joblib: Path,
    soft_joblib: Path,
    kg_joblib: Path,
    *,
    force: bool = False,
    metrics_csv: str = "metrics_summary.csv",
) -> Path:
    """Aggregate bootstrap stability, STRING overlap, latent-state Jaccard/Frobenius."""
    configure_logging("pgm", cfg)
    set_global_seed(cfg)
    t0 = time.perf_counter()
    outp = results_reports_dir(cfg) / metrics_csv
    ensure_parents(outp)
    if not force and outp.exists():
        logger.info("Evaluation skip %.2fs", time.perf_counter() - t0)
        return outp

    adata = sc.read_h5ad(clustered_h5ad)
    glob = joblib.load(global_joblib)
    soft = joblib.load(soft_joblib)
    kg_bundle = joblib.load(kg_joblib)
    genes = adata.var_names.to_numpy()

    triu_freq, _ = bootstrap_global_adjacency(adata, cfg)
    rows = np.triu(triu_freq, k=1)
    mean_stable = (
        float(np.mean(rows[rows > 0])) if np.any(rows > 0) else 0.0
    )

    deg_g = degree_stats(glob["adjacency"])
    metrics: dict[str, float | int | str] = {
        "bootstrap_mean_pos_freq": mean_stable,
        "global_mean_degree": deg_g["mean_degree"],
        "global_sparsity_triu": deg_g["sparsity_upper"],
    }

    gene_set = set(map(str, genes))
    truth = string_to_pairs(
        kg_bundle["edges"], gene_set, cfg.kg.confidence_threshold
    )
    pred_pairs = adjacency_to_pairs(glob["adjacency"], genes)
    prec, rec = precision_recall(pred_pairs, truth)
    metrics["string_precision_global"] = prec
    metrics["string_recall_global"] = rec

    adjs = [c["adjacency"] for c in soft]
    if len(adjs) >= 2:
        metrics["jac_0_1_soft"] = jaccard_triu(adjs[0], adjs[1])
        metrics["fro_k0_k1_soft"] = frobenius_diff(adjs[0], adjs[1])

    kgs = kg_bundle.get("kg_results", [])
    if len(kgs) >= 2:
        metrics["jac_kg_0_1"] = jaccard_triu(
            kgs[0]["adjacency"], kgs[1]["adjacency"]
        )

    pd.DataFrame([metrics]).to_csv(outp, index=False)
    logger.info(
        "Evaluation written %s in %.2fs", outp, time.perf_counter() - t0
    )
    return outp

