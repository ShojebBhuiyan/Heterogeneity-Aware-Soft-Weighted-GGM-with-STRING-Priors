"""Evaluation stage: stability, overlap with STRING, cross-state similarity."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scanpy as sc

from pgm.config.schemas import ProjectConfig
from pgm.clustering.heterogeneity import membership_entropy
from pgm.evaluation.overlap import adjacency_to_pairs, precision_recall, string_to_pairs
from pgm.evaluation.similarity import frobenius_diff, jaccard_triu
from pgm.evaluation.sparsity import degree_stats
from pgm.evaluation.stability import (
    bootstrap_global_adjacency,
    bootstrap_soft_component_adjacency,
)
from pgm.utils.logging_setup import configure_logging
from pgm.utils.paths import ensure_parents, results_reports_dir
from pgm.utils.seeds import set_global_seed

logger = logging.getLogger("pgm.pipelines.eval")


def _mean_positive_triu_freq(freq_upper: np.ndarray) -> float:
    rows = np.triu(freq_upper, k=1)
    mask = rows > 0
    return float(np.mean(rows[mask])) if np.any(mask) else 0.0


def _pairwise_jaccard_stats(adjs: list[np.ndarray]) -> tuple[float, float, float]:
    vals: list[float] = []
    for a in range(len(adjs)):
        for b in range(a + 1, len(adjs)):
            vals.append(jaccard_triu(adjs[a], adjs[b]))
    if not vals:
        return 0.0, 0.0, 0.0
    return float(min(vals)), float(np.mean(vals)), float(max(vals))


def _mean_absdiff_adj(a: np.ndarray, b: np.ndarray) -> float:
    ua = np.triu(np.asarray(a, dtype=np.float64), k=1)
    ub = np.triu(np.asarray(b, dtype=np.float64), k=1)
    return float(np.mean(np.abs(ua - ub)))


def _gmm_soft_stats(W: np.ndarray) -> dict[str, float]:
    W = np.asarray(W, dtype=np.float64)
    ent = membership_entropy(W)
    mass = W.sum(axis=0)
    ess_list = []
    floor_w = 5e-3
    for k in range(W.shape[1]):
        w = np.clip(W[:, k], floor_w, None)
        s = float(w.sum())
        if s <= 0:
            ess_list.append(0.0)
            continue
        wn = w / s
        ess_list.append(float(1.0 / np.dot(wn, wn)))
    return {
        "gmm_mean_entropy": float(ent.mean()),
        "gmm_median_entropy": float(np.median(ent)),
        "gmm_max_prob_mean": float(W.max(axis=1).mean()),
        "gmm_component_mass_min": float(mass.min()),
        "gmm_component_mass_max": float(mass.max()),
        "gmm_min_ess": float(min(ess_list) if ess_list else 0.0),
        "gmm_max_ess": float(max(ess_list) if ess_list else 0.0),
    }


def run_evaluation_pipeline(
    cfg: ProjectConfig,
    clustered_h5ad: Path,
    global_joblib: Path,
    soft_joblib: Path,
    kg_joblib: Path,
    *,
    force: bool = False,
    metrics_csv: str | None = None,
) -> Path:
    """Aggregate bootstrap stability, STRING overlap, latent-state similarity, GMM diagnostics."""
    configure_logging("pgm", cfg)
    set_global_seed(cfg)
    t0 = time.perf_counter()
    fname = metrics_csv if metrics_csv is not None else cfg.evaluation.metrics_csv
    outp = results_reports_dir(cfg) / fname
    ensure_parents(outp)
    if not force and outp.exists():
        logger.info("Evaluation skip %.2fs", time.perf_counter() - t0)
        return outp

    adata = sc.read_h5ad(clustered_h5ad)
    glob = joblib.load(global_joblib)
    soft = joblib.load(soft_joblib)
    kg_bundle = joblib.load(kg_joblib)
    genes = adata.var_names.to_numpy()

    metrics: dict[str, float | int | str] = {
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
    }

    triu_freq, _ = bootstrap_global_adjacency(adata, cfg)
    metrics["bootstrap_mean_pos_freq"] = _mean_positive_triu_freq(triu_freq)

    soft_boot_k = cfg.evaluation.bootstrap_soft_component
    if soft_boot_k is not None:
        triu_soft, _ = bootstrap_soft_component_adjacency(adata, cfg, soft_boot_k)
        metrics["bootstrap_soft_mean_pos_freq"] = _mean_positive_triu_freq(triu_soft)
        metrics["bootstrap_soft_component_k"] = int(soft_boot_k)
    else:
        metrics["bootstrap_soft_mean_pos_freq"] = float("nan")
        metrics["bootstrap_soft_component_k"] = -1

    deg_g = degree_stats(glob["adjacency"])
    metrics["global_mean_degree"] = deg_g["mean_degree"]
    metrics["global_sparsity_triu"] = deg_g["sparsity_upper"]
    g_adj = glob["adjacency"]
    gtri = g_adj[np.triu_indices_from(g_adj, k=1)]
    metrics["global_edge_count"] = int(gtri.sum())

    gene_set = set(map(str, genes))
    truth = string_to_pairs(
        kg_bundle["edges"], gene_set, cfg.kg.confidence_threshold
    )
    metrics["string_truth_pair_count"] = int(len(truth))

    pred_pairs = adjacency_to_pairs(glob["adjacency"], genes)
    prec, rec = precision_recall(pred_pairs, truth)
    metrics["string_precision_global"] = prec
    metrics["string_recall_global"] = rec

    adjs_soft = [c["adjacency"] for c in soft]
    metrics["n_soft_components"] = int(len(adjs_soft))

    soft_counts = [
        int(a[np.triu_indices_from(a, k=1)].sum()) for a in adjs_soft
    ]
    if soft_counts:
        metrics["soft_edges_min"] = int(min(soft_counts))
        metrics["soft_edges_max"] = int(max(soft_counts))
        metrics["soft_edges_mean"] = float(np.mean(soft_counts))
    else:
        metrics["soft_edges_min"] = 0
        metrics["soft_edges_max"] = 0
        metrics["soft_edges_mean"] = 0.0

    soft_union = set()
    for c in soft:
        soft_union |= adjacency_to_pairs(c["adjacency"], genes)
    metrics["soft_union_edge_count"] = int(len(soft_union))
    ps, rs = precision_recall(soft_union, truth)
    metrics["string_precision_soft_union"] = ps
    metrics["string_recall_soft_union"] = rs

    if len(adjs_soft) >= 2:
        metrics["jac_0_1_soft"] = jaccard_triu(adjs_soft[0], adjs_soft[1])
        metrics["fro_k0_k1_soft"] = frobenius_diff(adjs_soft[0], adjs_soft[1])
    jmin, jmean, jmax = _pairwise_jaccard_stats(adjs_soft)
    metrics["jac_soft_pairwise_min"] = jmin
    metrics["jac_soft_pairwise_mean"] = jmean
    metrics["jac_soft_pairwise_max"] = jmax

    kgs = kg_bundle.get("kg_results", [])
    metrics["n_kg_components"] = int(len(kgs))
    kg_adjs = [k["adjacency"] for k in kgs]
    kg_counts = [
        int(a[np.triu_indices_from(a, k=1)].sum()) for a in kg_adjs
    ]
    if kg_counts:
        metrics["kg_edges_min"] = int(min(kg_counts))
        metrics["kg_edges_max"] = int(max(kg_counts))
        metrics["kg_edges_mean"] = float(np.mean(kg_counts))
    else:
        metrics["kg_edges_min"] = 0
        metrics["kg_edges_max"] = 0
        metrics["kg_edges_mean"] = 0.0

    kg_union_pairs = set()
    for k in kgs:
        kg_union_pairs |= adjacency_to_pairs(k["adjacency"], genes)
    metrics["kg_union_edge_count"] = int(len(kg_union_pairs))
    pk, rk = precision_recall(kg_union_pairs, truth)
    metrics["string_precision_kg_union"] = pk
    metrics["string_recall_kg_union"] = rk

    kj_min, kj_mean, kj_max = _pairwise_jaccard_stats(kg_adjs)
    metrics["jac_kg_pairwise_min"] = kj_min
    metrics["jac_kg_pairwise_mean"] = kj_mean
    metrics["jac_kg_pairwise_max"] = kj_max

    if len(kgs) >= 2:
        metrics["jac_kg_0_1"] = jaccard_triu(
            kgs[0]["adjacency"], kgs[1]["adjacency"]
        )

    diffs = []
    for i in range(min(len(soft), len(kgs))):
        diffs.append(
            _mean_absdiff_adj(soft[i]["adjacency"], kgs[i]["adjacency"])
        )
    metrics["kg_vs_soft_mean_absdiff"] = (
        float(np.mean(diffs)) if diffs else 0.0
    )
    metrics["kg_vs_soft_max_absdiff"] = (
        float(np.max(diffs)) if diffs else 0.0
    )

    if "X_gmm_proba" in adata.obsm:
        metrics.update(_gmm_soft_stats(np.asarray(adata.obsm["X_gmm_proba"])))

    pd.DataFrame([metrics]).to_csv(outp, index=False)
    logger.info(
        "Evaluation written %s in %.2fs", outp, time.perf_counter() - t0
    )
    return outp
