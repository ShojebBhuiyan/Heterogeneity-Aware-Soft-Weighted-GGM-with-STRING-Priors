"""Soft-membership weighted GGM components."""

from __future__ import annotations

import logging
import time

import numpy as np

from pgm.config.schemas import ProjectConfig
from pgm.models.glm_utils import (
    expression_dense,
    graphical_lasso_from_covariance,
    precision_to_binary_adj,
    weighted_scatter_cov,
)

logger = logging.getLogger("pgm.models.soft_ggm")


def fit_soft_component_graphs(
    adata,
    cfg: ProjectConfig,
) -> list[dict]:
    """One graphical lasso per GMM latent state weighted by posterior membership."""
    if "X_gmm_proba" not in adata.obsm:
        raise KeyError("obsm['X_gmm_proba'] missing — run clustering first")
    W = np.asarray(adata.obsm["X_gmm_proba"])
    X = expression_dense(adata)
    K = W.shape[1]
    ridge = cfg.models.covariance_ridge
    t_run = time.perf_counter()
    ent = float(-(W * np.log(W + 1e-12)).sum(axis=1).mean())
    logger.info(
        "fit_soft_component_graphs start K=%d X.shape=%s mean_row_entropy=%.4f ridge=%.2e",
        K,
        X.shape,
        ent,
        ridge,
    )
    out = []
    for k in range(K):
        t_k = time.perf_counter()
        lbl = f"soft_k={k}"
        weights = np.clip(W[:, k], 5e-3, None)
        try:
            emp = weighted_scatter_cov(X, weights, ridge=ridge, log_label=lbl)
            theta = graphical_lasso_from_covariance(emp, cfg, log_label=lbl)
        except Exception as e:
            logger.error("Soft GGM component %d failed (%s)", k, e)
            raise
        adj = precision_to_binary_adj(theta, cfg.models.adjacency_tol)
        tri = adj[np.triu_indices_from(adj, k=1)]
        n_tri = int(tri.size)
        n_edge = int(tri.sum())
        dens = float(tri.mean()) if n_tri else 0.0
        logger.info(
            "[%s] component done wall=%.3fs triu_edges=%d/%d density=%.5f",
            lbl,
            time.perf_counter() - t_k,
            n_edge,
            n_tri,
            dens,
        )
        out.append({"k": k, "precision": theta, "adjacency": adj, "emp_cov": emp})
    logger.info(
        "Soft weighted GGM finished %d components on %s cells × %s genes total %.3fs",
        K,
        X.shape[0],
        X.shape[1],
        time.perf_counter() - t_run,
    )
    return out

