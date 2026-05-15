"""Knowledge-graph covariance inflation + graphical lasso (heuristic priors)."""

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

logger = logging.getLogger("pgm.models.kg_ggm")


def fit_kg_soft_graphs(
    adata,
    prior_mat: np.ndarray,
    cfg: ProjectConfig,
) -> list[dict]:
    """Per-GMM-component graphical lasso on ``Σ_k + λ P`` covariance blend."""
    if "X_gmm_proba" not in adata.obsm:
        raise KeyError("obsm['X_gmm_proba'] missing")
    W = np.asarray(adata.obsm["X_gmm_proba"])
    X = expression_dense(adata)
    genes = np.array(adata.var_names)
    if prior_mat.shape != (X.shape[1], X.shape[1]):
        raise ValueError(f"prior_mat {prior_mat.shape} vs genes {X.shape[1]}")

    scales = cfg.kg.prior_covariance_scale
    ridge = cfg.models.covariance_ridge
    K = W.shape[1]
    t_run = time.perf_counter()
    logger.info(
        "fit_kg_soft_graphs start K=%d genes=%d prior_scale=%.4f ridge=%.2e X.shape=%s",
        K,
        genes.size,
        scales,
        ridge,
        X.shape,
    )
    out_l: list[dict] = []
    for k in range(K):
        t_k = time.perf_counter()
        lbl = f"kg_k={k}"
        wt = np.clip(W[:, k], 5e-3, None)
        ws = float(wt.sum())
        w_norm = wt / ws if ws > 0 else wt
        ess = float(1.0 / np.dot(w_norm, w_norm)) if ws > 0 else 0.0
        emp = weighted_scatter_cov(X, wt, log_label=lbl)
        emp_blend = emp + scales * np.asarray(prior_mat, dtype=np.float64)
        theta = graphical_lasso_from_covariance(
            emp_blend, cfg, log_label=lbl, effective_n=ess
        )
        adj = precision_to_binary_adj(theta, cfg.models.adjacency_tol)
        tri = adj[np.triu_indices_from(adj, k=1)]
        logger.info(
            "[%s] KG component wall=%.3fs triu_density=%.5f",
            lbl,
            time.perf_counter() - t_k,
            float(tri.mean()) if tri.size else 0.0,
        )
        out_l.append(
            {
                "k": k,
                "precision": theta,
                "adjacency": adj,
                "emp_cov": emp_blend,
                "gene_names": genes,
            }
        )
    logger.info(
        "KG soft GGM blended %d states (λ=%.4f, genes=%d) total %.3fs",
        W.shape[1],
        scales,
        genes.size,
        time.perf_counter() - t_run,
    )
    return out_l

