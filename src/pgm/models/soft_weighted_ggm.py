"""Soft-membership weighted GGM components."""

from __future__ import annotations

import logging

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
    out = []
    for k in range(K):
        weights = np.clip(W[:, k], 5e-3, None)
        try:
            emp = weighted_scatter_cov(X, weights, ridge=ridge)
            theta = graphical_lasso_from_covariance(emp, cfg)
        except Exception as e:
            logger.error("Soft GGM component %d failed (%s)", k, e)
            raise
        adj = precision_to_binary_adj(theta, cfg.models.adjacency_tol)
        out.append({"k": k, "precision": theta, "adjacency": adj, "emp_cov": emp})
    logger.info(
        "Soft weighted GGM computed for %d components on %s cells × %s genes",
        K,
        X.shape[0],
        X.shape[1],
    )
    return out

