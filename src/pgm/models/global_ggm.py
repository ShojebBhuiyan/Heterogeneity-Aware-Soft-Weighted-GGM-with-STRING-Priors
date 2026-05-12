"""Global Gaussian graphical model (single precision over all cells)."""

from __future__ import annotations

import logging

import numpy as np

from pgm.config.schemas import ProjectConfig
from pgm.models.glm_utils import (
    expression_dense,
    fit_graphical_lasso_from_samples,
    precision_to_binary_adj,
)

logger = logging.getLogger("pgm.models.global_ggm")


def fit_global_graphical_lasso(adata, cfg: ProjectConfig) -> dict:
    """Graphical Lasso on all retained genes (scaled expression in ``adata.X``)."""
    X = expression_dense(adata)
    logger.info(
        "Global GGM fitting on X shape %s alpha=%s", X.shape, cfg.models.gl_alpha
    )
    model, theta = fit_graphical_lasso_from_samples(X, cfg)
    adj = precision_to_binary_adj(theta, cfg.models.adjacency_tol)
    genes = np.array(adata.var_names)
    triu_edges = int(adj[np.triu_indices_from(adj, k=1)].sum())
    logger.info(
        "Global GGM precision cond=%s edges=%d",
        f"{np.linalg.cond(theta):.2e}",
        triu_edges,
    )
    return {
        "model": model,
        "precision": theta,
        "adjacency": adj,
        "gene_names": genes,
        "covariance_model": getattr(model, "covariance_", None),
    }

