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
    shrink = cfg.models.covariance_shrinkage
    t_run = time.perf_counter()
    ent = float(-(W * np.log(W + 1e-12)).sum(axis=1).mean())
    banner = (
        f"[Soft GGM] fit_soft_component_graphs: K={K} matrix={X.shape[0]}x{X.shape[1]} "
        f"mean_assignment_entropy={ent:.4f} gl_mode={cfg.models.gl_mode!r} "
        f"gl_alpha={cfg.models.gl_alpha} ridge={ridge:.2e} shrinkage={shrink:.3f}"
    )
    print(banner, flush=True)
    logger.info(
        "fit_soft_component_graphs start K=%d X.shape=%s mean_row_entropy=%.4f "
        "gl_mode=%s gl_alpha=%s ridge=%.2e shrinkage=%.4f retry_mults=%s",
        K,
        X.shape,
        ent,
        cfg.models.gl_mode,
        cfg.models.gl_alpha,
        ridge,
        shrink,
        list(cfg.models.gl_alpha_retry_multipliers),
    )
    out = []
    for k in range(K):
        t_k = time.perf_counter()
        lbl = f"soft_k={k}"
        raw_col = np.asarray(W[:, k], dtype=np.float64)
        n_clip = int(np.sum(raw_col < 5e-3))
        weights = np.clip(raw_col, 5e-3, None)
        ws = float(weights.sum())
        w_norm = weights / ws
        ess = float(1.0 / np.dot(w_norm, w_norm)) if ws > 0 else 0.0
        step_msg = (
            f"[Soft GGM] component {k + 1}/{K}: "
            "weighted scatter -> stabilize covariance -> graphical lasso "
            f"(mode={cfg.models.gl_mode!r}, ess≈{ess:.1f} cells, "
            f"weights clipped at floor for {n_clip}/{len(weights)} cells)"
        )
        print(step_msg, flush=True)
        logger.info("%s", step_msg)
        try:
            emp = weighted_scatter_cov(X, weights, log_label=lbl)
            theta = graphical_lasso_from_covariance(
                emp, cfg, log_label=lbl, effective_n=ess
            )
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

