"""Bootstrap repetitions on global GGM."""

from __future__ import annotations

import logging

import numpy as np
from sklearn.covariance import empirical_covariance

from pgm.models.glm_utils import (
    expression_dense,
    graphical_lasso_from_covariance,
    precision_to_binary_adj,
    weighted_scatter_cov,
)

logger = logging.getLogger("pgm.eval.bootstrap")


def bootstrap_soft_component_adjacency(
    adata,
    cfg,
    component_k: int,
) -> tuple[np.ndarray, float]:
    """
    Bootstrap edge frequency for soft-weighted GLasso on mixture component ``component_k``.

    Each replicate subsamples cells (with replacement), recomputes weighted scatter
    covariance with weights ``clip(W[:, k], floor, ∞)``, then fits graphical lasso.
    """
    X_full = expression_dense(adata)
    if "X_gmm_proba" not in adata.obsm:
        raise KeyError("obsm['X_gmm_proba'] missing — run clustering first")
    W_full = np.asarray(adata.obsm["X_gmm_proba"])
    if component_k < 0 or component_k >= W_full.shape[1]:
        raise IndexError(f"component_k={component_k} out of range for W.shape={W_full.shape}")

    n, _ = X_full.shape
    b = cfg.evaluation.bootstrap_b
    frac = cfg.evaluation.bootstrap_fraction
    rng = np.random.default_rng(cfg.run.random_seed)
    tally = np.zeros((X_full.shape[1], X_full.shape[1]), dtype=np.float64)
    ok_runs = 0

    floor_w = 5e-3
    for _rep in range(b):
        idx = rng.choice(n, size=max(10, int(n * frac)), replace=True)
        Xb = X_full[idx]
        wb_raw = np.asarray(W_full[idx, component_k], dtype=np.float64)
        wb = np.clip(wb_raw, floor_w, None)
        ws = float(wb.sum())
        if ws <= 0:
            continue
        w_norm = wb / ws
        ess = float(1.0 / np.dot(w_norm, w_norm)) if ws > 0 else 0.0
        try:
            emp = weighted_scatter_cov(Xb, wb, log_label=f"boot_soft_k={component_k}")
            theta = graphical_lasso_from_covariance(
                emp,
                cfg,
                log_label=f"boot_soft_k={component_k}",
                effective_n=ess,
            )
            adj = precision_to_binary_adj(theta, cfg.models.adjacency_tol)
            tally += adj
            ok_runs += 1
        except (FloatingPointError, ValueError, np.linalg.LinAlgError) as exc:
            logger.debug("bootstrap soft rep skipped (%s)", exc)

    denom = max(ok_runs, 1)
    freq = tally / denom
    triu = np.triu(freq, k=1)
    logger.info(
        "bootstrap soft k=%d averaged over %d / %d successful folds",
        component_k,
        ok_runs,
        b,
    )
    return triu, 0.0


def bootstrap_global_adjacency(
    adata,
    cfg,
) -> tuple[np.ndarray, float]:
    """
    Fraction of bootstrap replicates selecting each unordered edge under global GLasso.

    Uses empirical covariance + PSD-stabilised ``graphical_lasso`` per replicate.
    Failed replicates (ill-conditioned subsets) are skipped.
    """
    X_full = expression_dense(adata)
    n, _ = X_full.shape
    b = cfg.evaluation.bootstrap_b
    frac = cfg.evaluation.bootstrap_fraction
    rng = np.random.default_rng(cfg.run.random_seed)
    tally = np.zeros((X_full.shape[1], X_full.shape[1]), dtype=np.float64)
    ok_runs = 0
    for _rep in range(b):
        idx = rng.choice(n, size=max(10, int(n * frac)), replace=True)
        Xb = X_full[idx]
        try:
            emp = empirical_covariance(Xb, assume_centered=False)
            theta = graphical_lasso_from_covariance(emp, cfg)
            adj = precision_to_binary_adj(theta, cfg.models.adjacency_tol)
            tally += adj
            ok_runs += 1
        except (FloatingPointError, ValueError, np.linalg.LinAlgError) as exc:
            logger.debug("bootstrap rep skipped (%s)", exc)

    denom = max(ok_runs, 1)
    freq = tally / denom
    triu = np.triu(freq, k=1)
    logger.info(
        "bootstrap adjacency averaged over %d / %d successful folds", ok_runs, b
    )
    return triu, 0.0
