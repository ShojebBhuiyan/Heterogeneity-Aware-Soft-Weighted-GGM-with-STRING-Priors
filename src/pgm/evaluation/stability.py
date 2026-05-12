"""Bootstrap repetitions on global GGM."""

from __future__ import annotations

import logging

import numpy as np
from sklearn.covariance import empirical_covariance

from pgm.models.glm_utils import expression_dense, graphical_lasso_from_covariance, precision_to_binary_adj

logger = logging.getLogger("pgm.eval.bootstrap")


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
