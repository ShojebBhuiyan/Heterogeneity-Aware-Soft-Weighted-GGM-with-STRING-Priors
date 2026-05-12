"""Shared helpers for inverse-covariance (graphical lasso) fits."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.covariance import GraphicalLasso, GraphicalLassoCV, graphical_lasso

from pgm.config.schemas import ProjectConfig

logger = logging.getLogger("pgm.models.glm")


def expression_dense(adata) -> np.ndarray:
    """Return ``float64`` (n_cells × n_genes) expression matrix."""
    import scanpy as sc  # noqa: F401 — type namespace
    X = adata.X
    from scipy import sparse

    if sparse.issparse(X):
        return np.asarray(X.toarray(), dtype=np.float64)
    return np.asarray(X, dtype=np.float64)


def weighted_scatter_cov(
    X: np.ndarray,
    weights: np.ndarray,
    *,
    ridge: float,
) -> np.ndarray:
    """
    Weighted covariance with weights summing arbitrarily (normalized internally).

    Uses ``Σ w_i (x_i-μ)(x_i-μ)ᵀ`` with ``w ← w/sum(w)``.
    Adds ``ridge · I``.
    """
    X = np.asarray(X, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64).ravel()
    s = float(w.sum())
    if s <= 0:
        raise ValueError("Weights must sum to a positive number")
    w = w / s
    mu = np.average(X, axis=0, weights=w)
    xc = X - mu
    z = np.sqrt(w)[:, np.newaxis] * xc
    sigma = z.T @ z + ridge * np.eye(X.shape[1], dtype=np.float64)
    return sigma


def fit_graphical_lasso_from_samples(
    X: np.ndarray, cfg: ProjectConfig
) -> tuple[Any, np.ndarray]:
    """
    Fit GLasso with CV if ``cfg.models.gl_alpha`` is unset, else fixed ``alpha``.
    Returns (fitted_estimator_or_none, precision_).
    """
    if cfg.models.gl_alpha is None:
        model = GraphicalLassoCV(
            cv=cfg.models.gl_cv_folds,
            tol=cfg.models.gl_tol,
            max_iter=cfg.models.gl_max_iter,
        )
        model.fit(X)
        return model, model.precision_
    model = GraphicalLasso(
        alpha=float(cfg.models.gl_alpha),
        tol=cfg.models.gl_tol,
        max_iter=cfg.models.gl_max_iter,
    )
    model.fit(X)
    return model, model.precision_


def graphical_lasso_from_covariance(
    emp_cov: np.ndarray, cfg: ProjectConfig
) -> np.ndarray:
    """
    Run Friedman graphical lasso on an empirical covariance (stabilized to SPD).

    Prefer ``cfg.models.gl_alpha``; otherwise uses a heuristic default.
    """
    emp_cov = np.asarray(emp_cov, dtype=np.float64)
    emp_cov = 0.5 * (emp_cov + emp_cov.T)
    eig_min = float(np.linalg.eigvalsh(emp_cov).min())
    floor = cfg.models.covariance_ridge
    if eig_min <= 0:
        floor += max(-2.0 * eig_min, 1e-8)
    p = emp_cov.shape[0]
    emp_cov = emp_cov + floor * np.eye(p)

    alpha_base = cfg.models.gl_alpha
    if alpha_base is None:
        alpha_base = float(np.logspace(-2, -0.2, num=12)[7])
        logger.warning(
            "gl_alpha unset; defaulting to %.6f on covariance fit", alpha_base
        )

    last_err: Exception | None = None
    for mult in (1.0, 4.0, 16.0, 64.0, 256.0):
        alpha = float(alpha_base) * mult
        try:
            prec, _ = graphical_lasso(
                emp_cov,
                alpha=max(alpha, floor * 2.0 + 1e-8),
                max_iter=max(500, cfg.models.gl_max_iter),
                tol=max(cfg.models.gl_tol, 1e-5),
                mode="lars",
            )
            return prec
        except Exception as exc:
            last_err = exc
            logger.warning("graphical_lasso retry mult=%s (%s)", mult, exc)

    if last_err is not None:
        raise last_err
    raise RuntimeError("graphical_lasso_from_covariance failed")


def precision_to_binary_adj(theta: np.ndarray, tol: float) -> np.ndarray:
    """Undirected edges where off-diagonal |precision| > tol."""
    p = theta.shape[0]
    adj = np.abs(theta) > tol
    adj = adj.astype(np.int8)
    np.fill_diagonal(adj, 0)
    return adj

