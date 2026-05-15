"""Shared helpers for inverse-covariance (graphical lasso) fits."""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
from sklearn.covariance import empirical_covariance, graphical_lasso

from pgm.config.schemas import ProjectConfig

logger = logging.getLogger("pgm.models.glm")


def _glm_log_prefix(log_label: str | None) -> str:
    return f"[{log_label}] " if log_label else ""


def expression_dense(adata) -> np.ndarray:
    """Return ``float64`` (n_cells × n_genes) expression matrix."""
    import scanpy as sc  # noqa: F401 — type namespace
    t0 = time.perf_counter()
    X = adata.X
    from scipy import sparse

    sparse_in = sparse.issparse(X)
    if sparse_in:
        X = np.asarray(X.toarray(), dtype=np.float64)
    else:
        X = np.asarray(X, dtype=np.float64)
    np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    dt = time.perf_counter() - t0
    logger.info(
        "expression_dense shape=%s storage=%s → float64 %.3fs (%.1f MiB)",
        X.shape,
        "sparse" if sparse_in else "dense",
        dt,
        X.nbytes / (1024 * 1024),
    )
    return X


def weighted_scatter_cov(
    X: np.ndarray,
    weights: np.ndarray,
    *,
    ridge: float,
    log_label: str | None = None,
) -> np.ndarray:
    """
    Weighted covariance with weights summing arbitrarily (normalized internally).

    Uses ``Σ w_i (x_i-μ)(x_i-μ)ᵀ`` with ``w ← w/sum(w)``.
    Adds ``ridge · I``.
    """
    tag = _glm_log_prefix(log_label)
    t0 = time.perf_counter()
    X = np.asarray(X, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64).ravel()
    s = float(w.sum())
    if s <= 0:
        raise ValueError("Weights must sum to a positive number")
    w_raw_min = float(w.min())
    w_raw_max = float(w.max())
    w = w / s
    mu = np.average(X, axis=0, weights=w)
    xc = X - mu
    z = np.sqrt(w)[:, np.newaxis] * xc
    sigma = z.T @ z + ridge * np.eye(X.shape[1], dtype=np.float64)
    dt = time.perf_counter() - t0
    logger.info(
        "%sweighted_scatter_cov n=%d p=%d ridge=%.2e weights[min,max,sum_raw]=[%.4e,%.4e,%.4e] %.3fs",
        tag,
        X.shape[0],
        X.shape[1],
        ridge,
        w_raw_min,
        w_raw_max,
        s,
        dt,
    )
    return sigma


def fit_graphical_lasso_from_samples(
    X: np.ndarray,
    cfg: ProjectConfig,
    *,
    log_label: str | None = None,
) -> tuple[Any, np.ndarray]:
    """
    Fit graphical lasso on empirical covariance (ridge / SPD + α retries).

    Matches the stabilized path used by component GGMs and avoids
    ``GraphicalLasso`` / ``GraphicalLassoCV`` sample-based fits that can hit
    ill-conditioned coordinate descent on hard problems.

    If ``cfg.models.gl_alpha`` is unset, ``graphical_lasso_from_covariance``
    chooses a heuristic default (with a log warning).

    Returns ``(None, precision_)`` — no sklearn estimator is attached.
    """
    tag = _glm_log_prefix(log_label)
    X = np.asarray(X, dtype=np.float64)
    t_emp = time.perf_counter()
    emp = empirical_covariance(X, assume_centered=True)
    logger.info(
        "%sempirical_covariance n=%d p=%d %.3fs",
        tag,
        X.shape[0],
        X.shape[1],
        time.perf_counter() - t_emp,
    )
    theta = graphical_lasso_from_covariance(emp, cfg, log_label=log_label)
    return None, theta


def graphical_lasso_from_covariance(
    emp_cov: np.ndarray,
    cfg: ProjectConfig,
    *,
    log_label: str | None = None,
) -> np.ndarray:
    """
    Run Friedman graphical lasso on an empirical covariance (stabilized to SPD).

    Prefer ``cfg.models.gl_alpha``; otherwise uses a heuristic default.
    """
    tag = _glm_log_prefix(log_label)
    t_all = time.perf_counter()
    emp_cov = np.asarray(emp_cov, dtype=np.float64)
    emp_cov = 0.5 * (emp_cov + emp_cov.T)
    p = emp_cov.shape[0]
    t_eig = time.perf_counter()
    eig_min = float(np.linalg.eigvalsh(emp_cov).min())
    dt_eig = time.perf_counter() - t_eig
    floor = cfg.models.covariance_ridge
    floor_extra = 0.0
    if eig_min <= 0:
        floor_extra = max(-2.0 * eig_min, 1e-8)
        floor += floor_extra
    emp_cov = emp_cov + floor * np.eye(p)

    alpha_base = cfg.models.gl_alpha
    if alpha_base is None:
        alpha_base = float(np.logspace(-2, -0.2, num=12)[7])
        logger.warning(
            "%sgl_alpha unset; defaulting to %.6f on covariance fit",
            tag,
            alpha_base,
        )

    gl_tol = max(cfg.models.gl_tol, 1e-5)
    gl_max_iter = max(500, cfg.models.gl_max_iter)
    logger.info(
        "%sgraphical_lasso_from_covariance start p=%d eig_min=%.4e cov_ridge_cfg=%.4e "
        "floor_applied=%.4e (extra %.4e) eigvalsh=%.3fs alpha_base=%s mode=lars max_iter=%d tol=%.2e",
        tag,
        p,
        eig_min,
        cfg.models.covariance_ridge,
        floor,
        floor_extra,
        dt_eig,
        alpha_base,
        gl_max_iter,
        gl_tol,
    )

    last_err: Exception | None = None
    for mult in (1.0, 4.0, 16.0, 64.0, 256.0):
        alpha = float(alpha_base) * mult
        alpha_eff = max(alpha, floor * 2.0 + 1e-8)
        t_gl = time.perf_counter()
        try:
            cov_out, prec, n_iter = graphical_lasso(
                emp_cov,
                alpha=alpha_eff,
                max_iter=gl_max_iter,
                tol=gl_tol,
                mode="lars",
                return_n_iter=True,
            )
            dt_gl = time.perf_counter() - t_gl
            n_iter_v = int(n_iter) if np.ndim(n_iter) == 0 else int(np.asarray(n_iter).max())
            tr_cov = float(np.trace(cov_out))
            tr_prec = float(np.trace(prec))
            fro_prec = float(np.linalg.norm(prec, ord="fro"))
            logger.info(
                "%sgraphical_lasso OK mult=%.4g alpha_req=%.6e alpha_eff=%.6e "
                "n_iter=%d wall=%.3fs tr(cov_out)=%.4e tr(precision)=%.4e ||precision||_F=%.4e",
                tag,
                mult,
                alpha,
                alpha_eff,
                n_iter_v,
                dt_gl,
                tr_cov,
                tr_prec,
                fro_prec,
            )
            logger.info(
                "%sgraphical_lasso_from_covariance done total=%.3fs",
                tag,
                time.perf_counter() - t_all,
            )
            return prec
        except Exception as exc:
            last_err = exc
            logger.warning(
                "%sgraphical_lasso FAILED mult=%.4g alpha_eff=%.6e wall=%.3fs: %s",
                tag,
                mult,
                alpha_eff,
                time.perf_counter() - t_gl,
                exc,
            )

    if last_err is not None:
        logger.error(
            "%sgraphical_lasso_from_covariance exhausted retries after %.3fs",
            tag,
            time.perf_counter() - t_all,
        )
        raise last_err
    raise RuntimeError("graphical_lasso_from_covariance failed")


def precision_to_binary_adj(theta: np.ndarray, tol: float) -> np.ndarray:
    """Undirected edges where off-diagonal |precision| > tol."""
    p = theta.shape[0]
    adj = np.abs(theta) > tol
    adj = adj.astype(np.int8)
    np.fill_diagonal(adj, 0)
    return adj

