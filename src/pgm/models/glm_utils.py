"""Shared helpers for inverse-covariance (graphical lasso) fits."""

from __future__ import annotations

import contextlib
import io
import logging
import time
from typing import Any

import numpy as np
from sklearn.covariance import empirical_covariance, graphical_lasso

from pgm.config.schemas import ProjectConfig

logger = logging.getLogger("pgm.models.glm")


def _glm_log_prefix(log_label: str | None) -> str:
    return f"[{log_label}] " if log_label else ""


def _covariance_to_correlation(cov: np.ndarray) -> np.ndarray:
    """SPD-safe correlation matrix from a covariance (unit diagonal)."""
    cov = np.asarray(cov, dtype=np.float64)
    d = np.sqrt(np.clip(np.diag(cov), 1e-15, None))
    inv = 1.0 / d
    r = inv[:, None] * cov * inv[None, :]
    r = 0.5 * (r + r.T)
    np.fill_diagonal(r, 1.0)
    np.clip(r, -1.0, 1.0, out=r)
    np.fill_diagonal(r, 1.0)
    return r


def _eigenvalues_floor_spd(mat: np.ndarray, p: int, tag: str, *, rel_floor: float) -> np.ndarray:
    """Raise tiny eigenvalues to ``rel_floor * tr(A)/p`` for solver stability."""
    mat = np.asarray(mat, dtype=np.float64)
    mat = 0.5 * (mat + mat.T)
    evals, evecs = np.linalg.eigh(mat)
    avg_ev = float(np.trace(mat)) / max(p, 1)
    ev_floor = max(float(np.finfo(np.float64).eps) * 1e3, avg_ev * rel_floor)
    emin = float(evals.min())
    if emin < ev_floor:
        logger.info(
            "%seigenvalue_floor rel=%.2e lowest=%.4e → %.4e (tr/n=%.4e)",
            tag,
            rel_floor,
            emin,
            ev_floor,
            avg_ev,
        )
        evals = np.maximum(evals, ev_floor)
        mat = (evecs * evals) @ evecs.T
        mat = 0.5 * (mat + mat.T)
    return mat


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
    log_label: str | None = None,
) -> np.ndarray:
    """
    Weighted scatter matrix ``Σ w_i (x_i-μ)(x_i-μ)ᵀ`` with ``w ← w/sum(w)``.

    Regularization (ridge / shrinkage) is applied later in
    ``stabilize_empirical_covariance`` before graphical lasso.
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
    sigma = z.T @ z
    dt = time.perf_counter() - t0
    logger.info(
        "%sweighted_scatter_cov n=%d p=%d weights[min,max,sum_raw]=[%.4e,%.4e,%.4e] %.3fs",
        tag,
        X.shape[0],
        X.shape[1],
        w_raw_min,
        w_raw_max,
        s,
        dt,
    )
    return sigma


def _solve_sklearn_graphical_lasso(
    emp_mat: np.ndarray,
    *,
    alpha_eff: float,
    gl_max_iter: int,
    tol_gl: float,
    mode_try: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    cov_out, prec, n_iter = graphical_lasso(
        emp_mat,
        alpha=alpha_eff,
        max_iter=gl_max_iter,
        tol=tol_gl,
        mode=mode_try,
        return_n_iter=True,
    )
    n_iter_v = int(n_iter) if np.ndim(n_iter) == 0 else int(np.asarray(n_iter).max())
    return cov_out, prec, n_iter_v


def _precision_from_gglasso_sol(sol: dict[str, np.ndarray]) -> np.ndarray:
    """Symmetric sparse-precision estimate from a ``gglasso`` ADMM solution dict."""
    if "Theta" in sol:
        t = np.asarray(sol["Theta"], dtype=np.float64)
    else:
        t = np.asarray(sol["Omega"], dtype=np.float64)
    t = 0.5 * (t + t.T)
    if not np.isfinite(t).all():
        raise FloatingPointError("gglasso returned non-finite precision estimate")
    return t


def _solve_gglasso_sgl(
    emp_mat: np.ndarray,
    *,
    alpha_eff: float,
    cfg: ProjectConfig,
    gl_max_iter: int,
    tol_gl: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Single graphical lasso via ``gglasso`` ADMM (optionally block-wise)."""
    from gglasso.solver.single_admm_solver import ADMM_SGL, block_SGL

    emp_mat = np.asarray(emp_mat, dtype=np.float64)
    p = emp_mat.shape[0]
    omega_0 = np.eye(p, dtype=np.float64)
    rtol = float(cfg.models.gglasso_rtol)
    rho = float(cfg.models.gglasso_rho)
    upd = bool(cfg.models.gglasso_update_rho)
    lam = float(alpha_eff)
    if lam <= 0.0:
        raise ValueError("gglasso lambda1 (alpha_eff) must be positive")

    kwargs: dict[str, Any] = dict(
        rho=rho,
        max_iter=int(gl_max_iter),
        tol=float(tol_gl),
        rtol=rtol,
        update_rho=upd,
        verbose=False,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        if cfg.models.gglasso_solver == "block_sgl":
            sol = block_SGL(emp_mat, lam, omega_0, **kwargs)
            prec = _precision_from_gglasso_sol(sol)
            return emp_mat, prec, -1
        sol, _info = ADMM_SGL(emp_mat, lam, omega_0, latent=False, **kwargs)
        prec = _precision_from_gglasso_sol(sol)
        return emp_mat, prec, -1


def stabilize_empirical_covariance(
    emp_cov: np.ndarray,
    cfg: ProjectConfig,
    *,
    log_label: str | None = None,
) -> tuple[np.ndarray, float]:
    """
    Symmetrize, sanitize, optional shrinkage toward ``(tr/p)·I``, then SPD ridge.

    Returns stabilized matrix and total diagonal ``floor`` applied (for α lower bound).
    """
    tag = _glm_log_prefix(log_label)
    t0 = time.perf_counter()
    emp = np.asarray(emp_cov, dtype=np.float64)
    emp = 0.5 * (emp + emp.T)
    np.nan_to_num(emp, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    p = emp.shape[0]
    trace_0 = float(np.trace(emp))

    s_sh = float(cfg.models.covariance_shrinkage)
    if s_sh > 0.0:
        mu_id = trace_0 / p if p > 0 else 0.0
        emp = (1.0 - s_sh) * emp + s_sh * mu_id * np.eye(p, dtype=np.float64)
        logger.info(
            "%scovariance_target_shrink s=%.4f tr_before=%.4e mu_tr_over_p=%.4e",
            tag,
            s_sh,
            trace_0,
            mu_id,
        )

    t_eig = time.perf_counter()
    eig_min = float(np.linalg.eigvalsh(emp).min())
    dt_eig = time.perf_counter() - t_eig
    floor_base = float(cfg.models.covariance_ridge)
    floor = floor_base
    floor_extra = 0.0
    if eig_min <= 0:
        floor_extra = max(-2.0 * eig_min, 1e-8)
        floor += floor_extra
    emp = emp + floor * np.eye(p, dtype=np.float64)
    # Shrink extreme condition numbers so sklearn GL CD/LARS does not hit non-SPD mid-solve.
    emp = _eigenvalues_floor_spd(emp, p, tag, rel_floor=1e-3)
    logger.info(
        "%sstabilize_cov p=%d eig_min_pre_floor=%.4e ridge_cfg=%.4e floor_applied=%.4e "
        "(extra %.4e) eigvalsh=%.3fs total=%.3fs tr_after_floor=%.4e",
        tag,
        p,
        eig_min,
        floor_base,
        floor,
        floor_extra,
        dt_eig,
        time.perf_counter() - t0,
        float(np.trace(emp)),
    )
    return emp, float(floor)


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
    effective_n: float | None = None,
) -> np.ndarray:
    """
    Run graphical lasso on an empirical covariance (stabilized to SPD).

    Backend is ``cfg.models.gl_backend``: ``sklearn`` uses Friedman
    coordinate descent / LARS; ``gglasso`` uses ADMM single graphical lasso
    (``block_SGL`` or ``ADMM_SGL``).

    Prefer ``cfg.models.gl_alpha``; otherwise uses a heuristic default.

    If ``effective_n`` is set (e.g. Kish ESS for mixture weights) and
    ``cfg.models.gl_soft_ess_min_alpha_scale > 0``, the solved α is at least
    ``scale · p / max(effective_n, 1)`` to avoid Graphical Lasso when ``p`` is
    much larger than the effective sample size.

    If sklearn coordinate descent fails with a non-SPD / ill-conditioned error
    after all α retries, the fit retries on a **correlation** matrix, then adds
    extra diagonal loading and alternates ``mode='lars'`` vs ``'cd'``, and
    finally applies a stronger eigenvalue floor before the same attempts. The
    same recovery phases apply when the **gglasso** backend raises.
    """
    tag = _glm_log_prefix(log_label)
    t_all = time.perf_counter()
    emp_cov_stab, floor_applied = stabilize_empirical_covariance(
        emp_cov, cfg, log_label=log_label
    )
    p = emp_cov_stab.shape[0]

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
    gl_mode = cfg.models.gl_mode
    multipliers = list(cfg.models.gl_alpha_retry_multipliers)
    warn_s = cfg.models.gl_attempt_warn_seconds

    ess_scale = float(cfg.models.gl_soft_ess_min_alpha_scale)
    ess_note = ""
    if (
        effective_n is not None
        and ess_scale > 0.0
        and np.isfinite(float(effective_n))
        and float(effective_n) > 0.0
    ):
        en = float(effective_n)
        alpha_ess_floor = ess_scale * float(p) / max(en, 1.0)
        ess_note = f" effective_n={en:.4g} ess_alpha_floor≥{alpha_ess_floor:.4e}"

    backend = cfg.models.gl_backend
    gglasso_note = (
        f" gglasso_solver={cfg.models.gglasso_solver!r}"
        if backend == "gglasso"
        else ""
    )

    logger.info(
        "%sgraphical_lasso_from_covariance begin p=%d backend=%s mode=%s%s alpha_base=%s "
        "retry_mults=%s max_iter=%d tol=%.2e%s",
        tag,
        p,
        backend,
        gl_mode,
        gglasso_note,
        alpha_base,
        multipliers,
        gl_max_iter,
        gl_tol,
        ess_note,
    )

    last_err: Exception | None = None
    alt_mode = "lars" if gl_mode == "cd" else "cd"
    modes_both: tuple[str, ...] = (gl_mode, alt_mode) if alt_mode != gl_mode else (gl_mode,)

    def run_gl_attempts(
        emp_mat: np.ndarray,
        modes: tuple[str, ...],
        phase: str,
        *,
        tol_use: float | None = None,
    ) -> np.ndarray | None:
        nonlocal last_err
        tol_gl = float(gl_tol if tol_use is None else tol_use)
        modes_eff: tuple[str, ...] = modes if backend == "sklearn" else ("n/a",)
        for mode_try in modes_eff:
            for mult in multipliers:
                alpha_req = float(alpha_base) * float(mult)
                alpha_eff = max(alpha_req, floor_applied * 2.0 + 1e-8)
                if (
                    effective_n is not None
                    and ess_scale > 0.0
                    and np.isfinite(float(effective_n))
                    and float(effective_n) > 0.0
                ):
                    en = float(effective_n)
                    alpha_ess_floor = ess_scale * float(p) / max(en, 1.0)
                    if alpha_ess_floor > alpha_eff:
                        logger.info(
                            "%sgraphical_lasso ess_alpha_floor p/max(ess,1)=%.4g scale=%.4g "
                            "→ alpha_eff raised from %.4e to %.4e",
                            tag,
                            float(p) / max(en, 1.0),
                            ess_scale,
                            alpha_eff,
                            alpha_ess_floor,
                        )
                    alpha_eff = max(alpha_eff, alpha_ess_floor)
                t_gl = time.perf_counter()
                mode_disp = mode_try if backend == "sklearn" else "n/a"
                logger.info(
                    "%sgraphical_lasso attempt (%s) mult=%.4g alpha_req=%.6e alpha_eff=%.6e mode=%s",
                    tag,
                    phase,
                    mult,
                    alpha_req,
                    alpha_eff,
                    mode_disp,
                )
                try:
                    if backend == "sklearn":
                        cov_out, prec, n_iter_v = _solve_sklearn_graphical_lasso(
                            emp_mat,
                            alpha_eff=alpha_eff,
                            gl_max_iter=gl_max_iter,
                            tol_gl=tol_gl,
                            mode_try=mode_try,
                        )
                    else:
                        cov_out, prec, n_iter_v = _solve_gglasso_sgl(
                            emp_mat,
                            alpha_eff=alpha_eff,
                            cfg=cfg,
                            gl_max_iter=gl_max_iter,
                            tol_gl=tol_gl,
                        )
                    dt_gl = time.perf_counter() - t_gl
                    if warn_s is not None and dt_gl > float(warn_s):
                        logger.warning(
                            "%sgraphical_lasso SLOW attempt mult=%.4g wall=%.3fs (threshold %.1fs)",
                            tag,
                            mult,
                            dt_gl,
                            warn_s,
                        )
                    tr_cov = float(np.trace(cov_out))
                    tr_prec = float(np.trace(prec))
                    fro_prec = float(np.linalg.norm(prec, ord="fro"))
                    n_iter_disp = str(n_iter_v) if n_iter_v >= 0 else "n/a"
                    logger.info(
                        "%sgraphical_lasso OK (%s) mult=%.4g alpha_req=%.6e alpha_eff=%.6e "
                        "n_iter=%s wall=%.3fs tr(cov_or_input)=%.4e tr(precision)=%.4e "
                        "||precision||_F=%.4e",
                        tag,
                        phase,
                        mult,
                        alpha_req,
                        alpha_eff,
                        n_iter_disp,
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
                    dt_gl = time.perf_counter() - t_gl
                    if warn_s is not None and dt_gl > float(warn_s):
                        logger.warning(
                            "%sgraphical_lasso FAILED SLOW mult=%.4g wall=%.3fs (threshold %.1fs)",
                            tag,
                            mult,
                            dt_gl,
                            warn_s,
                        )
                    logger.warning(
                        "%sgraphical_lasso FAILED (%s) mult=%.4g alpha_eff=%.6e wall=%.3fs: %s",
                        tag,
                        phase,
                        mult,
                        alpha_eff,
                        dt_gl,
                        exc,
                    )
        return None

    prec = run_gl_attempts(emp_cov_stab, (gl_mode,), "primary")
    if prec is not None:
        return prec

    trace_scale = float(np.trace(emp_cov_stab)) / max(p, 1)
    f_floor = float(floor_applied)
    # Same Markov structure as Σ under diagonal scaling; often numerically easier for GL.
    r_mat = _covariance_to_correlation(emp_cov_stab)
    r_mat = _eigenvalues_floor_spd(r_mat, p, tag, rel_floor=5e-4)
    jitter = max(1e-8, f_floor, trace_scale * 1e-6)
    logger.warning(
        "%sgraphical_lasso retry on correlation parameterization (jitter_diag=%.4e)",
        tag,
        jitter,
    )
    prec = run_gl_attempts(
        r_mat + jitter * np.eye(p, dtype=np.float64),
        modes_both,
        "correlation",
    )
    if prec is not None:
        return prec

    bumps = [
        max(trace_scale * 1e-2, f_floor * 20.0),
        max(trace_scale * 0.08, f_floor * 100.0),
        max(trace_scale * 0.25, f_floor * 400.0),
        max(trace_scale * 0.6, f_floor * 2000.0),
    ]
    for bump in bumps:
        logger.warning(
            "%sgraphical_lasso non-SPD / ill-conditioned; retry with +diag=%.4e (tr/n=%.4e)",
            tag,
            bump,
            trace_scale,
        )
        emp_bumped = emp_cov_stab + float(bump) * np.eye(p, dtype=np.float64)
        prec = run_gl_attempts(
            emp_bumped,
            modes_both,
            f"recovery_diag={bump:.4e}",
            tol_use=max(gl_tol, 5e-3),
        )
        if prec is not None:
            return prec

    logger.error(
        "%sgraphical_lasso last resort: heavy eigenvalue floor + large diagonal + loose tol",
        tag,
    )
    emp_lr = _eigenvalues_floor_spd(emp_cov_stab, p, tag, rel_floor=0.04)
    emp_lr = emp_lr + max(trace_scale * 0.5, f_floor * 5000.0) * np.eye(p, dtype=np.float64)
    emp_lr = _eigenvalues_floor_spd(emp_lr, p, tag, rel_floor=1e-3)
    prec = run_gl_attempts(
        emp_lr,
        modes_both,
        "last_resort_spd",
        tol_use=max(gl_tol, 0.02),
    )
    if prec is not None:
        return prec

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

