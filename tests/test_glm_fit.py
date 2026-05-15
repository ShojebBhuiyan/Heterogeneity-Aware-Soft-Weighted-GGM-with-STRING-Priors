"""Regression tests for graphical lasso sample fitting."""

from __future__ import annotations

import numpy as np

from pgm.config.schemas import ProjectConfig
from pgm.models.glm_utils import fit_graphical_lasso_from_samples


def _near_collinear_X(rng: np.random.Generator, n: int, p: int) -> np.ndarray:
    X = rng.standard_normal((n, p), dtype=np.float64)
    for j in range(1, min(6, p)):
        X[:, j] = X[:, 0] + 1e-6 * rng.standard_normal(n)
    return X


def test_fit_graphical_lasso_from_samples_heuristic_alpha():
    """Ill-conditioned X + unset gl_alpha should return finite precision (covariance path)."""
    cfg = ProjectConfig()
    cfg.models.gl_alpha = None
    rng = np.random.default_rng(42)
    X = _near_collinear_X(rng, n=80, p=25)
    model, theta = fit_graphical_lasso_from_samples(X, cfg)
    assert model is None
    assert theta.shape == (25, 25)
    assert np.isfinite(theta).all()


def test_fit_graphical_lasso_from_samples_fixed_alpha():
    cfg = ProjectConfig()
    cfg.models.gl_alpha = 0.25
    rng = np.random.default_rng(0)
    X = _near_collinear_X(rng, n=100, p=18)
    model, theta = fit_graphical_lasso_from_samples(X, cfg)
    assert model is None
    assert theta.shape == (18, 18)
    assert np.isfinite(theta).all()


def test_graphical_lasso_from_covariance_ess_alpha_floor():
    """Low ESS should raise alpha floor so GLasso does not use a too-small alpha."""
    from sklearn.covariance import empirical_covariance

    from pgm.models.glm_utils import graphical_lasso_from_covariance

    cfg = ProjectConfig()
    cfg.models.gl_alpha = 0.02
    cfg.models.gl_soft_ess_min_alpha_scale = 0.05
    rng = np.random.default_rng(2)
    n, p = 120, 35
    X = rng.standard_normal((n, p))
    emp = empirical_covariance(X, assume_centered=False)
    theta = graphical_lasso_from_covariance(emp, cfg, effective_n=5.0)
    assert theta.shape == (p, p)
    assert np.isfinite(theta).all()


def test_graphical_lasso_from_covariance_gglasso_block_sgl():
    """gglasso ADMM block path returns finite precision on near-collinear data."""
    from sklearn.covariance import empirical_covariance

    from pgm.models.glm_utils import graphical_lasso_from_covariance

    rng = np.random.default_rng(11)
    n, p = 120, 28
    X = rng.standard_normal((n, p))
    X[:, 1:] = X[:, :1] + 1e-4 * rng.standard_normal((n, p - 1))
    emp = empirical_covariance(X, assume_centered=False)

    cfg = ProjectConfig()
    cfg.models.gl_backend = "gglasso"
    cfg.models.gglasso_solver = "block_sgl"
    cfg.models.gl_alpha = 0.35

    theta = graphical_lasso_from_covariance(emp, cfg)
    assert theta.shape == (p, p)
    assert np.isfinite(theta).all()


def test_graphical_lasso_from_covariance_gglasso_admm_sgl():
    from sklearn.covariance import empirical_covariance

    from pgm.models.glm_utils import graphical_lasso_from_covariance

    rng = np.random.default_rng(13)
    n, p = 100, 15
    X = rng.standard_normal((n, p))
    emp = empirical_covariance(X, assume_centered=False)

    cfg = ProjectConfig()
    cfg.models.gl_backend = "gglasso"
    cfg.models.gglasso_solver = "admm_sgl"
    cfg.models.gl_alpha = 0.25

    theta = graphical_lasso_from_covariance(emp, cfg)
    assert theta.shape == (p, p)
    assert np.isfinite(theta).all()
