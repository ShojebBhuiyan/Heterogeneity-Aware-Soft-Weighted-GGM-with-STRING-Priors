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
