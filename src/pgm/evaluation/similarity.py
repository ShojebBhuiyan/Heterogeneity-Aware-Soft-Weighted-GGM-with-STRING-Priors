"""Graph similarity utilities."""

from __future__ import annotations

import numpy as np


def jaccard_triu(a: np.ndarray, b: np.ndarray) -> float:
    """Jaccard index on strict upper-triangular binary masks."""
    ua = np.triu(a, k=1) > 0
    ub = np.triu(b, k=1) > 0
    inter = np.logical_and(ua, ub).sum()
    union = np.logical_or(ua, ub).sum()
    return float(inter / union) if union > 0 else 0.0


def frobenius_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b, ord="fro"))

