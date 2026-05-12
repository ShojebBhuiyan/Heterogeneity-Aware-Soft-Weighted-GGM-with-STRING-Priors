"""Sparsity / degree summaries."""

from __future__ import annotations

import numpy as np


def degree_stats(adj: np.ndarray) -> dict[str, float]:
    """Undirected degree distribution on binary adjacency."""
    a = (np.triu(adj, k=1) > 0).astype(np.int32)
    deg = a.sum(axis=0) + a.sum(axis=1)
    return {
        "mean_degree": float(deg.mean()),
        "max_degree": float(deg.max()),
        "sparsity_upper": float(1.0 - a.sum() / max(a.shape[0] * (a.shape[0] - 1) / 2, 1)),
    }

