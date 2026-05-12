"""Dataset summarization helpers."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import scanpy as sc

logger = logging.getLogger("pgm.data.summary")


def sparsity_fraction(adata: sc.AnnData) -> float:
    """Fraction of matrix entries equal to zero."""
    X = adata.X
    n = adata.n_obs * adata.n_vars
    if hasattr(X, "nnz"):
        return 1.0 - (X.nnz / max(n, 1))
    return float(np.mean(X == 0))


def missing_value_fraction(adata: sc.AnnData) -> float:
    """Fraction of entries that are NaN (usually zero for count matrices)."""
    X = adata.X
    if hasattr(X, "data"):
        data = X.data
    else:
        data = np.asarray(X).ravel()
    if data.size == 0:
        return 0.0
    return float(np.isnan(data).mean())


def dataset_summary(adata: sc.AnnData) -> dict[str, Any]:
    """Return a JSON-serializable summary dict."""
    summ = {
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "sparsity": sparsity_fraction(adata),
        "missing_fraction": missing_value_fraction(adata),
    }
    logger.info("Summary: %s", summ)
    return summ


def summary_to_markdown(summary: dict[str, Any], title: str = "Dataset summary") -> str:
    """Render summary as markdown table."""
    df = pd.DataFrame([summary])
    return f"## {title}\n\n" + df.to_markdown(index=False)
