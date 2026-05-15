"""Lightweight AnnData / matrix summaries for notebooks and debugging."""

from __future__ import annotations

from typing import Any

import numpy as np
import scanpy as sc
from scipy import sparse


def summarize_adata_matrix(adata: sc.AnnData) -> dict[str, Any]:
    """
    Count NaN / Inf and summarize ``adata.X`` (cells × genes; dense or sparse).

    Useful after scaling and before GLasso to spot numerical issues.
    """
    X = adata.X
    if sparse.issparse(X):
        X = X.data
    else:
        X = np.asarray(X)
    finite = np.isfinite(X)
    out: dict[str, Any] = {
        "shape": (adata.n_obs, adata.n_vars),
        "n_nan": int(np.isnan(X).sum()),
        "n_pos_inf": int(np.isposinf(X).sum()),
        "n_neg_inf": int(np.isneginf(X).sum()),
        "n_nonfinite": int((~finite).sum()),
        "min": float(np.nanmin(X)) if X.size else float("nan"),
        "max": float(np.nanmax(X)) if X.size else float("nan"),
        "mean": float(np.nanmean(X)) if X.size else float("nan"),
    }
    if adata.n_vars > 0 and not sparse.issparse(adata.X):
        xd = np.asarray(adata.X)
        std = xd.std(axis=0, ddof=1)
        out["n_genes_zero_std"] = int(np.sum(np.isfinite(std) & (std <= 1e-12)))
        out["n_genes_nonfinite_std"] = int(np.sum(~np.isfinite(std)))
    return out


def format_matrix_summary(d: dict[str, Any]) -> str:
    lines = [
        f"X shape: {d['shape'][0]} × {d['shape'][1]}",
        f"non-finite values: {d['n_nonfinite']} (nan={d['n_nan']}, +inf={d['n_pos_inf']}, -inf={d['n_neg_inf']})",
        f"finite min/mean/max: {d['min']:.6g} / {d['mean']:.6g} / {d['max']:.6g}",
    ]
    if "n_genes_zero_std" in d:
        lines.append(
            f"genes with ~zero std (ddof=1): {d['n_genes_zero_std']}; "
            f"genes with non-finite std: {d['n_genes_nonfinite_std']}"
        )
    return "\n".join(lines)
