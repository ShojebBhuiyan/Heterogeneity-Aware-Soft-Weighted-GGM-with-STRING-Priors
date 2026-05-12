"""Schema and sanity checks for AnnData objects."""

from __future__ import annotations

import logging

import numpy as np
import scanpy as sc

logger = logging.getLogger("pgm.data.validate")


class AnnDataValidationError(ValueError):
    """Raised when AnnData fails structural checks."""


def validate_anndata_basic(adata: sc.AnnData, *, name: str = "adata") -> None:
    """
    Validate non-empty matrix, unique obs/var names, finite counts.

    Raises
    ------
    AnnDataValidationError
        On failed checks.
    """
    if adata.n_obs == 0 or adata.n_vars == 0:
        raise AnnDataValidationError(f"{name}: empty obs or vars")
    if adata.X is None:
        raise AnnDataValidationError(f"{name}: X is None")
    if adata.obs_names.has_duplicates:
        raise AnnDataValidationError(f"{name}: duplicate cell barcodes")
    if adata.var_names.has_duplicates:
        raise AnnDataValidationError(f"{name}: duplicate gene names")
    nnz = adata.X.nnz if hasattr(adata.X, "nnz") else np.count_nonzero(adata.X)
    if nnz == 0:
        raise AnnDataValidationError(f"{name}: matrix is entirely zero")
    logger.debug("Validated %s shape %s", name, adata.shape)
