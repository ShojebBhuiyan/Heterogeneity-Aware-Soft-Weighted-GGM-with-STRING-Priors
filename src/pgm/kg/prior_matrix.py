"""Convert STRING edge tables to graph objects and prior matrices."""

from __future__ import annotations

import logging

import networkx as nx
import numpy as np
import pandas as pd

logger = logging.getLogger("pgm.kg.prior")


def edges_to_sparse_prior(
    edges: pd.DataFrame,
    gene_order: list[str],
    threshold: float,
) -> np.ndarray:
    """
    Symmetric prior matrix ``P`` with ``P_ij = max(score)`` for edges above ``threshold``.
    Diagonal zero.
    """
    n = len(gene_order)
    idx = {g: i for i, g in enumerate(gene_order)}
    p = np.zeros((n, n), dtype=np.float64)
    if edges is None or edges.empty:
        return p
    need = {"preferredName_A", "preferredName_B", "score"}
    if not need.issubset(set(edges.columns)):
        raise ValueError(f"edges need columns {need}, got {list(edges.columns)}")
    for row in edges.itertuples(index=False):
        a, b, s = row.preferredName_A, row.preferredName_B, float(row.score)
        if s < threshold:
            continue
        ia, ib = idx.get(a), idx.get(b)
        if ia is None or ib is None or ia == ib:
            continue
        v = max(p[ia, ib], s)
        p[ia, ib] = v
        p[ib, ia] = v
    logger.debug("Prior nnz=%d", np.count_nonzero(p))
    return p


def prior_to_networkx(prior: np.ndarray, gene_names: np.ndarray | list[str]) -> nx.Graph:
    g = nx.Graph()
    gn = np.asarray(gene_names)
    thresh = np.finfo(np.float64).eps
    nz = zip(*np.where(np.triu(prior, k=1) > thresh))
    for i, j in nz:
        g.add_edge(str(gn[i]), str(gn[j]), weight=float(prior[i, j]))
    return g

