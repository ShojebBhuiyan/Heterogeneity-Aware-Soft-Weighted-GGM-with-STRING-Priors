"""Overlap between inferred networks and STRING edges."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def adjacency_to_pairs(adj: np.ndarray, gene_names: np.ndarray) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    gn = np.asarray(gene_names)
    tri = np.triu(adj.astype(bool), k=1)
    for i, j in zip(*np.where(tri)):
        a, b = str(gn[i]), str(gn[j])
        pairs.add(tuple(sorted((a, b))))
    return pairs


def string_to_pairs(
    edges: pd.DataFrame,
    gene_set: set[str],
    thr: float,
) -> set[tuple[str, str]]:
    if edges is None or edges.empty:
        return set()
    out: set[tuple[str, str]] = set()
    for row in edges.itertuples(index=False):
        s = float(row.score)
        if s < thr:
            continue
        a, b = str(row.preferredName_A), str(row.preferredName_B)
        if a not in gene_set or b not in gene_set:
            continue
        out.add(tuple(sorted((a, b))))
    return out


def precision_recall(pred: Iterable[tuple[str, str]], truth: set[tuple[str, str]]) -> tuple[float, float]:
    pred_s = set(pred)
    if not pred_s:
        return 0.0, 0.0
    tp = len(pred_s & truth)
    prec = tp / len(pred_s)
    rec = tp / len(truth) if truth else 0.0
    return float(prec), float(rec)

