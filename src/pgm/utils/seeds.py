"""Reproducible random seeds."""

from __future__ import annotations

import os
import random

import numpy as np

from pgm.config.schemas import ProjectConfig


def set_global_seed(cfg: ProjectConfig | int) -> int:
    """
    Seed Python, NumPy (and optionally scanpy) from config or bare int.

    Returns
    -------
    int
        The seed applied.
    """
    seed = cfg.run.random_seed if isinstance(cfg, ProjectConfig) else int(cfg)
    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed % (2**32)))
    try:
        import scanpy as sc

        sc.settings.seed = seed
    except ImportError:
        pass
    return seed
