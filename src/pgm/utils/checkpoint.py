"""Save and load checkpoints (AnnData, DataFrame, joblib-able, pickle)."""

from __future__ import annotations

import logging
import pickle
import re
from pathlib import Path
from typing import Any

import joblib

from pgm.config.schemas import ProjectConfig

from .paths import checkpoints_dir as ckpt_ns
from .paths import ensure_parents

logger = logging.getLogger("pgm.checkpoint")

_SAFE = re.compile(r"[^a-zA-Z0-9_.-]+")


def _sanitize(name: str) -> str:
    s = name.strip().replace(" ", "_")
    s = _SAFE.sub("-", s)
    return s or "artifact"


def _checkpoint_subdir(cfg: ProjectConfig) -> Path:
    """Same resolved dir as ``pgm.utils.paths.checkpoints_dir`` (subdir-aware)."""
    return checkpoints_dir(cfg)


def _base_path(cfg: ProjectConfig, name: str, ext: str) -> Path:
    sub = _checkpoint_subdir(cfg)
    sub.mkdir(parents=True, exist_ok=True)
    return sub / f"{_sanitize(name)}{ext}"


def checkpoints_dir(cfg: ProjectConfig) -> Path:
    return ckpt_ns(cfg)


def artifact_path_candidates(cfg: ProjectConfig, name: str) -> list[Path]:
    """Candidate paths used to detect existence (order preserved)."""
    sub = _checkpoint_subdir(cfg)
    base = _sanitize(name)
    return [
        sub / f"{base}.h5ad",
        sub / f"{base}.joblib",
        sub / f"{base}.pq",
        sub / f"{base}.pickle",
    ]


def checkpoint_exists(cfg: ProjectConfig, name: str) -> bool:
    return any(p.exists() for p in artifact_path_candidates(cfg, name))


def save_checkpoint(cfg: ProjectConfig, name: str, obj: Any) -> Path:
    """
    Persist an object to checkpoint storage.

    AnnData → ``.h5ad``, ``pandas.DataFrame`` → parquet ``.pq``,
    otherwise ``joblib``. Pickle fallback on failure after warning.
    """
    from importlib.util import find_spec

    path: Path | None = None

    typename = type(obj).__name__

    try:
        if find_spec("anndata"):
            import anndata as ad

            if isinstance(obj, ad.AnnData):
                path = _base_path(cfg, name, ".h5ad")
                ensure_parents(path)
                obj.write_h5ad(path)
                logger.info("Saved AnnData checkpoint %s (%s)", path, typename)
                return path
    except Exception as e:
        logger.warning("AnnData checkpoint failed (%s); trying next backend", e)

    try:
        import pandas as pd

        if isinstance(obj, pd.DataFrame):
            path = _base_path(cfg, name, ".pq")
            ensure_parents(path)
            obj.to_parquet(path)
            logger.info("Saved DataFrame parquet %s", path)
            return path
    except Exception as e:
        logger.warning("Parquet checkpoint failed (%s)", e)

    path = _base_path(cfg, name, ".joblib")
    ensure_parents(path)
    try:
        joblib.dump(obj, path)
        logger.info("Saved joblib checkpoint %s", path)
        return path
    except Exception as e:
        logger.warning("joblib dump failed (%s); using pickle", e)

    path = _base_path(cfg, name, ".pickle")
    ensure_parents(path)
    with path.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Saved pickle checkpoint %s", path)
    return path


def load_checkpoint(cfg: ProjectConfig, name: str) -> Any:
    """Load artifact if any checkpoint file exists for ``name``."""
    for cand in artifact_path_candidates(cfg, name):
        if not cand.exists():
            continue
        suf = cand.suffix.lower()
        if suf == ".h5ad":
            import scanpy as sc

            return sc.read_h5ad(cand)
        if suf == ".pq":
            import pandas as pd

            return pd.read_parquet(cand)
        if suf == ".joblib":
            return joblib.load(cand)
        if suf == ".pickle":
            with cand.open("rb") as f:
                return pickle.load(f)
    raise FileNotFoundError(f"No checkpoint named {name!r} found")


def optional_load(cfg: ProjectConfig, name: str, default: Any = None) -> Any:
    try:
        return load_checkpoint(cfg, name)
    except FileNotFoundError:
        return default
