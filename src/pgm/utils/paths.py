"""Resolve standard directories from project config."""

from __future__ import annotations

from pathlib import Path

from pgm.config.schemas import ProjectConfig


def checkpoints_dir(cfg: ProjectConfig) -> Path:
    """Resolved checkpoint directory, including optional ``checkpoint.subdirectory``."""
    base = cfg.resolve(cfg.data.checkpoint_dir)
    extra = (cfg.checkpoint.subdirectory or "").strip()
    return base / extra if extra else base


def interim_dir(cfg: ProjectConfig) -> Path:
    return cfg.resolve(cfg.data.interim_dir)


def processed_dir(cfg: ProjectConfig) -> Path:
    return cfg.resolve(cfg.data.processed_dir)


def processed_h5ad_path(cfg: ProjectConfig) -> Path:
    return cfg.resolve(cfg.data.processed_dir) / cfg.preprocessing.output_filename


def figures_dir(cfg: ProjectConfig) -> Path:
    return cfg.resolve(cfg.reports.figures_dir)


def eda_reports_dir(cfg: ProjectConfig) -> Path:
    return cfg.resolve(cfg.reports.eda_dir)


def results_reports_dir(cfg: ProjectConfig) -> Path:
    return cfg.resolve(cfg.reports.results_dir)


def tenx_mtx_path(cfg: ProjectConfig) -> Path:
    return cfg.resolve(cfg.data.tenx_mtx_dir)


def string_cache_dir(cfg: ProjectConfig) -> Path:
    return cfg.resolve(cfg.data.string_cache_dir)


def ensure_parents(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
