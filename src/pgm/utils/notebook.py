"""Notebook helpers: smoke detection and checkpoint skip patterns."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from pgm.config.loader import load_config
from pgm.config.schemas import ProjectConfig

from .checkpoint import checkpoint_exists, load_checkpoint, save_checkpoint


def is_smoke(cfg: ProjectConfig) -> bool:
    return bool(cfg.smoke_mode.enabled)


def load_notebook_config() -> ProjectConfig:
    """Convention: ``SMOKE=1`` enables smoke overlays."""
    return load_config()


@contextmanager
def skip_compute_if_ckpt(cfg: ProjectConfig, name: str) -> Generator[bool, None, None]:
    """
    Yield ``True`` when a checkpoint exists (compute should be skipped).

    Example
    -------
    >>> with skip_compute_if_ckpt(cfg, "eda_profile") as skip:
    ...     if skip:
    ...         pass
    ...     else:
    ...         run_expensive_profiling()
    """
    yield checkpoint_exists(cfg, name)


def load_or_compute(
    cfg: ProjectConfig,
    name: str,
    compute_fn: Callable[[], Any],
) -> Any:
    """Return checkpoint content if present, else compute, save, return."""
    if checkpoint_exists(cfg, name):
        return load_checkpoint(cfg, name)
    out = compute_fn()
    save_checkpoint(cfg, name, out)
    return out


def smoke_demo_writes_temp() -> Path:
    """Tiny demo for Phase 1: write a tempfile and return path."""
    fd, spath = tempfile.mkstemp(prefix="pgm_smoke_", suffix=".txt")
    os.close(fd)
    pth = Path(spath)
    pth.write_text("ok", encoding="utf-8")
    return pth
