"""Logging setup with Rich console and rotating file."""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler

from rich.logging import RichHandler

from pgm.utils.paths import ensure_parents

_LOGGER_CONFIGURED = False


def configure_logging(logger_name: str, cfg) -> logging.Logger:
    """
    Configure root logger once for the PGM package subtree.

    Parameters
    ----------
    logger_name
        Typically ``pgm``.
    cfg
        :class:`ProjectConfig`-like object with ``logging``, ``resolved_root``.
    """
    global _LOGGER_CONFIGURED
    log = logging.getLogger(logger_name)
    if _LOGGER_CONFIGURED:
        return log

    log.setLevel(getattr(logging, str(cfg.logging.level).upper(), logging.INFO))
    fmt = logging.Formatter(
        fmt="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if cfg.logging.console:
        rich = RichHandler(rich_tracebacks=True, markup=True)
        rich.setLevel(log.level)
        log.addHandler(rich)
    log_dir = cfg.resolve(cfg.logging.log_dir)
    ensure_parents(log_dir / "pgm.log")
    fh = RotatingFileHandler(
        log_dir / "pgm.log",
        maxBytes=cfg.logging.rotating_max_bytes,
        backupCount=cfg.logging.rotating_backup_count,
        encoding="utf-8",
    )
    fh.setFormatter(fmt)
    fh.setLevel(log.level)
    log.addHandler(fh)

    _LOGGER_CONFIGURED = True
    return log


def get_logger(name: str = "pgm") -> logging.Logger:
    return logging.getLogger(name)
