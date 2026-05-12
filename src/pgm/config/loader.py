"""Load YAML configs with optional smoke overlay."""

from __future__ import annotations

import os
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from pgm.config.schemas import ProjectConfig


def deep_merge_dicts(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge override into base (deep)."""
    out = deepcopy(dict(base))
    for k, v in override.items():
        if (
            k in out
            and isinstance(out[k], dict)
            and isinstance(v, Mapping)
        ):
            out[k] = deep_merge_dicts(out[k], v)  # type: ignore[arg-type]
        elif v != {}:
            out[k] = deepcopy(v)
    return out


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    return loaded if isinstance(loaded, dict) else {}


def config_paths(cfg_dir: Path) -> tuple[Path, Path]:
    return cfg_dir / "default.yaml", cfg_dir / "smoke.yaml"


def load_project_config(
    *,
    configs_dir: Path | None = None,
    smoke: bool | None = None,
    extra_overrides: Mapping[str, Any] | None = None,
) -> ProjectConfig:
    """
    Load `default.yaml`; merge `smoke.yaml` when ``smoke`` or env ``SMOKE=1``.

    Parameters
    ----------
    configs_dir
        Directory containing default.yaml / smoke.yaml. Defaults to
        `<resolved_root>/configs` after first load pass.
    extra_overrides
        Optional shallow/deep overrides (e.g. tests).
    """
    root_candidates = Path.cwd().resolve()
    smoke_env = os.environ.get("SMOKE", "").lower() in {"1", "true", "yes"}
    smoke = smoke if smoke is not None else smoke_env

    if configs_dir is None:
        configs_dir = _find_configs_dir(root_candidates)

    default_path = configs_dir / "default.yaml"
    smoke_path = configs_dir / "smoke.yaml"
    merged = _load_yaml(default_path)
    if smoke and smoke_path.is_file():
        merged = deep_merge_dicts(merged, _load_yaml(smoke_path))

    merged["smoke_mode"] = {"enabled": bool(smoke)}
    merged.setdefault("paths", {})
    merged["paths"].setdefault("project_root", None)
    cfg = ProjectConfig.model_validate(merged)

    resolved_root = cfg.resolved_root
    if configs_dir is None:
        configs_dir = _find_configs_dir(resolved_root)

    if extra_overrides:
        dumped = cfg.model_dump(mode="python")
        merged2 = deep_merge_dicts(dumped, dict(extra_overrides))
        cfg = ProjectConfig.model_validate(merged2)

    return cfg


def _find_configs_dir(start: Path) -> Path:
    for cand in [start] + list(start.parents):
        d = cand / "configs"
        if (d / "default.yaml").is_file():
            return d.resolve()
    return (start / "configs").resolve()


def load_config(
    *,
    configs_dir: Path | None = None,
    smoke: bool | None = None,
    extra_overrides: Mapping[str, Any] | None = None,
) -> ProjectConfig:
    """Alias for :func:`load_project_config` (notebook-friendly name)."""
    return load_project_config(
        configs_dir=configs_dir, smoke=smoke, extra_overrides=extra_overrides
    )
