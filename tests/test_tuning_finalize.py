"""Smoke tests for tuning sweep finalize (no full pipeline required)."""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import pandas as pd
import yaml


def _load_sweep_module(repo_root: Path):
    path = repo_root / "scripts" / "tuning_sweep.py"
    spec = importlib.util.spec_from_file_location("tuning_sweep", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_finalize_best_writes_yaml(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    mod = _load_sweep_module(repo_root)

    cfg_dst = tmp_path / "configs"
    cfg_dst.mkdir(parents=True)
    shutil.copy(repo_root / "configs" / "default.yaml", cfg_dst / "default.yaml")
    shutil.copy(repo_root / "configs" / "tuning_hvg300.yaml", cfg_dst / "tuning_hvg300.yaml")

    tuning_dir = tmp_path / "reports" / "results" / "tuning"
    tuning_dir.mkdir(parents=True)
    row = {
        "run_tag": "test::one",
        "phase": "hvg",
        "preset": "tuning_hvg300",
        "overrides_json": json.dumps({}),
        "score": 1.0,
        "m__global_edge_count": 12,
        "m__soft_union_edge_count": 30,
    }
    pd.DataFrame([row]).to_csv(tuning_dir / "sweep_results.csv", index=False)

    mod.finalize_best(tmp_path)

    best_path = cfg_dst / "tuning_best.yaml"
    assert best_path.is_file()
    data = yaml.safe_load(best_path.read_text(encoding="utf-8"))
    assert data["checkpoint"]["subdirectory"] == "tuning/best"
    assert data["preprocessing"]["output_filename"] == "pbmc3k_tuning_best.h5ad"
