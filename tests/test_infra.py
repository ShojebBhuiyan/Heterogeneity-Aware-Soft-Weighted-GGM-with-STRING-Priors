"""Tests for infra: config checkpointing, seeds."""

from __future__ import annotations

from pathlib import Path

import pytest

from pgm.config.loader import deep_merge_dicts, load_project_config
from pgm.config.schemas import ProjectConfig
from pgm.utils.checkpoint import checkpoint_exists, load_checkpoint, save_checkpoint
from pgm.utils.paths import checkpoints_dir


def test_deep_merge_nonempty_child():
    base = {"a": {"x": 1}, "b": 2}
    over = {"a": {"y": 3}}
    m = deep_merge_dicts(base, over)
    assert m["a"]["x"] == 1 and m["a"]["y"] == 3 and m["b"] == 2


def test_load_config_smoke_flag(tmp_path):
    """Write minimal default/smoke YAML in tmp and load with smoke."""
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    (cfg_dir / "default.yaml").write_text("run:\n  random_seed: 7\n", encoding="utf-8")
    (cfg_dir / "smoke.yaml").write_text(
        "clustering:\n  gmm_n_components: 2\n",
        encoding="utf-8",
    )
    cfg = load_project_config(configs_dir=cfg_dir, smoke=True)
    assert cfg.smoke_mode.enabled is True
    assert cfg.clustering.gmm_n_components == 2
    assert cfg.run.random_seed == 7


def test_resolve_paths(tmp_path: Path):
    cfg = ProjectConfig.model_validate({"paths": {"project_root": str(tmp_path)}})
    out = cfg.resolve(Path("relative/sub"))
    assert out == tmp_path / "relative" / "sub"


def test_checkpoint_roundtrip_numpy(tmp_path: Path):
    import numpy as np

    cfg = ProjectConfig.model_validate(
        {"paths": {"project_root": str(tmp_path)}, "checkpoint": {"subdirectory": "t"}}
    )
    checkpoints_dir(cfg).mkdir(parents=True, exist_ok=True)
    x = {"arr": np.arange(12).reshape(3, 4)}
    save_checkpoint(cfg, "trial", x)
    assert checkpoint_exists(cfg, "trial")
    loaded = load_checkpoint(cfg, "trial")
    assert "arr" in loaded
    assert np.allclose(loaded["arr"], x["arr"])


def test_seed_determinism(tmp_path: Path):
    import numpy as np

    from pgm.utils.seeds import set_global_seed

    cfg_a = ProjectConfig.model_validate({"paths": {"project_root": str(tmp_path)}})
    cfg_b = cfg_a.model_copy(update={"run": cfg_a.run.model_copy(update={"random_seed": 123})})
    assert set_global_seed(cfg_b) == 123
    a = np.random.rand(5).copy()

    cfg2 = cfg_b.model_copy()
    assert set_global_seed(cfg2) == 123
    b = np.random.rand(5)
    assert np.allclose(a, b)
