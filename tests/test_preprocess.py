"""Preprocessing smoke."""

from __future__ import annotations

import pytest

from pgm.config.loader import load_config
from pgm.pipelines.ingest import ingest_pbmc_pipeline
from pgm.pipelines.preprocess import run_preprocess_pipeline
from pgm.utils.paths import tenx_mtx_path


def _mtx(cfg):
    return tenx_mtx_path(cfg).is_dir() and (
        tenx_mtx_path(cfg) / "matrix.mtx"
    ).is_file()


@pytest.mark.smoke
def test_preprocess_smoke():
    cfg = load_config(smoke=True)
    if not _mtx(cfg):
        pytest.skip("no mtx")
    raw = ingest_pbmc_pipeline(cfg, force=False)
    out = run_preprocess_pipeline(cfg, raw, force=True)
    assert out.exists()
