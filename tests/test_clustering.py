"""Clustering smoke."""

from __future__ import annotations

import pytest

from pgm.config.loader import load_config
from pgm.pipelines.cluster import run_clustering_pipeline
from pgm.pipelines.ingest import ingest_pbmc_pipeline
from pgm.pipelines.preprocess import run_preprocess_pipeline
from pgm.utils.paths import tenx_mtx_path


def _mtx(cfg):
    return tenx_mtx_path(cfg).is_dir()


@pytest.mark.smoke
def test_cluster_smoke():
    cfg = load_config(smoke=True)
    if not _mtx(cfg):
        pytest.skip("no mtx")
    r = ingest_pbmc_pipeline(cfg, force=False)
    p = run_preprocess_pipeline(cfg, r, force=False)
    c = run_clustering_pipeline(cfg, p, force=True)
    assert c.exists()
