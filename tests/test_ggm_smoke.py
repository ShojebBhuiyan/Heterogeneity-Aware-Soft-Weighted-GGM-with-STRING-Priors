"""Smoke targets for graphical models without network."""

from __future__ import annotations

import pytest

from pgm.config.loader import load_config
from pgm.pipelines.cluster import run_clustering_pipeline
from pgm.pipelines.global_ggm import run_global_ggm_pipeline
from pgm.pipelines.ingest import ingest_pbmc_pipeline
from pgm.pipelines.preprocess import run_preprocess_pipeline
from pgm.utils.paths import tenx_mtx_path


def _mtx(cfg) -> bool:
    return tenx_mtx_path(cfg).is_dir()


def _chain(cfg):
    r = ingest_pbmc_pipeline(cfg, force=False)
    p = run_preprocess_pipeline(cfg, r, force=False)
    return run_clustering_pipeline(cfg, p, force=False)


@pytest.mark.smoke
def test_global_ggm_writes_joblib():
    cfg = load_config(smoke=True)
    if not _mtx(cfg):
        pytest.skip("no mtx")
    cl = _chain(cfg)
    out = run_global_ggm_pipeline(cfg, cl, force=True)
    assert out.exists()


@pytest.mark.smoke
def test_soft_ggm_writes_joblib():
    from pgm.pipelines.soft_ggm import run_soft_weighted_ggm_pipeline

    cfg = load_config(smoke=True)
    if not _mtx(cfg):
        pytest.skip("no mtx")
    cl = _chain(cfg)
    out = run_soft_weighted_ggm_pipeline(cfg, cl, force=True)
    assert out.exists()


@pytest.mark.smoke
def test_evaluation_writes_csv():
    import requests

    cfg = load_config(smoke=True)
    if not _mtx(cfg):
        pytest.skip("no mtx")
    try:
        requests.head("https://string-db.org/", timeout=6)
    except OSError:
        pytest.skip("offline")

    from pgm.pipelines.evaluation import run_evaluation_pipeline
    from pgm.pipelines.global_ggm import run_global_ggm_pipeline
    from pgm.pipelines.ingest import ingest_pbmc_pipeline
    from pgm.pipelines.kg_pipeline import run_kg_pipeline
    from pgm.pipelines.preprocess import run_preprocess_pipeline
    from pgm.pipelines.soft_ggm import run_soft_weighted_ggm_pipeline

    r = ingest_pbmc_pipeline(cfg, force=False)
    p = run_preprocess_pipeline(cfg, r, force=False)
    cl = run_clustering_pipeline(cfg, p, force=False)
    g = run_global_ggm_pipeline(cfg, cl, force=True)
    s = run_soft_weighted_ggm_pipeline(cfg, cl, force=True)
    k = run_kg_pipeline(cfg, cl, force=True)
    m = run_evaluation_pipeline(cfg, cl, g, s, k, force=True)
    txt = m.read_text(encoding="utf-8")
    assert "bootstrap_mean_pos_freq" in txt or "global_mean_degree" in txt
