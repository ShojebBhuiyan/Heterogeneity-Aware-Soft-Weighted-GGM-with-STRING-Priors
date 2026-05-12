"""EDA smoke on PBMC interim when present."""

from __future__ import annotations

import pytest

from pgm.config.loader import load_config
from pgm.pipelines.eda import run_eda
from pgm.pipelines.ingest import ingest_pbmc_pipeline
from pgm.utils.paths import tenx_mtx_path


def _has_mtx(cfg) -> bool:
    return tenx_mtx_path(cfg).is_dir() and (
        tenx_mtx_path(cfg) / "matrix.mtx"
    ).is_file()


@pytest.mark.smoke
def test_eda_smoke_reports():
    cfg = load_config(smoke=True)
    if not _has_mtx(cfg):
        pytest.skip("no mtx")
    h5 = ingest_pbmc_pipeline(cfg, force=False)
    summ = run_eda(cfg, h5, force=True)
    assert summ.exists()
    txt = summ.read_text(encoding="utf-8")
    assert "EDA summary" in txt
