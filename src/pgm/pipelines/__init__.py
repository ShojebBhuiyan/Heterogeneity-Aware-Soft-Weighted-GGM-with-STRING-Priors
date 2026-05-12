"""High-level pipeline entry points."""

from pgm.pipelines.eda import run_eda
from pgm.pipelines.ingest import ingest_pbmc_pipeline
from pgm.pipelines.preprocess import run_preprocess_pipeline

__all__ = ["ingest_pbmc_pipeline", "run_eda", "run_preprocess_pipeline"]
