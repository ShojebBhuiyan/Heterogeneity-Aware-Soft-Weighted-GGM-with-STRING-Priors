"""High-level pipeline entry points."""

from pgm.pipelines.cluster import run_clustering_pipeline
from pgm.pipelines.eda import run_eda
from pgm.pipelines.evaluation import run_evaluation_pipeline
from pgm.pipelines.full_pipeline import run_full_pipeline
from pgm.pipelines.global_ggm import run_global_ggm_pipeline
from pgm.pipelines.ingest import ingest_pbmc_pipeline
from pgm.pipelines.kg_pipeline import run_kg_pipeline
from pgm.pipelines.preprocess import run_preprocess_pipeline
from pgm.pipelines.soft_ggm import run_soft_weighted_ggm_pipeline

__all__ = [
    "ingest_pbmc_pipeline",
    "run_clustering_pipeline",
    "run_eda",
    "run_evaluation_pipeline",
    "run_full_pipeline",
    "run_global_ggm_pipeline",
    "run_kg_pipeline",
    "run_preprocess_pipeline",
    "run_soft_weighted_ggm_pipeline",
]
