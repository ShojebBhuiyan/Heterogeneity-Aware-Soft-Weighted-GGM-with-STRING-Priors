"""Typed project configuration (pydantic v2)."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, computed_field


class PathsConfig(BaseModel):
    """Root path resolution."""

    project_root: Path | None = None


class DataPathsConfig(BaseModel):
    """Input/output data locations (relative paths resolved vs project_root)."""

    tenx_mtx_dir: Path = Path("datasets/filtered_gene_bc_matrices/hg19")
    interim_dir: Path = Path("data/interim")
    processed_dir: Path = Path("data/processed")
    checkpoint_dir: Path = Path("data/checkpoints")
    external_dir: Path = Path("data/external")
    string_cache_dir: Path = Path("data/external/string_cache")


class ReportsPathsConfig(BaseModel):
    figures_dir: Path = Path("reports/figures")
    eda_dir: Path = Path("reports/eda")
    results_dir: Path = Path("reports/results")


class RunConfig(BaseModel):
    random_seed: int = 42


class LoggingConfig(BaseModel):
    level: str = "INFO"
    console: bool = True
    log_dir: Path = Path("logs")
    rotating_max_bytes: int = 10_485_760
    rotating_backup_count: int = 3


class CheckpointRuntimeConfig(BaseModel):
    subdirectory: str = ""


class EdaRuntimeConfig(BaseModel):
    n_cells_profiling_hint: int | None = None
    umap_neighbors: int = 15
    umap_min_dist: float = 0.5


class IngestionConfig(BaseModel):
    dataset_name: str = "pbmc3k"
    write_interim_filename: str = "pbmc3k_raw.h5ad"


class PreprocessingConfig(BaseModel):
    target_sum: float = 1e4
    min_genes: int = 200
    min_cells: int = 3
    max_pct_counts_mito: float = 20
    mitochondrial_prefix: str = r"^MT-"
    n_top_hvg: int = 2000
    n_pcs: int = 50


class ClusteringConfig(BaseModel):
    neighbor_n_neighbors: int = 15
    neighbor_n_pcs: int = 40
    leiden_resolution: float = 0.5
    random_state_leiden: int = 0
    gmm_n_components: int = 8
    gmm_covariance_type: str = "full"
    gmm_pca_representation: str = "X_pca"


class ModelsConfig(BaseModel):
    """Graphical model hyperparameters."""

    gl_alpha: float | None = None
    gl_cv_folds: int = 5
    gl_max_iter: int = 200
    gl_tol: float = 1e-3
    adjacency_tol: float = 1e-4
    ledoitwolf_shrink: bool = True
    covariance_ridge: float = 1e-6


class KgConfig(BaseModel):
    species_id: int = 9606
    confidence_threshold: float = 0.7
    batch_size: int = 400
    request_timeout_seconds: int = 60
    max_retries: int = 5
    cache_enabled: bool = True
    max_genes_for_query: int = 500
    #: Add ``scale * prior`` to weighted covariance before graphical lasso (uncertain heuristic).
    prior_covariance_scale: float = 0.02

class EvaluationConfig(BaseModel):
    bootstrap_b: int = 50
    bootstrap_fraction: float = 0.8


class SmokeFlag(BaseModel):
    """Internal: set when merging smoke YAML or SMOKE=1."""

    enabled: bool = False


class ProjectConfig(BaseModel):
    """Unified configuration."""

    paths: PathsConfig = Field(default_factory=PathsConfig)
    data: DataPathsConfig = Field(default_factory=DataPathsConfig)
    reports: ReportsPathsConfig = Field(default_factory=ReportsPathsConfig)
    run: RunConfig = Field(default_factory=RunConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    checkpoint: CheckpointRuntimeConfig = Field(default_factory=CheckpointRuntimeConfig)
    eda: EdaRuntimeConfig = Field(default_factory=EdaRuntimeConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    clustering: ClusteringConfig = Field(default_factory=ClusteringConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    kg: KgConfig = Field(default_factory=KgConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    smoke_mode: SmokeFlag = Field(default_factory=SmokeFlag)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def resolved_root(self) -> Path:
        """Absolute project root for resolving relative paths."""
        root = self.paths.project_root
        if root is not None:
            return root.expanduser().resolve()
        cwd = Path.cwd().resolve()
        for cand in [cwd] + list(cwd.parents):
            if (cand / "configs" / "default.yaml").is_file():
                return cand
        return cwd

    def resolve(self, relative: Path) -> Path:
        """Resolve a configured path relative to project root."""
        if relative.is_absolute():
            return relative
        return (self.resolved_root / relative).resolve()
