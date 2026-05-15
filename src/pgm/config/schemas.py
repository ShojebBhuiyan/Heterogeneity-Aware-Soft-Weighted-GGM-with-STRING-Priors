"""Typed project configuration (pydantic v2)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, computed_field, field_validator


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
    #: Subsample to at most this many cells after load (None = use all). Applies even when
    #: smoke mode is off; use for faster dev runs without tightening EDA / PCA smoke behavior.
    max_cells: int | None = None
    #: Optional cap for smoke / interactive runs (None = all cells).
    smoke_max_cells: int | None = None
    #: Output filename relative to ``data/processed``.
    output_filename: str = "pbmc3k_processed.h5ad"


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

    #: ``sklearn`` = Friedman coordinate descent / LARS on covariance; ``gglasso`` = ADMM SGL solvers.
    gl_backend: Literal["sklearn", "gglasso"] = "sklearn"
    #: Which ``gglasso`` single-graphical-lasso routine to call when ``gl_backend='gglasso'``.
    gglasso_solver: Literal["block_sgl", "admm_sgl"] = "block_sgl"
    #: ADMM dual residual tolerance (gglasso); forwarded to ``block_SGL`` / ``ADMM_SGL``.
    gglasso_rtol: float = 1e-4
    #: ADMM augmented Lagrangian step ``rho`` (gglasso).
    gglasso_rho: float = 1.0
    #: Whether gglasso updates ``rho`` inside ADMM (Boyd-style).
    gglasso_update_rho: bool = True

    gl_alpha: float | None = None
    #: sklearn ``graphical_lasso(..., mode=...)`` for covariance fits; ``cd`` is usually more stable than ``lars``.
    gl_mode: Literal["cd", "lars"] = "cd"
    #: Multiply ``gl_alpha`` (or heuristic default) by each factor until a solve succeeds.
    gl_alpha_retry_multipliers: list[float] = Field(
        default_factory=lambda: [1.0, 2.0, 4.0, 8.0]
    )
    #: Convex shrink toward scaled identity: ``(1-s)Σ + s·(tr(Σ)/p)I`` before ridge; ``0`` disables.
    covariance_shrinkage: float = 0.1
    #: When ``effective_n`` (e.g. mixture ESS) is passed to covariance GLasso, enforce
    #: ``α ≥ scale · p / max(effective_n, 1)`` so fits with large ``p`` relative to ESS are
    #: not attempted with tiny ``α``. ``0`` disables.
    gl_soft_ess_min_alpha_scale: float = 0.05
    gl_cv_folds: int = 5
    gl_max_iter: int = 200
    gl_tol: float = 1e-3
    adjacency_tol: float = 1e-4
    ledoitwolf_shrink: bool = True
    covariance_ridge: float = 1e-4
    #: Log a warning after each solve attempt if wall time exceeds this (seconds); ``None`` disables.
    gl_attempt_warn_seconds: float | None = 60.0

    @field_validator("gl_alpha_retry_multipliers")
    @classmethod
    def _positive_multipliers(cls, v: list[float]) -> list[float]:
        if not v:
            raise ValueError("gl_alpha_retry_multipliers must be non-empty")
        if any(float(m) <= 0 for m in v):
            raise ValueError("gl_alpha_retry_multipliers must be positive")
        return [float(m) for m in v]

    @field_validator("covariance_shrinkage")
    @classmethod
    def _shrink_bounds(cls, v: float) -> float:
        fv = float(v)
        if not 0.0 <= fv <= 1.0:
            raise ValueError("covariance_shrinkage must be in [0, 1]")
        return fv

    @field_validator("gl_soft_ess_min_alpha_scale")
    @classmethod
    def _ess_alpha_scale_nonneg(cls, v: float) -> float:
        fv = float(v)
        if fv < 0.0:
            raise ValueError("gl_soft_ess_min_alpha_scale must be >= 0")
        return fv

    @field_validator("gglasso_rtol")
    @classmethod
    def _gglasso_rtol_positive(cls, v: float) -> float:
        fv = float(v)
        if fv <= 0.0:
            raise ValueError("gglasso_rtol must be positive")
        return fv

    @field_validator("gglasso_rho")
    @classmethod
    def _gglasso_rho_positive(cls, v: float) -> float:
        fv = float(v)
        if fv <= 0.0:
            raise ValueError("gglasso_rho must be positive")
        return fv


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
    #: Output CSV basename under ``reports/results`` (parents created as needed).
    metrics_csv: str = "metrics_summary.csv"
    #: If set, bootstrap stability for soft GMM component ``k`` (weighted covariance GLasso).
    bootstrap_soft_component: int | None = None


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
