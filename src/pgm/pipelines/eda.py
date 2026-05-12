"""EDA plotting and optional HTML profiler."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import seaborn as sns

from pgm.config.schemas import ProjectConfig
from pgm.utils.checkpoint import checkpoint_exists, save_checkpoint
from pgm.utils.logging_setup import configure_logging
from pgm.utils.notebook import is_smoke
from pgm.utils.paths import eda_reports_dir, figures_dir
from pgm.utils.seeds import set_global_seed
from pgm.visualization.theme import apply_publication_theme, save_dual_format

logger = logging.getLogger("pgm.pipelines.eda")


def run_eda(
    cfg: ProjectConfig,
    adata_input_path: Path,
    *,
    force: bool = False,
    checkpoint_name: str = "eda_processed_adata_preview",
) -> Path:
    """
    Generate QC-style EDA plots, optional HTML profile, markdown summary.

    Writes under ``reports/eda`` and ``reports/figures``.
    """
    configure_logging("pgm", cfg)
    set_global_seed(cfg)
    t0 = time.perf_counter()
    ed_dir = eda_reports_dir(cfg)
    fig_dir = figures_dir(cfg) / "eda"
    ed_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    summary_path = ed_dir / "summary.md"

    if (
        not force
        and checkpoint_exists(cfg, checkpoint_name)
        and summary_path.exists()
    ):
        logger.info("EDA skip (checkpoint + summary)")
        return summary_path

    adata = sc.read_h5ad(adata_input_path)
    hint = cfg.eda.n_cells_profiling_hint or adata.n_obs
    target_n = min(int(hint), adata.n_obs)
    seed = cfg.run.random_seed
    rng = np.random.default_rng(seed)
    if target_n < adata.n_obs:
        idx = rng.choice(adata.n_obs, target_n, replace=False)
        adata = adata[idx].copy()
        logger.info("Subsampled to %d cells for EDA", target_n)

    Tot = np.asarray(adata.X.sum(axis=1)).ravel()

    apply_publication_theme()
    fig_tot, ax_tot = plt.subplots(figsize=(5, 3.2))
    sns.histplot(Tot, bins=60, ax=ax_tot)
    ax_tot.set_xlabel("UMI counts per cell")
    save_dual_format(fig_tot, fig_dir / "total_counts_hist")
    plt.close(fig_tot)

    gg = min(50, adata.n_vars)
    cg = min(50, adata.n_obs)
    sub = adata[:cg, :gg]
    Xs = sub.X.toarray() if hasattr(sub.X, "toarray") else np.asarray(sub.X)
    fig_h, ax_h = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(Xs, cmap="viridis", ax=ax_h)
    ax_h.set_title("Expression snippet")
    save_dual_format(fig_h, fig_dir / "sparsity_snippet")
    plt.close(fig_h)

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    n_comp = min(25, max(adata.n_vars // 10, 2)) if is_smoke(cfg) else min(
        50, max(adata.n_vars // 10, 2)
    )
    n_neighbors = max(5, min(cfg.eda.umap_neighbors, 10 if is_smoke(cfg) else 15))
    sc.pp.pca(adata, n_comps=n_comp)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep="X_pca")
    sc.tl.umap(adata, min_dist=cfg.eda.umap_min_dist)
    fig_um, ax_um = plt.subplots(figsize=(5.5, 4.8))
    sc.pl.umap(adata, ax=ax_um, show=False)
    fig_um.tight_layout()
    save_dual_format(fig_um, fig_dir / "umap_quick")
    plt.close(fig_um)

    prof_path = ed_dir / "profile_report.html"
    try:
        from ydata_profiling import ProfileReport

        obs_df = adata.obs.copy()
        n_expr = min(30, adata.n_vars)
        expr_df = adata[:, :n_expr].to_df()
        ProfileReport(
            obs_df.join(expr_df.add_prefix("g_")), title="scRNA EDA", minimal=is_smoke(cfg)
        ).to_file(prof_path)
        logger.info("Wrote ydata-profiling report %s", prof_path)
    except Exception as e:
        logger.warning("ydata-profiling unavailable (%s); writing stub HTML", e)
        prof_path.write_text(
            "<html><body><h1>EDA profile placeholder</h1>"
            "<p>Install ydata-profiling when available on your Python build.</p></body></html>",
            encoding="utf-8",
        )

    tot_log = np.asarray(adata.X.sum(axis=1)).ravel().mean()
    summary_lines = [
        "# EDA summary",
        "",
        f"- cells (used): {adata.n_obs}",
        f"- genes: {adata.n_vars}",
        f"- mean log1p count sum: {tot_log:.3f}",
        "",
        "Artifacts: `reports/figures/eda/`, HTML `reports/eda/profile_report.html`.",
    ]
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    save_checkpoint(cfg, checkpoint_name, adata)
    logger.info("EDA complete in %.2fs", time.perf_counter() - t0)
    return summary_path

