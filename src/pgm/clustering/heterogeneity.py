"""Leiden clustering and GMM soft assignment."""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from sklearn import mixture

from pgm.config.schemas import ProjectConfig
from pgm.utils.paths import figures_dir
from pgm.utils.seeds import set_global_seed
from pgm.visualization.theme import apply_publication_theme, save_dual_format

logger = logging.getLogger("pgm.clustering")


def membership_entropy(soft: np.ndarray) -> np.ndarray:
    """Per-cell Shannon entropy across mixture components."""
    eps = 1e-12
    s = np.clip(soft, eps, 1.0)
    return -(s * np.log(s)).sum(axis=1)


def neighbors_leiden_umap(adata: sc.AnnData, cfg: ProjectConfig) -> None:
    """Compute neighbors, Leiden, UMAP embeddings (writes to ``adata.obsm`` / ``obs``)."""
    n_pcs_actual = min(
        cfg.clustering.neighbor_n_pcs,
        adata.obsm["X_pca"].shape[1],
    )
    sc.pp.neighbors(
        adata,
        n_neighbors=cfg.clustering.neighbor_n_neighbors,
        n_pcs=n_pcs_actual,
        random_state=cfg.clustering.random_state_leiden,
    )
    sc.tl.leiden(
        adata,
        resolution=cfg.clustering.leiden_resolution,
        key_added="leiden",
        random_state=cfg.clustering.random_state_leiden,
        flavor="igraph",
        directed=False,
        n_iterations=2,
    )
    sc.tl.umap(adata, random_state=cfg.clustering.random_state_leiden)


def fit_gmm_soft_labels(adata: sc.AnnData, cfg: ProjectConfig) -> None:
    """
    Gaussian mixture on PCA scores; writes ``obs['gmm_hard']``, ``obsm['X_gmm_proba']``.
    """
    set_global_seed(cfg)
    rep = cfg.clustering.gmm_pca_representation
    if rep not in adata.obsm:
        raise KeyError(f"representation {rep!r} missing")
    pcs = min(cfg.clustering.neighbor_n_pcs, adata.obsm[rep].shape[1])
    X = np.asarray(adata.obsm[rep][:, :pcs], dtype=float)
    gmm = mixture.GaussianMixture(
        n_components=cfg.clustering.gmm_n_components,
        covariance_type=cfg.clustering.gmm_covariance_type,
        random_state=cfg.run.random_seed,
        max_iter=500,
    )
    gmm.fit(X)
    proba = gmm.predict_proba(X)
    adata.obs["gmm_hard"] = gmm.predict(X).astype(np.int32).astype(str)
    adata.obsm["X_gmm_proba"] = proba
    mean_ent = float(membership_entropy(proba).mean())
    logger.info("GMM fit done; mean soft assignment entropy %.4f", mean_ent)


def plot_cluster_visuals(adata: sc.AnnData, cfg: ProjectConfig, tag: str = "") -> None:
    """Save UMAP by Leiden/GMM + entropy histogram under ``figures/clustering``."""
    out = figures_dir(cfg) / "clustering"
    out.mkdir(parents=True, exist_ok=True)
    pref = (tag + "_") if tag else ""
    fig1, ax1 = plt.subplots(figsize=(6, 4.8))
    sc.pl.umap(
        adata,
        color="leiden",
        ax=ax1,
        show=False,
        title="Leiden (UMAP)",
    )
    fig1.tight_layout()
    save_dual_format(fig1, out / f"{pref}umap_leiden")
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(6, 4.8))
    sc.pl.umap(
        adata,
        color="gmm_hard",
        ax=ax2,
        show=False,
        title="GMM labels (hard, UMAP)",
    )
    fig2.tight_layout()
    save_dual_format(fig2, out / f"{pref}umap_gmm")
    plt.close(fig2)

    if "X_gmm_proba" in adata.obsm:
        ent = membership_entropy(np.asarray(adata.obsm["X_gmm_proba"]))
        fig3, ax3 = plt.subplots(figsize=(5, 3.5))
        ax3.hist(ent, bins=40)
        ax3.set_xlabel("Soft assignment entropy")
        ax3.set_ylabel("# cells")
        apply_publication_theme()
        save_dual_format(fig3, out / f"{pref}entropy_hist")
        plt.close(fig3)
