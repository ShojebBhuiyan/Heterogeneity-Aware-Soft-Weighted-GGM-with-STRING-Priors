"""Matplotlib styling for publication figures."""

from __future__ import annotations

from pathlib import Path

from matplotlib.figure import Figure

from pgm.utils.paths import ensure_parents


def apply_publication_theme() -> None:
    """Tune rcParams for consistent figures (readable without heavy styling)."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.figsize": (6.0, 4.5),
            "axes.linewidth": 0.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_dual_format(fig: Figure, base_path: Path) -> tuple[Path, Path]:
    """
    Save ``base_path.png`` and ``base_path.svg`` alongside each other.

    Returns
    -------
    tuple[pathlib.Path, pathlib.Path]
        PNG and SVG paths.
    """
    ensure_parents(Path(base_path).with_suffix(".png"))
    png = Path(str(Path(base_path).with_suffix(".png")))
    svg = Path(str(Path(base_path).with_suffix(".svg")))
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    return png, svg
