"""Shared figure style: PDF, single-column ACL width, consistent colors across figures."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .config import FIGURES  # noqa: E402

COL_W = 3.03  # ACL single-column width, inches

# Fixed entity → color mapping (validated categorical pair + neutrals); identity is never
# color-only: every series also differs by marker / line style / direct label.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e4e3df"
COLORS = {
    "ours": BLUE, "workday": BLUE, "offpeak": BLUE, "pearson": BLUE,
    "baseline": ORANGE, "nonworkday": ORANGE, "peak": ORANGE, "spearman": ORANGE,
    "neutral": MUTED,
}

plt.rcParams.update({
    "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "lines.linewidth": 1.4, "lines.markersize": 3.5,
    "legend.frameon": False, "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def new_fig(height: float = 2.2, ncols: int = 1, nrows: int = 1, **kw):
    return plt.subplots(nrows, ncols, figsize=(COL_W, height), **kw)


def save(fig, name: str, out_dir=None) -> None:
    out_dir = out_dir or FIGURES
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.pdf")
    fig.savefig(out_dir / f"{name}.png", dpi=200)  # preview only; the paper uses the PDF
    plt.close(fig)
