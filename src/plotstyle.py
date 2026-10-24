"""One shared look for every chart, so the thesis reads as a single piece of work.

Colours are chosen to stay distinguishable in greyscale and for the most common forms
of colour blindness, because a printed or photocopied thesis loses colour entirely.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")      # draw to files only; never try to open a window

import matplotlib.pyplot as plt  # noqa: E402

import config as cfg  # noqa: E402

NAVY = "#2E5C8A"
AMBER = "#D97E2B"
TEAL = "#2F7F7F"
PLUM = "#8A4F7D"
GREY = "#6E6E6E"
RED = "#B5443A"

BAND_COLOURS = {"Lower": NAVY, "Middle": AMBER, "Higher": TEAL}
MODEL_COLOURS = {
    "Median guess": GREY,
    "Linear Regression": PLUM,
    "Random Forest": NAVY,
    "Gradient Boosting": AMBER,
}


def setup() -> None:
    """Apply the shared typography and grid settings."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#CCCCCC",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#E8E8E8",
        "grid.linewidth": 0.8,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "legend.frameon": False,
        "figure.dpi": 110,
    })


def save(fig, filename: str) -> None:
    """Write the chart into outputs/figures and close it to free memory."""
    cfg.FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = cfg.FIG_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {filename}")


def money(ax, axis: str = "y") -> None:
    """Label an axis in dollars with thousands separators."""
    fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f"${v:,.0f}")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)
