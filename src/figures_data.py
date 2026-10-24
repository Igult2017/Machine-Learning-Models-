"""Charts describing the data itself, before any model is involved.

These answer Objective 2: what the listings look like, what is missing from them, and
how asking price moves with the characteristics the models will be given.
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as cfg
import plotstyle as ps


def band_cuts(prices: pd.Series) -> list[float]:
    """Read the training cut-offs if training has run, otherwise derive them."""
    meta_file = cfg.TAB_DIR / "run_metadata.json"
    if meta_file.exists():
        return json.loads(meta_file.read_text())["band_cuts"]
    return [float(prices.quantile(q)) for q in cfg.BAND_QUANTILES]


def fig_funnel() -> None:
    """How many listings survived each cleaning rule."""
    t = pd.read_csv(cfg.TAB_DIR / "table1_cleaning_funnel.csv")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.barh(t["Step"], t["Rows remaining"], color=ps.NAVY)
    ax.invert_yaxis()
    ax.set_xlabel("Listings remaining")
    ax.set_title("Cleaning reduced 426,880 listings to 209,164 individual cars")
    for y, v in enumerate(t["Rows remaining"]):
        ax.text(v + 4000, y, f"{v:,}", va="center", fontsize=8)
    ax.set_xlim(0, t["Rows remaining"].max() * 1.18)
    ps.save(fig, "fig01_cleaning_funnel.png")


def fig_missing() -> None:
    """Which columns are blank, and how often, before cleaning."""
    t = pd.read_csv(cfg.TAB_DIR / "table2_missingness.csv")
    t = t[t["Percent blank"] > 0]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    colours = [ps.RED if v > 40 else ps.AMBER if v > 15 else ps.NAVY
               for v in t["Percent blank"]]
    ax.barh(t["Column"], t["Percent blank"], color=colours)
    ax.invert_yaxis()
    ax.set_xlabel("Percent of listings with no value")
    ax.set_title("Gaps in the raw listings")
    for y, v in enumerate(t["Percent blank"]):
        ax.text(v + 1, y, f"{v:.1f}%", va="center", fontsize=8)
    ax.set_xlim(0, 112)
    ps.save(fig, "fig02_missing_data.png")


def fig_price_distribution(df: pd.DataFrame) -> None:
    """The shape of asking prices, with the three bands marked."""
    cuts = band_cuts(df[cfg.TARGET])
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.hist(df[cfg.TARGET], bins=90, range=(0, 80000), color=ps.NAVY, alpha=0.85)
    for cut, label in zip(cuts, ["33rd percentile", "67th percentile"]):
        ax.axvline(cut, color=ps.RED, linestyle="--", linewidth=1.4)
        ax.text(cut, ax.get_ylim()[1] * 0.92, f"  {label}\n  ${cut:,.0f}",
                color=ps.RED, fontsize=8, va="top")
    ax.set_xlabel("Asking price")
    ax.set_ylabel("Number of cars")
    ax.set_title("Asking prices lean towards the cheaper end (long tail to the right)")
    ps.money(ax, "x")
    ps.save(fig, "fig03_price_distribution.png")


def fig_price_vs_mileage(df: pd.DataFrame) -> None:
    """Density of cars across mileage and price - 209,164 dots would be a blob."""
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    hb = ax.hexbin(df["odometer"], df[cfg.TARGET], gridsize=55, cmap="Blues",
                   bins="log", extent=(0, 300000, 0, 70000))
    fig.colorbar(hb, ax=ax, label="Cars per cell (log scale)")
    ax.set_xlabel("Mileage")
    ax.set_ylabel("Asking price")
    ax.set_title("Asking price falls as mileage rises")
    ps.money(ax, "y")
    ax.xaxis.set_major_formatter(
        plt.matplotlib.ticker.FuncFormatter(lambda v, _: f"{v/1000:,.0f}k"))
    ps.save(fig, "fig04_price_vs_mileage.png")


def fig_price_vs_age(df: pd.DataFrame) -> None:
    """Median price by age, with the middle half of cars shaded."""
    g = df[df["age"].between(0, 30)].groupby("age")[cfg.TARGET]
    stat = g.agg(median="median", lo=lambda s: s.quantile(0.25),
                 hi=lambda s: s.quantile(0.75), n="size")
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.fill_between(stat.index, stat["lo"], stat["hi"], color=ps.NAVY, alpha=0.18,
                    label="Middle half of cars")
    ax.plot(stat.index, stat["median"], color=ps.NAVY, linewidth=2, label="Median price")
    ax.set_xlabel("Age at time of listing (years)")
    ax.set_ylabel("Asking price")
    ax.set_title("Price drops steeply for the first decade, then flattens")
    ax.legend()
    ps.money(ax, "y")
    ps.save(fig, "fig05_price_vs_age.png")


def _box_by_category(df: pd.DataFrame, column: str, title: str, filename: str,
                     top: int = 15) -> None:
    keep = df[column].value_counts().head(top).index
    sub = df[df[column].isin(keep)]
    order = sub.groupby(column)[cfg.TARGET].median().sort_values().index
    data = [sub.loc[sub[column] == k, cfg.TARGET].to_numpy() for k in order]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    bp = ax.boxplot(data, vert=False, tick_labels=[str(k) for k in order],
                    showfliers=False, patch_artist=True, widths=0.65)
    for patch in bp["boxes"]:
        patch.set(facecolor=ps.NAVY, alpha=0.65, edgecolor=ps.NAVY)
    for med in bp["medians"]:
        med.set(color=ps.AMBER, linewidth=1.8)
    ax.set_xlabel("Asking price")
    ax.set_title(title)
    ps.money(ax, "x")
    ps.save(fig, filename)


def main() -> None:
    cfg.ensure_dirs()
    ps.setup()
    df = pd.read_parquet(cfg.CLEAN_FILE)
    print(f"Drawing data charts from {len(df):,} cars")
    fig_funnel()
    fig_missing()
    fig_price_distribution(df)
    fig_price_vs_mileage(df)
    fig_price_vs_age(df)
    _box_by_category(df, "manufacturer",
                     "Asking price by make (15 commonest makes)",
                     "fig06_price_by_manufacturer.png")
    _box_by_category(df, "type",
                     "Asking price by body type",
                     "fig07_price_by_type.png", top=13)


if __name__ == "__main__":
    main()
