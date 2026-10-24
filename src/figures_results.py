"""Charts of the model results, including the two that carry the main argument.

Figures 9 and 10 are the pair that matters: the same errors measured in dollars and as
a share of each car's price. If they point in opposite directions, that contrast is the
study's central finding, and no single overall score could have revealed it.
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as cfg
import plotstyle as ps


def _load():
    pred = pd.read_parquet(cfg.PRED_FILE)
    meta = json.loads((cfg.TAB_DIR / "run_metadata.json").read_text())
    best = json.loads((cfg.TAB_DIR / "best_model.json").read_text())["best"]
    return pred, meta, best


def _grouped_bars(ax, table: pd.DataFrame, value: str, models: list[str]) -> None:
    """One cluster of bars per price band, one bar per model."""
    width = 0.8 / len(models)
    x = np.arange(len(cfg.BAND_LABELS))
    for i, name in enumerate(models):
        sub = table[table.Model == name].set_index("Price band").loc[cfg.BAND_LABELS]
        ax.bar(x + i * width - 0.4 + width / 2, sub[value], width,
               label=name, color=ps.MODEL_COLOURS.get(name, ps.GREY))
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b}\nprice band" for b in cfg.BAND_LABELS])


def fig_overall(n_test: int) -> None:
    t = pd.read_csv(cfg.TAB_DIR / "table3_overall_metrics.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.1))
    for ax, col, title in [(axes[0], "MAE", "Average miss (lower is better)"),
                           (axes[1], "RMSE", "Miss with big errors punished harder")]:
        colours = [ps.MODEL_COLOURS.get(m, ps.GREY) for m in t.Model]
        ax.bar(t.Model, t[col], color=colours)
        ax.set_title(title)
        ax.set_ylabel("Dollars")
        ps.money(ax, "y")
        ax.tick_params(axis="x", rotation=20)
        for i, v in enumerate(t[col]):
            ax.text(i, v, f"${v:,.0f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle(f"Overall accuracy on {n_test:,} cars the models never saw",
                 fontweight="bold")
    ps.save(fig, "fig08_model_comparison.png")


def fig_band_dollars(models: list[str]) -> None:
    t = pd.read_csv(cfg.TAB_DIR / "table4_band_metrics.csv")
    fig, ax = plt.subplots(figsize=(8, 4.3))
    _grouped_bars(ax, t, "MAE", models)
    ax.set_ylabel("Average miss (dollars)")
    ax.set_title("Measured in dollars, the misses grow with the price of the car")
    ps.money(ax, "y")
    ax.legend(ncol=len(models), loc="upper left")
    ps.save(fig, "fig09_mae_by_band_dollars.png")


def fig_band_percent(models: list[str]) -> None:
    t = pd.read_csv(cfg.TAB_DIR / "table4_band_metrics.csv")
    fig, ax = plt.subplots(figsize=(8, 4.3))
    _grouped_bars(ax, t, "MdAPE_pct", models)
    ax.set_ylabel("Typical miss as a share of price (%)")
    ax.set_title("Measured against each car's own price, the picture reverses")
    ax.yaxis.set_major_formatter(
        plt.matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.legend(ncol=len(models), loc="upper right")
    ps.save(fig, "fig10_error_by_band_percent.png")


def fig_predicted_vs_actual(pred: pd.DataFrame, best: str) -> None:
    actual, guess = pred[cfg.TARGET], pred[f"pred::{best}"]
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    hb = ax.hexbin(actual, guess, gridsize=55, cmap="Blues", bins="log",
                   extent=(0, 80000, 0, 80000))
    fig.colorbar(hb, ax=ax, label="Cars per cell (log scale)")
    ax.plot([0, 80000], [0, 80000], color=ps.RED, linestyle="--", linewidth=1.3,
            label="Perfect prediction")
    ax.set_xlabel("Actual asking price")
    ax.set_ylabel("Predicted asking price")
    ax.set_title(f"{best}: predicted against actual")
    ax.legend(loc="upper left")
    ps.money(ax, "x")
    ps.money(ax, "y")
    ps.save(fig, "fig11_predicted_vs_actual.png")


def fig_residuals(pred: pd.DataFrame, best: str, cuts: list[float]) -> None:
    """Whether the model prices cars too high or too low, band by band."""
    error = pred[cfg.TARGET] - pred[f"pred::{best}"]
    bands = pd.cut(pred[cfg.TARGET], bins=[-np.inf, cuts[0], cuts[1], np.inf],
                   labels=cfg.BAND_LABELS)
    data = [error[bands == b].to_numpy() for b in cfg.BAND_LABELS]
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    bp = ax.boxplot(data, tick_labels=cfg.BAND_LABELS, showfliers=False,
                    patch_artist=True, widths=0.55)
    for patch, b in zip(bp["boxes"], cfg.BAND_LABELS):
        patch.set(facecolor=ps.BAND_COLOURS[b], alpha=0.65,
                  edgecolor=ps.BAND_COLOURS[b])
    for med in bp["medians"]:
        med.set(color="black", linewidth=1.6)
    ax.axhline(0, color=ps.RED, linewidth=1.2, linestyle="--")
    ax.set_ylabel("Actual minus predicted (dollars)")
    ax.set_xlabel("Price band")
    ax.set_title(f"{best}: above the line means the car was priced too low")
    ps.money(ax, "y")
    ps.save(fig, "fig12_residuals_by_band.png")


def fig_importance(best: str) -> None:
    t = pd.read_csv(cfg.TAB_DIR / "table6_permutation_importance.csv")
    t = t.sort_values("MAE increase when shuffled")
    t["Feature"] = [cfg.label(f) for f in t.Feature]
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.barh(t.Feature, t["MAE increase when shuffled"], color=ps.NAVY,
            xerr=t["Std dev"], error_kw={"ecolor": ps.GREY, "lw": 1})
    ax.set_xlabel("Dollars added to the average miss when this column is scrambled")
    ax.set_title(f"What {best} leans on most")
    ps.money(ax, "x")
    ps.save(fig, "fig13_permutation_importance.png")


def fig_large_errors() -> None:
    t = pd.read_csv(cfg.TAB_DIR / "table5_large_error_profile.csv")
    t = t[t.Kind == "share"]
    y = np.arange(len(t))
    fig, ax = plt.subplots(figsize=(8, 4.3))
    ax.barh(y + 0.19, t["Large-error cars"] * 100, 0.38, color=ps.RED,
            label="Badly predicted cars")
    ax.barh(y - 0.19, t["All other cars"] * 100, 0.38, color=ps.NAVY,
            label="All other cars")
    ax.set_yticks(y)
    ax.set_yticklabels(t.Characteristic)
    ax.invert_yaxis()
    ax.set_xlabel("Share of cars (%)")
    ax.set_title("Missing information is commoner among the badly predicted cars")
    ax.legend()
    ps.save(fig, "fig14_large_error_profile.png")


def fig_dedup_effect() -> None:
    d = json.loads((cfg.TAB_DIR / "table7_dedup_effect.json").read_text())
    labels = ["Repeat adverts left in\n(what a careless study reports)",
              "Repeat adverts removed\n(the honest score)"]
    values = [d["repeats_left_mae"], d["deduplicated_mae"]]
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.bar(labels, values, color=[ps.RED, ps.NAVY], width=0.55)
    for i, v in enumerate(values):
        ax.text(i, v, f"${v:,.0f}", ha="center", va="bottom", fontweight="bold")
    ax.set_ylabel("Average miss (dollars)")
    ax.set_title("Leaving repeat adverts in makes the model look "
                 f"${d['apparent_improvement_usd']:,.0f} more accurate than it is")
    ps.money(ax, "y")
    ps.save(fig, "fig15_dedup_effect.png")


def main() -> None:
    cfg.ensure_dirs()
    ps.setup()
    pred, meta, best = _load()
    models = [c.split("::", 1)[1] for c in pred.columns if c.startswith("pred::")]
    print(f"Drawing result charts (best model: {best})")

    fig_overall(len(pred))
    fig_band_dollars(models)
    fig_band_percent(models)
    fig_predicted_vs_actual(pred, best)
    fig_residuals(pred, best, meta["band_cuts"])
    fig_importance(best)
    fig_large_errors()
    if (cfg.TAB_DIR / "table7_dedup_effect.json").exists():
        fig_dedup_effect()


if __name__ == "__main__":
    main()
