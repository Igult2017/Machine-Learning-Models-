"""Score the models on the held-back cars and report where they go wrong.

Two analyses live here - overall accuracy, and accuracy split by price band. The error
dissection lives in error_analysis.py.

The price-band table reports the miss in dollars AND as a share of each car's own
price, because the two disagree and the disagreement is the point: a $2,000 miss on a
$5,000 car is 40% wrong, while the same miss on a $50,000 car is 4% wrong. Measured
only in dollars, "expensive cars are harder to predict" is almost guaranteed and tells
the reader nearly nothing.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import config as cfg
import error_analysis as ea

PRED_PREFIX = "pred::"


def load_predictions() -> tuple[pd.DataFrame, dict]:
    df = pd.read_parquet(cfg.PRED_FILE)
    meta = json.loads((cfg.TAB_DIR / "run_metadata.json").read_text())
    return df, meta


def model_names(df: pd.DataFrame) -> list[str]:
    return [c[len(PRED_PREFIX):] for c in df.columns if c.startswith(PRED_PREFIX)]


def scores(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    """The four numbers every comparison in this study rests on."""
    error = actual - predicted
    abs_error = error.abs()
    return {
        # average size of the miss, in dollars
        "MAE": float(abs_error.mean()),
        # like MAE but punishes occasional very bad misses far more harshly
        "RMSE": float(np.sqrt((error ** 2).mean())),
        # the typical miss expressed as a share of what the car actually asks
        "MdAPE_pct": float((abs_error / actual * 100).median()),
        # positive means the model priced cars too low on average
        "MeanResidual": float(error.mean()),
    }


def overall_table(df: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """One row per model, across all held-back cars."""
    actual = df[cfg.TARGET]
    rows = []
    for name in names:
        pred = df[PRED_PREFIX + name]
        s = scores(actual, pred)
        sse = ((actual - pred) ** 2).sum()
        sst = ((actual - actual.mean()) ** 2).sum()
        # share of the price variation the model accounts for; 0 means no better than
        # always guessing the average price
        s["R2"] = float(1 - sse / sst)
        rows.append({"Model": name, **s})
    return pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)


def assign_bands(prices: pd.Series, cuts: list[float]) -> pd.Series:
    """Label each car cheap, mid-range or expensive using TRAINING cut-offs."""
    return pd.cut(prices, bins=[-np.inf, cuts[0], cuts[1], np.inf],
                  labels=cfg.BAND_LABELS)


def band_table(df: pd.DataFrame, names: list[str], cuts: list[float]) -> pd.DataFrame:
    """Accuracy per price band.

    R-squared is deliberately absent. Slicing by price squeezes the price variation
    inside each slice, so the yardstick R-squared compares against becomes trivially
    easy to match and the number collapses toward zero for every model in every band -
    it would mislead rather than inform.
    """
    bands = assign_bands(df[cfg.TARGET], cuts)
    rows = []
    for name in names:
        for band in cfg.BAND_LABELS:
            m = bands == band
            s = scores(df.loc[m, cfg.TARGET], df.loc[m, PRED_PREFIX + name])
            rows.append({"Model": name, "Price band": band, "Cars": int(m.sum()), **s})
    return pd.DataFrame(rows)


def main() -> None:
    cfg.ensure_dirs()
    df, meta = load_predictions()
    names = model_names(df)
    cuts, threshold = meta["band_cuts"], meta["large_error_threshold"]

    assert df[cfg.TARGET].notna().all(), "a held-back car has no actual asking price"
    print(f"Scoring {len(df):,} held-back cars across {len(names)} models\n")

    overall = overall_table(df, names)
    overall.to_csv(cfg.TAB_DIR / "table3_overall_metrics.csv", index=False)
    print("OVERALL ACCURACY")
    print(overall.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))

    best = overall.iloc[0]["Model"]
    floor = overall.loc[overall.Model == "Median guess", "MAE"].iloc[0]
    print(f"\nBest model: {best}")
    for _, r in overall.iterrows():
        if r.Model != "Median guess" and r.MAE >= floor:
            print(f"  WARNING: {r.Model} failed to beat the median-guess floor")

    bands = band_table(df, names, cuts)
    bands.to_csv(cfg.TAB_DIR / "table4_band_metrics.csv", index=False)
    print("\nACCURACY BY PRICE BAND")
    print(bands.to_string(index=False, float_format=lambda v: f"{v:,.1f}"))

    pipe = ea.load_best_pipeline(best)
    kept = ea.kept_model_names(pipe) if pipe is not None else None
    if kept is None:
        print("\n(model grouping unavailable - skipping the uncommon-model row)")

    profile = ea.large_error_profile(df, best, threshold, kept)
    profile.to_csv(cfg.TAB_DIR / "table5_large_error_profile.csv", index=False)
    print(f"\nWHAT THE WORST-PREDICTED CARS HAVE IN COMMON "
          f"(a miss above ${threshold:,.0f})")
    print(ea.format_profile(profile).to_string(index=False))

    examples, extreme = ea.extreme_misses(df, best, threshold)
    examples.to_csv(cfg.TAB_DIR / "table9_extreme_misses.csv", index=False)
    (cfg.TAB_DIR / "extreme_stats.json").write_text(json.dumps(extreme, indent=2))
    print("\nTHE EIGHT LARGEST MISSES")
    print(examples.to_string(index=False))
    print(f"  prices that look typed : {extreme['junk_price_cars']} cars "
          f"({extreme['junk_price_share_all']*100:.3f}% of held-back cars, "
          f"{extreme['junk_price_share_big']*100:.2f}% of large errors)")
    print(f"  asking over ${extreme['dear_threshold']:,}  : "
          f"{extreme['dear_share_big']*100:.1f}% of large errors against "
          f"{extreme['dear_share_all']*100:.1f}% overall")

    imp = ea.importance_table(best, df)
    imp.to_csv(cfg.TAB_DIR / "table6_permutation_importance.csv", index=False)
    print(f"\nWHAT {best.upper()} RELIES ON")
    print(imp.to_string(index=False, float_format=lambda v: f"{v:,.1f}"))

    # Recorded here because the pipeline is already loaded. Without it the document
    # build would have to decompress the 530 MB forest purely to count these names.
    (cfg.TAB_DIR / "best_model.json").write_text(json.dumps(
        {"best": best, "models_kept": len(kept) if kept else 0}, indent=2))


if __name__ == "__main__":
    main()
