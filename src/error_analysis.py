"""The two error analyses, which answer two different questions.

Objective 5 - what do the badly predicted cars have in common?
    Split the held-back cars into those missed by more than the pre-set threshold and
    everyone else, then compare the two groups characteristic by characteristic.

Objective 6 - what does the model lean on?
    Scramble one column at a time and record how much accuracy worsens.

These are NOT the same question, and conflating them is a common mistake. A model can
lean heavily on age while its worst misses are concentrated among cars that never
stated a condition - scrambling columns would never reveal that.
"""
from __future__ import annotations

import joblib
import pandas as pd
from sklearn.inspection import permutation_importance

import config as cfg

PRED_PREFIX = "pred::"
IMPORTANCE_SAMPLE = 10_000   # keeps the repeated predicting affordable
IMPORTANCE_REPEATS = 5


def large_error_profile(df: pd.DataFrame, best: str, threshold: float,
                        kept_models: set | None) -> pd.DataFrame:
    """Compare the worst-predicted cars against all the rest.

    Returns raw numbers, not formatted text, so the table and the chart built from it
    can never disagree. "Kind" records how each row should be displayed: a share of
    cars, a plain average, or an amount of money.

    The threshold was fixed from training data before any test result was seen, so this
    comparison cannot be tuned to produce a tidier story.
    """
    big = (df[cfg.TARGET] - df[PRED_PREFIX + best]).abs() > threshold

    specs: list[tuple[str, pd.Series, str]] = [
        ("Condition not stated", df["condition"].isna(), "share"),
        ("Drive type not stated", df["drive"].isna(), "share"),
        ("Body type not stated", df["type"].isna(), "share"),
        ("Cylinders not stated", df["cylinders"].isna(), "share"),
        ("Salvage or rebuilt title",
         df["title_status"].isin(["salvage", "rebuilt"]), "share"),
        ("Average age (years)", df["age"], "number"),
        ("Average mileage", df["odometer"], "number"),
        ("Average asking price", df[cfg.TARGET], "money"),
    ]
    if kept_models is not None:
        rare = ~df["model"].astype("string").str.lower().str.strip().isin(kept_models)
        specs.insert(4, ("Uncommon model (pooled as Other)", rare, "share"))

    rows = [{"Characteristic": "Number of cars", "Kind": "count",
             "Large-error cars": float(big.sum()),
             "All other cars": float((~big).sum())}]
    for label, series, kind in specs:
        rows.append({"Characteristic": label, "Kind": kind,
                     "Large-error cars": float(series[big].mean()),
                     "All other cars": float(series[~big].mean())})

    out = pd.DataFrame(rows)
    out["Difference"] = out["Large-error cars"] - out["All other cars"]
    return out


def format_profile(t: pd.DataFrame) -> pd.DataFrame:
    """Render the numeric profile for human eyes."""
    def row(r):
        a, b, d = r["Large-error cars"], r["All other cars"], r["Difference"]
        if r["Kind"] == "count":
            return f"{a:,.0f}", f"{b:,.0f}", ""
        if r["Kind"] == "share":
            return f"{a*100:.1f}%", f"{b*100:.1f}%", f"{d*100:+.1f} pts"
        if r["Kind"] == "money":
            return f"${a:,.0f}", f"${b:,.0f}", f"${d:+,.0f}"
        return f"{a:,.1f}", f"{b:,.1f}", f"{d:+,.1f}"

    rendered = t.apply(row, axis=1, result_type="expand")
    rendered.columns = ["Large-error cars", "All other cars", "Difference"]
    return pd.concat([t[["Characteristic"]], rendered], axis=1)


def extreme_misses(df: pd.DataFrame, best: str, threshold: float, top: int = 8
                   ) -> tuple[pd.DataFrame, dict]:
    """Describe the very worst predictions, and what kind of car they are.

    Two things turn up at the extreme. Genuine exotics, which the model has too few
    examples of and so prices towards the ordinary, and adverts whose price was clearly
    typed rather than meant. Separating them matters: one is a limit of the model, the
    other a limit of the data.
    """
    err = df[cfg.TARGET] - df[PRED_PREFIX + best]
    work = df.assign(_abs=err.abs(), _err=err)
    big = work[work._abs > threshold]

    junk_all = work[cfg.TARGET].map(cfg.is_placeholder_price)
    junk_big = big[cfg.TARGET].map(cfg.is_placeholder_price)
    stats = {
        "junk_price_cars": int(junk_all.sum()),
        "junk_price_share_all": float(junk_all.mean()),
        "junk_price_share_big": float(junk_big.mean()),
        "junk_prices_found": sorted({int(p) for p in
                                     work.loc[junk_all, cfg.TARGET].unique()}),
        "dear_share_big": float((big[cfg.TARGET] > cfg.DEAR_PRICE).mean()),
        "dear_share_all": float((work[cfg.TARGET] > cfg.DEAR_PRICE).mean()),
        "underpriced_share_big": float((big._err > 0).mean()),
        "dear_threshold": cfg.DEAR_PRICE,
    }

    worst = work.reindex(work._abs.sort_values(ascending=False).index).head(top)
    examples = pd.DataFrame({
        "Make and model": [f"{r.manufacturer} {r.model}".title()
                           for _, r in worst.iterrows()],
        "Age": worst.age.map(lambda v: f"{v:.0f}y"),
        "Mileage": worst.odometer.map("{:,.0f}".format),
        "Advertised": worst[cfg.TARGET].map("${:,.0f}".format),
        "Predicted": worst[PRED_PREFIX + best].map("${:,.0f}".format),
        "Miss": worst._abs.map("${:,.0f}".format),
        "Price looks typed": worst[cfg.TARGET].map(
            lambda p: "yes" if cfg.is_placeholder_price(p) else ""),
    })
    return examples, stats


def load_best_pipeline(best: str):
    """Load the fitted pipeline saved by train.py, or None if it is missing."""
    path = cfg.MODEL_DIR / f"{best.replace(' ', '_').lower()}.joblib"
    try:
        return joblib.load(path)
    except (FileNotFoundError, EOFError):
        return None


def kept_model_names(pipe) -> set | None:
    """Which model names survived the rare-category grouping during training."""
    try:
        return pipe.named_steps["prep"].named_steps["group_rare"].keep_
    except (AttributeError, KeyError):
        return None


def importance_table(best: str, df: pd.DataFrame) -> pd.DataFrame:
    """Shuffle one column at a time and see how much the average miss worsens.

    A bigger jump means the model relies on that column more. It says nothing about
    cause: scrambling mileage hurts the model, but that does not prove that changing a
    car's mileage would change its asking price.
    """
    pipe = load_best_pipeline(best)
    if pipe is None:
        raise FileNotFoundError(f"no saved model for {best}; run train.py first")
    sample = df.sample(n=min(IMPORTANCE_SAMPLE, len(df)), random_state=cfg.RANDOM_SEED)
    result = permutation_importance(
        pipe, sample[cfg.FEATURES], sample[cfg.TARGET],
        scoring="neg_mean_absolute_error", n_repeats=IMPORTANCE_REPEATS,
        random_state=cfg.RANDOM_SEED, n_jobs=1,
    )
    return (pd.DataFrame({"Feature": cfg.FEATURES,
                          "MAE increase when shuffled": result.importances_mean,
                          "Std dev": result.importances_std})
            .sort_values("MAE increase when shuffled", ascending=False)
            .reset_index(drop=True))
