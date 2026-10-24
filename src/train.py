"""Split the cars, search for good settings, fit the four models, save predictions.

Order of operations matters and is deliberate:

1. Split first. 80% of cars are used for everything that follows; the remaining 20% are
   locked away and touched exactly once, at the very end.
2. Search for model settings using cross-validation (repeatedly holding back a slice of
   the TRAINING data) on a 50,000-car subsample, for speed. The winning settings are
   then refitted on the whole training set.
3. Only then predict the held-back test cars.

Nothing in steps 1-2 ever sees the test prices. That is what makes the final score an
honest estimate of performance on cars the model has never met.
"""
from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline

import config as cfg
import features as ft

SEARCH_SPACES = {
    "Random Forest": {
        "regressor__n_estimators": [100, 200, 300],
        "regressor__max_depth": [10, 20, 30, None],
        "regressor__min_samples_leaf": [1, 2, 5, 10],
        "regressor__max_features": ["sqrt", 0.5, 1.0],
    },
    "Gradient Boosting": {
        "regressor__learning_rate": [0.03, 0.05, 0.1, 0.2],
        "regressor__max_iter": [200, 400, 600],
        "regressor__max_depth": [None, 4, 6, 8],
        "regressor__min_samples_leaf": [10, 20, 50],
        "regressor__max_leaf_nodes": [15, 31, 63],
    },
}


def build_models() -> dict[str, Pipeline]:
    """The four competitors, each with the preparation its own family needs."""
    seed = cfg.RANDOM_SEED
    return {
        # Predicts the same price for every car - the floor any real model must beat.
        # The median is used because the score being optimised is the average miss.
        "Median guess": Pipeline([
            ("regressor", DummyRegressor(strategy="median")),
        ]),
        "Linear Regression": Pipeline([
            ("prep", ft.build_preprocessor(one_hot=True)),
            ("regressor", LinearRegression()),
        ]),
        "Random Forest": Pipeline([
            ("prep", ft.build_preprocessor(one_hot=False)),
            ("regressor", RandomForestRegressor(random_state=seed, n_jobs=-1)),
        ]),
        "Gradient Boosting": Pipeline([
            ("prep", ft.build_preprocessor(one_hot=False)),
            ("regressor", HistGradientBoostingRegressor(random_state=seed)),
        ]),
    }


def make_split(df: pd.DataFrame):
    """Hold back 20% of cars, then prove no car sits on both sides of the line."""
    X, y = ft.split_xy(df)
    idx_train, idx_test = train_test_split(
        df.index, test_size=cfg.TEST_SIZE, random_state=cfg.RANDOM_SEED
    )
    vin_train = set(df.loc[idx_train, cfg.DEDUP_KEY].dropna())
    vin_test = set(df.loc[idx_test, cfg.DEDUP_KEY].dropna())
    overlap = vin_train & vin_test
    assert not overlap, (
        f"{len(overlap):,} cars appear in both training and test data - the "
        "de-duplication step failed and every score below would be inflated"
    )
    print(f"Split: {len(idx_train):,} training cars / {len(idx_test):,} test cars")
    print(f"Cars appearing on both sides: {len(overlap)} (must be 0)")
    return X.loc[idx_train], X.loc[idx_test], y.loc[idx_train], y.loc[idx_test]


def tune(name: str, pipe: Pipeline, X_train: pd.DataFrame,
         y_train: pd.Series) -> tuple[Pipeline, dict]:
    """Search settings on a subsample, then refit the winner on all training cars.

    Returns the configured pipeline and just the settings that were chosen - not every
    default - so the record of what was run stays readable and re-usable.
    """
    if name not in SEARCH_SPACES:
        print(f"  {name}: nothing to tune")
        return pipe, {}

    n = min(cfg.TUNE_SUBSAMPLE, len(X_train))
    sub = X_train.sample(n=n, random_state=cfg.RANDOM_SEED)
    search = RandomizedSearchCV(
        pipe,
        SEARCH_SPACES[name],
        n_iter=cfg.TUNE_CANDIDATES,
        cv=cfg.CV_FOLDS,
        scoring="neg_mean_absolute_error",
        random_state=cfg.RANDOM_SEED,
        n_jobs=1,          # the forest itself is already parallel; avoids memory blow-up
        refit=False,
    )
    t0 = time.time()
    search.fit(sub, y_train.loc[sub.index])
    best = {k.replace("regressor__", ""): v for k, v in search.best_params_.items()}
    print(f"  {name}: searched {cfg.TUNE_CANDIDATES} settings on {n:,} cars "
          f"in {time.time()-t0:.0f}s")
    print(f"    best (cross-validated average miss ${-search.best_score_:,.0f}): {best}")
    pipe.set_params(**search.best_params_)
    best["_cv_mae"] = round(-search.best_score_, 1)
    return pipe, best


def main() -> None:
    cfg.ensure_dirs()
    df = ft.load_clean()
    X_train, X_test, y_train, y_test = make_split(df)

    band_cuts = [float(y_train.quantile(q)) for q in cfg.BAND_QUANTILES]
    print(f"Price bands from TRAINING prices only: "
          f"<= ${band_cuts[0]:,.0f} / ${band_cuts[0]:,.0f}-${band_cuts[1]:,.0f} "
          f"/ > ${band_cuts[1]:,.0f}")

    out = X_test.copy()
    out[cfg.TARGET] = y_test
    # carried so a badly predicted car can be traced back to its original advert
    out[cfg.ID_COL] = df.loc[X_test.index, cfg.ID_COL]
    best_params, timings, fitted = {}, {}, {}

    print("\nTuning and fitting")
    for name, pipe in build_models().items():
        pipe, chosen = tune(name, pipe, X_train, y_train)
        t0 = time.time()
        pipe.fit(X_train, y_train)
        timings[name] = round(time.time() - t0, 1)
        fitted[name] = pipe
        out[f"pred::{name}"] = pipe.predict(X_test)
        best_params[name] = chosen
        print(f"  {name}: fitted on {len(X_train):,} cars in {timings[name]}s")

    # The "large error" line is drawn using TRAINING data only, before any test result
    # is looked at, so it cannot be chosen to flatter the findings.
    print("\nSetting the large-error threshold from training data (cross-validated)")
    cv_pred = cross_val_predict(fitted["Gradient Boosting"], X_train, y_train,
                                cv=cfg.CV_FOLDS, n_jobs=1)
    cv_abs_err = np.abs(y_train.to_numpy() - cv_pred)
    threshold = float(np.quantile(cv_abs_err, cfg.LARGE_ERROR_QUANTILE))
    print(f"  a miss above ${threshold:,.0f} counts as a large error "
          f"({cfg.LARGE_ERROR_QUANTILE:.0%} of training misses are smaller)")

    out.to_parquet(cfg.PRED_FILE, index=False)
    for name, pipe in fitted.items():
        slug = name.replace(" ", "_").lower()
        joblib.dump(pipe, cfg.MODEL_DIR / f"{slug}.joblib", compress=3)

    meta = {
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "band_cuts": band_cuts,
        "large_error_threshold": threshold,
        "fit_seconds": timings,
        "best_params": best_params,
        "source": cfg.SOURCE,
        "source_md5": cfg.SOURCE_MD5,
    }
    (cfg.TAB_DIR / "run_metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"\nSaved test predictions for {len(out):,} cars to {cfg.PRED_FILE.name}")


if __name__ == "__main__":
    main()
