"""Two checks that test whether the headline findings survive different choices.

Check 1 - what repeat adverts do to the score.
    The cleaning step removed 137,525 repeat adverts for cars already in the data. This
    re-runs the best model on the data WITHOUT that step, so the same physical car can
    sit in both the training and the test set. The gap between the two scores is the
    amount by which a study that skipped de-duplication would have flattered itself.
    This is reported as a finding, not a footnote.

Check 2 - predicting the logarithm of the price instead of the price.
    Asking prices are lopsided: many cheap cars, a few very expensive ones. Working with
    the logarithm evens that out, which usually helps. But it also quietly reduces the
    penalty for being wrong about expensive cars - the very effect this study sets out
    to measure. So it is run as a check, not as the headline, and the misses are
    converted back into dollars before scoring.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import clean
import config as cfg
import evaluate as ev
import features as ft
import train as tr

CHECK_MODEL = "Gradient Boosting"


def _fit_and_score(X_train, y_train, X_test, y_test, log_target: bool = False):
    """Fit the check model and return its per-car predictions in dollars."""
    pipe = tr.build_models()[CHECK_MODEL]
    if log_target:
        pipe.fit(X_train, np.log(y_train))
        pred = np.exp(pipe.predict(X_test))
    else:
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
    return pd.Series(pred, index=X_test.index)


def _split(df: pd.DataFrame):
    X, y = ft.split_xy(df)
    itr, ite = train_test_split(df.index, test_size=cfg.TEST_SIZE,
                                random_state=cfg.RANDOM_SEED)
    return X.loc[itr], X.loc[ite], y.loc[itr], y.loc[ite]


def check_deduplication() -> dict:
    """Score the model with and without repeat adverts removed."""
    print("Check 1: the cost of leaving repeat adverts in")

    deduped = ft.load_clean()
    Xtr, Xte, ytr, yte = _split(deduped)
    pred_clean = _fit_and_score(Xtr, ytr, Xte, yte)
    clean_scores = ev.scores(yte, pred_clean)
    print(f"  de-duplicated   : {len(deduped):,} cars, "
          f"average miss ${clean_scores['MAE']:,.0f}")

    raw = clean.load_raw()
    raw, _ = clean.apply_quality_rules(raw)
    raw = clean.add_age(raw).reset_index(drop=True)
    Xtr2, Xte2, ytr2, yte2 = _split(raw)
    pred_dirty = _fit_and_score(Xtr2, ytr2, Xte2, yte2)
    dirty_scores = ev.scores(yte2, pred_dirty)
    print(f"  repeats left in : {len(raw):,} cars, "
          f"average miss ${dirty_scores['MAE']:,.0f}")

    inflation = clean_scores["MAE"] - dirty_scores["MAE"]
    pct = inflation / clean_scores["MAE"] * 100
    print(f"  leaving repeats in makes the model look ${inflation:,.0f} more accurate "
          f"({pct:.1f}% better than it really is)")
    return {
        "deduplicated_rows": int(len(deduped)),
        "deduplicated_mae": clean_scores["MAE"],
        "repeats_left_rows": int(len(raw)),
        "repeats_left_mae": dirty_scores["MAE"],
        "apparent_improvement_usd": inflation,
        "apparent_improvement_pct": pct,
    }


def check_log_target(cuts: list[float]) -> pd.DataFrame:
    """Compare raw-price and log-price training, per price band, in dollars."""
    print("\nCheck 2: predicting the logarithm of the price instead")
    df = ft.load_clean()
    Xtr, Xte, ytr, yte = _split(df)

    rows = []
    for label, use_log in [("Trained on price", False), ("Trained on log price", True)]:
        pred = _fit_and_score(Xtr, ytr, Xte, yte, log_target=use_log)
        bands = ev.assign_bands(yte, cuts)
        overall = ev.scores(yte, pred)
        rows.append({"Version": label, "Price band": "All cars",
                     "Cars": len(yte), **overall})
        for band in cfg.BAND_LABELS:
            m = bands == band
            rows.append({"Version": label, "Price band": band, "Cars": int(m.sum()),
                         **ev.scores(yte[m], pred[m])})
        print(f"  {label:22s} average miss ${overall['MAE']:,.0f}, "
              f"typical miss {overall['MdAPE_pct']:.1f}% of price")
    return pd.DataFrame(rows)


def main() -> None:
    cfg.ensure_dirs()
    meta = json.loads((cfg.TAB_DIR / "run_metadata.json").read_text())
    cuts = meta["band_cuts"]

    dedup = check_deduplication()
    (cfg.TAB_DIR / "table7_dedup_effect.json").write_text(json.dumps(dedup, indent=2))

    log_table = check_log_target(cuts)
    log_table.to_csv(cfg.TAB_DIR / "table8_log_target.csv", index=False)
    print("\n" + log_table.to_string(index=False, float_format=lambda v: f"{v:,.1f}"))


if __name__ == "__main__":
    main()
