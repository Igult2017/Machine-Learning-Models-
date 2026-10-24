"""Reduce every result file to the handful of numbers the written sections need.

Kept apart from report.py so that assembling the numbers and rendering the document are
separate jobs. Everything here is read from disk at build time, which is what stops the
written text from drifting away from the run that produced it.
"""
from __future__ import annotations

import json

import pandas as pd

import config as cfg
import error_analysis as ea
import report_tables


def _profile_value(profile: pd.DataFrame, label: str, column: str) -> float:
    """Pull one cell out of the large-error comparison, or NaN if the row is absent."""
    hit = profile.loc[profile.Characteristic == label, column]
    return float(hit.iloc[0]) if len(hit) else float("nan")


def gather() -> tuple[dict, dict[str, pd.DataFrame]]:
    """Return the facts dictionary used by the prose, and the tables to embed."""
    t = cfg.TAB_DIR
    funnel = pd.read_csv(t / "table1_cleaning_funnel.csv")
    overall = pd.read_csv(t / "table3_overall_metrics.csv")
    bands = pd.read_csv(t / "table4_band_metrics.csv")
    profile = pd.read_csv(t / "table5_large_error_profile.csv")
    imp = pd.read_csv(t / "table6_permutation_importance.csv")
    meta = json.loads((t / "run_metadata.json").read_text())
    best_info = json.loads((t / "best_model.json").read_text())
    best = best_info["best"]
    dedup = json.loads((t / "table7_dedup_effect.json").read_text())

    log_table = pd.read_csv(t / "table8_log_target.csv")
    gaps = pd.read_csv(t / "table2_missingness.csv")
    extremes = pd.read_csv(t / "table9_extreme_misses.csv")
    extreme_stats = json.loads((t / "extreme_stats.json").read_text())

    clean = pd.read_parquet(cfg.CLEAN_FILE)
    pred = pd.read_parquet(cfg.PRED_FILE)
    price = clean[cfg.TARGET]

    row = overall.set_index("Model")
    best_row = row.loc[best]
    floor = float(row.loc["Median guess", "MAE"])
    bb = bands[bands.Model == best].set_index("Price band")
    mae_by_band = {b: float(bb.loc[b, "MAE"]) for b in cfg.BAND_LABELS}
    mdape_by_band = {b: float(bb.loc[b, "MdAPE_pct"]) for b in cfg.BAND_LABELS}
    residual_by_band = {b: float(bb.loc[b, "MeanResidual"]) for b in cfg.BAND_LABELS}

    log_rows = log_table.set_index(["Version", "Price band"])
    log_mae = float(log_rows.loc[("Trained on log price", "All cars"), "MAE"])
    raw_mae = float(log_rows.loc[("Trained on price", "All cars"), "MAE"])
    log_mdape = float(log_rows.loc[("Trained on log price", "All cars"), "MdAPE_pct"])
    raw_mdape = float(log_rows.loc[("Trained on price", "All cars"), "MdAPE_pct"])

    def _per_band(version: str, column: str) -> dict[str, float]:
        return {b: float(log_rows.loc[(version, b), column]) for b in cfg.BAND_LABELS}

    log_band_pct = _per_band("Trained on log price", "MdAPE_pct")
    raw_band_pct = _per_band("Trained on price", "MdAPE_pct")
    log_band_mae = _per_band("Trained on log price", "MAE")
    raw_band_mae = _per_band("Trained on price", "MAE")

    threshold = meta["large_error_threshold"]
    big = (pred[cfg.TARGET] - pred[f"pred::{best}"]).abs() > threshold

    # evaluate.py records this while it has the pipeline open. Falling back to loading
    # the saved model costs several minutes, so it is only a safety net.
    models_kept = best_info.get("models_kept")
    if models_kept is None:
        pipe = ea.load_best_pipeline(best)
        kept = ea.kept_model_names(pipe) if pipe is not None else None
        models_kept = len(kept) if kept else 0

    after_quality = int(funnel.iloc[5]["Rows remaining"])
    dollar_sorted = sorted(mae_by_band, key=mae_by_band.get)
    pct_sorted = sorted(mdape_by_band, key=mdape_by_band.get)

    # Rank what actually separates the badly predicted cars, biggest gap first, so the
    # written text describes the real pattern rather than an expected one.
    shares = profile[profile.Kind == "share"].copy()
    shares["gap"] = shares["Difference"].abs()
    ranked = [{"label": r.Characteristic, "large": r["Large-error cars"],
               "rest": r["All other cars"], "diff": r["Difference"]}
              for _, r in shares.sort_values("gap", ascending=False).iterrows()]

    runner_up = overall.iloc[1]["Model"]
    imp_map = dict(zip(imp.Feature, imp["MAE increase when shuffled"]))
    missing_pct = dict(zip(gaps.Column, gaps["Percent blank"]))
    # how often each predictor is blank, looked up under the column it comes from
    feature_missing = {f: float(missing_pct.get(cfg.MISSING_SOURCE.get(f, f), 0.0))
                       for f in cfg.FEATURES}

    facts = {
        "source": cfg.SOURCE, "n_raw": cfg.RAW_ROWS, "n_clean": len(clean),
        "pct_kept": len(clean) / cfg.RAW_ROWS * 100,
        "quality_removed": cfg.RAW_ROWS - after_quality,
        "dedup_removed": after_quality - len(clean),
        "vin_repeats": int(funnel.iloc[6]["Removed by this step"]),
        "fp_repeats": int(funnel.iloc[7]["Removed by this step"]),
        "n_train": meta["n_train"], "n_test": meta["n_test"],
        "band_cuts": meta["band_cuts"], "large_threshold": threshold,
        "model_min": cfg.MODEL_MIN_COUNT, "models_kept": models_kept,
        "tune_candidates": cfg.TUNE_CANDIDATES, "tune_subsample": cfg.TUNE_SUBSAMPLE,
        "price_median": float(price.median()), "price_mean": float(price.mean()),
        "price_skew": float(price.skew()),
        "price_q25": float(price.quantile(0.25)),
        "price_q75": float(price.quantile(0.75)),
        "best": best, "floor_mae": floor,
        "best_mae": float(best_row["MAE"]), "best_rmse": float(best_row["RMSE"]),
        "best_r2": float(best_row["R2"]), "best_mdape": float(best_row["MdAPE_pct"]),
        "best_vs_floor_pct": (1 - float(best_row["MAE"]) / floor) * 100,
        "linear_mae": float(row.loc["Linear Regression", "MAE"]),
        "mae_by_band": mae_by_band, "mdape_by_band": mdape_by_band,
        "residual_by_band": residual_by_band,
        "log_mae": log_mae, "raw_mae": raw_mae,
        "log_mdape": log_mdape, "raw_mdape": raw_mdape,
        "log_better": log_mae < raw_mae, "log_gap": abs(log_mae - raw_mae),
        "log_band_pct": log_band_pct, "raw_band_pct": raw_band_pct,
        "log_band_mae": log_band_mae, "raw_band_mae": raw_band_mae,
        # the spread between the best- and worst-predicted band, as a share of price:
        # it shows how much the unevenness itself depends on what the model minimises
        "raw_band_spread": max(raw_band_pct.values()) - min(raw_band_pct.values()),
        "log_band_spread": max(log_band_pct.values()) - min(log_band_pct.values()),
        "dirty_worse": dedup["repeats_left_mae"] > dedup["deduplicated_mae"],
        "underpriced_bands": [b for b in cfg.BAND_LABELS if residual_by_band[b] > 0],
        "overpriced_bands": [b for b in cfg.BAND_LABELS if residual_by_band[b] < 0],
        "price_min": cfg.PRICE_MIN, "price_max": cfg.PRICE_MAX,
        "odo_min": cfg.ODO_MIN, "odo_max": cfg.ODO_MAX,
        "year_min": cfg.YEAR_MIN, "year_max": cfg.YEAR_MAX,
        "seed": cfg.RANDOM_SEED, "source_md5": cfg.SOURCE_MD5,
        "dollar_best": dollar_sorted[0], "dollar_worst": dollar_sorted[-1],
        "pct_best": pct_sorted[0], "pct_worst": pct_sorted[-1],
        "reversal": dollar_sorted[-1] != pct_sorted[-1],
        "big_n": int(big.sum()), "big_pct": float(big.mean() * 100),
        "big_cond_missing": _profile_value(profile, "Condition not stated",
                                           "Large-error cars"),
        "rest_cond_missing": _profile_value(profile, "Condition not stated",
                                            "All other cars"),
        "big_rare_model": _profile_value(profile, "Uncommon model (pooled as Other)",
                                         "Large-error cars"),
        "rest_rare_model": _profile_value(profile, "Uncommon model (pooled as Other)",
                                          "All other cars"),
        "big_price": _profile_value(profile, "Average asking price",
                                    "Large-error cars"),
        "rest_price": _profile_value(profile, "Average asking price", "All other cars"),
        "top_features": [cfg.label(f) for f in imp.Feature.head(3)],
        "top_importance": float(imp.iloc[0]["MAE increase when shuffled"]),
        "importance": imp_map, "missing_pct": missing_pct,
        "feature_missing": feature_missing, "label": cfg.label,
        "junk_price_cars": extreme_stats["junk_price_cars"],
        "junk_share_all": extreme_stats["junk_price_share_all"],
        "junk_share_big": extreme_stats["junk_price_share_big"],
        "junk_prices_found": extreme_stats["junk_prices_found"],
        "dear_share_big": extreme_stats["dear_share_big"],
        "dear_share_all": extreme_stats["dear_share_all"],
        "dear_threshold": extreme_stats["dear_threshold"],
        "underpriced_share_big": extreme_stats["underpriced_share_big"],
        "profile_ranked": ranked,
        "big_age": _profile_value(profile, "Average age (years)", "Large-error cars"),
        "rest_age": _profile_value(profile, "Average age (years)", "All other cars"),
        "big_odo": _profile_value(profile, "Average mileage", "Large-error cars"),
        "rest_odo": _profile_value(profile, "Average mileage", "All other cars"),
        "runner_up": runner_up,
        "runner_mae": float(row.loc[runner_up, "MAE"]),
        "runner_rmse": float(row.loc[runner_up, "RMSE"]),
        "runner_r2": float(row.loc[runner_up, "R2"]),
        "floor_r2": float(row.loc["Median guess", "R2"]),
        "rmse_winner": str(overall.loc[overall.RMSE.idxmin(), "Model"]),
        "r2_winner": str(overall.loc[overall.R2.idxmax(), "Model"]),
        "clean_mae": dedup["deduplicated_mae"], "dirty_mae": dedup["repeats_left_mae"],
        "dedup_inflation": dedup["apparent_improvement_usd"],
        "dedup_pct": dedup["apparent_improvement_pct"],
    }

    frames = {"funnel": funnel, "overall": overall, "bands": bands, "bb": bb,
              "profile": profile, "imp": imp, "log": log_table,
              "extremes": extremes}
    tables = report_tables.build(frames, feature_missing)
    return facts, tables
