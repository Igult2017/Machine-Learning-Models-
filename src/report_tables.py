"""Format the result tables for the Word document.

Separated from report_facts.py, which gathers the numbers the prose quotes. This file
only decides how figures are displayed: dollars with separators, percentages with one
decimal, column names in plain English rather than the dataset's own abbreviations.
"""
from __future__ import annotations

import pandas as pd

import config as cfg
import error_analysis as ea


def _money(value: float) -> str:
    return f"${value:,.0f}"


def _pct(value: float) -> str:
    return f"{value:.1f}%"


def build(frames: dict[str, pd.DataFrame], feature_missing: dict[str, float]
          ) -> dict[str, pd.DataFrame]:
    """Turn the raw result tables into display-ready ones, keyed by the name the
    section text refers to them by."""
    funnel, overall = frames["funnel"], frames["overall"]
    bands, bb = frames["bands"], frames["bb"]
    profile, imp, log_table = frames["profile"], frames["imp"], frames["log"]

    return {
        "funnel": pd.DataFrame({
            "Step": funnel.Step,
            "Adverts remaining": funnel["Rows remaining"].map("{:,}".format),
            "Removed by this step": funnel["Removed by this step"].map("{:,}".format),
            "Percent of raw kept": funnel["Percent of raw kept"].map(_pct),
        }),
        "overall": pd.DataFrame({
            "Model": overall.Model,
            "Average miss": overall.MAE.map(_money),
            "Miss (large errors weighted)": overall.RMSE.map(_money),
            "Variation explained": overall.R2.map(lambda v: _pct(v * 100)),
            "Typical miss vs price": overall.MdAPE_pct.map(_pct),
        }),
        "bands": pd.DataFrame({
            "Price band": bb.index,
            "Cars": bb.Cars.map("{:,}".format),
            "Average miss": bb.MAE.map(_money),
            "Typical miss vs price": bb.MdAPE_pct.map(_pct),
            "Priced below its advert by, on average":
                bb.MeanResidual.map(lambda v: f"${v:,.0f}"),
        }),
        "bands_all": pd.DataFrame({
            "Model": bands.Model,
            "Price band": bands["Price band"],
            "Cars": bands.Cars.map("{:,}".format),
            "Average miss": bands.MAE.map(_money),
            "Typical miss vs price": bands.MdAPE_pct.map(_pct),
        }),
        "profile": ea.format_profile(profile),
        # already formatted when it was written out, so it passes through unchanged
        "extremes": frames["extremes"],
        "importance": pd.DataFrame({
            "Characteristic": [cfg.label(f) for f in imp.Feature],
            "Added to the average miss when scrambled":
                imp["MAE increase when shuffled"].map(_money),
            "Blank in this share of adverts":
                [_pct(feature_missing.get(f, 0.0)) for f in imp.Feature],
        }),
        "log_check": pd.DataFrame({
            "Version": log_table.Version,
            "Price band": log_table["Price band"],
            "Average miss": log_table.MAE.map(_money),
            "Typical miss vs price": log_table.MdAPE_pct.map(_pct),
        }),
    }
