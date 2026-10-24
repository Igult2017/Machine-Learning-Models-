"""Prose for the two robustness checks in "Data Analysis and Results".

Check 1 asks what repeat adverts cost; check 2 asks whether predicting the logarithm of
the price changes the answer. Both are written from the measured result, in whichever
direction it came out - the de-duplication check in particular came out opposite to the
expectation, and says so.
"""
from __future__ import annotations


def _dedup_blocks(F: dict) -> list[tuple]:
    """Report the de-duplication check in whichever direction it actually came out."""
    gap = abs(F["dirty_mae"] - F["clean_mae"])
    if F["dirty_worse"]:
        topic = ("Removing repeat adverts barely moved the headline score, and moved "
                 "it in the opposite direction to the obvious expectation.")
        bullets = [
            f"With repeat adverts removed the model misses by ${F['clean_mae']:,.0f}; "
            f"with them left in it misses by ${F['dirty_mae']:,.0f}. Leaving them in "
            f"made the model look ${gap:,.0f} WORSE, not better.",
            "The expectation was the reverse, because a car present in both the "
            "training and the testing data can be recalled rather than predicted.",
            f"Two effects pull against each other. Leaving repeats in should flatter "
            f"the score, but it also adds {F['dedup_removed']:,} rows dominated by "
            f"heavily re-posted cars, and those appear to be harder to price - a car "
            f"advertised 261 times is quite likely mis-priced. The second effect "
            f"outweighed the first here.",
            "The comparison therefore cannot isolate the recall effect on its own: the "
            "two runs differ in size, in composition and in overlap all at once. It is "
            "reported as measured rather than presented as a clean estimate.",
            "De-duplication remains the correct choice regardless of which way the "
            "number moves. A score measured partly on cars the model has already seen "
            "does not answer the research question, which asks about unseen vehicles.",
            "The useful finding is a different one: the repeatedly re-posted cars are "
            "systematically unlike the rest, so any study using this data inherits "
            "whatever those adverts are, and should say how it handled them.",
        ]
    else:
        topic = ("Leaving repeat adverts in the data would have overstated accuracy "
                 f"by ${gap:,.0f} per car.")
        bullets = [
            f"Trained and tested without removing repeat adverts, the same model "
            f"appears to miss by only ${F['dirty_mae']:,.0f} rather than "
            f"${F['clean_mae']:,.0f}.",
            f"That is an apparent improvement of {abs(F['dedup_pct']):.1f}% "
            f"reflecting no real skill, only cars already shown to the model.",
            "Because ordinary duplicate removal catches just 37 of the 426,880 rows, "
            "a study relying on it would publish the inflated figure in good faith.",
        ]
    bullets.append("Both sides use the same untuned settings, so they are comparable "
                   "with each other even though neither matches Table 2 exactly.")
    return [("topic", topic), ("bullets", bullets),
            ("figure", "fig15_dedup_effect.png",
             "Figure 14. Average miss with and without repeat adverts removed.")]


def _log_blocks(F: dict) -> list[tuple]:
    """The target transformation barely moves the overall score but reshapes the bands."""
    rp, lp = F["raw_band_pct"], F["log_band_pct"]
    rm, lm = F["raw_band_mae"], F["log_band_mae"]
    return [
        ("topic", "Predicting the logarithm of the price instead of the price leaves "
                  "overall accuracy almost unchanged but materially reshapes which "
                  "cars are predicted well."),
        ("bullets", [
            f"Overall the two are close: an average miss of ${F['log_mae']:,.0f} "
            f"working in logarithms against ${F['raw_mae']:,.0f} working in dollars, "
            f"and a typical miss of {F['log_mdape']:.1f}% against "
            f"{F['raw_mdape']:.1f}% of price.",
            f"Band by band they differ sharply. Cheap cars improve from "
            f"{rp['Lower']:.1f}% to {lp['Lower']:.1f}% of price (${rm['Lower']:,.0f} "
            f"to ${lm['Lower']:,.0f}), while expensive cars worsen from "
            f"{rp['Higher']:.1f}% to {lp['Higher']:.1f}% (${rm['Higher']:,.0f} to "
            f"${lm['Higher']:,.0f}).",
            f"The unevenness across bands shrinks from {F['raw_band_spread']:.1f} "
            f"percentage points to {F['log_band_spread']:.1f}. Working in logarithms "
            f"penalises proportional error, so it buys accuracy on cheap cars by "
            f"giving it up on expensive ones - exactly the trade predicted before the "
            f"check was run.",
            "This is the most important thing the check shows: part of the uneven "
            "accuracy is a consequence of what the model is asked to minimise, not "
            "only a property of the market. The direction of the headline finding "
            "survives, but its size does not.",
            "The plain-dollar version was kept as the headline because it is directly "
            "interpretable, and because converting logged predictions back into "
            "dollars introduces a bias of its own.",
            "A study reporting accuracy on this kind of data should therefore state "
            "which quantity it trained on, as plainly as it states its metrics.",
        ]),
        ("table", "log_check",
         "Table 8. Predicting the price against predicting its logarithm."),
    ]


def blocks(F: dict) -> list[tuple]:
    """Both checks, in the order they are discussed in the thesis."""
    return _dedup_blocks(F) + _log_blocks(F)
