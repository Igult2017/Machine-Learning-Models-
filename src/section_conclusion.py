"""Prose for "Conclusion and Recommendations".

Split out of section_discussion.py so each file keeps one job and stays within the
project line limit. Every figure quoted here is read from the result tables at build
time, so the conclusion cannot claim more than the run produced.
"""
from __future__ import annotations


def conclusion_blocks(F: dict) -> list[tuple]:
    best = F["best"]
    return [
        ("topic", f"Machine learning predicts used-car asking prices for unseen cars "
                  f"to within {F['best_mdape']:.1f}% of the asking price typically, "
                  f"and its accuracy is uneven across the market."),
        ("bullets", [
            f"{best} was the most accurate of the four, missing by "
            f"${F['best_mae']:,.0f} on average and accounting for "
            f"{F['best_r2']*100:.1f}% of the variation in asking prices across "
            f"{F['n_test']:,} cars it had never seen.",
            f"Both tree-based models clearly beat linear regression "
            f"(${F['linear_mae']:,.0f}), and all three beat the floor of "
            f"${F['floor_mae']:,.0f}.",
            f"Errors are largest in dollars for {F['dollar_worst'].lower()}-priced "
            f"cars and largest in proportion for {F['pct_worst'].lower()}-priced ones, "
            f"so neither measure alone describes how useful the model is.",
            "The errors also have a consistent direction within each band, meaning "
            "they are partly systematic rather than purely random.",
        ]),

        ("topic", "Five recommendations follow for anyone building or publishing a "
                  "price model on listing data."),
        ("bullets", [
            "Report errors by price band, in both dollars and as a share of price. A "
            "single overall figure hides which customers the model fails.",
            "Remove repeat adverts at the vehicle level before splitting the data, and "
            "state the method used - otherwise the published accuracy is not "
            "comparable with anyone else's.",
            "Report the direction of the errors, not only their size, so users know "
            "whether estimates lean high or low for their kind of car.",
            "Record missing information as its own category rather than filling it in, "
            "because the absence itself carries signal.",
            "Describe the output as an estimate of an advertised price, never as a "
            "valuation, so users do not read more into it than the data supports.",
        ]),

        ("topic", "Four extensions would test these conclusions further."),
        ("bullets", [
            "Repeating the analysis on data spanning several years would show whether "
            "accuracy decays as the market moves, which this snapshot cannot.",
            "Adding the free-text description, excluded here, would test whether what "
            "sellers write recovers some of the missing condition information.",
            "Comparing advertised prices against completed sale prices, where such "
            "data can be obtained, would establish how far an asking-price model can "
            "stand in for a valuation.",
            "Fitting a separate model per price band would test whether the uneven "
            "accuracy found here is a limitation of using one model across the whole "
            "market.",
        ]),
    ]
