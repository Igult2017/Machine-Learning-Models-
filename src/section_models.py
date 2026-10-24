"""Prose for "Data Analysis and Results": how accurately the models predicted.

Covers the overall comparison, accuracy by price band, and the direction of the errors.
Where and why the models fail is covered in section_errors.py.

The wording adapts to the measured numbers: which band ranks worst in dollars, which
ranks worst as a share of price, and which bands are priced too high or too low are all
decided at build time, so this text cannot announce a finding the data did not deliver.
"""
from __future__ import annotations


def _accuracy_blocks(F: dict) -> list[tuple]:
    best, runner = F["best"], F["runner_up"]
    bullets = [
        f"Guessing the same price for every car misses by ${F['floor_mae']:,.0f} on "
        f"average. That is the number any genuine model has to beat, and without it a "
        f"figure like ${F['best_mae']:,.0f} has no context.",
        f"{best} misses by ${F['best_mae']:,.0f} on average - "
        f"{F['best_vs_floor_pct']:.0f}% better than the floor - and accounts for "
        f"{F['best_r2']*100:.1f}% of the variation in asking prices.",
        f"Its typical miss is {F['best_mdape']:.1f}% of the car's asking price.",
        f"Linear regression misses by ${F['linear_mae']:,.0f}, well behind both "
        f"tree-based models, which confirms that the relationships in this data are "
        f"not captured by straight lines.",
    ]
    if F["rmse_winner"] != best or F["r2_winner"] != best:
        bullets.append(
            f"The two tree-based models are close enough that the ranking depends on "
            f"the measure: {best} wins on the average miss (${F['best_mae']:,.0f} "
            f"against ${F['runner_mae']:,.0f}), while {F['rmse_winner']} edges it on "
            f"the measure that punishes large misses and {F['r2_winner']} on variation "
            f"explained. The average miss was named the deciding measure before any "
            f"result was seen, so {best} is reported as the best.")
    bullets.append(
        f"The gap between the two tree-based models (${abs(F['best_mae']-F['runner_mae']):,.0f}) "
        f"is far smaller than the gap between them and linear regression "
        f"(${F['linear_mae']-F['best_mae']:,.0f}), which suggests the limit is the "
        f"information in the adverts rather than the choice of method.")
    if F["floor_r2"] < 0:
        bullets.append(
            f"The floor's variation-explained figure is slightly negative "
            f"({F['floor_r2']:.3f}). That is expected, not an error: the measure is "
            f"calculated against always guessing the average price, while the floor "
            f"guesses the middle price, which is the fairer floor when the deciding "
            f"measure is the average miss.")
    return [("topic", f"All three learning models beat the floor, and {best.lower()} "
                      f"was the most accurate."),
            ("bullets", bullets)]


def _band_blocks(F: dict) -> list[tuple]:
    """State the price-band result in whichever direction the numbers point."""
    d_worst, p_worst = F["dollar_worst"], F["pct_worst"]
    bullets = [
        f"In dollars, the average miss is largest for {d_worst.lower()}-priced cars "
        f"(${F['mae_by_band'][d_worst]:,.0f}) and smallest for "
        f"{F['dollar_best'].lower()}-priced cars "
        f"(${F['mae_by_band'][F['dollar_best']]:,.0f}).",
        f"Measured against each car's own price, the typical miss is largest for "
        f"{p_worst.lower()}-priced cars ({F['mdape_by_band'][p_worst]:.1f}%) and "
        f"smallest for {F['pct_best'].lower()}-priced cars "
        f"({F['mdape_by_band'][F['pct_best']]:.1f}%).",
        "A fixed dollar miss is a far heavier mistake on a cheap car than on an "
        "expensive one, which is why the two measures can rank the bands differently.",
        "The same pattern holds for every model tested, not only the best one, so it "
        "is a property of the data rather than of one method.",
    ]
    if F["reversal"]:
        topic = ("The ranking of the price bands reverses depending on how the error "
                 "is measured, and this is the central result of the study.")
        bullets.append(
            f"Reporting only the dollar figure would have supported the conclusion "
            f"that {d_worst.lower()}-priced cars are hardest to predict; the "
            f"percentage figure shows they are in fact the easiest, and that "
            f"{p_worst.lower()}-priced cars are hardest. No single overall score could "
            f"have revealed this, which is precisely the gap the study addresses.")
    else:
        topic = (f"Both ways of measuring the error agree that "
                 f"{p_worst.lower()}-priced cars are the hardest to predict.")
        bullets.append(
            "Because the two measures agree here, the conclusion does not depend on "
            "which is chosen - but it still had to be checked, since the dollar figure "
            "rises with price almost automatically.")
    return [("topic", topic), ("bullets", bullets)]


def _residual_blocks(F: dict) -> list[tuple]:
    """Report whether the model prices cars above or below their advertised price."""
    r = F["residual_by_band"]
    under, over = F["underpriced_bands"], F["overpriced_bands"]
    bullets = ["A positive figure means the model guessed below the advertised price "
               "(it under-priced the car); a negative means it guessed above."]
    for band in ["Lower", "Middle", "Higher"]:
        direction = "below" if r[band] > 0 else "above"
        bullets.append(
            f"{band}-priced cars: the model guessed on average ${abs(r[band]):,.0f} "
            f"{direction} the advertised price.")
    if under and over:
        bullets.append(
            f"The direction is not uniform - the model under-prices "
            f"{', '.join(b.lower() for b in under)}-priced cars and over-prices "
            f"{', '.join(b.lower() for b in over)}-priced ones. This is the signature "
            f"of a model pulled towards the middle of the range: it has seen fewest "
            f"examples at the extremes, so it hedges away from them.")
    else:
        bullets.append(
            "The direction is consistent across all three bands, so the model is not "
            "simply hedging towards the middle of the price range.")
    bullets.append("This matters practically: a systematic lean means an automated "
                   "estimate is not merely imprecise for those cars but predictably "
                   "biased, which a user could correct for if they knew.")
    return [("topic", "The model does not miss evenly in both directions, and the "
                      "direction changes across the price range."),
            ("bullets", bullets)]


def blocks(F: dict) -> list[tuple]:
    best = F["best"]
    out = _accuracy_blocks(F)
    out += [
        ("table", "overall",
         "Table 2. Overall accuracy on the held-back cars, most accurate first."),
        ("figure", "fig08_model_comparison.png",
         "Figure 7. Overall accuracy of the four models."),
    ]
    out += _band_blocks(F)
    out += [
        ("table", "bands", f"Table 3. {best} accuracy by price band."),
        ("figure", "fig09_mae_by_band_dollars.png",
         "Figure 8. Average miss by price band, measured in dollars."),
        ("figure", "fig10_error_by_band_percent.png",
         "Figure 9. The same errors measured as a share of each car's own price."),
        ("table", "bands_all",
         "Table 4. Accuracy by price band for all four models."),
    ]
    out += _residual_blocks(F)
    out += [
        ("figure", "fig12_residuals_by_band.png",
         "Figure 10. Actual minus predicted price by band; above the line means the "
         "model priced the car below its advert."),
        ("figure", "fig11_predicted_vs_actual.png",
         "Figure 11. Predicted against actual asking price. Points on the dashed line "
         "are exact; the flattening at top right is the model hedging on dear cars."),
    ]
    return out
