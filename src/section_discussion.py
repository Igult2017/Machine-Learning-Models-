"""Prose for "Discussion of Findings" and "Conclusion and Recommendations".

Interpretation is where a thesis most easily overclaims, so every statement here is
tied to a number this run produced, and the limits of what advertised-price data can
support are stated plainly rather than buried at the end.
"""
from __future__ import annotations


def discussion_blocks(F: dict) -> list[tuple]:
    hedging = bool(F["underpriced_bands"] and F["overpriced_bands"])
    return [
        ("topic", "The models are accurate enough to screen a listing but not to "
                  "value a car."),
        ("bullets", [
            f"A typical miss of {F['best_mdape']:.1f}% of the asking price is useful "
            f"for flagging an advert that is far out of line with the market, and far "
            f"too wide to settle a price between a buyer and a seller.",
            f"On a ${F['price_median']:,.0f} car - the middle of this market - that "
            f"percentage is roughly "
            f"${F['price_median']*F['best_mdape']/100:,.0f} of uncertainty, which is "
            f"more than most buyers would treat as a rounding error.",
            f"The improvement from linear regression to the best model is substantial "
            f"(${F['linear_mae']:,.0f} down to ${F['best_mae']:,.0f}), but the gap "
            f"between the two tree-based models is small by comparison.",
            "That pattern points to the information in the adverts as the limiting "
            "factor rather than the sophistication of the method, which means a more "
            "elaborate model would buy very little.",
        ]),

        ("topic", "Average accuracy conceals which customers the model serves worst, "
                  "and that is the gap this study set out to fill."),
        ("bullets", [
            f"Measured in dollars the {F['dollar_worst'].lower()} band looks worst; "
            f"measured against each car's own price the {F['pct_worst'].lower()} band "
            f"does.",
            "Both numbers are correct, and they answer different questions. A dollar "
            "figure matters to someone deciding how much to set aside; a percentage "
            "matters to someone judging whether an asking price is fair.",
            "A study reporting a single overall figure can answer neither, and a "
            "reader cannot reconstruct the answer from what such a study publishes - "
            "which is why two models with identical overall scores can be unequally "
            "useful in practice.",
            f"Practically, a buyer looking at a {F['pct_worst'].lower()}-priced car "
            f"should treat an automated estimate with the most caution, because that "
            f"is where the estimate is furthest out relative to the money involved.",
        ]),

        ("topic", "The model's errors have a direction as well as a size, which makes "
                  "them partly correctable."),
        ("bullets", [
            f"Lower-priced cars are priced on average "
            f"${abs(F['residual_by_band']['Lower']):,.0f} "
            f"{'below' if F['residual_by_band']['Lower'] > 0 else 'above'} their "
            f"advert, and higher-priced cars "
            f"${abs(F['residual_by_band']['Higher']):,.0f} "
            f"{'below' if F['residual_by_band']['Higher'] > 0 else 'above'} theirs.",
            ("The opposing directions are the signature of a model pulled towards the "
             "middle of the range: it has seen fewest examples at the extremes, so it "
             "hedges away from them. This is a known consequence of minimising average "
             "error, not a coding fault." if hedging else
             "The direction is consistent across the bands, so the model is not simply "
             "hedging towards the middle of the price range - the bias has some other "
             "source and is worth investigating further."),
            "A predictable lean is more tractable than random noise: a practitioner "
            "who knew the direction could adjust for it, whereas scatter cannot be "
            "adjusted away.",
            "It also means that reporting only the average size of the miss hides "
            "something a user would want to know, in the same way that reporting only "
            "the overall figure hides the band differences.",
        ]),

        ("topic", "What limits accuracy is how unusual a car is and how little its "
                  "advert says, not the choice of algorithm."),
        ("bullets", [
            f"Being an uncommon model is the strongest single marker of a badly "
            f"predicted car ({F['big_rare_model']*100:.1f}% of them against "
            f"{F['rest_rare_model']*100:.1f}% of the rest). The model simply has few "
            f"comparable cars to learn from.",
            f"Gaps in the advert come next: cars with no stated condition are "
            f"over-represented among the worst predictions "
            f"({F['big_cond_missing']*100:.1f}% against "
            f"{F['rest_cond_missing']*100:.1f}%), and the pattern repeats for "
            f"cylinders and drive type.",
            f"That combination is awkward, because two of the four characteristics the "
            f"model relies on most - {F['top_features'][2]} among them - are blank for "
            f"roughly a third to a half of adverts. The model is most deprived of "
            f"exactly the information it would use.",
            "Accident history, service records, number of previous owners and any "
            "reserve price are absent from the dataset altogether, and those are what "
            "a human valuer would ask for first.",
            "This reframes what would improve such a system: better information "
            "captured at the point of listing would help far more than a more "
            "elaborate algorithm, given how little separates the two tree models.",
        ]),

        ("topic", "The sheer scale of repeat adverts is a warning about how studies on "
                  "this kind of data are normally conducted - though not for the "
                  "reason expected."),
        ("bullets", [
            f"Repeat postings accounted for {F['dedup_removed']:,} rows, more than "
            f"half of every advert reporting a VIN, with one car posted 261 times. "
            f"Ordinary duplicate removal catches 37 rows out of 426,880, because every "
            f"advert carries its own reference number.",
            f"The expectation was that leaving them in would flatter the score. "
            f"Measured, it did the opposite by "
            f"${abs(F['dirty_mae'] - F['clean_mae']):,.0f} per car, because the "
            f"un-cleaned data is also larger and dominated by heavily re-posted cars "
            f"that are themselves harder to price.",
            "That comparison mixes overlap, sample size and composition together, so "
            "it is not a clean estimate of what recalling a seen car is worth. "
            "Reporting it as one would mean presenting a confounded number as a "
            "finding.",
            "The case for removing repeats is therefore a logical one rather than an "
            "empirical one: a score measured partly on cars the model has already seen "
            "does not answer a question about unseen cars, whichever way the figure "
            "happens to move.",
            "The practical lesson holds either way. Any accuracy figure quoted for "
            "listing data should state how repeat items were identified and handled, "
            "or it cannot meaningfully be compared with another study's figure.",
        ]),

        ("topic", "Part of the uneven accuracy comes from what the model is asked to "
                  "minimise, not only from the market itself."),
        ("bullets", [
            f"Training on the logarithm of the price rather than the price narrowed "
            f"the spread between the best- and worst-predicted bands from "
            f"{F['raw_band_spread']:.1f} percentage points to "
            f"{F['log_band_spread']:.1f}, while leaving overall accuracy almost "
            f"unchanged.",
            "Working in logarithms penalises proportional error, so it buys accuracy "
            "on cheap cars at the expense of expensive ones. The direction of the "
            "headline finding survives that change; its size does not.",
            "So the answer to the research question depends partly on a modelling "
            "choice that studies rarely state. That is an argument for reporting the "
            "quantity trained on as plainly as the metrics themselves.",
        ]),

        ("topic", "Five limitations bound what these results can claim."),
        ("bullets", [
            "The target is the advertised asking price, not an agreed sale price. "
            "Nothing here measures what a car is actually worth, and the gap between "
            "the two is a negotiation this data cannot see.",
            "The data is a 30-day snapshot from April and May 2021. It cannot show "
            "whether accuracy holds as market conditions move, and a date-based test "
            "was impossible for exactly that reason.",
            "Around 38% of adverts carry no VIN, so some repeat vehicles certainly "
            "remain in the sample and the de-duplication is a floor, not a ceiling.",
            f"Model names were pooled below {F['model_min']} listings, so the roughly "
            f"one-third of cars in the \"Other\" group are predicted with less "
            f"specific information than the rest.",
            "The findings describe one national market at one time, and should not be "
            "assumed to transfer to other countries, other price ranges or other "
            "periods without being re-tested.",
        ]),
    ]
