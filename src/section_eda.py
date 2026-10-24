"""Prose for the first half of "Data Analysis and Results" - what the listings look like.

This is the exploratory half, answering Objective 2: the relationships between vehicle
characteristics and asking price, and the data-quality problems that shape everything
after it. The model results follow in section_models.py.
"""
from __future__ import annotations


def blocks(F: dict) -> list[tuple]:
    return [
        ("topic", "Asking prices lean heavily towards the cheaper end of the market."),
        ("bullets", [
            f"The middle car asks ${F['price_median']:,.0f} while the average is "
            f"${F['price_mean']:,.0f}. The average sits higher because a small number "
            f"of very expensive vehicles pull it upwards (a right-skewed "
            f"distribution, measured at {F['price_skew']:.2f}, where 0 would mean "
            f"perfectly even).",
            f"Half of all cars ask between ${F['price_q25']:,.0f} and "
            f"${F['price_q75']:,.0f}.",
            "This lopsidedness is why the price bands are set by percentile rather "
            "than in equal dollar steps, which would have left the top band nearly "
            "empty.",
            "It is also why the error analysis reports percentages as well as dollars: "
            "the cars are not spread evenly across the price range, so an average "
            "dollar error is dominated by whichever end holds most of the cars.",
        ]),
        ("figure", "fig03_price_distribution.png",
         "Figure 1. Distribution of asking prices, with the two band cut-offs marked."),

        ("topic", "The fields describing a car's condition are the ones most often "
                  "left blank, and that shapes what any model can achieve."),
        ("bullets", [
            "Condition is blank in 40.8% of raw adverts, cylinders in 41.6%, drive "
            "type in 30.6%, paint colour in 30.5% and body type in 21.8%.",
            "County is blank in all 426,880 adverts and size in 71.8%, which is why "
            "both were discarded outright rather than filled in.",
            "Mileage, year, make and model are nearly complete, so the "
            "characteristics that move price hardest are also the best recorded - "
            "which is fortunate, and is part of why the models work at all.",
            "The gaps are not random. A private seller in a hurry leaves fields "
            "blank; a dealer listing stock fills them in. That makes a blank field a "
            "signal in its own right, which is why blanks were labelled \"Unknown\" "
            "rather than guessed at.",
        ]),
        ("figure", "fig02_missing_data.png",
         "Figure 2. Share of adverts with no value, field by field, before cleaning."),

        ("topic", "Asking price falls with age, steeply at first and then gently."),
        ("bullets", [
            "The median price drops sharply across roughly the first decade of a "
            "car's life, then flattens out, so the relationship is a curve rather "
            "than a straight line.",
            "That curvature is the main reason the two tree-based models can beat "
            "linear regression, which has to fit one straight slope across the whole "
            "range.",
            "The spread of prices at any single age is wide, which means age alone "
            "explains only part of the price and the other characteristics carry real "
            "weight.",
        ]),
        ("figure", "fig05_price_vs_age.png",
         "Figure 3. Median asking price by age, with the middle half of cars shaded."),

        ("topic", "Price falls as mileage rises, but the two are not "
                  "interchangeable."),
        ("bullets", [
            "The densest part of the market sits between roughly 50,000 and 150,000 "
            "miles, and prices decline across that span.",
            "At any given mileage the spread of prices remains wide, because a "
            "high-mileage luxury vehicle and a low-mileage economy car can ask the "
            "same amount.",
            "Age and mileage usually move together but not always, and cars where "
            "they disagree - an old car with very low mileage, or a nearly new one "
            "with very high mileage - are among the harder cases to price.",
        ]),
        ("figure", "fig04_price_vs_mileage.png",
         "Figure 4. Density of cars across mileage and asking price."),

        ("topic", "Make and body type both separate the market clearly, which is why "
                  "they earn their place as predictors."),
        ("bullets", [
            "Median asking prices differ substantially between the commonest makes, "
            "so the make of a car carries real information about its price.",
            "Body type separates the market even more sharply: pickups and larger "
            "vehicles ask considerably more than sedans and hatchbacks of comparable "
            "age.",
            "The ranges overlap heavily, so neither field determines a price on its "
            "own - they shift the starting point, and age and mileage do the rest.",
        ]),
        ("figure", "fig06_price_by_manufacturer.png",
         "Figure 5. Asking price by make, for the fifteen commonest makes."),
        ("figure", "fig07_price_by_type.png",
         "Figure 6. Asking price by body type."),
    ]
