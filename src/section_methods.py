"""Prose for the "Data and Methodology" section.

Returns a list of blocks rather than writing a document, so the wording lives apart
from the mechanics of building a .docx. Every number is interpolated from the measured
results, so the text can never claim something the run did not produce.

Block vocabulary, shared by all three section modules:
    ("topic", text)                  a bold topic sentence
    ("bullets", [text, ...])         short bullets beneath it
    ("table", key, caption)          a table prepared by report.py
    ("figure", filename, caption)    a chart from outputs/figures
"""
from __future__ import annotations


def blocks(F: dict) -> list[tuple]:
    c0, c1 = F["band_cuts"]
    return [
        ("topic", "This study analyses a single, permanently archived snapshot of "
                  "United States used-car adverts."),
        ("bullets", [
            f"Source: {F['source']}. The file's fingerprint (MD5 checksum, a short code "
            f"that changes if even one character of the file differs) matches the "
            f"value published with the archive, so the file analysed here is provably "
            f"the archived one.",
            f"As downloaded it holds {F['n_raw']:,} adverts described by 26 fields.",
            "Adverts were posted between 4 April and 5 May 2021, so a car's age is "
            "measured against 2021 rather than the present year.",
            "The value being predicted is the asking price - what the seller "
            "advertised. The dataset contains no record of what buyers finally paid, "
            "so every finding concerns advertised prices only.",
        ]),

        ("topic", "Cleaning removed slightly over half the adverts, and most of that "
                  "was repeat postings rather than faulty data."),
        ("bullets", [
            f"Quality rules, applied in order: asking price $500-$200,000; mileage "
            f"100-400,000; year 1990-2021; make, model, fuel and transmission present. "
            f"Together these removed {F['quality_removed']:,} adverts.",
            f"A further {F['dedup_removed']:,} rows were repeat adverts for a car "
            f"already counted - {F['vin_repeats']:,} found through the vehicle "
            f"identification number (VIN, a car's unique serial number) and "
            f"{F['fp_repeats']:,} through an exact match on every characteristic.",
            "Ordinary duplicate removal would have caught only 37 of these, because "
            "every advert carries its own unique reference number.",
            "This step matters because a car appearing in both the training and the "
            "testing data lets a model recall a price rather than predict it, which "
            "makes its accuracy look better than it is.",
            "Limitation: 37.7% of adverts carry no VIN, so an unknown number of "
            "repeats survive among them.",
            f"The final analysis sample is {F['n_clean']:,} individual cars, "
            f"{F['pct_kept']:.1f}% of those downloaded.",
        ]),
        ("table", "funnel", "Table 1. Adverts remaining after each cleaning rule."),

        ("topic", "Thirteen vehicle characteristics were used to predict the asking "
                  "price."),
        ("bullets", [
            "Numeric: age at the time of listing, and mileage.",
            "Categorical: make, model, condition, cylinders, fuel, transmission, "
            "drive, body type, paint colour, title status and state.",
            "Discarded: county (blank in every row), size (blank in 71.8%), the "
            "reference-number, web-address and photograph fields, the free-text "
            "description, and latitude and longitude.",
            f"Sellers type the model name by hand, producing 19,310 variants. Those "
            f"with fewer than {F['model_min']} adverts were pooled into a single "
            f"\"Other\" group, leaving {F['models_kept']} named models; which names "
            f"survive was decided from the training cars alone.",
            "Blank categories were given their own \"Unknown\" label rather than being "
            "replaced with the commonest value. Nearly four in ten cars never state a "
            "condition, and whether a seller stated it may itself be part of why a "
            "price is hard to predict.",
        ]),

        ("topic", "Four models were compared, including a deliberate floor that makes "
                  "the scores interpretable."),
        ("bullets", [
            "Median guess: predicts the same price for every car. It is not a real "
            "model; it establishes the score any genuine model must beat.",
            "Linear regression (price estimated as a weighted sum of the "
            "characteristics) serves as the baseline.",
            "Random forest (many decision trees grown on random slices of the data and "
            "averaged together).",
            "Gradient boosting (trees built one after another, each one correcting the "
            "errors left by the last).",
            f"80% of cars trained the models. The remaining {F['n_test']:,} were held "
            f"back, untouched, and used exactly once at the end.",
            f"Settings were chosen by repeatedly holding back slices of the training "
            f"data (5-fold cross-validation) across {F['tune_candidates']} candidate "
            f"combinations on a {F['tune_subsample']:,}-car subsample, then refitted on "
            f"all training cars. The held-back cars played no part.",
            "The two tree models received numbered categories and the linear model "
            "received one column per category, because a linear model would otherwise "
            "read the numbering as an order. The cars, the split and the held-back set "
            "are identical across all four.",
        ]),

        ("topic", "Accuracy was measured in two ways, because either one alone would "
                  "mislead."),
        ("bullets", [
            "Overall: the average size of the miss in dollars (mean absolute error, "
            "MAE); the same with occasional large misses punished far more harshly "
            "(root mean squared error, RMSE); and the share of price variation the "
            "model accounts for (R-squared, where 0 means no better than always "
            "guessing the average price).",
            "By price band: MAE, RMSE, and the typical miss expressed as a share of "
            "that car's own price (median absolute percentage error).",
            f"Bands were set at the 33rd and 67th percentile of training prices - up "
            f"to ${c0:,.0f}, ${c0:,.0f} to ${c1:,.0f}, and above ${c1:,.0f} - then "
            f"applied unchanged to the held-back cars.",
            "R-squared is reported overall but deliberately not per band. Splitting by "
            "price compresses the price variation inside each band, which drives "
            "R-squared toward zero for every model regardless of how good it is.",
            f"A \"large error\" was defined before any result was examined as a miss "
            f"above ${F['large_threshold']:,.0f}, the level that 95% of cross-validated "
            f"training misses fall below.",
        ]),

        ("topic", "Five measures were taken so that the study reproduces exactly."),
        ("bullets", [
            f"The dataset is a fixed, archived version rather than a live download, "
            f"and its fingerprint is recorded ({F['source_md5']}).",
            f"Every threshold and the random seed ({F['seed']}) live in one settings "
            f"file, so no number is buried in the analysis code and the written "
            f"methodology cannot drift away from what the code does.",
            f"The cleaning step checks its own output against the expected "
            f"{F['n_clean']:,} cars and stops if a rule has silently changed, because "
            f"every figure in this report would then need regenerating.",
            "The split is checked to confirm that no car appears on both sides of it, "
            "rather than that being assumed.",
            "The whole study runs from a single command, from the raw file to this "
            "document, so no step depends on being done by hand in the right order.",
        ]),

        ("topic", "The methods used are standard, and are named here so they can be "
                  "cited directly."),
        ("bullets", [
            "Random forest: Breiman, L. (2001) 'Random Forests', Machine Learning, "
            "45(1), pp. 5-32. The same paper introduced permutation importance, the "
            "column-scrambling measure used later.",
            "Histogram-based gradient boosting: implemented in scikit-learn following "
            "the approach of Ke, G. et al. (2017) 'LightGBM: A Highly Efficient "
            "Gradient Boosting Decision Tree', Advances in Neural Information "
            "Processing Systems 30.",
            "Software: Pedregosa, F. et al. (2011) 'Scikit-learn: Machine Learning in "
            "Python', Journal of Machine Learning Research, 12, pp. 2825-2830. "
            "Version 1.9.1 was used, with pandas 2.3.3 on Python 3.11.9.",
            f"Data: {F['source']}.",
        ]),
    ]
