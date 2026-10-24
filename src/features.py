"""Turn the cleaned columns into something the models can read.

Three jobs, all of them wrapped inside scikit-learn pipelines so that every value used
to fill a gap or build a category is learned from the training data alone. That is what
stops information from the test set leaking backwards into training and flattering the
final score.

Two different pipelines are built, one per model family:

- Linear Regression gets one-hot encoding: each category becomes its own yes/no column
  (about 731 of them). A linear model needs this, because feeding it numbered
  categories would imply Ford is "less than" Toyota.
- The two tree models get numbered categories instead (13 columns). Trees are not
  harmed by the arbitrary numbering, and one-hot encoding for them would not fit in
  this machine's 7.9 GB of memory.

Same cars, same split, same test rows - only each model's own preparation differs.
"""
from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

import config as cfg


class TextNormaliser(BaseEstimator, TransformerMixin):
    """Lower-case and trim the free-text columns.

    Sellers type the model name by hand, so 'F-150', 'f-150 ' and 'f-150' arrive as
    three different values. Normalising first means they are counted as one car.
    """

    def __init__(self, columns: tuple[str, ...] = ("model", "manufacturer")):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.columns:
            if col in X.columns:
                X[col] = X[col].astype("string").str.lower().str.strip()
        return X


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """Replace values that are rare in the TRAINING data with a single 'Other' label.

    The model column holds 19,310 distinct names. Most appear a handful of times, which
    teaches the model nothing and would create a column for each. Keeping only names
    with at least `min_count` listings leaves 581 named models covering roughly
    two-thirds of cars; the rest are pooled.

    Which names survive is decided on the training data only, then applied unchanged to
    the test data - so the test set never influences the feature set.
    """

    def __init__(self, column: str = "model", min_count: int = 50, other: str = "Other"):
        self.column = column
        self.min_count = min_count
        self.other = other

    def fit(self, X: pd.DataFrame, y=None):
        counts = X[self.column].value_counts()
        self.keep_ = set(counts[counts >= self.min_count].index)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        col = X[self.column]
        X[self.column] = col.where(col.isin(self.keep_), self.other).astype(object)
        return X


def _numeric_branch(scale: bool) -> Pipeline:
    """Fill missing mileage or age with the training median, optionally rescale.

    The median is used rather than the mean because a few extreme mileages would drag
    a mean upwards. Rescaling matters only for the linear model, where age (0-31) and
    mileage (100-400,000) would otherwise sit on wildly different scales.
    """
    steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scale", StandardScaler()))
    return Pipeline(steps)


def _categorical_branch(one_hot: bool) -> Pipeline:
    """Give missing categories their own label, then encode.

    Missing values become an explicit "Unknown" category rather than being replaced
    with the commonest value. Nearly 4 in 10 cars never state a condition, and whether
    a car reported its condition may itself be what makes its price hard to predict.
    Overwriting that with "good" would destroy the very evidence the error analysis is
    looking for.
    """
    steps = [("impute", SimpleImputer(strategy="constant", fill_value=cfg.UNKNOWN))]
    if one_hot:
        # handle_unknown="ignore" so a category seen only in the test set is simply
        # all-zeros rather than crashing the model.
        steps.append(("encode", OneHotEncoder(handle_unknown="ignore",
                                              sparse_output=True)))
    else:
        steps.append(("encode", OrdinalEncoder(handle_unknown="use_encoded_value",
                                               unknown_value=-1)))
    return Pipeline(steps)


def build_preprocessor(one_hot: bool) -> Pipeline:
    """Assemble the full preparation chain for one model family."""
    column_work = ColumnTransformer([
        ("numeric", _numeric_branch(scale=one_hot), cfg.NUMERIC),
        ("categorical", _categorical_branch(one_hot=one_hot), cfg.CATEGORICAL),
    ])
    return Pipeline([
        ("normalise", TextNormaliser()),
        ("group_rare", RareCategoryGrouper(column="model",
                                           min_count=cfg.MODEL_MIN_COUNT)),
        ("prepare", column_work),
    ])


def load_clean() -> pd.DataFrame:
    """Read the analysis sample produced by clean.py."""
    return pd.read_parquet(cfg.CLEAN_FILE)


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate the 13 predictors from the asking price being predicted."""
    return df[cfg.FEATURES].copy(), df[cfg.TARGET].copy()
