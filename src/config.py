"""Every rule, threshold and path used by the used-car asking-price study.

No other script hard-codes a number. Keeping them here means the methodology chapter
and the code cannot drift apart: if a threshold changes, it changes in one place and
every table and chart that reports it updates with it.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RAW_CSV = ROOT / "data" / "raw" / "vehicles.csv"
CLEAN_FILE = ROOT / "data" / "processed" / "clean.parquet"
PRED_FILE = ROOT / "data" / "processed" / "test_predictions.parquet"
FIG_DIR = ROOT / "outputs" / "figures"
TAB_DIR = ROOT / "outputs" / "tables"
MODEL_DIR = ROOT / "outputs" / "models"
DOC_DIR = ROOT / "docs"

RANDOM_SEED = 42

# ---------------------------------------------------------------- provenance
# Recorded so the thesis can state exactly which file was analysed.
SOURCE = "Zenodo record 10457828 - Craigslist Used Cars and Trucks (CC BY 4.0)"
SOURCE_MD5 = "d7462a2fb71459c76170c24a6fd0578a"
SOURCE_DOI = "https://doi.org/10.5281/zenodo.10457828"
SOURCE_FILE_URL = ("https://zenodo.org/records/10457828/files/"
                   "Used%20Cars%20Dataset.zip?download=1")
SNAPSHOT_YEAR = 2021          # listings were posted 2021-04-04 to 2021-05-05
RAW_ROWS = 426_880
EXPECTED_CLEAN_ROWS = 209_164  # asserted by clean.py so silent drift is impossible

# ---------------------------------------------------------------- columns
TARGET = "price"               # the advertised asking price, never a sale price
DEDUP_KEY = "VIN"              # used only to spot repeat listings, never as a predictor
ORDER_COL = "posting_date"     # used only to pick the earliest listing of a car

NUMERIC = ["age", "odometer"]
CATEGORICAL = ["manufacturer", "model", "condition", "cylinders", "fuel",
               "transmission", "drive", "type", "paint_color", "title_status", "state"]
FEATURES = NUMERIC + CATEGORICAL

# 'description' and the url columns are deliberately never read: they are most of the
# 1.35 GB file and would not fit comfortably in 7.9 GB of memory.
READ_COLS = ["id", TARGET, "year", "odometer", "manufacturer", "model", "condition",
             "cylinders", "fuel", "transmission", "drive", "type", "paint_color",
             "title_status", "state", "region", DEDUP_KEY, ORDER_COL]

# 'id' is the advert's own reference number. It is never a predictor; it is kept so a
# badly predicted car can be traced back to the exact advert it came from, and as a
# tie-breaker that makes de-duplication give the same answer on every run.
ID_COL = "id"

# ---------------------------------------------------------------- cleaning rules
PRICE_MIN, PRICE_MAX = 500, 200_000      # $0 adverts mean "call me"; the max removes $3.7bn typos
ODO_MIN, ODO_MAX = 100, 400_000          # the raw file reaches 10,000,000 miles
YEAR_MIN, YEAR_MAX = 1990, 2021          # the raw file reaches back to 1900

# Two different listings of the same car, where no VIN is given, are treated as
# repeats only if every one of these matches - mileage to the exact mile included.
FINGERPRINT = ["region", "price", "year", "manufacturer", "model", "odometer",
               "condition", "fuel", "transmission", "type", "paint_color"]

# ---------------------------------------------------------------- features
MODEL_MIN_COUNT = 50     # sellers type 'model' freely; 19,310 distinct values exist
UNKNOWN = "Unknown"      # missing categories get their own label, not the commonest value

# ---------------------------------------------------------------- split and tuning
TEST_SIZE = 0.20
CV_FOLDS = 5
TUNE_SUBSAMPLE = 50_000  # settings are searched on a slice, then refitted on all of it
TUNE_CANDIDATES = 20

# ---------------------------------------------------------------- evaluation
BAND_QUANTILES = (1 / 3, 2 / 3)          # cut-offs taken from the training set only
BAND_LABELS = ["Lower", "Middle", "Higher"]
LARGE_ERROR_QUANTILE = 0.95              # fixed before any test result is looked at


# A price counts as an obvious placeholder only when its digits run in sequence
# (123,456 or 12,345) - the pattern left by someone dragging a finger across the number
# row. Deliberately narrow: repeated digits were considered and rejected, because
# $99,999 is ordinary psychological pricing, exactly like the $69,999 beside it in the
# data, and flagging it would have been a false accusation. The rule therefore
# under-counts, and other nonsense prices certainly survive undetected.
PLACEHOLDER_MIN_DIGITS = 5

# Above this, a car counts as expensive for the purpose of describing large errors.
DEAR_PRICE = 60_000


def is_placeholder_price(price: float) -> bool:
    """True when a price was clearly typed across the keyboard rather than meant."""
    digits = str(int(price))
    if len(digits) < PLACEHOLDER_MIN_DIGITS:
        return False
    return all(int(b) - int(a) == 1 for a, b in zip(digits, digits[1:]))


# Plain-English names for the column names, used in all prose, tables and charts.
# The raw names are what the dataset calls them; nobody reading a thesis should have to
# know that "odometer" means mileage or that "paint_color" is the colour.
FEATURE_LABELS = {
    "age": "age", "odometer": "mileage", "manufacturer": "make", "model": "model",
    "condition": "condition", "cylinders": "cylinders", "fuel": "fuel type",
    "transmission": "transmission", "drive": "drive type", "type": "body type",
    "paint_color": "paint colour", "title_status": "title status", "state": "state",
}

# 'age' is calculated from 'year', so the gaps that matter for age are the gaps in year.
MISSING_SOURCE = {"age": "year"}


def label(column: str) -> str:
    """Plain-English name for a column, falling back to the raw name."""
    return FEATURE_LABELS.get(column, column)


def ensure_dirs() -> None:
    """Create every output folder. Safe to call repeatedly."""
    for d in (CLEAN_FILE.parent, FIG_DIR, TAB_DIR, MODEL_DIR, DOC_DIR):
        d.mkdir(parents=True, exist_ok=True)
