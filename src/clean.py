"""Turn the raw Craigslist listings into the analysis sample.

Two jobs, in this order:

1. Apply the documented quality rules, recording how many rows each one removes so the
   thesis can report the full audit trail.
2. Remove repeat adverts for the same physical car. This is the step that matters most.
   Ordinary duplicate removal finds only 37 rows out of 426,880, because every advert
   carries its own unique id - yet more than half of all cars that report a VIN (the
   car's unique serial number) are advertised more than once, one of them 261 times.
   Left in, the same car would appear in both the training and the test set, and the
   test score would measure recall rather than prediction.

De-duplication runs BEFORE the train/test split, never after.
"""
from __future__ import annotations

import pandas as pd

import config as cfg


def require_raw_file() -> None:
    """Explain how to obtain the data instead of dying with a stack trace.

    The raw file is 1.35 GB and is deliberately not distributed with this code. It is
    cited by its permanent identifier instead, so anyone can fetch the identical bytes
    and confirm them against the checksum below.
    """
    if cfg.RAW_CSV.exists():
        return
    raise SystemExit(
        f"\nThe raw dataset is not present at:\n"
        f"  {cfg.RAW_CSV}\n\n"
        f"It is 1.35 GB and is not shipped with this code. Download the archived copy -\n"
        f"the exact file these results were produced from:\n\n"
        f"  {cfg.SOURCE}\n"
        f"  {cfg.SOURCE_DOI}\n\n"
        f'  curl -L -o used-cars.zip "{cfg.SOURCE_FILE_URL}"\n'
        f'  unzip used-cars.zip -d "{cfg.RAW_CSV.parent}"\n\n'
        f"Check it before use. The MD5 checksum must be:\n"
        f"  {cfg.SOURCE_MD5}\n"
    )


def load_raw() -> pd.DataFrame:
    """Read only the columns the study uses, then parse the posting date."""
    df = pd.read_csv(cfg.RAW_CSV, usecols=cfg.READ_COLS, low_memory=False)
    df[cfg.ORDER_COL] = pd.to_datetime(df[cfg.ORDER_COL], errors="coerce", utc=True)
    return df


def missingness_report() -> pd.DataFrame:
    """How much of each raw column is blank, before any cleaning.

    Read in its own pass, skipping only the four text-heavy columns, because the gaps in
    columns the study ends up discarding are themselves a finding worth reporting.
    """
    heavy = ["url", "region_url", "image_url", "description"]
    wanted = [c for c in pd.read_csv(cfg.RAW_CSV, nrows=1).columns if c not in heavy]
    raw = pd.read_csv(cfg.RAW_CSV, usecols=wanted, low_memory=False)
    pct = (raw.isna().mean() * 100).sort_values(ascending=False)
    return pd.DataFrame({"Column": pct.index, "Percent blank": pct.to_numpy().round(1)})


def apply_quality_rules(df: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple[str, int]]]:
    """Drop implausible and incomplete rows, logging the count after each rule."""
    funnel = [("0. Raw listings", len(df))]

    df = df[df[cfg.TARGET].between(cfg.PRICE_MIN, cfg.PRICE_MAX)]
    funnel.append((f"1. Asking price ${cfg.PRICE_MIN:,}-${cfg.PRICE_MAX:,}", len(df)))

    df = df[df["odometer"].between(cfg.ODO_MIN, cfg.ODO_MAX)]
    funnel.append((f"2. Mileage {cfg.ODO_MIN:,}-{cfg.ODO_MAX:,} miles", len(df)))

    df = df[df["year"].between(cfg.YEAR_MIN, cfg.YEAR_MAX)]
    funnel.append((f"3. Year {cfg.YEAR_MIN}-{cfg.YEAR_MAX}", len(df)))

    df = df[df["manufacturer"].notna() & df["model"].notna()]
    funnel.append(("4. Make and model present", len(df)))

    df = df[df["fuel"].notna() & df["transmission"].notna()]
    funnel.append(("5. Fuel and transmission present", len(df)))

    return df, funnel


def remove_repeat_listings(
    df: pd.DataFrame, funnel: list[tuple[str, int]]
) -> tuple[pd.DataFrame, list[tuple[str, int]], dict[str, int]]:
    """Keep one row per physical car.

    Cars that report a VIN are collapsed to their earliest advert. Cars with no VIN
    cannot be traced that way, so two rows are treated as the same car only when every
    field in the fingerprint matches, mileage to the exact mile included.
    """
    # A stable sort with an explicit tie-breaker. pandas sorts with an unstable method
    # by default, so two adverts posted at the same second could change places between
    # runs and a different one would be kept - which would silently alter the sample.
    df = df.sort_values([cfg.ORDER_COL, cfg.ID_COL], kind="mergesort",
                        na_position="last")

    with_vin = df[df[cfg.DEDUP_KEY].notna()]
    collapsed = with_vin.drop_duplicates(subset=cfg.DEDUP_KEY, keep="first")
    without_vin = df[df[cfg.DEDUP_KEY].isna()]
    funnel.append(("6. One advert per VIN (earliest kept)",
                   len(collapsed) + len(without_vin)))

    deduped_no_vin = without_vin.drop_duplicates(subset=cfg.FINGERPRINT, keep="first")
    out = pd.concat([collapsed, deduped_no_vin], ignore_index=True)
    funnel.append(("7. Fingerprint match on rows with no VIN", len(out)))

    stats = {
        "rows_with_vin": len(with_vin),
        "distinct_vins": len(collapsed),
        "vin_repeats_removed": len(with_vin) - len(collapsed),
        "no_vin_rows": len(without_vin),
        "no_vin_repeats_removed": len(without_vin) - len(deduped_no_vin),
    }
    return out, funnel, stats


def add_age(df: pd.DataFrame) -> pd.DataFrame:
    """Vehicle age at the time of the snapshot, not at today's date.

    The listings were posted in 2021. Measuring age against the current calendar year
    would add several years to every car in the study.
    """
    df = df.copy()
    df["age"] = cfg.SNAPSHOT_YEAR - df["year"]
    return df


def funnel_frame(funnel: list[tuple[str, int]]) -> pd.DataFrame:
    """Build the audit-trail table: rows surviving and removed at each step."""
    rows, previous, start = [], None, funnel[0][1]
    for label, count in funnel:
        rows.append({
            "Step": label,
            "Rows remaining": count,
            "Removed by this step": 0 if previous is None else previous - count,
            "Percent of raw kept": round(count / start * 100, 1),
        })
        previous = count
    return pd.DataFrame(rows)


def main() -> pd.DataFrame:
    cfg.ensure_dirs()
    require_raw_file()
    print(f"Reading {cfg.RAW_CSV.name} ...")

    gaps = missingness_report()
    gaps.to_csv(cfg.TAB_DIR / "table2_missingness.csv", index=False)
    worst = gaps.head(6).itertuples(index=False)
    print("Blankest raw columns: "
          + ", ".join(f"{c.Column} {c._1}%" for c in worst))

    df = load_raw()

    df, funnel = apply_quality_rules(df)
    df, funnel, stats = remove_repeat_listings(df, funnel)
    df = add_age(df)

    table = funnel_frame(funnel)
    table.to_csv(cfg.TAB_DIR / "table1_cleaning_funnel.csv", index=False)
    print("\n" + table.to_string(index=False))

    print("\nRepeat-advert detail")
    print(f"  rows reporting a VIN            : {stats['rows_with_vin']:,}")
    print(f"  distinct cars among them        : {stats['distinct_vins']:,}")
    print(f"  repeat adverts removed by VIN   : {stats['vin_repeats_removed']:,}")
    print(f"  rows with no VIN                : {stats['no_vin_rows']:,}")
    print(f"  repeats removed by fingerprint  : {stats['no_vin_repeats_removed']:,}")

    keep = cfg.FEATURES + [cfg.TARGET, cfg.DEDUP_KEY, cfg.ID_COL, "year"]
    df[keep].to_parquet(cfg.CLEAN_FILE, index=False)

    assert len(df) == cfg.EXPECTED_CLEAN_ROWS, (
        f"cleaning produced {len(df):,} rows, expected {cfg.EXPECTED_CLEAN_ROWS:,} - "
        "a rule changed, so every figure in the write-up must be regenerated"
    )
    print(f"\nSaved {len(df):,} cars to {cfg.CLEAN_FILE.name}")
    return df


if __name__ == "__main__":
    main()
