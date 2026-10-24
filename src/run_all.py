"""Run the whole study end to end, from the raw CSV to the finished Word document.

    python run_all.py              every step, in order
    python run_all.py train        that step and everything after it

Each step writes files the next one reads, so the order is not optional. Running this
from scratch reproduces every number in the document, because the random seed and all
thresholds are fixed in config.py.
"""
from __future__ import annotations

import sys
import time

import clean
import evaluate
import figures_data
import figures_results
import report
import robustness
import train

STEPS = [
    ("clean", "Clean the listings and remove repeat adverts", clean.main),
    ("train", "Split, tune and fit the four models", train.main),
    ("evaluate", "Score the models and dissect the errors", evaluate.main),
    ("robustness", "Run the two robustness checks", robustness.main),
    ("figures_data", "Draw the charts describing the data", figures_data.main),
    ("figures_results", "Draw the charts of the results", figures_results.main),
    ("report", "Build the Word document", report.main),
]


def main(start: str | None = None) -> None:
    names = [name for name, _, _ in STEPS]
    if start and start not in names:
        sys.exit(f"unknown step '{start}'. choose one of: {', '.join(names)}")
    begin = names.index(start) if start else 0

    overall = time.time()
    for name, description, func in STEPS[begin:]:
        print("\n" + "=" * 78)
        print(f"STEP: {description}  ({name})")
        print("=" * 78)
        t0 = time.time()
        func()
        print(f"-- {name} finished in {time.time() - t0:.0f}s")

    print(f"\nAll steps complete in {(time.time() - overall) / 60:.1f} minutes.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
