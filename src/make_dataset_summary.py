"""Regenerate results/dataset_summary.csv from the cleaned datasets.

Run from the project root:
    python -m src.make_dataset_summary
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.data import load_dataset

RESULTS_PATH = Path(__file__).resolve().parent.parent / "results" / "dataset_summary.csv"

DATASET_NAMES: List[str] = [
    "pima_diabetes",
    "heart_cleveland",
    "ckd",
    "ilpd",
    "breast_wisconsin",
]


def build_summary() -> pd.DataFrame:
    """Return one summary row per dataset, computed from `load_dataset` output."""
    rows: List[Dict[str, Any]] = []
    for name in DATASET_NAMES:
        X, _, meta = load_dataset(name)
        rows.append(
            {
                "dataset": name,
                "n_rows_before_dedup": meta["n_rows_before_dedup"],
                "n_duplicates_found": meta["n_duplicates_found"],
                "n_rows": meta["n_rows"],
                "n_features": meta["n_features"],
                "n_numeric": len(meta["numeric_cols"]),
                "n_categorical": len(meta["categorical_cols"]),
                "positive_fraction": round(meta["class_balance"], 4),
                # Fraction of all feature cells that are missing, across the table.
                "missing_fraction": round(float(X.isna().to_numpy().mean()), 4),
                "n_conflicting_label_rows": meta["n_conflicting_label_rows"],
                "source": meta["source"],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """Write the summary CSV and print it."""
    summary = build_summary()
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(RESULTS_PATH, index=False)
    print(f"\nWrote {RESULTS_PATH}\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
