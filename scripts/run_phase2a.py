"""Run the Phase 2a classical-model benchmark.

Examples, from the project root:
    python scripts/run_phase2a.py --smoke
    python scripts/run_phase2a.py --workers 4

Re-running resumes: tasks that already finished are skipped.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# One thread per model fit, so parallel tasks don't compete for cores and fit times stay
# comparable across models. This must be set before numpy is imported.
for _thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_thread_variable, "1")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from src.models import MODEL_REGISTRY
from src.runner import is_constant_prediction, make_result_dirs, run_task_safe

ALL_DATASETS: List[str] = ["pima_diabetes", "heart_cleveland", "ckd", "ilpd", "breast_wisconsin"]
ALL_MODELS: List[str] = list(MODEL_REGISTRY)


def parse_splits(values: List[str]) -> List[int]:
    """Turn split arguments such as ["0-4"] or ["0", "3"] into a list of split indexes."""
    splits: List[int] = []
    for value in values:
        first, _, last = value.partition("-")
        splits.extend(range(int(first), int(last or first) + 1))
    if not all(0 <= split <= 9 for split in splits):
        raise argparse.ArgumentTypeError(f"Splits must be between 0 and 9, got {values}")
    return splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--datasets", nargs="+", default=ALL_DATASETS, choices=ALL_DATASETS)
    parser.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS)
    parser.add_argument(
        "--splits", nargs="+", default=["0-9"], help="Split indexes or ranges, e.g. 0-9 or 0 3 5"
    )
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Quick check in results_smoke/: pima_diabetes, logreg and xgb, split 0, 3 trials. "
        "Overrides --datasets, --models, --splits and --n-trials.",
    )
    return parser.parse_args()


def aggregate_results(results_dir: Path) -> pd.DataFrame:
    """Combine every raw CSV under `results_dir` into phase2a_all_runs.csv."""
    frames = [pd.read_csv(path) for path in sorted((results_dir / "raw").glob("*.csv"))]
    if not frames:
        raise RuntimeError(f"No raw result files found in {results_dir / 'raw'}")
    combined = pd.concat(frames, ignore_index=True).sort_values(
        ["dataset", "model", "split_idx", "fraction"], ignore_index=True
    )
    combined.to_csv(results_dir / "phase2a_all_runs.csv", index=False)
    return combined


def main() -> int:
    args = parse_args()
    if args.smoke:
        datasets, models, splits, n_trials = ["pima_diabetes"], ["logreg", "xgb"], [0], 3
        results_dir = PROJECT_ROOT / "results_smoke"
    else:
        datasets, models, n_trials = args.datasets, args.models, args.n_trials
        splits = parse_splits(args.splits)
        results_dir = PROJECT_ROOT / "results"
    make_result_dirs(results_dir)

    tasks: List[Dict[str, Any]] = [
        {
            "dataset": dataset,
            "model_name": model,
            "split_idx": split,
            "n_trials": n_trials,
            "results_dir": str(results_dir),
        }
        for dataset in datasets
        for model in models
        for split in splits
    ]
    print(f"Running {len(tasks)} tasks with {args.workers} worker(s), writing to {results_dir}")

    start = time.perf_counter()
    crashes: List[str] = []
    results = Parallel(n_jobs=args.workers, return_as="generator_unordered")(
        delayed(run_task_safe)(**task) for task in tasks
    )
    for result in tqdm(results, total=len(tasks), desc="Tasks"):
        if result["crash"]:
            outcome = f"CRASHED ({result['crash']})"
            crashes.append(f"{result['stem']}: {result['crash']}")
        elif result["skipped"]:
            outcome = "skipped, already complete"
        else:
            outcome = f"{result['n_ok']} ok, {result['n_failed']} failed"
        tqdm.write(f"{result['stem']}: {result['elapsed_s']:.1f}s, {outcome}")
    total_runtime = time.perf_counter() - start

    combined = aggregate_results(results_dir)
    print(f"\nAggregated into {results_dir / 'phase2a_all_runs.csv'}")
    print(f"Rows: {len(combined)}")
    print(f"Status ok: {(combined['status'] == 'ok').sum()}")
    print(f"Status failed: {(combined['status'] == 'failed').sum()}")
    print(f"Total runtime: {total_runtime:.1f}s ({total_runtime / 60:.1f} min)")

    constant_rows = combined[is_constant_prediction(combined)]
    print(f"\nRows with constant_prediction True: {len(constant_rows)}")
    if len(constant_rows) > 0:
        columns = ["dataset", "model", "split_idx", "fraction", "n_train"]
        print(constant_rows[columns].to_string(index=False))

    if crashes:
        print(f"\n{len(crashes)} TASK(S) CRASHED AND HAVE NO RESULTS:")
        for crash in crashes:
            print(f"  {crash}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
