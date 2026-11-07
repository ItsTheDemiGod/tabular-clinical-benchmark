"""Fixed orderings and parameters shared by every Phase 3a subcommand."""
from __future__ import annotations

from pathlib import Path

SEED: int = 42
BOOTSTRAP_B: int = 5000

# Fixed model order used in every table and figure.
MODELS: list[str] = ["logreg", "rf", "xgb", "lgbm", "mlp", "ftt", "tabnet"]

# Families, for plotting only (colour/line-style grouping).
FAMILY: dict[str, str] = {
    "logreg": "linear",
    "rf": "trees",
    "xgb": "trees",
    "lgbm": "trees",
    "mlp": "deep",
    "ftt": "deep",
    "tabnet": "deep",
}

DATASETS_ALL5: list[str] = [
    "pima_diabetes",
    "heart_cleveland",
    "ckd",
    "ilpd",
    "breast_wisconsin",
]
DATASETS_NO_CKD: list[str] = [d for d in DATASETS_ALL5 if d != "ckd"]

FRACTIONS_TEST: list[float] = [1.0, 0.1]
FRACTIONS_ALL: list[float] = [0.1, 0.25, 0.5, 1.0]

METRICS_TESTED: list[str] = ["auroc", "log_loss", "brier"]
# True if a larger value is better for that metric.
HIGHER_IS_BETTER: dict[str, bool] = {"auroc": True, "log_loss": False, "brier": False}

ALL_RUNS_PATH: Path = Path("results/all_runs.csv")
RAW_DIR: Path = Path("results/raw")
OUTPUT_DIR: Path = Path("results/phase3a")

# Colour-blind-safe (Okabe-Ito) palette, one colour per model, plus a distinct
# marker per model so points remain distinguishable in black-and-white print.
MODEL_COLOR: dict[str, str] = {
    "logreg": "#000000",
    "rf": "#E69F00",
    "xgb": "#56B4E9",
    "lgbm": "#009E73",
    "mlp": "#D55E00",
    "ftt": "#CC79A7",
    "tabnet": "#0072B2",
}
MODEL_MARKER: dict[str, str] = {
    "logreg": "o",
    "rf": "s",
    "xgb": "^",
    "lgbm": "D",
    "mlp": "v",
    "ftt": "P",
    "tabnet": "X",
}
# Solid for linear/trees, dashed for deep models (per family).
FAMILY_LINESTYLE: dict[str, str] = {"linear": "-", "trees": "-", "deep": "--"}

METRIC_LABEL: dict[str, str] = {
    "auroc": "AUROC",
    "log_loss": "Log loss",
    "brier": "Brier score",
}
