"""CSV + booktabs-LaTeX table writers (no jinja2; underscores escaped by hand)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def escape_latex(text: str) -> str:
    return str(text).replace("_", r"\_")


def bold_best_mask(values: pd.DataFrame, higher_is_better: bool) -> pd.DataFrame:
    """Boolean mask, True at the best value(s) in each row (ties all marked bold)."""
    if higher_is_better:
        best = values.max(axis=1)
    else:
        best = values.min(axis=1)
    return values.eq(best, axis=0)


def write_csv_and_tex(
    name: str,
    display: pd.DataFrame,
    output_dir: Path,
    bold_mask: pd.DataFrame | None = None,
    index_label: str = "",
) -> tuple[Path, Path]:
    """Write `display` (a DataFrame of pre-formatted strings) as both CSV and a
    booktabs LaTeX table. `bold_mask`, if given, must align with `display` and marks
    which cells get \\textbf{}. Returns (csv_path, tex_path).
    """
    csv_path = output_dir / f"{name}.csv"
    tex_path = output_dir / f"{name}.tex"

    display.to_csv(csv_path, index=True, index_label=index_label or display.index.name or "")

    n_cols = len(display.columns)
    lines = [
        r"\begin{tabular}{l" + "r" * n_cols + "}",
        r"\toprule",
    ]
    header = [escape_latex(index_label)] + [escape_latex(c) for c in display.columns]
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\midrule")
    for row_label, row in display.iterrows():
        cells = []
        for col in display.columns:
            cell = escape_latex(row[col])
            if bold_mask is not None and bold_mask.loc[row_label, col]:
                cell = r"\textbf{" + cell + "}"
            cells.append(cell)
        lines.append(" & ".join([escape_latex(row_label)] + cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, tex_path


def format_mean_std(mean: float, std: float, decimals: int) -> str:
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def format_float(value: float, decimals: int) -> str:
    if np.isnan(value):
        return "nan"
    return f"{value:.{decimals}f}"
