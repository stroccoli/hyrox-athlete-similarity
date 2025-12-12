"""Recompute athlete specialty labels using z-scores of run vs working totals.

The script standardizes `run_total` (X) and `working_total` (Y) across the
provided dataset, assigns the lower (better) z-score as the athlete's
`specialty` for each race, and writes the updated CSV.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


RUN_LABEL = "Runner"
WORK_LABEL = "Working Station Specialist"


def _compute_zscores(values: pd.Series) -> pd.Series:
    """Return z-scores; fall back to zeros when std is 0 to avoid NaNs."""
    std = values.std(ddof=0)
    if std == 0:
        return pd.Series(np.zeros(len(values)), index=values.index)
    return (values - values.mean()) / std


def add_specialty(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return df copy with specialty plus the z-scores for run/working.

    The best (lower) z-score decides the specialty. In ties we mark as Runner
    to keep a deterministic outcome.
    """
    run_z = _compute_zscores(df["run_total"])
    working_z = _compute_zscores(df["working_total"])

    labeled = df.copy()
    labeled["specialty"] = np.where(run_z <= working_z, RUN_LABEL, WORK_LABEL)
    return labeled, run_z, working_z


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        default="data/hyrox_elite15_synthetic_dataset.csv",
        help="Ruta al CSV de entrada con columnas run_total, working_total y specialty.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Ruta de salida. Si no se indica, sobrescribe el CSV de entrada.",
    )
    parser.add_argument(
        "--keep-zscores",
        action="store_true",
        help="Si se indica, agrega columnas run_zscore y working_zscore al CSV resultante.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    output_path = Path(args.output_csv) if args.output_csv else input_path

    df = pd.read_csv(input_path)

    # Recalcular especialidad.
    labeled_df, run_z, working_z = add_specialty(df)

    if args.keep_zscores:
        output_df = labeled_df.assign(run_zscore=run_z, working_zscore=working_z)
    else:
        # Mantener el orden original de columnas, reemplazando specialty.
        output_df = df.copy()
        output_df["specialty"] = labeled_df["specialty"]

    output_df.to_csv(output_path, index=False)
    print(f"Especialidad recalculada y guardada en {output_path}")


if __name__ == "__main__":
    main()
