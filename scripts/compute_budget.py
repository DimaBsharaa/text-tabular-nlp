"""Predicted TAR fine-tuning cost for every text dataset -> reports/compute_budget.csv.

Model: time per fine-tune = E5 training cells × ms per cell per epoch × epochs. Measured on a T4 (old path, E4):
6.3 ms/cell/epoch on Book Price fold 0, 20 epochs. Cost per cell really depends on text length, so this is a
first-order estimate: it overstates datasets with short cells and understates long ones (see chars_per_cell;
Book Price, where 6.3 ms was measured, has ~220 chars/cell). With length bucketing, cost follows tokens, not cells. Pass the speedup measured by scripts/speed_benchmark.py to fill in the fast-path columns:
    python scripts/compute_budget.py --speedup 8.5
"""
from __future__ import annotations

import argparse
from os.path import abspath, dirname, join

import numpy as np
import pandas as pd

REPO_ROOT = dirname(dirname(abspath(__file__)))
PILOT = ["REG_TEXT_BOOK_PRICE", "MUL_TEXT_PRODUCT_SENTIMENT", "MUL_TEXT_DATA_SCIENTIST_SALARY", "REG_TEXT_BABIES_PRICES",
         "REG_TEXT_BOOK_READABILITY", "REG_TEXT_MONTGOMERY_SALARIES"]
TABPFN_SKIPS = ["MUL_TEXT_SPOTIFY_GENRES", "MUL_TEXT_WINE_REVIEW"]
TRAIN_CAP, TEST_RATIO, MAX_TEST, VAL_RATIO, MAX_VAL = 10_000, 0.1, 2000, 0.1, 1000  # upstream split constants


def _val(n: int) -> int:
    return min(int(n * VAL_RATIO), MAX_VAL)


def _chars_per_cell() -> dict:
    """Mean characters per text cell, for datasets whose raw data is in the local Kaggle cache (Phase 0 code)."""
    import sys
    sys.path.insert(0, REPO_ROOT)
    from scripts.phase0_diagnostics import dataset_overview
    _, lengths = dataset_overview()
    if lengths.empty:
        return {}
    w = lengths.assign(chars=lengths.mean_chars * lengths.n_nonnull)
    g = w.groupby("dataset")
    return (g.chars.sum() / g.n_nonnull.sum()).to_dict()


def budget(ms_per_cell_epoch: float, epochs: int, speedup: float | None) -> tuple[pd.DataFrame, dict]:
    o = pd.read_csv(join(REPO_ROOT, "reports", "phase0", "dataset_overview.csv"))
    rows = []
    for _, r in o.iterrows():
        n_test = min(int(r.N * TEST_RATIO), MAX_TEST)
        n_train = min(r.N - n_test, TRAIN_CAP)
        fit_gbdt = n_train - _val(n_train)  # CatBoost / LightGBM / TabM split off their own val rows first
        e5_gbdt = fit_gbdt - _val(fit_gbdt)  # then TAR splits 90/10 for early stopping
        e5_tabpfn = n_train - _val(n_train)  # TabPFN: no learner val split
        row = {"dataset": r.dataset, "task": r.task, "N": r.N, "n_text_cols": r.n_text_cols, "train_rows": n_train,
               "train_cells": n_train * r.n_text_cols,  # simple definition: training rows (capped) × text columns
               "e5_train_cells": e5_gbdt * r.n_text_cols, "e5_train_cells_tabpfn": e5_tabpfn * r.n_text_cols,
               "in_pilot": r.dataset in PILOT, "tabpfn_baseline": r.dataset not in TABPFN_SKIPS}
        row["min_per_finetune_old"] = row["e5_train_cells"] * ms_per_cell_epoch * epochs / 60_000
        row["min_per_finetune_tabpfn_old"] = row["e5_train_cells_tabpfn"] * ms_per_cell_epoch * epochs / 60_000
        if speedup:
            row["min_per_finetune_fast"] = row["min_per_finetune_old"] / speedup
            row["min_per_finetune_tabpfn_fast"] = row["min_per_finetune_tabpfn_old"] / speedup
        rows.append(row)
    df = pd.DataFrame(rows)
    df["chars_per_cell"] = df["dataset"].map(_chars_per_cell())  # NaN = raw data not cached locally
    hours = lambda col, mask=None, folds=5: (df.loc[mask, col] if mask is not None else df[col]).sum() * folds / 60
    summary = {}
    for path in ["old"] + (["fast"] if speedup else []):
        summary[f"pilot_hours_{path} (6 datasets x 3 folds, CatBoost)"] = hours(f"min_per_finetune_{path}", df.in_pilot, 3)
        summary[f"full_hours_{path} (20 x 5 folds, one fine-tune per dataset-fold)"] = hours(f"min_per_finetune_{path}")
        summary[f"full_hours_{path} (+ TabPFN's separate fine-tune, 18 x 5)"] = (
            hours(f"min_per_finetune_{path}") + hours(f"min_per_finetune_tabpfn_{path}", df.tabpfn_baseline))
    return df, summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ms_per_cell_epoch", type=float, default=6.3, help="old path, T4, measured in E4")
    p.add_argument("--epochs", type=int, default=20, help="epochs per fine-tune (E4: 20 = best 17 + patience 3)")
    p.add_argument("--speedup", type=float, default=None, help="fast/old per-epoch speedup from speed_benchmark.py")
    a = p.parse_args()
    df, summary = budget(a.ms_per_cell_epoch, a.epochs, a.speedup)
    out = join(REPO_ROOT, "reports", "compute_budget.csv")
    df.round(1).to_csv(out, index=False)
    pd.set_option("display.width", 250)
    cols = ["dataset", "N", "n_text_cols", "chars_per_cell", "e5_train_cells", "min_per_finetune_old"] + \
           (["min_per_finetune_fast"] if a.speedup else [])
    print(df.sort_values("e5_train_cells")[cols].round(1).to_string(index=False))
    print()
    for k, v in summary.items():
        print(f"{k}: {v:.1f} GPU-hours")
    print(f"assumptions: {a.ms_per_cell_epoch} ms/cell/epoch, {a.epochs} epochs/fine-tune, speedup {a.speedup}")
    print("wrote", out)
