"""Pilot analysis (PROGRESS.md P1): engine-bias check, A/A control, and the advance rule for variants.

Reads result JSONs only (pandas + scipy; no torch). Every comparison is paired on the same (dataset, fold).
    python scripts/pilot_report.py --out RESULTS                      # engine bias + A/A (whatever exists)
    python scripts/pilot_report.py --out RESULTS --variant tar_ctx    # + a variant vs tar_repro
Writes CSVs to --report_dir (default reports/pilot/).
"""
from __future__ import annotations
import os as _os
import sys as _sys
_REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # archive/scripts -> repo
_sys.path[:0] = [_REPO, _os.path.join(_REPO, "scripts"), _os.path.dirname(_os.path.abspath(__file__))]

import argparse
import glob
import json
import os
from os.path import abspath, basename, dirname, join

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = dirname(dirname(dirname(abspath(__file__))))
PILOT_DATASETS = ["REG_TEXT_BOOK_PRICE", "MUL_TEXT_PRODUCT_SENTIMENT", "MUL_TEXT_DATA_SCIENTIST_SALARY",
                  "REG_TEXT_BABIES_PRICES", "REG_TEXT_BOOK_READABILITY", "REG_TEXT_MONTGOMERY_SALARIES"]
PILOT_FOLDS = [0, 1, 2]
ADVANCE_MIN_GAIN = 0.005  # P1 + amendment 1
ADVANCE_MIN_WINS = 4      # of 6 datasets
HEAD_SPLITS = ("e5_train", "e5_val", "learner_val", "test")


def load_runs(out: str, run_name: str, learner: str = "catboost") -> pd.DataFrame:
    """One row per finished pilot job of a run (e.g. tar_repro, tar_repro__s43)."""
    rows = []
    for path in sorted(glob.glob(join(out, run_name, learner, "*", "fold*.json"))):
        if not basename(path)[len("fold"):-len(".json")].isdigit():
            continue  # e.g. fold0.tar_engine-1.json: kept old-engine results are not part of the run
        r = json.load(open(path))
        engine = r.get("engine_version") or ((r.get("tar") or {}).get("config") or {}).get("engine_version")
        hq = (r.get("tar") or {}).get("head_quality", {})
        rows.append({"run_name": run_name, "dataset": r["dataset"], "fold": r["fold"], "score": r["score"],
                     "csv_tar": r["reference"].get("joint_tar"), "engine": engine,
                     "seed": r["config"]["seed"], "commit": r["git"]["commit"][:7],
                     **{f"head_{k}": (hq.get(k) or {}).get("score") for k in HEAD_SPLITS}})
    df = pd.DataFrame(rows, columns=["run_name", "dataset", "fold", "score", "csv_tar", "engine", "seed", "commit"]
                      + [f"head_{k}" for k in HEAD_SPLITS])
    return df[df.dataset.isin(PILOT_DATASETS) & df.fold.isin(PILOT_FOLDS)].reset_index(drop=True)


def load_control(out: str, run_names: list[str], learner: str = "catboost") -> pd.DataFrame:
    """P1 amendment 3: control score per (dataset, fold) = mean over the given runs (tar_repro seeds 42 and 43).
    Only (dataset, fold) pairs present in ALL runs are kept, so an incomplete control can never pass a variant."""
    runs = [load_runs(out, n, learner)[["dataset", "fold", "score", "csv_tar"]] for n in run_names]
    m = runs[0]
    for i, r in enumerate(runs[1:], 1):
        m = m.merge(r[["dataset", "fold", "score"]], on=["dataset", "fold"], suffixes=("", f"_{i}"))
    cols = ["score"] + [f"score_{i}" for i in range(1, len(runs))]
    m["score"] = m[cols].mean(axis=1)
    m["run_name"] = "+".join(run_names)
    return m[["run_name", "dataset", "fold", "score", "csv_tar"]]


def _wilcoxon(d: np.ndarray) -> dict:
    d = np.asarray(d, dtype=float)
    if len(d) < 2 or np.allclose(d, 0):
        return {"n": int(len(d)), "W": None, "p_value": None}
    w = stats.wilcoxon(d, alternative="two-sided")
    return {"n": int(len(d)), "W": float(w.statistic), "p_value": float(w.pvalue)}


def engine_bias(runs: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """(tar_repro − leaderboard ft) per dataset (over folds) and overall, with paired two-sided Wilcoxon tests."""
    r = runs.dropna(subset=["csv_tar"]).assign(diff=lambda d: d.score - d.csv_tar)
    per_ds = r.groupby("dataset").agg(n_folds=("fold", "count"), mean_diff=("diff", "mean"), std_diff=("diff", "std"),
                                      ours=("score", "mean"), leaderboard=("csv_tar", "mean")).reset_index()
    overall = {"n_jobs": int(len(r)), "mean_diff": float(r["diff"].mean()), "std_diff": float(r["diff"].std()),
               "n_negative": int((r["diff"] < 0).sum()),
               "wilcoxon_jobs": _wilcoxon(r["diff"].to_numpy()),          # 18 paired (dataset, fold) scores
               "wilcoxon_datasets": _wilcoxon(per_ds["mean_diff"].to_numpy())}  # 6 fold-averaged differences
    return per_ds, overall


def leakage_summary(runs: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Head quality of the fine-tuning head (AUROC or R²) on E5's own training rows vs unseen rows, per dataset.
    gap = e5_train − test: how much better the fine-tuned encoder 'knows' rows it was trained on."""
    r = runs.assign(gap=lambda d: d.head_e5_train - d.head_test,
                    gap_lval=lambda d: d.head_e5_train - d.head_learner_val)
    per_ds = r.groupby("dataset").agg(n_folds=("fold", "count"), e5_train=("head_e5_train", "mean"),
                                      learner_val=("head_learner_val", "mean"), test=("head_test", "mean"),
                                      gap=("gap", "mean"), gap_std=("gap", "std"),
                                      gap_vs_learner_val=("gap_lval", "mean")).reset_index()
    per_ds["rel_gap"] = per_ds.gap / per_ds.test.abs()
    overall = {"n_jobs": int(r.gap.notna().sum()), "mean_gap": float(r.gap.mean()), "std_gap": float(r.gap.std()),
               "n_positive_gap": int((r.gap > 0).sum()), "wilcoxon_jobs": _wilcoxon(r.gap.dropna().to_numpy())}
    return per_ds, overall


def advance_rule(variant: pd.DataFrame, baseline: pd.DataFrame, min_gain: float = ADVANCE_MIN_GAIN,
                 min_wins: int = ADVANCE_MIN_WINS) -> tuple[pd.DataFrame, dict]:
    """P1: per-dataset gain = mean over folds 0–2 of (variant − baseline) on the same folds; mean over datasets;
    win = gain > 0. Pass iff complete (all 6 × 3 pairs) and mean gain ≥ min_gain and wins ≥ min_wins."""
    m = variant.merge(baseline, on=["dataset", "fold"], suffixes=("_v", "_b"))
    m["diff"] = m.score_v - m.score_b
    per_ds = m.groupby("dataset").agg(n_folds=("fold", "count"), gain=("diff", "mean"),
                                      fold_diff_std=("diff", "std")).reset_index()
    complete = len(m) == len(PILOT_DATASETS) * len(PILOT_FOLDS)
    mean_gain = float(per_ds.gain.mean()) if len(per_ds) else float("nan")
    wins = int((per_ds.gain > 0).sum())
    summary = {"n_pairs": int(len(m)), "complete": complete, "mean_gain": mean_gain, "wins": wins,
               "n_datasets": int(len(per_ds)), "mean_abs_gain": float(per_ds.gain.abs().mean()) if len(per_ds) else None,
               "max_abs_gain": float(per_ds.gain.abs().max()) if len(per_ds) else None,
               "fold_diff_std": float(m["diff"].std()) if len(m) > 1 else None,
               "min_gain": min_gain, "min_wins": min_wins,
               "passes": bool(complete and mean_gain >= min_gain and wins >= min_wins)}
    return per_ds, summary


def aa_threshold_candidates(aa_per_ds: pd.DataFrame, aa: dict) -> dict:
    """Numbers to choose a tighter advance threshold from (amendment 2). Proposals only — Dima decides."""
    se_mean = aa["fold_diff_std"] / np.sqrt(aa["n_pairs"]) if aa["fold_diff_std"] else None
    return {"aa_mean_abs_gain": aa["mean_abs_gain"], "aa_max_abs_gain": aa["max_abs_gain"],
            "aa_noise_sd_of_mean": se_mean,
            "candidate_2x_mean_abs": 2 * aa["mean_abs_gain"],
            "candidate_mean_abs_plus_0.005": aa["mean_abs_gain"] + 0.005,
            "candidate_3sd_of_mean": 3 * se_mean if se_mean else None,
            "candidate_max_abs": aa["max_abs_gain"]}


def _show(title: str, df: pd.DataFrame | None = None, d: dict | None = None) -> None:
    print(f"\n=== {title}")
    if df is not None:
        print(df.to_string(index=False, float_format=lambda v: f"{v:+.4f}" if abs(v) < 1 else f"{v:.4f}"))
    if d is not None:
        for k, v in d.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="results root (the folder containing tar_repro/)")
    p.add_argument("--baseline", default="tar_repro")
    p.add_argument("--aa", default="tar_repro__s43", help="run name of the A/A control")
    p.add_argument("--variant", action="append", default=[], help="run name(s) of variants to test vs the control")
    p.add_argument("--control", default="tar_repro,tar_repro__s43",
                   help="comma-separated runs averaged per (dataset, fold) = the P1 control (amendment 3)")
    p.add_argument("--report_dir", default=join(REPO_ROOT, "archive", "reports", "pilot"))
    a = p.parse_args()
    os.makedirs(a.report_dir, exist_ok=True)
    base = load_runs(a.out, a.baseline)
    print(f"{a.baseline}: {len(base)} / {len(PILOT_DATASETS) * len(PILOT_FOLDS)} pilot jobs, engines {sorted(set(base.engine))}")

    per_ds, overall = engine_bias(base)
    _show(f"Engine bias: {a.baseline} − leaderboard ft (per dataset over folds)", per_ds)
    _show("Engine bias overall (paired, two-sided Wilcoxon)", d={**{k: v for k, v in overall.items() if not k.startswith("wilcoxon")},
          **{f"jobs_{k}": v for k, v in overall["wilcoxon_jobs"].items()},
          **{f"datasets_{k}": v for k, v in overall["wilcoxon_datasets"].items()}})
    per_ds.to_csv(join(a.report_dir, "engine_bias.csv"), index=False)
    json.dump(overall, open(join(a.report_dir, "engine_bias_overall.json"), "w"), indent=1)

    lk_ds, lk = leakage_summary(base)
    _show(f"Leakage diagnostic: head quality e5_train vs test ({a.baseline})", lk_ds)
    _show("Leakage overall (gap = e5_train − test; Wilcoxon: is the gap different from 0?)",
          d={**{k: v for k, v in lk.items() if k != "wilcoxon_jobs"}, **{f"jobs_{k}": v for k, v in lk["wilcoxon_jobs"].items()}})
    lk_ds.to_csv(join(a.report_dir, "leakage.csv"), index=False)

    aa_runs = load_runs(a.out, a.aa)
    if len(aa_runs):
        aa_ds, aa = advance_rule(aa_runs, base)
        _show(f"A/A control: {a.aa} vs {a.baseline} under the advance rule", aa_ds, aa)
        cand = aa_threshold_candidates(aa_ds, aa)
        _show("A/A numbers for a tighter threshold (proposals; amendment 2 says: show Dima before freezing)", d=cand)
        aa_ds.to_csv(join(a.report_dir, "aa_control.csv"), index=False)
        json.dump({**aa, "threshold_candidates": cand}, open(join(a.report_dir, "aa_control.json"), "w"), indent=1)
        if aa["passes"]:
            print("\n  !! The A/A control PASSES the advance rule -> amendment 2: tighten the rule before any variant runs.")
    control = load_control(a.out, a.control.split(","))
    print(f"\ncontrol ({a.control}): {len(control)} / 18 (dataset, fold) pairs with all runs present")
    if not a.variant:  # default: every run found under --out except the control runs (and _logs, _cache, ...)
        skip = set(a.control.split(",")) | {a.baseline, a.aa}
        a.variant = sorted(d for d in os.listdir(a.out) if not d.startswith("_") and d not in skip
                           and glob.glob(join(a.out, d, "*", "*", "fold*.json")))
        print(f"variants found: {a.variant}")
    for v in a.variant:
        v_ds, vs = advance_rule(load_runs(a.out, v), control)
        _show(f"Variant {v} vs control ({a.control}) under the advance rule", v_ds, vs)
        v_ds.to_csv(join(a.report_dir, f"advance_{v}.csv"), index=False)
        json.dump(vs, open(join(a.report_dir, f"advance_{v}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
