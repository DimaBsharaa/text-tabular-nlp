"""Full-run analysis for the frozen claim C1. Written before any full-run result exists.

Primary: CatBoost, frozen_xfit_pred vs the leaderboard Joint TAR (`ft`), two-sided Wilcoxon signed-rank over the 20
datasets on fold-averaged differences; pass = p < 0.05 AND mean difference > 0.
Secondary (all reported regardless of outcome): LightGBM, TabM, TabPFN-v2, TabPFN-v2.5 (n = 18) vs the leaderboard
`ft`, Holm-corrected over these 4; and CatBoost frozen_xfit_pred vs our own tar_repro (CatBoost).
For every comparison: n, mean / median difference, 95% bootstrap CI of the mean (10,000 resamples of datasets,
seed 42), Wilcoxon W and p, matched-pairs rank-biserial r, Cohen's d_z, wins.
A dataset enters a comparison only if all 5 folds exist on both sides (listed if excluded).

    python scripts/full_report.py --out MyDrive/.../results_full [--report_dir reports/full]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from os.path import abspath, basename, dirname, join

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = dirname(dirname(abspath(__file__)))
METHOD = "frozen_xfit_pred"
PRIMARY = "catboost"
SECONDARY = ["lgbm", "tabm", "tabpfnv2", "tabpfnv2p5"]
MODEL_NAMES = {"catboost": "CatBoost", "lgbm": "LightGBM", "tabm": "TabM", "tabpfnv2": "TabPFN-v2",
               "tabpfnv2p5": "TabPFN-v2p5"}
FOLDS = [0, 1, 2, 3, 4]
ALPHA = 0.05


def leaderboard(learner: str) -> pd.DataFrame:
    """Upstream `ft` (Joint TAR) scores per (dataset, fold) for one learner."""
    rows = []
    for path in glob.glob(join(REPO_ROOT, "multabench", "leaderboard", "results", "text", "*.csv")):
        df = pd.read_csv(path)
        df = df[(df.multimodal_state == "ft") & (df.model.str.rsplit(" ", n=1).str[0] == MODEL_NAMES[learner])]
        rows += [{"dataset": basename(path)[:-4], "fold": int(r.fold), "score": float(r.test_score)} for r in df.itertuples()]
    return pd.DataFrame(rows, columns=["dataset", "fold", "score"])


def ours(out: str, run: str, learner: str) -> pd.DataFrame:
    rows = []
    for path in glob.glob(join(out, run, learner, "*", "fold*.json")):
        if not basename(path)[4:-5].isdigit():
            continue
        r = json.load(open(path))
        rows.append({"dataset": r["dataset"], "fold": r["fold"], "score": r["score"], "gpu": r.get("gpu")})
    return pd.DataFrame(rows, columns=["dataset", "fold", "score", "gpu"])


def fold_averaged_differences(a: pd.DataFrame, b: pd.DataFrame) -> tuple[pd.Series, list[str]]:
    """Per dataset: mean over folds 0-4 of (a - b); only datasets with all 5 folds on both sides."""
    m = a.merge(b, on=["dataset", "fold"], suffixes=("_a", "_b"))
    m = m[m.fold.isin(FOLDS)]
    complete = m.groupby("dataset").fold.nunique() == len(FOLDS)
    keep = complete[complete].index
    excluded = sorted((set(a.dataset) | set(b.dataset)) - set(keep))
    d = (m[m.dataset.isin(keep)].assign(diff=lambda x: x.score_a - x.score_b).groupby("dataset")["diff"].mean())
    return d, excluded


def compare(d: pd.Series, n_boot: int = 10_000, seed: int = 42) -> dict:
    v = d.to_numpy(dtype=float)
    out = {"n": int(len(v)), "mean_diff": float(v.mean()) if len(v) else None,
           "median_diff": float(np.median(v)) if len(v) else None, "wins": int((v > 0).sum())}
    if len(v) < 2:
        return out
    rng = np.random.default_rng(seed)
    boots = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    out["ci95"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    nz = v[v != 0]
    w = stats.wilcoxon(v, alternative="two-sided")
    ranks = stats.rankdata(np.abs(nz))
    w_plus, w_minus = ranks[nz > 0].sum(), ranks[nz < 0].sum()
    out.update({"wilcoxon_W": float(w.statistic), "p_value": float(w.pvalue),
                "rank_biserial_r": float((w_plus - w_minus) / (w_plus + w_minus)) if len(nz) else 0.0,
                "cohens_dz": float(v.mean() / v.std(ddof=1)) if v.std(ddof=1) > 0 else None})
    return out


def holm(pvals: dict) -> dict:
    """Holm-Bonferroni adjusted p-values (monotone), keyed like the input."""
    items = sorted((p, k) for k, p in pvals.items() if p is not None)
    m, adj, running = len(items), {}, 0.0
    for i, (p, k) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adj[k] = running
    return adj


def full_report(out: str) -> dict:
    rep = {"method": METHOD, "primary": {}, "secondary": {}, "vs_tar_repro": {}}
    gpus = set()
    for learner in [PRIMARY] + SECONDARY:
        mine = ours(out, METHOD, learner)
        gpus |= set(mine.gpu.dropna())
        d, excl = fold_averaged_differences(mine, leaderboard(learner))
        res = {**compare(d), "excluded_datasets": excl, "per_dataset": {k: round(float(v), 5) for k, v in d.items()}}
        (rep["primary"] if learner == PRIMARY else rep["secondary"])[learner] = res
    adj = holm({k: v.get("p_value") for k, v in rep["secondary"].items()})
    for k, v in rep["secondary"].items():
        v["p_holm"] = adj.get(k)
        v["significant_holm"] = bool(v["p_holm"] is not None and v["p_holm"] < ALPHA)
    p = rep["primary"][PRIMARY]
    p["passes"] = bool(p.get("p_value") is not None and p["p_value"] < ALPHA and (p["mean_diff"] or 0) > 0)
    tr = ours(out, "tar_repro", PRIMARY)
    gpus |= set(tr.gpu.dropna())
    d, excl = fold_averaged_differences(ours(out, METHOD, PRIMARY), tr)
    rep["vs_tar_repro"] = {**compare(d), "excluded_datasets": excl,
                           "per_dataset": {k: round(float(v), 5) for k, v in d.items()}}
    rep["gpu_types"] = sorted(gpus)
    rep["same_gpu_type"] = len(gpus) <= 1
    return rep


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--report_dir", default=join(REPO_ROOT, "reports", "full"))
    a = p.parse_args()
    rep = full_report(a.out)
    os.makedirs(a.report_dir, exist_ok=True)
    json.dump(rep, open(join(a.report_dir, "c1_report.json"), "w"), indent=1)
    pr = rep["primary"][PRIMARY]
    print(f"PRIMARY (CatBoost vs leaderboard ft): n={pr['n']} mean={pr.get('mean_diff')} CI={pr.get('ci95')} "
          f"p={pr.get('p_value')} r={pr.get('rank_biserial_r')} -> {'PASS' if pr['passes'] else 'no pass'}")
    for k, v in rep["secondary"].items():
        print(f"secondary {k:<11} n={v['n']} mean={v.get('mean_diff')} p={v.get('p_value')} p_holm={v.get('p_holm')}")
    v = rep["vs_tar_repro"]
    print(f"vs tar_repro (CatBoost): n={v['n']} mean={v.get('mean_diff')} CI={v.get('ci95')} p={v.get('p_value')}")
    print("GPU types:", rep["gpu_types"], "| same type:", rep["same_gpu_type"])


if __name__ == "__main__":
    main()
