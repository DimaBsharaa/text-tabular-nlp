"""P17 analysis, written before any P17 result. C1-C11 stay as reported.

C12: is the gain robust to the method's own randomness, the assignment of training rows to the 5 cross-fitting folds?
     cte_depth_top3 with cross-fitting fold seeds 43 and 44 (methods cte_depth_top3_xs43 / _xs44), LightGBM, the 20
     datasets, folds 0-4, each vs the leaderboard Joint TAR of LightGBM; two-sided Wilcoxon, Holm over the 2 seeds;
     pass = Holm p < 0.05 AND positive mean for BOTH seeds. Descriptive: the same comparison for seed 42 (C10), and per
     dataset the spread of the fold-averaged gain over the 3 seeds.
    python scripts/p17_report.py --out <export>/results
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from os.path import abspath, dirname, join

import numpy as np

sys.path.insert(0, dirname(abspath(__file__)))
import full_report as fr  # noqa: E402
from p5_report import comparison, leaderboard_state  # noqa: E402

REPO_ROOT = dirname(dirname(abspath(__file__)))
RUNS = {42: "cte_depth_top3", 43: "cte_depth_top3_xs43", 44: "cte_depth_top3_xs44"}


def p17_report(out: str) -> dict:
    lb = leaderboard_state("lgbm", "ft")
    per_seed = {s: comparison(fr.ours(out, run, "lgbm"), lb) for s, run in RUNS.items()}
    rep = {"label": "P17 - claim C12 (cross-fitting fold seeds 43, 44; LightGBM); C1-C11 unchanged",
           "C12": {f"seed{s}": per_seed[s] for s in (43, 44)}, "seed42_C10": per_seed[42]}
    adj = fr.holm({k: c.get("p_value") for k, c in rep["C12"].items() if c["n"] == 20})
    for k, c in rep["C12"].items():
        c["p_holm"] = adj.get(k)
        c["passes"] = None if c["n"] != 20 else bool(c["p_holm"] < fr.ALPHA and (c.get("mean_diff") or 0) > 0)
    passes = [c["passes"] for c in rep["C12"].values()]
    rep["C12_passes"] = None if None in passes else all(passes)
    common = sorted(set.intersection(*[set(c["per_dataset"]) for c in per_seed.values()]))
    if common:
        g = np.array([[per_seed[s]["per_dataset"][d] for s in RUNS] for d in common])  # datasets x seeds
        rep["seed_spread"] = {
            "datasets": len(common),
            "mean_gain_per_seed": {f"seed{s}": round(float(g[:, i].mean()), 5) for i, s in enumerate(RUNS)},
            "per_dataset_sd_over_seeds_median": round(float(np.median(g.std(axis=1, ddof=1))), 5),
            "per_dataset_sd_over_seeds_max": round(float(g.std(axis=1, ddof=1).max()), 5),
            "datasets_same_sign_all_seeds": int((np.sign(g) == np.sign(g[:, [0]])).all(axis=1).sum()),
        }
    return rep


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--report_dir", default=join(REPO_ROOT, "reports", "p17"))
    a = p.parse_args()
    rep = p17_report(a.out)
    os.makedirs(a.report_dir, exist_ok=True)
    json.dump(rep, open(join(a.report_dir, "p17_report.json"), "w"), indent=1)
    print(rep["label"], "| C12 passes:", rep["C12_passes"])
    for k, c in {**rep["C12"], "seed42 (C10)": rep["seed42_C10"]}.items():
        print(f"{k:<13} n={c['n']} mean={c.get('mean_diff')} median={c.get('median_diff')} p={c.get('p_value')} "
              f"p_holm={c.get('p_holm')} wins={c.get('wins')} passes={c.get('passes')}")
    print("seed spread:", rep.get("seed_spread"))


if __name__ == "__main__":
    main()
