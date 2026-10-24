"""Every row of reports/tables/fold_scores.csv against the committed raw result JSON it was collected from: the final
method and the depth-probe version (incl. other cross-fitting seeds) in results/raw/, the earlier development runs in
archive/results/raw/<export>/. Checks the score, the leaderboard references and the split sizes.
    python scripts/check_fold_scores.py
"""
from __future__ import annotations

import json
import math
from os.path import abspath, dirname, join

import pandas as pd

REPO_ROOT = dirname(dirname(abspath(__file__)))


def raw_path(r) -> str:
    rel = join(r.method, r.learner, r.dataset, f"fold{r.fold}.json")
    if r.method.startswith("cte_depth_top3"):
        return join(REPO_ROOT, "results", "raw", rel)
    return join(REPO_ROOT, "archive", "results", "raw", r.export, rel)


def same(a, b) -> bool:
    if a is None or (isinstance(a, float) and math.isnan(a)):
        return b is None or (isinstance(b, float) and math.isnan(b))
    return b is not None and abs(float(a) - float(b)) <= 1e-9


def main() -> list[str]:
    fs = pd.read_csv(join(REPO_ROOT, "reports", "tables", "fold_scores.csv"))
    bad = []
    for r in fs.itertuples():
        d = json.load(open(raw_path(r)))
        ref = d.get("reference") or {}
        checks = [(r.score, d["score"]), (r.lb_joint_tar, ref.get("joint_tar")), (r.lb_joint_frozen, ref.get("joint_frozen")),
                  (r.n_train, d["n_train"]), (r.n_test, d["n_test"])]
        if (d["method"], d["learner"], d["dataset"], d["fold"]) != (r.method, r.learner, r.dataset, r.fold) or \
                not all(same(a, b) for a, b in checks):
            bad.append(f"{r.method} {r.learner} {r.dataset} fold {r.fold}")
    print(f"{len(fs)} rows checked against their raw result files: {len(bad)} mismatches")
    return bad


if __name__ == "__main__":
    raise SystemExit(1 if main() else 0)
