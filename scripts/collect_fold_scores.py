"""One row per result JSON (method, learner, dataset, fold): score, leaderboard references, runtimes, engine, commit.
Writes a small CSV the analysis notebook reads, so the notebook needs no Drive export.
    python scripts/collect_fold_scores.py --export .explore/export_20261011_200658/results \
        --export .explore/export_20261014_095322/results --out reports/tables/fold_scores.csv
Later exports win for the same (method, learner, dataset, fold).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from os.path import dirname, join

import pandas as pd


def rows(export: str) -> list[dict]:
    out = []
    for p in sorted(glob.glob(join(export, "*", "*", "*", "fold[0-9].json"))):
        d = json.load(open(p))
        st = d.get("stacking") or {}
        imp = (st.get("catboost_importance") or {}).get("group_share") or {}
        rt = d["runtime_s"]
        out.append({
            "method": d["method"], "learner": d["learner"], "dataset": d["dataset"], "fold": d["fold"],
            "task": d["task"], "score": d["score"],
            "lb_joint_tar": d["reference"].get("joint_tar"), "lb_joint_frozen": d["reference"].get("joint_frozen"),
            "n_train": d["n_train"], "n_test": d["n_test"], "job_seconds": rt["total"],
            "tar_train_seconds": rt.get("tar_train"), "tar_encode_seconds": rt.get("tar_encode"),
            "tar_lower_cache_seconds": rt.get("tar_lower_cache"),
            "emb_seconds": (st.get("seconds") or {}).get("frozen_embeddings"),
            "expert_seconds": (st.get("seconds") or {}).get("experts"),
            "lex_alpha": (st.get("lex_alpha") or {}).get("chosen"),
            **{f"share_{k}": v for k, v in imp.items()},
            "engine_version": d["engine_version"], "commit": d["git"].get("commit", "")[:7], "gpu": d["gpu"],
            "learner_threads": d.get("learner_threads"), "export": os.path.basename(dirname(export.rstrip("/"))),
        })
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--export", action="append", required=True, help="an export's results/ folder (repeat)")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    df = pd.DataFrame([r for e in a.export for r in rows(e)])
    df = df.drop_duplicates(["method", "learner", "dataset", "fold"], keep="last")
    df = df.sort_values(["method", "learner", "dataset", "fold"]).reset_index(drop=True)
    os.makedirs(dirname(a.out), exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"{len(df)} rows -> {a.out}")
    print(df.groupby(["method", "learner"]).size().to_string())


if __name__ == "__main__":
    main()
