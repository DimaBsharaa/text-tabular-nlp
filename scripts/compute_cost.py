"""Time to build the text features: Joint TAR's fine-tuning vs. our method, per (dataset, fold), on the same Colab A100
machine -> results/compute_cost.csv (read by notebooks/02_results.ipynb).
    python scripts/compute_cost.py --export <P16 export>/results
- Joint TAR: our re-implementation's fine-tuning + encoding seconds (GPU; reports/tables/fold_scores.csv, method
  tar_repro). Its fast path is about 11x faster per epoch than upstream's implementation.
- Ours: the cte_depth_top3 feature seconds (experts, probe, layer states, depth probes; CPU, 6 threads, 2 jobs in
  parallel) of the TabPFN-v2 run, the one run in which every expert and probe was built from scratch (no cache hits;
  the frozen embeddings were cached and take seconds).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from os.path import abspath, dirname, join

import pandas as pd

REPO_ROOT = dirname(dirname(abspath(__file__)))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--export", required=True, help="the P16 export's results/ folder")
    a = p.parse_args()
    rows = []
    for path in glob.glob(join(a.export, "cte_depth_top3", "tabpfnv2", "*", "fold[0-4].json")):
        r = json.load(open(path))
        st = r["stacking"]
        assert not any(st["cte"]["expert_cache_hits"].values()), path  # built from scratch
        rows.append({"dataset": r["dataset"], "fold": int(r["fold"]), "n_train": r["n_train"],
                     "ours_seconds": st["seconds"]["base"] + st["seconds"]["probe"] + st["depth"]["seconds"]["states"]
                     + st["depth"]["seconds"]["probes"]})
    fs = pd.read_csv(join(REPO_ROOT, "reports", "tables", "fold_scores.csv"))
    tar = fs[fs.method == "tar_repro"].assign(
        tar_seconds=lambda x: x.tar_train_seconds + x.tar_encode_seconds + x.tar_lower_cache_seconds.fillna(0))
    out = pd.DataFrame(rows).merge(tar[["dataset", "fold", "tar_seconds"]], on=["dataset", "fold"])
    out = out.sort_values(["dataset", "fold"]).round(1)
    os.makedirs(join(REPO_ROOT, "results"), exist_ok=True)
    out.to_csv(join(REPO_ROOT, "results", "compute_cost.csv"), index=False)
    print(f"{len(out)} jobs: Joint TAR {out.tar_seconds.sum() / 60:.0f} min, ours {out.ours_seconds.sum() / 60:.0f} min; "
          f"median per job {out.tar_seconds.median():.0f} s vs {out.ours_seconds.median():.0f} s")


if __name__ == "__main__":
    main()
