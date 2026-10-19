"""P11/P12 figure: where in the frozen encoder the target signal lives (training rows only).

One small panel per dataset: x = layer of frozen e5-small-v2 (1..12), y = training-row out-of-fold quality of that
layer's depth probe MINUS the quality of cte_probe's final-layer probe (AUROC for classification, Spearman for
regression; > 0 = the layer reads more target signal than the final embedding). The shaded band marks the 3 layers
Joint TAR fine-tunes with LoRA. Inputs, either or both:
  --results  a results root with cte_depth* result JSONs (folds averaged; diagnostics written by depth.run_depth)
  --profile  a CSV from scripts/depth_profile.py (mean pooling rows)
    python scripts/depth_figures.py --results <results_cte> --out reports/figures/fig7_depth_profile
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from os.path import abspath, dirname, join

import numpy as np
import pandas as pd

REPO_ROOT = dirname(dirname(abspath(__file__)))
INK, INK_2, LINE, BAND = "#0b0b0b", "#52514e", "#2a78d6", "#e9e8e4"


def from_results(root: str, run: str = "cte_depth_sl", learner: str = "catboost") -> pd.DataFrame:
    rows = []
    for path in glob.glob(join(root, run, learner, "*", "fold[0-9].json")):
        r = json.load(open(path))
        dd = (r.get("stacking") or {}).get("depth")
        if not dd:
            continue
        final = max(dd["probe_oof_metric_final_layer_cte_probe"])
        rows += [{"dataset": r["dataset"], "fold": r["fold"], "layer": int(l), "delta": float(m) - final}
                 for l, m in dd["layer_oof_metric"].items()]
    df = pd.DataFrame(rows, columns=["dataset", "fold", "layer", "delta"])
    return df.groupby(["dataset", "layer"], as_index=False).delta.mean()


def from_profile(path: str) -> pd.DataFrame:
    p = pd.read_csv(path)
    final = p[p.expert == "probe_final"].set_index("dataset").oof_metric
    d = p[(p.expert == "depth") & (p.pooling == "mean")].copy()
    d["delta"] = d.oof_metric - d.dataset.map(final)
    return d[["dataset", "layer", "delta"]]


def short(name: str) -> str:
    return name.split("_TEXT_", 1)[-1].replace("_", " ").title()


def plot(df: pd.DataFrame, out: str, tar_layers=(10, 11, 12)) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ds = sorted(df.dataset.unique(), key=lambda d: -df[df.dataset == d].delta.max())
    n = len(ds)
    cols = min(5, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.1 * cols, 1.55 * rows + 0.5), sharex=True, squeeze=False)
    lo, hi = min(df.delta.min(), -0.005), max(df.delta.max(), 0.005)
    for ax, d in zip(axes.flat, ds):
        g = df[df.dataset == d].sort_values("layer")
        ax.axvspan(tar_layers[0] - 0.5, tar_layers[-1] + 0.5, color=BAND, lw=0, zorder=0)
        ax.axhline(0, color=INK_2, lw=0.8, ls=(0, (3, 2)), zorder=1)
        ax.plot(g.layer, g.delta, color=LINE, lw=1.6, zorder=2)
        b = g.loc[g.delta.idxmax()]
        ax.plot([b.layer], [b.delta], "o", ms=4.5, color=LINE, mec="white", mew=1.0, zorder=3)
        ax.annotate(f"L{int(b.layer)} {b.delta:+.3f}", (b.layer, b.delta), xytext=(0, 5), textcoords="offset points",
                    ha="center", fontsize=6.5, color=INK)
        ax.set_title(short(d), fontsize=7.5, color=INK, pad=3)
        ax.set_ylim(lo - 0.004, hi + 0.008)
        ax.set_xticks([1, 4, 8, 12])
        ax.tick_params(labelsize=6.5, colors=INK_2, length=2)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#c3c2b7")
    for ax in axes[:, 0]:
        ax.set_ylabel("OOF quality vs\nfinal-layer probe", fontsize=7, color=INK_2)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.supxlabel("layer of frozen e5-small-v2 (shaded: the 3 layers Joint TAR fine-tunes)", fontsize=7.5, color=INK_2)
    fig.tight_layout(pad=0.4)
    os.makedirs(dirname(out) or ".", exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=200)
    df.to_csv(f"{out}.csv", index=False)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default=None)
    p.add_argument("--run", default="cte_depth_sl")
    p.add_argument("--profile", default=None)
    p.add_argument("--out", default=join(REPO_ROOT, "reports", "figures", "fig7_depth_profile"))
    a = p.parse_args()
    parts = ([from_results(a.results, a.run)] if a.results else []) + ([from_profile(a.profile)] if a.profile else [])
    df = pd.concat(parts).groupby(["dataset", "layer"], as_index=False).delta.mean()
    plot(df, a.out)
    best = df.loc[df.groupby("dataset").delta.idxmax()]
    print(best.to_string(index=False))


if __name__ == "__main__":
    main()
