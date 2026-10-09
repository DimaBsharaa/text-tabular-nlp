"""Phase 0 diagnostics: what the precomputed MulTaBench baselines tell us before we build anything.

Reads only upstream's precomputed results (`multabench/leaderboard/results/`), plus raw data that is already in the
local Kaggle cache. CPU only: no E5 embeddings, no fine-tuning, no downloads.

Usage (writes CSVs to reports/phase0/):
    python scripts/phase0_diagnostics.py
"""
from __future__ import annotations

import ast
import glob
import json
import os
from os.path import abspath, dirname, exists, join

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = dirname(dirname(abspath(__file__)))
RESULTS_DIR = join(REPO_ROOT, "multabench", "leaderboard", "results")
TEXT_RESULTS_DIR = join(RESULTS_DIR, "text")
OUT_DIR = join(REPO_ROOT, "reports", "phase0")
KAGGLE_CACHE = os.environ.get("KAGGLEHUB_CACHE", join(os.path.expanduser("~"), ".cache", "kagglehub"))

# Upstream state name -> name used in the paper / our brief.
STATES = {
    "no_text": "Unimodal Structured",
    "text_only": "Unimodal Text",
    "all": "Joint Frozen",
    "ft": "Joint TAR",
}
LEARNERS = ["CatBoost", "LightGBM", "TabM", "TabPFN-v2", "TabPFN-v2p5"]
N_FOLDS = 5
ALPHA = 0.05
SEED = 42


def _clean_learner(name: str) -> str:
    """'CatBoost 😸' -> 'CatBoost' (upstream appends an emoji to every model name)."""
    return name.rsplit(" ", 1)[0] if " " in name else name


def friendly_names() -> dict[str, str]:
    """Dataset ID -> paper display name, parsed (not imported) from upstream's do_dataset_summary.py."""
    path = join(REPO_ROOT, "multabench", "scripts", "do_dataset_summary.py")
    tree = ast.parse(open(path).read())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "_FRIENDLY_NAMES":
            return ast.literal_eval(node.value)
    raise KeyError("_FRIENDLY_NAMES not found")


def load_results() -> pd.DataFrame:
    """Long table of every precomputed text-benchmark run: dataset, learner, state, fold, score."""
    frames = [pd.read_csv(f) for f in sorted(glob.glob(join(TEXT_RESULTS_DIR, "*.csv")))]
    df = pd.concat(frames, ignore_index=True)
    df = df[df["State"] == "finished"]
    out = pd.DataFrame({
        "dataset": df["dataset"],
        "learner": df["model"].map(_clean_learner),
        "state": df["multimodal_state"],
        "fold": df["fold"].astype(int),
        "score": df["test_score"].astype(float),
    })
    dup = out.duplicated(["dataset", "learner", "state", "fold"])
    assert not dup.any(), f"{dup.sum()} duplicate (dataset, learner, state, fold) rows"
    return out.sort_values(["dataset", "learner", "state", "fold"]).reset_index(drop=True)


def _task(dataset_id: str) -> str:
    return {"BIN": "binary", "MUL": "multiclass", "REG": "regression"}[dataset_id[:3]]


def _paired(results: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    """Per (dataset, learner, fold): score of state a and state b side by side (only folds where both exist)."""
    wide = results.pivot_table(index=["dataset", "learner", "fold"], columns="state", values="score")
    return wide[[a, b]].dropna().reset_index()


def gain_table(results: pd.DataFrame) -> pd.DataFrame:
    """Per dataset x learner: Joint Frozen vs Joint TAR over the 5 folds.

    gain = TAR - Frozen, computed per fold (same split for both) and then averaged.
    gain_std is the fold std of that paired difference; tar_std is the fold std of TAR's own score.
    exceeds_fold_std flags |gain mean| > tar_std, i.e. the gain is bigger than ordinary split-to-split variation.
    """
    p = _paired(results, "all", "ft")
    p["gain"] = p["ft"] - p["all"]
    g = p.groupby(["dataset", "learner"]).agg(
        n_folds=("fold", "count"),
        frozen_mean=("all", "mean"), frozen_std=("all", "std"),
        tar_mean=("ft", "mean"), tar_std=("ft", "std"),
        gain_mean=("gain", "mean"), gain_std=("gain", "std"),
        gain_folds_positive=("gain", lambda s: int((s > 0).sum())),
    ).reset_index()
    g["exceeds_fold_std"] = g["gain_mean"].abs() > g["tar_std"]
    g.insert(1, "task", g["dataset"].map(_task))
    return g


def condition_means(results: pd.DataFrame) -> pd.DataFrame:
    """Per dataset x learner: fold-mean score of each of the four conditions (columns named as in the paper)."""
    m = results.groupby(["dataset", "learner", "state"])["score"].mean().unstack("state")
    return m[list(STATES)].rename(columns=STATES).reset_index()


def headroom(results: pd.DataFrame) -> pd.DataFrame:
    """Per dataset (averaged over learners): Joint TAR vs the best of the four conditions, and structured vs text.

    For each learner we take the condition with the highest fold-mean score; headroom = best - TAR (>= 0).
    headroom > 0 means some *other* condition beats TAR on that dataset, e.g. dropping the text helps.
    text_minus_struct > 0 means the text alone is more predictive than the structured columns alone.
    """
    m = condition_means(results)
    cond = list(STATES.values())
    m["best_condition"] = m[cond].idxmax(axis=1)
    m["best_score"] = m[cond].max(axis=1)
    m["headroom"] = m["best_score"] - m["Joint TAR"]
    agg = m.groupby("dataset").agg(
        n_learners=("learner", "count"),
        struct=("Unimodal Structured", "mean"),
        text=("Unimodal Text", "mean"),
        frozen=("Joint Frozen", "mean"),
        tar=("Joint TAR", "mean"),
        headroom=("headroom", "mean"),
        learners_tar_is_best=("best_condition", lambda s: int((s == "Joint TAR").sum())),
        most_common_best=("best_condition", lambda s: s.value_counts().index[0]),
    ).reset_index()
    agg["text_minus_struct"] = agg["text"] - agg["struct"]
    agg.insert(1, "task", agg["dataset"].map(_task))
    return agg


def baseline_noise(results: pd.DataFrame) -> pd.DataFrame:
    """Per dataset (averaged over learners): how noisy Joint TAR is across folds, and what that implies.

    tar_fold_std: std of TAR's score over the 5 folds (split-to-split variation).
    paired_diff_std: std over folds of (TAR - Frozen). Two methods evaluated on the *same* splits share most of the
      split noise, so this is the realistic noise level for comparing our method to TAR fold by fold.
    mde_paired: smallest mean improvement over 5 paired folds that a paired t-test would call significant on this
      dataset alone: t(0.975, df=4) * paired_diff_std / sqrt(5).
    mde_unpaired: same, if we ignored the pairing (two independent samples of 5): t(0.975, df=8) * tar_fold_std * sqrt(2/5).
    """
    g = gain_table(results)
    agg = g.groupby("dataset").agg(
        tar_mean=("tar_mean", "mean"),
        tar_fold_std=("tar_std", "mean"),
        paired_diff_std=("gain_std", "mean"),
    ).reset_index()
    t_paired = stats.t.ppf(1 - ALPHA / 2, df=N_FOLDS - 1)
    t_unpaired = stats.t.ppf(1 - ALPHA / 2, df=2 * N_FOLDS - 2)
    agg["mde_paired"] = t_paired * agg["paired_diff_std"] / np.sqrt(N_FOLDS)
    agg["mde_unpaired"] = t_unpaired * agg["tar_fold_std"] * np.sqrt(2 / N_FOLDS)
    agg.insert(1, "task", agg["dataset"].map(_task))
    return agg


def _wilcoxon_critical_w(n: int, alpha: float = ALPHA) -> int:
    """Largest W (smaller rank sum) still significant for a two-sided exact Wilcoxon test with n pairs (no ties)."""
    max_sum = n * (n + 1) // 2
    counts = np.zeros(max_sum + 1)
    counts[0] = 1
    for r in range(1, n + 1):  # number of sign assignments giving each rank sum
        counts[r:] = counts[r:] + counts[:-r].copy()
    cdf = np.cumsum(counts) / 2 ** n
    return int(np.searchsorted(cdf, alpha / 2, side="right") - 1)


def _bootstrap_ci(x: np.ndarray, n_boot: int = 10_000, seed: int = SEED) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def significance_baseline(results: pd.DataFrame) -> pd.DataFrame:
    """Our pre-registered test (claim C1), applied to Joint TAR minus Joint Frozen, one row per learner.

    Per dataset: d = mean over folds(TAR) - mean over folds(Frozen). Then a two-sided Wilcoxon signed-rank test over
    the datasets. w_critical: the test is significant iff min(W+, W-) <= w_critical.
    """
    g = gain_table(results)
    rows = []
    for learner in LEARNERS:
        d = g.loc[g["learner"] == learner, "gain_mean"].to_numpy()
        w = stats.wilcoxon(d, alternative="two-sided")
        lo, hi = _bootstrap_ci(d)
        rows.append({
            "learner": learner,
            "n_datasets": len(d),
            "mean_diff": d.mean(),
            "median_diff": float(np.median(d)),
            "ci95_low": lo,
            "ci95_high": hi,
            "n_positive": int((d > 0).sum()),
            "wilcoxon_W": float(w.statistic),
            "w_critical": _wilcoxon_critical_w(len(d)),
            "p_value": float(w.pvalue),
            "significant": bool(w.pvalue < ALPHA),
        })
    return pd.DataFrame(rows)


def _cached_data_path(slug: str) -> str | None:
    versions = sorted(glob.glob(join(KAGGLE_CACHE, "datasets", "chico89", slug, "versions", "*", "data.csv")))
    return versions[-1] if versions else None


def dataset_overview() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(overview, text_lengths).

    overview: one row per text dataset with task, classes, N, #structured and #text columns (from upstream's
      datasets_summary.csv), plus whether the raw data is cached locally.
    text_lengths: characters per text cell (mean/median per text column), only for datasets already in the local
      Kaggle cache. Text columns are detected with upstream's own detect_text_features.
    """
    from multabench.datasets.all_datasets import MulTaBenchDatasetID
    from multabench.preprocessing.feat_types import detect_text_features

    names = friendly_names()
    ids = sorted(os.path.basename(f)[:-4] for f in glob.glob(join(TEXT_RESULTS_DIR, "*.csv")))
    summary = pd.read_csv(join(RESULTS_DIR, "datasets_summary.csv")).set_index("Dataset")
    rows, lengths = [], []
    for dataset_id in ids:
        s = summary.loc[names[dataset_id]]
        slug = MulTaBenchDatasetID[dataset_id].value
        path = _cached_data_path(slug)
        rows.append({
            "dataset": dataset_id,
            "name": names[dataset_id],
            "task": _task(dataset_id),
            "classes": int(s["Classes"]) if pd.notna(s["Classes"]) else None,
            "N": int(s["N"]),
            "n_struct_cols": int(s["Struct."]),
            "n_text_cols": int(s["Text cols"]),
            "raw_cached": path is not None,
        })
        if path is None:
            continue
        target = json.load(open(join(dirname(path), "metadata.json")))["target"]
        x = pd.read_csv(path).drop(columns=[target])
        for col in detect_text_features(x):
            n_chars = x[col].dropna().astype(str).str.len()
            lengths.append({"dataset": dataset_id, "column": col, "n_nonnull": int(n_chars.size),
                            "mean_chars": float(n_chars.mean()), "median_chars": float(n_chars.median()),
                            "max_chars": int(n_chars.max())})
    overview = pd.DataFrame(rows)
    overview["classes"] = overview["classes"].astype("Int64")
    return overview, pd.DataFrame(lengths)


def summary_table(results: pd.DataFrame) -> pd.DataFrame:
    """One row per dataset: TAR mean, gain over Frozen, noise and headroom (all averaged over learners)."""
    h = headroom(results).set_index("dataset")
    n = baseline_noise(results).set_index("dataset")
    g = gain_table(results).groupby("dataset")["gain_mean"].mean()
    out = pd.DataFrame({
        "task": h["task"],
        "tar_mean": h["tar"],
        "gain_over_frozen": g,
        "tar_fold_std": n["tar_fold_std"],
        "mde_paired": n["mde_paired"],
        "headroom": h["headroom"],
    })
    return out.reset_index()


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    results = load_results()
    overview, lengths = dataset_overview()
    tables = {
        "dataset_overview": overview,
        "text_lengths": lengths,
        "gain_table": gain_table(results),
        "headroom": headroom(results),
        "baseline_noise": baseline_noise(results),
        "significance_baseline": significance_baseline(results),
        "summary": summary_table(results),
    }
    for name, df in tables.items():
        df.to_csv(join(OUT_DIR, f"{name}.csv"), index=False)
        print(f"wrote reports/phase0/{name}.csv  ({len(df)} rows)")
    if lengths.empty:
        print("text_lengths: no raw data cached locally — skipped")
    else:
        print(f"text_lengths: computed for cached datasets only: {sorted(lengths['dataset'].unique())}")


if __name__ == "__main__":
    main()
