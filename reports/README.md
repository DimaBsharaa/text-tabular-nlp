# reports/

The pre-registered analysis of each final claim, written by its `scripts/pNN_report.py` (committed before the run),
plus the raw tables and checks the results are built from. The summary numbers for readers are in
[`results/`](../results/); this folder is the evidence behind them.

| Folder | Contents |
|---|---|
| [`p14/`](p14/) | C9: final method with CatBoost on the 20 datasets; C7: the 36 further datasets never used in development |
| [`p15/`](p15/) | C10: LightGBM and TabM |
| [`p16/`](p16/) | C11: TabPFN-v2 and TabPFN-2.5; Holm correction over all 17 pre-registered tests of the project |
| `p17/` | Robustness to the cross-fitting fold seed (added when that run is analysed) |
| [`tables/`](tables/) | `fold_scores.csv`: one row per run (method, learner, dataset, fold) of every full run, with the leaderboard scores of the same fold; `table_ablations.csv`: the expert ablations (CatBoost) |
| [`checks/`](checks/) | `label_shuffle.csv`: the leakage check (shuffled labels → chance) |
| [`phase0/`](phase0/) | What the published MulTaBench baselines show before building anything (fine-tuning gain, fold noise); notebook 01 |
| [`depth/`](depth/) | The first depth profile (4 datasets, development run) |
| [`figures/`](figures/) | The depth profile from that development run (`fig7_depth_profile_dev.*`) and from all 20 datasets of the final run (`fig7_depth_profile_20.*`, the data of notebook 02's heatmap in [`results/figures/`](../results/figures/)) |
