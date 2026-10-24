# reports/

The pre-registered analysis of each final claim, written by its `scripts/pNN_report.py` (committed before the run),
plus the raw tables and checks the results are built from. The summary numbers for readers are in
[`results/`](../results/); this folder is the evidence behind them.

| Folder | Contents |
|---|---|
| [`p14/`](p14/) | C9: the depth-probe version (`cte_depth_top3`, the final method without the attention probe) with CatBoost on the 20 datasets; C7: the 36 further datasets never used in development |
| [`p15/`](p15/) | C10: LightGBM and TabM |
| [`p16/`](p16/) | C11: TabPFN-v2 and TabPFN-2.5 (depth-probe version) |
| [`p17/`](p17/) | Robustness to the cross-fitting fold seed: the depth-probe version with LightGBM and fold seeds 43 and 44 (C12; both pass, Holm p = 0.019) |
| [`p18/`](p18/) | C13: the final method (`cte_depth_top3_attn`, with the attention probe) with all five learners; paired comparison with the depth-probe version; Holm correction over all 24 pre-registered tests of the project; `a100_check.csv` (+ `a100_check/`): 19 of its Tesla T4 jobs re-run on an A100 from the published code; `provenance_t4.csv`: the source version of each Tesla T4 result |
| [`dev/`](dev/) | Development rounds on splits of the training rows: depth-probe aggregations (`e51/`) and the attention probe (`e52/`, with its label-shuffle check) |
| [`tables/`](tables/) | `fold_scores.csv`: one row per run (method, learner, dataset, fold) of every full run, with the leaderboard scores of the same fold; `table_ablations.csv`: the expert ablations (CatBoost) |
| [`checks/`](checks/) | `label_shuffle.csv`: the leakage check of the depth-probe version (shuffled labels → chance); the final method's is in `dev/e52/` |
| [`phase0/`](phase0/) | What the published MulTaBench baselines show before building anything (fine-tuning gain, fold noise); notebook 01 |
| [`depth/`](depth/) | The first depth profile (4 datasets, development run) |
| [`figures/`](figures/) | The depth profile from that development run (`fig7_depth_profile_dev.*`) and from all 20 datasets of the final CatBoost run (`fig7_depth_profile_20.*`, the data of notebook 02's heatmap in [`results/figures/`](../results/figures/)) |
