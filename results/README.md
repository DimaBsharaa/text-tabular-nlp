# results/

The evaluation results of the final method (`cte_depth_top3_attn`) on the 20 MulTaBench text-tabular datasets, next to
the published Joint TAR and Joint Frozen scores of the same learner, dataset and fold. AUROC for classification
(macro one-vs-rest for multiclass), R² for regression, computed by the benchmark's own code.

| File | Contents |
|---|---|
| [`results_summary.csv`](results_summary.csv) | Our scores only, one row per learner: mean over all datasets (as in the paper's Table 1), mean AUROC on classification, mean R² on regression |
| [`results_per_dataset.csv`](results_per_dataset.csv) | Our scores only, one row per dataset: task type, metric and the fold-averaged score of each learner |
| [`results_per_fold.csv`](results_per_fold.csv) | Our scores only, one row per run (dataset, learner, fold): the 480 final runs |
| [`summary.csv`](summary.csv) | One row per learner: mean scores, mean / median gain over Joint TAR, 95% bootstrap interval, datasets won, two-sided Wilcoxon p over datasets |
| [`scores_per_dataset.csv`](scores_per_dataset.csv) | One row per (learner, dataset): scores averaged over the 5 folds |
| [`scores_per_fold.csv`](scores_per_fold.csv) | One row per (learner, dataset, fold): the 480 final runs |
| [`class_balance.csv`](class_balance.csv) | Class shares of the 10 classification datasets (from the benchmark's loader) |
| [`compute_cost.csv`](compute_cost.csv) | Feature-building time of the depth-probe version vs Joint TAR fine-tuning, per (dataset, fold), same machine |
| [`figures/`](figures/) | The figures of [notebook 02](../notebooks/02_results.ipynb) |
| [`raw/`](raw/) | One result JSON per run (score, leaderboard references, config, git commit, hardware): the final method (`cte_depth_top3_attn`, all 480 runs on an NVIDIA A100) and the depth-probe version (`cte_depth_top3`, also with cross-fitting seeds 43 / 44). |

The submitted results zip contains the three `results_*.csv` files; the other files here are in the code zip.

TabPFN cannot run Spotify Genres (114 classes) and Wine Review (30 classes), so MulTaBench has no TabPFN baseline there
and its rows cover 18 datasets.

Rebuild and check:

```bash
.venv/bin/python scripts/make_results.py          # reports/tables/fold_scores.csv -> the three scores CSVs
.venv/bin/python scripts/check_fold_scores.py     # every fold_scores.csv row against its raw JSON
.venv/bin/python scripts/check_recompute.py --method cte_depth_top3_attn --learner catboost   # from raw/ and the leaderboard only
.venv/bin/python scripts/build_results_notebook.py   # notebook 02 and figures/
```
