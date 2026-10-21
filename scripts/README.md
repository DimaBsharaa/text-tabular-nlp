# scripts/

| Group | Script | What it does |
|---|---|---|
| **Use the method** | [`train.py`](train.py) | Train the final method (`cte_depth_top3`) with any of the 5 learners on a MulTaBench dataset or any CSV; saves a model folder |
| | [`predict.py`](predict.py) | Predict new rows with a saved model folder (nothing is re-fitted) |
| **Run the benchmark** | [`run_experiment.py`](run_experiment.py) | One (method, learner, dataset, fold) job, or a whole jobs file from [`jobs/`](../jobs/); one result JSON per job, resumable |
| | [`dev_eval.py`](dev_eval.py) | Development comparisons on splits of the *training* rows only (test rows never used) |
| | [`progress_table.py`](progress_table.py) | Which jobs of a jobs file are done, with their scores |
| **Final analysis** | [`p14_report.py`](p14_report.py) | Claim C9 (CatBoost, 20 datasets) and C7 (36 further datasets) → [`reports/p14/`](../reports/p14/) |
| | [`p15_report.py`](p15_report.py) | Claim C10 (LightGBM, TabM) → [`reports/p15/`](../reports/p15/) |
| | [`p16_report.py`](p16_report.py) | Claim C11 (TabPFN-v2, TabPFN-2.5) and the Holm correction over all tests → [`reports/p16/`](../reports/p16/) |
| | [`p17_report.py`](p17_report.py) | Robustness to the cross-fitting fold seed (LightGBM, seeds 43 / 44) → `reports/p17/` |
| | [`collect_fold_scores.py`](collect_fold_scores.py) | Result JSONs → [`reports/tables/fold_scores.csv`](../reports/tables/fold_scores.csv) (every run, one row each) |
| | [`make_results.py`](make_results.py) | `fold_scores.csv` → [`results/`](../results/) (final method vs Joint TAR / Joint Frozen, per fold, dataset, learner) |
| | [`build_results_notebook.py`](build_results_notebook.py) | Builds and runs [`notebooks/02_results.ipynb`](../notebooks/02_results.ipynb) from `results/` and `reports/` |
| | [`depth_figures.py`](depth_figures.py) | The depth-profile figure (which encoder layer predicts the target best) |
| | [`compute_cost.py`](compute_cost.py), [`class_balance.py`](class_balance.py) | `results/compute_cost.csv`, `results/class_balance.csv` |
| **Trust checks** | [`check_recompute.py`](check_recompute.py) | Recomputes every headline number from the raw result files with no project code |
| | [`check_label_shuffle.py`](check_label_shuffle.py) | Trains with shuffled training labels: the method must fall to chance (it does) |
| **Earlier stages** (kept because the final reports import their tested statistics) | [`full_report.py`](full_report.py), [`p5_report.py`](p5_report.py), [`p11_report.py`](p11_report.py) | Analyses of earlier claims: `full_report.py` C1, `p5_report.py` the first P5 plan, `p11_report.py` the first C7 plan and C8; `full_report.py` holds the shared Wilcoxon / bootstrap / Holm code |
| | [`phase0_diagnostics.py`](phase0_diagnostics.py) | What the published baselines show before any method was built → [`reports/phase0/`](../reports/phase0/), notebook 01 |
| | [`depth_profile.py`](depth_profile.py) | Layer-by-layer probe scores on training rows, used while developing the depth probes |

The `pNN` names refer to the pre-registration in [`archive/PROGRESS.md`](../archive/PROGRESS.md) that each analysis
belongs to; each report script was committed before its run.
