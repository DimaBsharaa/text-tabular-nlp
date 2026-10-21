# jobs/

Each file lists the benchmark jobs of one final run, one per line: `method learner dataset fold`. Run with
`python scripts/run_experiment.py --jobs_file jobs/<file> --out <results root>` (or the Colab runner,
[`notebooks/colab_run.ipynb`](../notebooks/colab_run.ipynb)). Finished jobs are skipped.

| File | Run | Analysis |
|---|---|---|
| [`p14_A.txt`](p14_A.txt) | Final method, CatBoost, 20 datasets × 5 folds (100 jobs) | `reports/p14/` (C9) |
| [`p14_B.txt`](p14_B.txt) | Final method and its earlier version without depth probes (`cte_probe`), CatBoost, 36 further datasets × folds 0–2 (216 jobs) | `reports/p14/` (C7) |
| [`p15.txt`](p15.txt) | Final method, LightGBM and TabM, 20 × 5 (200 jobs) | `reports/p15/` (C10) |
| [`p16.txt`](p16.txt) | Final method, TabPFN-v2 and TabPFN-2.5, 18 × 5 (180 jobs) | `reports/p16/` (C11) |
| [`p16_timing.txt`](p16_timing.txt) | 6 jobs used to measure TabPFN runtimes before the full run | – |
| [`p17.txt`](p17.txt) | Final method, LightGBM, cross-fitting fold seeds 43 and 44, 20 × 5 each (200 jobs) | `reports/p17/` |

Development runs (earlier methods) are in [`archive/jobs/`](../archive/jobs/).
