"""Class balance of the MulTaBench classification datasets (target only, label-free statistics of the loaded data)
-> results/class_balance.csv, used by notebooks/02_results.ipynb.
    python scripts/class_balance.py
Loads each dataset with the benchmark's own loader (as in scripts/run_experiment.py); a dataset that cannot be
downloaded here is listed and skipped.
"""
from __future__ import annotations

import os
import sys
from os.path import abspath, dirname, join

REPO_ROOT = dirname(dirname(abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.environ.setdefault("WANDB_MODE", "disabled")


def main() -> None:
    import pandas as pd
    from multabench.datasets.downloading import download_dataset
    from multabench.datasets.multimodal import MultimodalState
    from multabench.datasets.utils import dataset_from_name
    scores = pd.read_csv(join(REPO_ROOT, "results", "scores_per_dataset.csv"))
    names = sorted(d for d in scores.dataset.unique() if not d.startswith("REG_"))
    rows = []
    for name in names:
        try:
            ds = download_dataset(dataset_id=dataset_from_name(name=name), multimodal_state=MultimodalState.ALL,
                                  target_override=None)
        except Exception as e:  # noqa: BLE001
            print(f"{name}: skipped ({type(e).__name__}: {str(e)[:120]})")
            continue
        share = ds.y.astype(str).value_counts(normalize=True)
        rows.append({"dataset": name, "rows": len(ds.y), "n_classes": len(share), "minority_share": share.min(),
                     "majority_share": share.max()})
        print(f"{name}: {len(share)} classes, minority {share.min():.3f}, majority {share.max():.3f}")
    pd.DataFrame(rows).round(6).to_csv(join(REPO_ROOT, "results", "class_balance.csv"), index=False)


if __name__ == "__main__":
    main()
