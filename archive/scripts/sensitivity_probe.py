"""Sensitivity probe: how much does each method's TEXT embedding move
when one structured feature of a row changes?

For N test rows and each of the top-k structured features (full-fit CatBoost importance on the E5 pool), make a
perturbed copy of the rows — numeric: move the value one pool quantile bin (1/16) up (down at the top end);
categorical: replace it by the most frequent other category — and pass the copies to a normal job as extra row
sets. They get embeddings through the regular path but are never trained on. Output: per method / feature / text
column, the mean and median cosine change 1 − cos(e_original, e_perturbed). By construction tar_repro and offset
give 0 (their encoder never sees structured values); prefix / context show how much the text representation
depends on the structure.

    python scripts/sensitivity_probe.py --dataset REG_TEXT_BOOK_PRICE --fold 0 --methods tar_repro prefix context \\
        --n_rows 200 --out results/_probe
Each method fine-tunes once (no result cache: the probe rows change the cache key). Writes <out>/sensitivity_<ds>_fold<k>.csv.
"""
from __future__ import annotations
import os as _os
import sys as _sys
_REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # archive/scripts -> repo
_sys.path[:0] = [_REPO, _os.path.join(_REPO, "scripts"), _os.path.dirname(_os.path.abspath(__file__))]

import argparse
import os
import sys
from os.path import abspath, dirname, join

REPO_ROOT = dirname(dirname(dirname(abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, join(REPO_ROOT, "scripts"))
os.environ.setdefault("WANDB_MODE", "disabled")

PROBE_INDEX_OFFSET = 10 ** 9  # probe copies get disjoint row indices


def perturb(x_rows, x_ref, feature: str, kind: str):
    import numpy as np
    import pandas as pd
    out = x_rows.copy()
    if kind == "num":
        ref = pd.to_numeric(x_ref[feature], errors="coerce").dropna().to_numpy(dtype=float)
        v = pd.to_numeric(out[feature], errors="coerce").to_numpy(dtype=float)
        q = np.searchsorted(np.sort(ref), v, side="right") / max(len(ref), 1)  # empirical quantile
        q_new = np.where(q + 1 / 16 <= 1, q + 1 / 16, q - 1 / 16)
        new = np.quantile(ref, np.clip(q_new, 0, 1)) if len(ref) else v
        out[feature] = np.where(np.isnan(v), v, new)
    else:
        top = x_ref[feature].astype(str).value_counts().index.tolist()
        out[feature] = [next((c for c in top if c != str(v)), str(v)) for v in out[feature]]
    return out


def main() -> int:
    import numpy as np
    import pandas as pd
    import torch
    from tabstar.preprocessing.splits import split_to_test
    from tabstar.training.devices import get_device

    from multabench.baselines.preprocessing.sampling import subsample_dataset
    from multabench.constants import DEVICE
    from multabench.datasets.downloading import download_dataset
    from multabench.datasets.multimodal import MultimodalState
    from multabench.datasets.utils import dataset_from_name
    from multabench.ours.learners import make_learner
    from multabench.ours.archive.structured_offset import fit_structured_offset
    from multabench.ours.tar_engine import TarConfig, set_seed
    from speed_benchmark import prepare

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--learner", default="catboost")
    p.add_argument("--methods", nargs="+", default=["tar_repro", "offset", "prefix", "context"])
    p.add_argument("--n_rows", type=int, default=200)
    p.add_argument("--top_k", type=int, default=3, help="perturb each of the top-k structured features in turn")
    p.add_argument("--train_examples", type=int, default=10_000)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--micro_batch_size", type=int, default=256)
    p.add_argument("--lower_cache_dir", default=None)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    device = get_device(device=DEVICE)
    dataset_id = dataset_from_name(name=a.dataset)
    ds = download_dataset(dataset_id=dataset_id, multimodal_state=MultimodalState.ALL, target_override=None)
    x, y = subsample_dataset(x=ds.x, y=ds.y, is_cls=ds.is_cls, train_examples=a.train_examples, fold=a.fold)
    x_train, x_test, y_train, y_test = split_to_test(x=x, y=y, is_cls=ds.is_cls, fold=a.fold, train_examples=a.train_examples)

    # Which structured features matter (same pool, same detection as the variants)
    x_fit, y_fit, text_cols, is_cls = prepare(a.dataset, a.fold, a.learner, a.train_examples, device)
    probe_learner = make_learner(a.learner)(problem_type=ds.task_type, device=device, dataset=dataset_id,
                                            image_folder=ds.image_folder, tune_e5=False)
    probe_learner.do_model_agnostic_preprocessing(x=x_train.copy(), y=y_train.copy())
    num, cat = set(probe_learner.numerical_features), set(probe_learner.categorical_features)
    struct = sorted((num | cat) - set(text_cols))
    if not struct:
        print("no structured columns: nothing to probe")
        return 0
    so = fit_structured_offset(x_fit[struct], y_fit, struct, sorted(cat & set(struct)), is_cls,
                               sorted(y_fit.astype(str).unique()) if is_cls else [])
    feats = so.top_features(a.top_k)
    rows = x_test.sample(n=min(a.n_rows, len(x_test)), random_state=0).index
    probe_sets = {}
    for j, f in enumerate(feats):
        xp = perturb(x_test.loc[rows], x_train, f, "cat" if f in cat else "num")
        xp.index = xp.index + PROBE_INDEX_OFFSET * (j + 1)
        probe_sets[f"probe:{f}"] = (xp, pd.Series(y_test.loc[rows].to_numpy(), index=xp.index))
    print(f"probing features {feats} on {len(rows)} test rows; methods {a.methods}", flush=True)

    out_rows = []
    for method in a.methods:
        set_seed(42)
        model = make_learner(a.learner)(problem_type=ds.task_type, device=device, verbose=False, dataset=dataset_id,
                                        image_folder=ds.image_folder, tune_e5=False, pca_components=30, no_pca=False)
        cfg = TarConfig(epochs=a.epochs, micro_batch_size=a.micro_batch_size, lower_cache_dir=a.lower_cache_dir,
                        eval_batch_size=256 if device.type == "cuda" else 16)
        model.configure_ours(variant=method, cfg=cfg, cache_dir=None, dataset=a.dataset, fold=a.fold,
                             test=(x_test, y_test), probe_sets=probe_sets)
        model.fit(x_train, y_train)
        res = model.tar_result
        for j, f in enumerate(feats):
            for col in res.text_cols:
                e0 = res.row_embeddings(col, rows)
                e1 = res.row_embeddings(col, rows + PROBE_INDEX_OFFSET * (j + 1))
                change = 1 - (e0 * e1).sum(axis=1)  # embeddings are L2-normalized
                out_rows.append({"method": method, "dataset": a.dataset, "fold": a.fold, "feature": f,
                                 "kind": "cat" if f in cat else "num", "text_col": col, "n_rows": len(rows),
                                 "mean_cos_change": float(change.mean()), "median_cos_change": float(np.median(change)),
                                 "max_cos_change": float(change.max())})
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    df = pd.DataFrame(out_rows)
    os.makedirs(a.out, exist_ok=True)
    path = join(a.out, f"sensitivity_{a.dataset}_fold{a.fold}.csv")
    df.to_csv(path, index=False)
    print(df.groupby(["method", "feature"])[["mean_cos_change", "max_cos_change"]].mean().round(5).to_string())
    print("wrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
