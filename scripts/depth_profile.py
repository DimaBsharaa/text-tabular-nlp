"""P11 depth profile: how much target signal a cross-fitted linear probe can read from EACH layer of the frozen e5,
on the learner's TRAINING rows only (5-fold out-of-fold metric; test rows are never loaded into the probe).

For each dataset (fold 0 by default) the exact inputs of our learner's text-encoder step are captured (upstream loading,
subsampling, split_to_test, learner-val split, model-agnostic preprocessing), and then:
  lexical          : CTE's TF-IDF + Ridge expert (reference)
  probe_final      : cte_probe's probe (Ridge on the L2-normalised final-layer embedding)
  depth[pool, l]   : the depth expert of layer l (multabench/ours/depth.py), pooling mean / meanmax
  depth_top3       : mean of the 3 best layers' OOF predictions (selection on the same OOF metric: slightly optimistic)
One CSV row per (dataset, expert). Metric = CTE's selection metric (AUROC / macro-OvR AUROC / Spearman).
    python scripts/depth_profile.py --datasets BIN_TEXT_JIGSAW_TOXICITY REG_TEXT_BOOK_PRICE --out reports/depth/profile.csv \
        --cache_dir .explore/depth/_cache
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from os.path import abspath, dirname

REPO_ROOT = dirname(dirname(abspath(__file__)))
sys.path[:0] = [REPO_ROOT, dirname(abspath(__file__))]
os.environ.setdefault("WANDB_MODE", "disabled")


class _Captured(Exception):
    pass


def capture_text_step_inputs(dataset: str, fold: int, device):
    """The kwargs our learner passes to run_stacking for (dataset, fold), without fitting anything."""
    from dev_eval import load_train_rows
    from multabench.ours import stacking
    from multabench.ours.learners import make_learner
    from multabench.ours.tar_engine import TarConfig
    ds, x_train, y_train = load_train_rows(dataset, fold)
    box = {}

    def grab(method, **kw):
        box.update(kw)
        raise _Captured()
    real = stacking.run_stacking
    stacking.run_stacking = grab
    try:
        model = make_learner("catboost")(problem_type=ds.task_type, device=device, verbose=False, tune_e5=False,
                                         pca_components=30, no_pca=False)
        model.configure_ours(variant="cte_probe", cfg=TarConfig(), cache_dir=None, dataset=dataset, fold=fold,
                             test=(x_train.iloc[:5], y_train.iloc[:5]))
        try:
            model.fit(x_train, y_train)
        except _Captured:
            pass
    finally:
        stacking.run_stacking = real
    return ds, box


def main(argv=None) -> None:
    import numpy as np
    import pandas as pd
    import torch
    from multabench.ours import cte, depth, p7
    from multabench.ours import tar_engine as te
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--poolings", nargs="+", default=["mean", "meanmax"])
    p.add_argument("--out", required=True)
    p.add_argument("--cache_dir", required=True)
    a = p.parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    done = set()
    if os.path.exists(a.out):
        done = set(pd.read_csv(a.out).dataset)
    for dataset in a.datasets:
        if dataset in done:
            continue
        t0 = time.time()
        ds, kw = capture_text_step_inputs(dataset, a.fold, device)
        x, y_fit, text_cols, is_cls = kw["x_fit"], kw["y_fit"], sorted(kw["text_cols"]), kw["is_cls"]
        if is_cls:
            classes = sorted(y_fit.astype(str).unique())
            y = y_fit.astype(str).map({k: i for i, k in enumerate(classes)}).to_numpy()
        else:
            classes, y = [], pd.to_numeric(y_fit).to_numpy(dtype=float)
        nc = len(classes) if is_cls else 1
        cfg = te.TarConfig(eval_batch_size=256 if device.type == "cuda" else 16)
        rows = np.asarray(x.index)
        out = []

        def put(expert, pooling, layer, metric, alpha, secs):
            out.append({"dataset": dataset, "fold": a.fold, "task": "cls" if is_cls else "reg", "n_classes": nc,
                        "n_rows": len(y), "n_text_cols": len(text_cols), "expert": expert, "pooling": pooling,
                        "layer": layer, "oof_metric": round(float(metric), 6), "alpha": alpha, "seconds": round(secs, 1)})

        t1 = time.time()  # lexical reference (CTE's expert)
        texts = {c: te.encoding_texts(x, c) for c in text_cols}
        concat = [" | ".join(f"{c}: {texts[c][i]}" for c in text_cols) for i in range(len(y))]
        X, _ = cte.tfidf_matrices(concat, {}, cte.CTE_CONFIG)
        oof, _, meta = p7.xfit_linear(X, {}, y, is_cls, nc)
        put("lexical", "", -1, max(meta["oof_metric"]), meta["alpha"], time.time() - t1)
        t1 = time.time()  # cte_probe's probe (final layer, L2-normalised)
        frozen, _ = cte.frozen_embeddings_cached([x[text_cols]], text_cols, cfg, device, a.cache_dir, dataset)
        E = np.hstack([frozen.row_embeddings(c, rows) for c in text_cols])
        _, _, meta = p7.xfit_linear(E, {}, y, is_cls, nc)
        put("probe_final", "", -1, max(meta["oof_metric"]), meta["alpha"], time.time() - t1)
        mm = None  # meanmax states contain the mean pool (first half): encode once when both are profiled
        for pooling in sorted(a.poolings, key=lambda q: q != "meanmax"):
            t1 = time.time()
            if pooling == "mean" and mm is not None:
                states = {c: (s[..., :s.shape[-1] // 2], u) for c, (s, u) in mm.items()}
            else:
                states = depth.layer_states_cached([x[text_cols]], text_cols, cfg, device, a.cache_dir, dataset, pooling)
                mm = states if pooling == "meanmax" else mm
            t_enc = time.time() - t1
            n_layers = next(iter(states.values()))[0].shape[1]
            oofs, mets = {}, {}
            for layer in range(1, n_layers + 1):
                t2 = time.time()
                Xl, _ = depth.layer_matrix(states, text_cols, rows, layer - 1)
                oofs[layer], _, meta = p7.xfit_linear(Xl, {}, y, is_cls, nc, depth.DepthSpec().cte_config())
                mets[layer] = max(meta["oof_metric"])
                put("depth", pooling, layer, mets[layer], meta["alpha"], time.time() - t2 + (t_enc if layer == 1 else 0))
            top = sorted(mets, key=lambda l: (mets[l], l), reverse=True)[:3]
            P = np.mean([oofs[l] for l in top], axis=0)
            full = P if not is_cls else (np.hstack([1 - P.sum(1, keepdims=True), P]) if P.shape[1] == nc - 1 else None)
            if full is not None:
                put("depth_top3", pooling, int(top[0]), cte.oof_metric(full, y, is_cls, nc), "", 0.0)
        pd.DataFrame(out).to_csv(a.out, mode="a", header=not os.path.exists(a.out), index=False)
        best = {r["pooling"]: r for r in sorted((r for r in out if r["expert"] == "depth"), key=lambda r: r["oof_metric"])}
        print(f"{dataset}: lexical {out[0]['oof_metric']:.4f} | probe_final {out[1]['oof_metric']:.4f} | "
              + " | ".join(f"best {k} layer {v['layer']} {v['oof_metric']:.4f}" for k, v in best.items())
              + f"  ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
