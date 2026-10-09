"""Time TAR fine-tuning epochs with the old full-model path vs the fast path (cached frozen layers + bucketing).

Same data preparation as a real job (upstream splits + the learner's preprocessing). The LR schedule is the full
50-epoch one; we only stop after --epochs_to_time epochs, so epoch-1 val losses are comparable to a real run.

    python scripts/speed_benchmark.py --dataset REG_TEXT_BOOK_PRICE --fold 0 --out results
Writes <out>/_bench/speed_<dataset>_fold<k>.json.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from os.path import abspath, dirname, join

REPO_ROOT = dirname(dirname(abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.environ.setdefault("WANDB_MODE", "disabled")


def prepare(dataset_name: str, fold: int, learner: str, train_examples: int, device):
    """x_fit / y_fit / text columns exactly as the learner hands them to the text-encoder step."""
    from tabstar.preprocessing.splits import split_to_test, split_to_val

    from multabench.baselines.preprocessing.sampling import subsample_dataset
    from multabench.datasets.downloading import download_dataset
    from multabench.datasets.multimodal import MultimodalState
    from multabench.datasets.utils import dataset_from_name
    from multabench.ours.learners import make_learner

    dataset_id = dataset_from_name(name=dataset_name)
    ds = download_dataset(dataset_id=dataset_id, multimodal_state=MultimodalState.ALL, target_override=None)
    x, y = subsample_dataset(x=ds.x, y=ds.y, is_cls=ds.is_cls, train_examples=train_examples, fold=fold)
    x_train, _, y_train, _ = split_to_test(x=x, y=y, is_cls=ds.is_cls, fold=fold, train_examples=train_examples)
    model = make_learner(learner)(problem_type=ds.task_type, device=device, dataset=dataset_id,
                                  image_folder=ds.image_folder, tune_e5=False)
    if model.USE_VAL_SPLIT:
        x_train, _, y_train, _ = split_to_val(x=x_train, y=y_train, is_cls=model.is_cls)
    x_fit, y_fit = model.do_model_agnostic_preprocessing(x=x_train.copy(), y=y_train.copy())
    return x_fit, y_fit, sorted(model.text_features), model.is_cls


def main() -> int:
    import torch
    from tabstar.training.devices import get_device
    from transformers import AutoTokenizer

    from multabench.constants import DEVICE
    from multabench.ours.tar_engine import (TarConfig, TokenStore, build_model, make_source, prepare_e5_data,
                                            set_seed, train_tar)

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="REG_TEXT_BOOK_PRICE")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--learner", default="catboost")
    p.add_argument("--train_examples", type=int, default=10_000)
    p.add_argument("--epochs_to_time", type=int, default=1)
    p.add_argument("--assumed_epochs", type=int, default=20, help="for the amortized speedup (cache built once)")
    p.add_argument("--micro_batch_size", type=int, default=256)
    p.add_argument("--lower_cache_dir", default=None)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    device = get_device(device=DEVICE)
    x_fit, y_fit, text_cols, is_cls = prepare(args.dataset, args.fold, args.learner, args.train_examples, device)
    base = TarConfig(stop_after_epochs=args.epochs_to_time, micro_batch_size=args.micro_batch_size,
                     eval_batch_size=256 if device.type == "cuda" else 16, lower_cache_dir=args.lower_cache_dir)
    d = prepare_e5_data(x_fit, y_fit, text_cols, is_cls, base.n_bins)
    tok = AutoTokenizer.from_pretrained(base.model_name)
    t0 = time.time()
    tokens = TokenStore(list(dict.fromkeys(d["train_texts"] + d["val_texts"])), tok, base.max_length)
    t_tok = time.time() - t0
    print(f"{args.dataset} fold {args.fold}: {len(d['train_texts'])} train cells, {len(tokens.strings)} unique strings, "
          f"tokenized in {t_tok:.1f}s, device {device}", flush=True)

    out = {"dataset": args.dataset, "fold": args.fold, "n_train_cells": len(d["train_texts"]),
           "n_unique_strings": len(tokens.strings), "tokenize_seconds": round(t_tok, 1),
           "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu", "paths": {}}
    for name, fast in (("old", False), ("fast", True)):
        cfg = dataclasses.replace(base, fast=fast)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        set_seed(cfg.seed)
        model = build_model(cfg, d_output=len(d["classes"]), device=device)
        t1 = time.time()
        source = make_source(model, tokens, cfg, device)
        t_cache = time.time() - t1
        print(f"--- {name} path ({'cached lower layers + bucketing' if source.fast else 'full model'})", flush=True)
        log = train_tar(cfg, model, source, tokens.uids(d["train_texts"]), d["train_labels"],
                        tokens.uids(d["val_texts"]), d["val_labels"], device)
        epoch_s = sum(h["seconds"] for h in log["history"]) / len(log["history"])
        out["paths"][name] = {"fast_path_used": log["fast_path"], "epoch_seconds": round(epoch_s, 1),
                              "lower_cache_seconds": round(t_cache, 1), "val_loss_epoch1": log["history"][0]["val_loss"],
                              "peak_gpu_gb": log["peak_gpu_gb"],
                              "ms_per_cell_epoch": round(1000 * epoch_s / len(d["train_texts"]), 3)}
        del model, source
    o, f = out["paths"]["old"], out["paths"]["fast"]
    n = args.assumed_epochs
    out["speedup_per_epoch"] = round(o["epoch_seconds"] / f["epoch_seconds"], 2)
    out[f"speedup_amortized_{n}_epochs"] = round(n * o["epoch_seconds"] / (f["lower_cache_seconds"] + n * f["epoch_seconds"]), 2)
    out["val_loss_epoch1_diff"] = f["val_loss_epoch1"] - o["val_loss_epoch1"]
    print("\npath   epoch_s  ms/cell/epoch  cache_build_s  peak_GPU_GB  val_loss@1")
    for name in ("old", "fast"):
        r = out["paths"][name]
        print(f"{name:<6} {r['epoch_seconds']:>7.1f}  {r['ms_per_cell_epoch']:>13.3f}  {r['lower_cache_seconds']:>13.1f}  "
              f"{str(r['peak_gpu_gb']):>11}  {r['val_loss_epoch1']:.4f}")
    print(f"\nspeedup per epoch: {out['speedup_per_epoch']}x | amortized over {n} epochs (cache built once): "
          f"{out[f'speedup_amortized_{n}_epochs']}x | val_loss@1 fast - old: {out['val_loss_epoch1_diff']:+.4f}")
    path = join(args.out, "_bench", f"speed_{args.dataset}_fold{args.fold}.json")
    os.makedirs(dirname(path), exist_ok=True)
    json.dump(out, open(path, "w"), indent=1)
    print("wrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
