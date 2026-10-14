"""Inference on new rows for a fitted CTE / P6 learner (scripts/train.py saves it, scripts/predict.py loads it).

New rows are treated exactly like the test rows of a training run: the same CTE code (run_stacking) is re-run on the
stored training pool with the new rows as an extra split. That gives them the predictions of experts fitted on all
training rows and their frozen embeddings. The fitted PCA and learner are reused unchanged. Nothing is re-fitted on
the new rows, and their labels (if any) are never read.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from pandas import DataFrame

from multabench.ours.stacking import STACK_METHODS, run_stacking


def predict_new(model, x_new: DataFrame, cache_dir: Optional[str] = None) -> DataFrame:
    """Predictions for x_new (raw feature columns, as in the dataset, without the target).
    Classification: one probability column per class (named by the class label); regression: column "prediction"."""
    x = x_new.copy()
    variant = model.ours_variant
    if variant in STACK_METHODS:
        inp = model.ours_stack_inputs
        known = [np.asarray(inp["x_fit"].index, dtype=np.int64)]
        known += [np.fromiter(c.row_uid.keys(), dtype=np.int64) for c in model.tar_result.columns.values()]
        start = max(int(k.max()) for k in known if len(k)) + 1  # new row ids never collide with known rows
        x.index = pd.RangeIndex(start, start + len(x))
        cols = inp["text_cols"] + inp["struct_cols"]
        pre = model._ours_preprocess_like_transform(x)[cols]
        res = run_stacking(variant, x_fit=inp["x_fit"], y_fit=inp["y_fit"], text_cols=inp["text_cols"],
                           struct_cols=inp["struct_cols"], cat_cols=inp["cat_cols"], is_cls=inp["is_cls"],
                           extra={"new": (pre, None)}, cfg=model.ours_cfg, device=model.device, cache_root=cache_dir,
                           dataset=model.ours_dataset, fold=model.ours_fold)
        for col, enc in model.text_transformers.items():
            enc.result = res.embedding_result
            enc.features = res.features[col]
            enc.encoder.features = res.features[col]
    elif variant != "catboost_native_text":
        raise ValueError(f"inference on new rows is implemented for CTE / P6 methods, not {variant!r}")
    pred = np.asarray(model.predict(x))
    if not model.is_cls:
        return DataFrame({"prediction": pred}, index=x_new.index)
    classes = [str(c) for c in model.target_transformer.classes_]
    probs = np.column_stack([1 - pred, pred]) if pred.ndim == 1 else pred
    return DataFrame(probs, columns=[f"proba_{c}" for c in classes], index=x_new.index)
