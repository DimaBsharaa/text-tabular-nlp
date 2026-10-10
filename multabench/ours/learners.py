"""Upstream learners with only the text-encoder step swapped for our TAR engine.

`make_learner("catboost")` returns a subclass of upstream's CatBoost (etc.). Everything except the text-encoder
step is inherited unchanged: splits, model-agnostic preprocessing, target encoding, the learner, predict, metrics.
"""
from __future__ import annotations

from typing import Dict, Optional, Type

import numpy as np
import torch
from pandas import DataFrame, Series
from sklearn.decomposition import PCA
from tabstar.constants import SEED
from tabstar.preprocessing.dates import transform_date_features
from tabstar.preprocessing.sparse import densify_objects
from tabstar.preprocessing.splits import split_to_val

from multabench.baselines.abstract_model import TabularModel
from multabench.baselines.preprocessing.categorical import fit_categorical_encoders
from multabench.baselines.preprocessing.feature_types import transform_feature_types
from multabench.baselines.preprocessing.numerical import fit_numerical_median
from multabench.baselines.preprocessing.text_embeddings import _IdentityTransform
from multabench.ours.tar_engine import TarConfig, TarResult, encoding_texts, run_tar
from multabench.utils.pca_logging import log_pca_variance


class CachedColumnEncoder:
    """Stands in for upstream's E5ColumnEncoder: same interface (encode_texts / transform / n_components), but
    embeddings come from a TarResult looked up by dataset row index (and verified against the text)."""

    def __init__(self, result: TarResult, col_name: str, encoder):
        self.result = result
        self.col_name = col_name
        self.encoder = encoder
        self.n_components = encoder.n_components
        self.current_rows: Optional[np.ndarray] = None

    def encode_texts(self, texts: list[str], device: torch.device) -> np.ndarray:
        assert self.current_rows is not None and len(self.current_rows) == len(texts), "rows not set before encoding"
        return self.result.row_embeddings(self.col_name, self.current_rows, texts=texts)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.encoder.transform(X)


class _PCAWithPredictions:
    """What upstream's transform_text_features calls as `wrapper.encoder.transform(embeddings)`: the PCA output
    followed by the head prediction features of the rows being transformed."""

    def __init__(self, pca, features, owner: "PredColumnEncoder"):
        self.pca, self.features, self.owner = pca, features, owner
        self.n_components = pca.n_components + features.shape[1]

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.hstack([self.pca.transform(X), self.features.loc[self.owner.current_rows].to_numpy()])


class PredColumnEncoder(CachedColumnEncoder):
    """CachedColumnEncoder whose output is [PCA(embedding), head prediction features] for the current rows. The
    learner sees n_components = PCA components + prediction features (upstream names them <col>_txt_pca_<i>)."""

    def __init__(self, result: TarResult, col_name: str, encoder, features):
        super().__init__(result=result, col_name=col_name, encoder=_PCAWithPredictions(encoder, features, self))
        self.features = features


class OursTextMixin:
    """Mixin placed before an upstream TabularModel subclass. Call configure_ours(...) before fit()."""

    def configure_ours(self, variant: str, cfg: TarConfig, cache_dir: Optional[str], dataset: str, fold: int,
                       test: tuple[DataFrame, Series], probe_sets: Optional[dict] = None):
        """probe_sets (scripts/sensitivity_probe.py only): extra raw row sets that get embeddings but are never
        trained on; empty in normal runs."""
        self.ours_variant, self.ours_cfg, self.ours_cache_dir = variant, cfg, cache_dir
        self.ours_dataset, self.ours_fold, self.ours_test = dataset, fold, test
        self.ours_probe_sets = dict(probe_sets or {})
        self.ours_learner_val: Optional[tuple[DataFrame, Series]] = None
        self.tar_result: Optional[TarResult] = None

    def fit(self, x: DataFrame, y: Series):
        if self.USE_VAL_SPLIT:
            # Same call TabularModel.fit makes (deterministic), so these are exactly the learner's val rows.
            _, x_val, _, y_val = split_to_val(x=x, y=y, is_cls=self.is_cls)
            self.ours_learner_val = (x_val, y_val)
        return super().fit(x, y)

    def fit_preprocessor(self, x_train: DataFrame, y_train: Series):
        # Identical to TabularModel.fit_preprocessor except the fit_text_encoders(...) call.
        x_train, y_train = self.do_model_agnostic_preprocessing(x=x_train, y=y_train)
        if self.USE_MEDIAN_FILLING:
            self.numerical_medians = fit_numerical_median(x=x_train, numerical_features=self.numerical_features)
        if self.USE_CATEGORICAL_ENCODING:
            self.categorical_encoders = fit_categorical_encoders(x=x_train, categorical_features=self.categorical_features)
        if self.USE_TEXT_EMBEDDINGS:
            self.text_transformers = self._ours_fit_text_encoders(x_train=x_train, y_train=y_train)
            self.vprint(f"📝 Detected {len(self.text_transformers)} text features: {sorted(self.text_transformers)}")
        self.fit_internal_preprocessor(x=x_train, y=y_train)

    def transform_preprocessor(self, x: DataFrame, y: Optional[Series]):
        for enc in self.text_transformers.values():
            enc.current_rows = np.asarray(x.index)
        return super().transform_preprocessor(x=x, y=y)

    def _ours_preprocess_like_transform(self, x: DataFrame) -> DataFrame:
        """The steps transform_preprocessor applies before the text encoders (text datasets have no images)."""
        assert not self.image_transformers, "image columns are not supported by our TAR engine"
        x, _ = densify_objects(x=x.copy(), y=None)
        x = transform_date_features(x=x, date_transformers=self.date_transformers)
        return transform_feature_types(x=x, numerical_features=self.numerical_features, image_features=set())

    def _ours_fit_text_encoders(self, x_train: DataFrame, y_train: Series) -> Dict[str, CachedColumnEncoder]:
        text_cols = sorted(self.text_features)
        if not text_cols:
            return {}
        extra = {"test": self.ours_test}
        if self.ours_learner_val is not None:
            extra["learner_val"] = self.ours_learner_val
        extra.update(getattr(self, "ours_probe_sets", {}))  # empty except in the sensitivity probe
        if self.ours_variant == "tar_repro":  # frozen path (tests/test_tar_repro_frozen.py)
            extra = {k: (self._ours_preprocess_like_transform(x)[text_cols], y) for k, (x, y) in extra.items()}
            self.tar_result = run_tar(x_fit=x_train[text_cols], y_fit=y_train, text_cols=text_cols, is_cls=self.is_cls,
                                      extra=extra, cfg=self.ours_cfg, device=self.device, cache_root=self.ours_cache_dir,
                                      dataset=self.ours_dataset, fold=self.ours_fold, variant=self.ours_variant)
        else:  # variants also see the structured columns (docs/method_spec.md, PROGRESS.md P3)
            from multabench.ours.stacking import STACK_METHODS, run_stacking
            from multabench.ours.variants import run_variant
            struct_cols = sorted((set(self.numerical_features) | set(self.categorical_features)) - set(text_cols))
            cols = text_cols + struct_cols
            extra = {k: (self._ours_preprocess_like_transform(x)[cols], y) for k, (x, y) in extra.items()}
            kw = dict(x_fit=x_train[cols], y_fit=y_train, text_cols=text_cols, struct_cols=struct_cols,
                      cat_cols=sorted(self.categorical_features), is_cls=self.is_cls, extra=extra, cfg=self.ours_cfg,
                      device=self.device, cache_root=self.ours_cache_dir, dataset=self.ours_dataset, fold=self.ours_fold)
            if self.ours_variant in STACK_METHODS:
                self.stack_result = run_stacking(self.ours_variant, **kw)
                self.tar_result = self.stack_result.embedding_result
            else:
                self.tar_result = run_variant(self.ours_variant, **kw)
        encoders = {}
        for col in text_cols:
            # PCA exactly as upstream: fit on the embeddings of all training rows, in their order.
            col_emb = self.tar_result.row_embeddings(col, np.asarray(x_train.index), texts=encoding_texts(x_train, col))
            if self.no_pca:
                pca = _IdentityTransform(n_components=col_emb.shape[1])
            else:
                pca = PCA(n_components=self.pca_components, random_state=SEED)
                pca.fit(col_emb)
                log_pca_variance(pca=pca, col_name=col)
            stack = getattr(self, "stack_result", None)
            if stack is None:
                encoders[col] = CachedColumnEncoder(result=self.tar_result, col_name=col, encoder=pca)
            else:  # P3: the head's prediction features are appended after this column's PCA components
                encoders[col] = PredColumnEncoder(result=self.tar_result, col_name=col, encoder=pca,
                                                  features=stack.features[col])
        return encoders

    def stack_feature_importance(self) -> Optional[dict]:
        """P3 diagnostic: share of CatBoost's feature importance on the appended prediction columns."""
        stack = getattr(self, "stack_result", None)
        if stack is None or not hasattr(self.model_, "get_feature_importance"):
            return None
        imp = dict(zip(self.model_.feature_names_, self.model_.get_feature_importance()))
        pred_cols = [f"{c}_txt_pca_{self.pca_components + j}" for c in stack.features
                     for j in range(stack.features[c].shape[1])]
        total = sum(imp.values()) or 1.0
        return {"pred_columns": {c: round(float(imp.get(c, 0.0)), 3) for c in pred_cols},
                "pred_share": round(sum(imp.get(c, 0.0) for c in pred_cols) / total, 4),
                "top10": sorted(((k, round(float(v), 3)) for k, v in imp.items()), key=lambda kv: -kv[1])[:10]}


def _upstream_learners() -> Dict[str, Type[TabularModel]]:
    from multabench.baselines.catboost import CatBoost
    from multabench.baselines.lgbm import LightGBM
    from multabench.baselines.tabm import TabM
    from multabench.baselines.tabpfnv2 import TabPFNv2, TabPFNv2p5
    return {"catboost": CatBoost, "lgbm": LightGBM, "tabm": TabM, "tabpfnv2": TabPFNv2, "tabpfnv2p5": TabPFNv2p5}


LEARNER_NAMES = ("catboost", "lgbm", "tabm", "tabpfnv2", "tabpfnv2p5")


def make_learner(name: str) -> Type[TabularModel]:
    base = _upstream_learners()[name]
    return type(f"Ours{base.__name__}", (OursTextMixin, base), {})
