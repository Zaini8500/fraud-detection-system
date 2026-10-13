"""
Vendored, dependency-free XGBoost inference runtime.

Loads the compact booster JSON exported by models/train.py and scores rows
using only numpy. This is what runs inside the Vercel serverless functions,
where xgboost / shap / pandas are too heavy to bundle.

Prediction is mathematically identical to booster.predict():
    raw = base_score + sum over trees of leaf_value
    probability = sigmoid(raw)                      (binary:logistic)

The compact format is documented in models/train.py::export_compact_booster.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # Numerically stable sigmoid
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


class XGBCompactPredictor:
    """Pure-numpy inference for a compact-exported XGBoost booster."""

    def __init__(self, payload: dict):
        if payload.get("format") != "xgb-compact-v1":
            raise ValueError(f"Unsupported compact format: {payload.get('format')}")
        if payload.get("objective") != "binary:logistic":
            raise ValueError(f"Unsupported objective: {payload.get('objective')}")
        self.feature_names: list[str] = list(payload["feature_names"])
        self.n_features = len(self.feature_names)
        self.base_score = float(payload["base_score"])
        # Per tree, parallel numpy arrays (vectorised walk below)
        self.trees: list[tuple[np.ndarray, ...]] = []
        for t in payload["trees"]:
            # float32 everywhere: matches XGBoost's internal precision and the
            # DMatrix input quantization, so branching is bit-identical.
            self.trees.append(
                (
                    np.asarray(t["feature"], dtype=np.int32),
                    np.asarray(t["threshold"], dtype=np.float32),
                    np.asarray(t["left"], dtype=np.int32),
                    np.asarray(t["right"], dtype=np.int32),
                    np.asarray(t["missing"], dtype=np.int32),
                    np.asarray(t["leaf_value"], dtype=np.float32),
                )
            )
        self.n_trees = len(self.trees)

    @classmethod
    def from_file(cls, path: str | Path) -> "XGBCompactPredictor":
        with open(path, "r", encoding="utf-8") as fh:
            return cls(json.load(fh))

    def _walk_tree(self, tree: tuple[np.ndarray, ...], X: np.ndarray) -> np.ndarray:
        """Vectorised walk of one tree for all rows; returns leaf values (float32)."""
        feat, thr, left, right, missing, leaf = tree
        n = X.shape[0]
        node = np.zeros(n, dtype=np.int32)
        while True:
            internal = feat[node] >= 0
            if not np.any(internal):
                break
            idx = np.nonzero(internal)[0]
            cur = node[idx]
            f = feat[cur]
            v = X[idx, f]  # float32, exactly like DMatrix/C++
            is_nan = np.isnan(v)
            # NaN -> missing child; otherwise left if v < threshold else right
            go_left = (~is_nan) & (v < thr[cur])
            nxt = np.where(is_nan, missing[cur], np.where(go_left, left[cur], right[cur]))
            node[idx] = nxt
        return leaf[node]

    def predict_margin(self, X: np.ndarray) -> np.ndarray:
        # Cast to float32: XGBoost stores features and split thresholds as
        # float32 (DMatrix quantizes inputs), so float32 comparison reproduces
        # the exact branching the C++ predictor takes. The margin is
        # accumulated in float32 in tree order, exactly like the booster.
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} features, got {X.shape[1]}"
            )
        raw = np.full(X.shape[0], np.float32(self.base_score), dtype=np.float32)
        for tree in self.trees:
            raw += self._walk_tree(tree, X)
        return raw

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Fraud probability for each row."""
        return _sigmoid(self.predict_margin(X))
