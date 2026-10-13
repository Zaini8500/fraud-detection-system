"""
Train the fraud detection models.

Pipeline
--------
1. Load the synthetic transactions CSV (see data/generate.py).
2. Stratified train/validation/test split (keeps the ~0.5% fraud rate in each).
3. Baselines for comparison:
     - DummyClassifier (stratified)        -> what "guessing" scores
     - RandomForest (class_weight=balanced_subsample)
     - IsolationForest trained on legitimate txns only (unsupervised anomaly baseline)
     - XGBoost + SMOTE (resampling experiment)
4. Final model: XGBoost with scale_pos_weight = neg/pos (no resampling needed).
5. Threshold tuning: pick the probability threshold that maximises F1 on the
   validation PR curve (accuracy is meaningless at 0.5% fraud).
6. Evaluate once on the held-out test set: precision / recall / F1 / AUC-PR /
   ROC-AUC / confusion matrix. Never accuracy alone.
7. SHAP: exact TreeExplainer values on a test sample -> mean|SHAP| importance
   + summary plot (saved to frontend/public for the dashboard).
8. Export a compact, dependency-free booster JSON + medians + metrics, copied
   to api/artifacts/ so the FastAPI backend and the Vercel serverless
   functions serve the exact same model.

Usage:
    python models/train.py --data data/transactions.csv --artifacts models/artifacts
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import joblib
import matplotlib

# Repo root on sys.path so `api._xgb_runtime` (vendored predictor) is
# importable when this script is run as `python models/train.py`.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

matplotlib.use("Agg")  # headless: no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit

TARGET = "is_fraud"


# --------------------------------------------------------------------------- #
# Metrics helpers
# --------------------------------------------------------------------------- #
def pr_metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "auc_pr": float(average_precision_score(y_true, proba)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
        },
    }


def tune_threshold_f1(y_val: np.ndarray, proba_val: np.ndarray) -> tuple[float, dict]:
    """Pick the threshold maximising F1 along the validation PR curve."""
    prec, rec, thr = precision_recall_curve(y_val, proba_val)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    best_idx = int(np.argmax(f1))
    # precision_recall_curve returns len(thresholds) == len(precision) - 1
    best_thr = float(thr[best_idx]) if best_idx < len(thr) else 1.0
    curve = [
        {"recall": float(r), "precision": float(p)}
        for r, p in zip(rec[:: max(1, len(rec) // 200)], prec[:: max(1, len(prec) // 200)])
    ]
    return best_thr, {"best_f1": float(f1[best_idx]), "pr_curve": curve}


# --------------------------------------------------------------------------- #
# Compact XGBoost export (dependency-free serving)
# --------------------------------------------------------------------------- #
def _collect_nodes(node: dict, acc: dict) -> None:
    acc[node["nodeid"]] = node
    for child in node.get("children", []):
        _collect_nodes(child, acc)


def export_compact_booster(booster, feature_names: list[str],
                           n_trees: int | None = None,
                           calibrate_X: np.ndarray | None = None) -> dict:
    """Convert an XGBoost booster into compact JSON trees walkable with numpy.

    Each tree becomes parallel arrays:
        feature[i]   int   feature index, or -1 for a leaf
        threshold[i] float split threshold
        left[i]      int   child index when x[feature] < threshold (or missing->left)
        right[i]     int   child index otherwise
        missing[i]   int   child index when the feature value is NaN
        leaf_value[i]float leaf score (0 for internal nodes)

    Prediction: raw = base_score + sum(tree leaf values); p = sigmoid(raw).

    n_trees: with early stopping, predict() only uses the first
    n_trees (= best_iteration + 1); exporting exactly those keeps the
    vendored predictor bit-identical to the real model.
    """
    name_to_idx = {name: i for i, name in enumerate(feature_names)}
    trees = []
    dump = booster.get_dump(dump_format="json")
    if n_trees is not None:
        dump = dump[:n_trees]  # match predict()'s iteration range after early stopping
    for tree_str in dump:
        root = json.loads(tree_str)
        nodes: dict[int, dict] = {}
        _collect_nodes(root, nodes)

        # BFS order so children always come after parents (tidy, not required)
        order, queue = [], [root["nodeid"]]
        while queue:
            nid = queue.pop(0)
            order.append(nid)
            for child in nodes[nid].get("children", []):
                queue.append(child["nodeid"])
        remap = {nid: i for i, nid in enumerate(order)}

        feature, threshold, left, right, missing, leaf_value = [], [], [], [], [], []
        for nid in order:
            nd = nodes[nid]
            if "leaf" in nd:
                feature.append(-1)
                threshold.append(0.0)
                left.append(-1)
                right.append(-1)
                missing.append(-1)
                # float32-exact: the dump round-trips float32, so parse via
                # np.float32 to recover the exact bit pattern.
                leaf_value.append(float(np.float32(nd["leaf"])))
            else:
                feature.append(name_to_idx[nd["split"]])
                threshold.append(float(np.float32(nd["split_condition"])))
                left.append(remap[nd["yes"]])
                right.append(remap[nd["no"]])
                missing.append(remap[nd["missing"]])
                leaf_value.append(0.0)
        trees.append(
            {
                "feature": feature,
                "threshold": threshold,
                "left": left,
                "right": right,
                "missing": missing,
                "leaf_value": leaf_value,
            }
        )

    # base_score: calibrated empirically against the real booster.
    # The config value is unreliable across sklearn versions (observed: config
    # says 0.5 while predict() behaves as 0.0), so we measure the residual
    #   booster.predict(output_margin=True) - compact_tree_walk
    # on a sample and use its median. It must be ~constant; we report its
    # spread as a sanity check.
    base_score = 0.0
    calib_std = 0.0
    if calibrate_X is not None and len(calibrate_X):
        import xgboost as _xgb

        from api._xgb_runtime import XGBCompactPredictor as _Predictor

        probe = _Predictor(
            {
                "format": "xgb-compact-v1",
                "objective": "binary:logistic",
                "base_score": 0.0,
                "feature_names": feature_names,
                "trees": trees,
            }
        )
        Xs = np.asarray(calibrate_X, dtype=np.float32)
        dmat = _xgb.DMatrix(Xs, feature_names=feature_names)
        it_range = (0, n_trees) if n_trees else None
        if it_range:
            true_margin = booster.predict(
                dmat, iteration_range=it_range, output_margin=True
            )
        else:
            true_margin = booster.predict(dmat, output_margin=True)
        walk = probe.predict_margin(Xs).astype(np.float64)
        resid = true_margin.astype(np.float64) - walk
        base_score = float(np.median(resid))
        calib_std = float(resid.std())
        print(f"base_score calibrated: {base_score:.6f} "
              f"(residual std {calib_std:.2e} over {len(Xs)} rows)")

    return {
        "format": "xgb-compact-v1",
        "objective": "binary:logistic",
        "base_score": base_score,
        "base_score_calibration_std": calib_std,
        "feature_names": feature_names,
        "trees": trees,
    }


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train fraud detection models.")
    parser.add_argument("--data", default="data/transactions.csv")
    parser.add_argument("--artifacts", default="models/artifacts")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    df = pd.read_csv(repo_root / args.data)
    feature_names = [c for c in df.columns if c != TARGET]
    X = df[feature_names].to_numpy(dtype=np.float64)
    y = df[TARGET].to_numpy(dtype=np.int64)
    print(f"Loaded {len(df):,} rows, fraud rate {y.mean():.4%}, {len(feature_names)} features")

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=args.seed)
    train_val_idx, test_idx = next(splitter.split(X, y))
    X_tv, y_tv = X[train_val_idx], y[train_val_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    splitter2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=args.seed)
    tr_idx, va_idx = next(splitter2.split(X_tv, y_tv))
    X_train, y_train = X_tv[tr_idx], y_tv[tr_idx]
    X_val, y_val = X_tv[va_idx], y_tv[va_idx]
    print(f"Split: train {len(X_train):,} / val {len(X_val):,} / test {len(X_test):,}")

    neg, pos = int((y_train == 0).sum()), int((y_train == 1).sum())
    scale_pos_weight = neg / max(pos, 1)
    print(f"Imbalance: {neg:,} legit vs {pos:,} fraud -> scale_pos_weight={scale_pos_weight:.1f}")

    comparison: dict[str, dict] = {}

    # --- 1. Dummy baseline ------------------------------------------------- #
    dummy = DummyClassifier(strategy="stratified", random_state=args.seed)
    dummy.fit(X_train, y_train)
    comparison["dummy_stratified"] = pr_metrics(y_test, dummy.predict_proba(X_test)[:, 1], 0.5)

    # --- 2. RandomForest with balanced class weights ------------------------ #
    rf = RandomForestClassifier(
        n_estimators=120,
        max_depth=12,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=args.seed,
    )
    rf.fit(X_train, y_train)
    thr_rf, _ = tune_threshold_f1(y_val, rf.predict_proba(X_val)[:, 1])
    comparison["random_forest_balanced"] = pr_metrics(
        y_test, rf.predict_proba(X_test)[:, 1], thr_rf
    )

    # --- 3. IsolationForest: unsupervised anomaly baseline ------------------ #
    # Trained on legitimate transactions only; flags the top ~fraud_rate as anomalies.
    iso = IsolationForest(
        n_estimators=150,
        contamination=float(y_train.mean()),
        random_state=args.seed,
        n_jobs=-1,
    )
    iso.fit(X_train[y_train == 0])
    iso_scores = -iso.decision_function(X_test)  # higher = more anomalous
    iso_thr = np.quantile(iso_scores, 1.0 - float(y_test.mean()))
    comparison["isolation_forest"] = pr_metrics(y_test, iso_scores, iso_thr)

    # --- 4. XGBoost + SMOTE experiment -------------------------------------- #
    try:
        from imblearn.over_sampling import SMOTE

        sm = SMOTE(random_state=args.seed)
        X_res, y_res = sm.fit_resample(X_train, y_train)
        from xgboost import XGBClassifier

        xgb_sm = XGBClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="aucpr", n_jobs=-1, random_state=args.seed,
        )
        xgb_sm.fit(X_res, y_res)
        thr_sm, _ = tune_threshold_f1(y_val, xgb_sm.predict_proba(X_val)[:, 1])
        comparison["xgboost_smote"] = pr_metrics(
            y_test, xgb_sm.predict_proba(X_test)[:, 1], thr_sm
        )
        print(f"SMOTE resampled train: {len(X_res):,} rows "
              f"(fraud {y_res.mean():.1%})")
    except Exception as exc:  # imbalanced-learn optional
        print(f"SMOTE experiment skipped: {exc}")

    # --- 5. Final model: XGBoost + scale_pos_weight -------------------------- #
    from xgboost import XGBClassifier

    xgb = XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        n_jobs=-1,
        random_state=args.seed,
        early_stopping_rounds=40,
    )
    try:
        xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    except TypeError:  # older xgboost without early_stopping_rounds kwarg
        xgb.set_params(early_stopping_rounds=None)
        xgb.fit(X_train, y_train)
    print(f"XGBoost trained: {xgb.n_estimators} trees "
          f"(best iteration {getattr(xgb, 'best_iteration', 'n/a')})")

    proba_val = xgb.predict_proba(X_val)[:, 1]
    threshold, tuning = tune_threshold_f1(y_val, proba_val)
    proba_test = xgb.predict_proba(X_test)[:, 1]
    final_metrics = pr_metrics(y_test, proba_test, threshold)
    final_metrics["pr_curve"] = tuning["pr_curve"]
    final_metrics["threshold_note"] = (
        f"Threshold {threshold:.4f} maximises F1 on the validation PR curve."
    )
    comparison["xgboost_scale_pos_weight_FINAL"] = final_metrics

    print("\n=== Test-set comparison (threshold tuned on validation) ===")
    for name, m in comparison.items():
        cm = m["confusion_matrix"]
        print(
            f"{name:32s} P={m['precision']:.3f} R={m['recall']:.3f} "
            f"F1={m['f1']:.3f} AUC-PR={m['auc_pr']:.3f} "
            f"TP={cm['tp']} FP={cm['fp']} FN={cm['fn']}"
        )

    # --- 6. SHAP explanations (exact, TreeExplainer) -------------------------- #
    print("\nComputing SHAP values (TreeExplainer, test sample)...")
    import shap

    sample_idx = np.random.default_rng(args.seed).choice(
        len(X_test), size=min(2000, len(X_test)), replace=False
    )
    explainer = shap.TreeExplainer(xgb)
    shap_values = explainer.shap_values(X_test[sample_idx])
    mean_abs = np.abs(shap_values).mean(axis=0)
    shap_importance = sorted(
        ({"feature": f, "mean_abs_shap": float(v)}
         for f, v in zip(feature_names, mean_abs)),
        key=lambda d: d["mean_abs_shap"],
        reverse=True,
    )
    final_metrics["shap_importance"] = shap_importance
    print("Top SHAP features:",
          ", ".join(f"{d['feature']}={d['mean_abs_shap']:.3f}"
                     for d in shap_importance[:5]))

    # SHAP summary plot for the dashboard
    shap_fig_path = repo_root / "frontend" / "public" / "shap_summary.png"
    shap_fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 6))
    shap.summary_plot(shap_values, X_test[sample_idx],
                      feature_names=feature_names, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(shap_fig_path, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"Saved SHAP summary plot -> {shap_fig_path}")

    # --- 7. Export artifacts ------------------------------------------------- #
    artifacts = repo_root / args.artifacts
    artifacts.mkdir(parents=True, exist_ok=True)

    booster = xgb.get_booster()
    booster.feature_names = feature_names  # name the features for the dump
    # With early stopping, predict() only uses trees up to best_iteration --
    # export exactly those so the vendored predictor matches bit-for-bit.
    best_iteration = getattr(xgb, "best_iteration", None)
    n_export_trees = (best_iteration + 1) if best_iteration is not None else None
    compact = export_compact_booster(
        booster, feature_names,
        n_trees=n_export_trees,
        calibrate_X=X_train[:2000],
    )
    if n_export_trees is not None:
        print(f"Exporting {n_export_trees} trees (best_iteration={best_iteration})")

    medians = {f: float(np.median(X_train[:, i])) for i, f in enumerate(feature_names)}

    payload_meta = {
        "model": "xgboost",
        "n_estimators": int(xgb.n_estimators),
        "max_depth": 5,
        "learning_rate": 0.05,
        "scale_pos_weight": float(scale_pos_weight),
        "threshold": float(threshold),
        "n_features": len(feature_names),
        "train_rows": len(X_train),
        "fraud_rate_train": float(y_train.mean()),
        "seed": args.seed,
    }

    (artifacts / "booster_compact.json").write_text(json.dumps(compact))
    (artifacts / "feature_names.json").write_text(json.dumps(feature_names))
    (artifacts / "feature_medians.json").write_text(json.dumps(medians))
    (artifacts / "metrics.json").write_text(
        json.dumps({"final": final_metrics, "comparison": comparison,
                    "meta": payload_meta}, indent=2)
    )
    # Full native booster too (for the parity test and further analysis)
    booster.save_model(str(artifacts / "xgb_model.ubj"))
    joblib.dump(xgb, artifacts / "xgb_sklearn.joblib")
    print(f"\nArtifacts written to {artifacts}/ "
          f"(booster_compact.json: {(artifacts/'booster_compact.json').stat().st_size/1024:.0f} KB)")

    # Mirror the serving artifacts into api/ so Vercel bundles them
    api_artifacts = repo_root / "api" / "artifacts"
    api_artifacts.mkdir(parents=True, exist_ok=True)
    for name in ["booster_compact.json", "feature_names.json",
                 "feature_medians.json", "metrics.json"]:
        shutil.copy2(artifacts / name, api_artifacts / name)
    print(f"Mirrored serving artifacts -> {api_artifacts}/")
    print("\nDone. Next: start the backend (backend/app.py) or run tests/test_parity.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
