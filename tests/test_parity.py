"""
Parity test: the vendored pure-numpy predictor must agree with the real
XGBoost booster on held-out rows.

Run from the repo root:
    .venv/bin/python tests/test_parity.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from api._xgb_runtime import XGBCompactPredictor  # noqa: E402


def main() -> int:
    artifacts = REPO_ROOT / "models" / "artifacts"
    booster_path = artifacts / "xgb_model.ubj"
    compact_path = REPO_ROOT / "api" / "artifacts" / "booster_compact.json"

    import xgboost as xgb

    booster = xgb.Booster()
    booster.load_model(str(booster_path))
    predictor = XGBCompactPredictor.from_file(compact_path)

    feature_names = json.loads((artifacts / "feature_names.json").read_text())

    # Deterministic test rows: 5 fraud + 5 legit from the generated dataset
    df = pd.read_csv(REPO_ROOT / "data" / "transactions.csv")
    fraud = df[df["is_fraud"] == 1].head(5)
    legit = df[df["is_fraud"] == 0].head(5)
    sample = pd.concat([fraud, legit]).reset_index(drop=True)
    X = sample[feature_names].to_numpy(dtype=np.float64)

    dmat = xgb.DMatrix(X, feature_names=feature_names)
    # The exported model contains trees [0, best_iteration]; predict with the
    # same iteration range the sklearn API uses after early stopping.
    best_iter = int(booster.attr("best_iteration") or -1)
    it_range = (0, best_iter + 1) if best_iter >= 0 else None
    expected = (booster.predict(dmat, iteration_range=it_range) if it_range
                else booster.predict(dmat))   # reference: real XGBoost
    actual = predictor.predict_proba(X)       # vendored numpy predictor
    assert predictor.n_trees == (it_range[1] if it_range else len(
        booster.get_dump())), "exported tree count mismatch"

    diffs = np.abs(expected - actual)
    print(f"{'row':>4} {'label':>6} {'xgboost':>10} {'vendored':>10} {'|diff|':>10}")
    for i in range(len(X)):
        print(f"{i:>4} {int(sample['is_fraud'][i]):>6} "
              f"{expected[i]:>10.6f} {actual[i]:>10.6f} {diffs[i]:>10.2e}")

    max_diff = float(diffs.max())
    print(f"\nmax |diff| = {max_diff:.2e} over {len(X)} rows "
          f"({predictor.n_trees} trees, {predictor.n_features} features)")
    # Both sides now walk float32 trees identically; the only slack is the
    # final sigmoid precision. 1e-6 is far below any displayed score digit.
    assert max_diff < 1e-6, f"Parity FAILED: max diff {max_diff:.2e} >= 1e-6"

    # Sanity: feature-contribution path runs and is finite
    sys.path.insert(0, str(REPO_ROOT / "api"))
    from _model import feature_contributions

    contribs = feature_contributions(X[0])
    assert len(contribs) == 8 and all(np.isfinite(c["contribution"]) for c in contribs)
    print("feature_contributions: OK (8 finite contributions)")

    print("\nPARITY TEST PASSED ✔ vendored predictor == XGBoost booster")
    return 0


if __name__ == "__main__":
    sys.exit(main())
