"""
Shared serving logic for fraud inference.

Used by BOTH:
  - backend/app.py            (local FastAPI dev server)
  - api/predict.py etc.       (Vercel Python serverless functions)

so predictions are identical everywhere.

Feature encoding (must match data/generate.py exactly):
  - numeric features passed through as floats
  - merchant_category one-hot encoded into mcc_* columns
  - amount_to_avg_ratio derived from amount / (avg_amount_30d + 1)

Explanations: training uses exact SHAP (shap.TreeExplainer, see
models/train.py). At serving time we use a lightweight, dependency-free
baseline-ablation approximation that is SHAP-consistent in spirit:
    contribution_i = M(x) - M(x with feature i replaced by its training median)
where M is the log-odds margin (the space SHAP values natively live in).
Positive contribution pushes the score toward fraud, negative away from it.
This is honest approximation, not exact SHAP -- documented in the README.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

try:
    from _xgb_runtime import XGBCompactPredictor  # Vercel: api/ on sys.path
except ImportError:  # local: imported as api._model
    from api._xgb_runtime import XGBCompactPredictor

_HERE = Path(__file__).resolve().parent
_ARTIFACTS = _HERE / "artifacts"

# Raw transaction fields accepted by /predict (before one-hot encoding).
RAW_FIELDS = [
    "amount",
    "hour",
    "day_of_week",
    "card_present",
    "is_international",
    "txn_velocity_1h",
    "amount_velocity_1h",
    "distance_from_home_km",
    "failed_attempts_24h",
    "device_changed",
    "avg_amount_30d",
    "merchant_category",
]

MERCHANT_CATEGORIES = [
    "grocery", "fuel", "restaurant", "travel",
    "electronics", "fashion", "atm", "online",
]


def _load_json(name: str):
    with open(_ARTIFACTS / name, "r", encoding="utf-8") as fh:
        return json.load(fh)


@lru_cache(maxsize=1)
def get_predictor() -> XGBCompactPredictor:
    return XGBCompactPredictor.from_file(_ARTIFACTS / "booster_compact.json")


@lru_cache(maxsize=1)
def get_feature_names() -> list[str]:
    return list(_load_json("feature_names.json"))


@lru_cache(maxsize=1)
def get_medians() -> dict:
    return dict(_load_json("feature_medians.json"))


@lru_cache(maxsize=1)
def get_metrics() -> dict:
    return dict(_load_json("metrics.json"))


def _coerce_number(value, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def transaction_to_vector(txn: dict) -> np.ndarray:
    """Encode a raw transaction dict into the model's feature vector."""
    names = get_feature_names()
    vec = np.zeros(len(names), dtype=np.float64)
    lookup = {name: i for i, name in enumerate(names)}

    def setf(name: str, value: float) -> None:
        if name in lookup:
            vec[lookup[name]] = value

    amount = _coerce_number(txn.get("amount"))
    avg30 = _coerce_number(txn.get("avg_amount_30d"), default=50.0)
    setf("amount", amount)
    setf("hour", _coerce_number(txn.get("hour")))
    setf("day_of_week", _coerce_number(txn.get("day_of_week")))
    setf("card_present", _coerce_number(txn.get("card_present"), default=1.0))
    setf("is_international", _coerce_number(txn.get("is_international")))
    setf("txn_velocity_1h", _coerce_number(txn.get("txn_velocity_1h")))
    setf("amount_velocity_1h", _coerce_number(txn.get("amount_velocity_1h")))
    setf("distance_from_home_km", _coerce_number(txn.get("distance_from_home_km")))
    setf("failed_attempts_24h", _coerce_number(txn.get("failed_attempts_24h")))
    setf("device_changed", _coerce_number(txn.get("device_changed")))
    setf("avg_amount_30d", avg30)
    setf("amount_to_avg_ratio", amount / (avg30 + 1.0))

    mcc = str(txn.get("merchant_category", "grocery")).lower().strip()
    if mcc not in MERCHANT_CATEGORIES:
        mcc = "grocery"  # unknown categories fall back gracefully
    setf(f"mcc_{mcc}", 1.0)
    return vec


def risk_label(probability: float) -> str:
    """Risk band relative to the tuned decision threshold."""
    threshold = float(get_metrics()["final"]["threshold"])
    if probability >= threshold:
        return "high"
    if probability >= threshold / 2:
        return "medium"
    return "low"


def feature_contributions(vector: np.ndarray, top_k: int = 8) -> list[dict]:
    """Baseline-ablation contributions (SHAP-consistent approximation).

    contribution_i = M(x) - M(x with feature i := training median),
    where M is the raw margin (log-odds). Margins are used instead of
    probabilities because probabilities saturate near 0/1 and hide the
    signal; this also matches shap.TreeExplainer, whose values for XGBoost
    are natively in log-odds space. Positive contribution pushes the score
    toward fraud, negative away from it.
    """
    predictor = get_predictor()
    names = get_feature_names()
    medians = get_medians()
    base_margin = float(predictor.predict_margin(vector.reshape(1, -1))[0])

    median_vec = np.array([medians[n] for n in names], dtype=np.float64)
    ablated = np.tile(vector, (len(names), 1))
    ablated[np.arange(len(names)), np.arange(len(names))] = median_vec
    ablated_margin = predictor.predict_margin(ablated)

    contribs = [
        {
            "feature": name,
            "value": float(vector[i]),
            "contribution": float(base_margin - ablated_margin[i]),
        }
        for i, name in enumerate(names)
    ]
    contribs.sort(key=lambda d: abs(d["contribution"]), reverse=True)
    return contribs[:top_k]


def score_transaction(txn: dict, top_k: int = 8) -> dict:
    """Score one raw transaction -> probability, risk label, explanations."""
    vector = transaction_to_vector(txn)
    probability = float(get_predictor().predict_proba(vector.reshape(1, -1))[0])
    threshold = float(get_metrics()["final"]["threshold"])
    return {
        "fraud_probability": probability,
        "fraud_score": round(probability * 100, 2),
        "risk": risk_label(probability),
        "threshold": threshold,
        "flagged": probability >= threshold,
        "top_features": feature_contributions(vector, top_k=top_k),
    }


def demo_transaction() -> dict:
    """A realistic high-risk example so Live Check works out of the box."""
    return {
        "amount": 1849.99,
        "hour": 3,
        "day_of_week": 6,
        "card_present": 0,
        "is_international": 1,
        "txn_velocity_1h": 6,
        "amount_velocity_1h": 9200.0,
        "distance_from_home_km": 1240.5,
        "failed_attempts_24h": 3,
        "device_changed": 1,
        "avg_amount_30d": 64.2,
        "merchant_category": "electronics",
    }


def legit_transaction() -> dict:
    """A realistic low-risk example for contrast."""
    return {
        "amount": 24.5,
        "hour": 13,
        "day_of_week": 2,
        "card_present": 1,
        "is_international": 0,
        "txn_velocity_1h": 1,
        "amount_velocity_1h": 24.5,
        "distance_from_home_km": 3.2,
        "failed_attempts_24h": 0,
        "device_changed": 0,
        "avg_amount_30d": 58.0,
        "merchant_category": "grocery",
    }
