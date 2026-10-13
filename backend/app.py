"""
Fraud Detection System -- local development API (FastAPI).

Full dev version of the inference service. It reuses api/_model.py, so it
serves the exact same model and predictions as the Vercel serverless
functions in api/.

Endpoints:
    GET  /health    service + model status
    GET  /metrics   model performance report (precision/recall/F1/AUC-PR, ...)
    POST /predict   score one transaction (JSON)
    POST /batch     score many transactions (JSON {"csv": "<csv text>"})

Run:
    uvicorn backend.app:app --reload --port 8000
"""

from __future__ import annotations

import csv
import io
import sys
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Make the vendored serving code importable (same code Vercel bundles).
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from api._model import (  # noqa: E402
    RAW_FIELDS,
    demo_transaction,
    get_feature_names,
    get_metrics,
    get_predictor,
    legit_transaction,
    risk_label,
    score_transaction,
    transaction_to_vector,
)

app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud scoring with explainable predictions.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Transaction(BaseModel):
    """A single raw transaction. All fields optional; missing -> defaults."""

    amount: float = Field(default=0.0, ge=0)
    hour: int = Field(default=12, ge=0, le=23)
    day_of_week: int = Field(default=1, ge=0, le=6)
    card_present: int = Field(default=1, ge=0, le=1)
    is_international: int = Field(default=0, ge=0, le=1)
    txn_velocity_1h: int = Field(default=0, ge=0)
    amount_velocity_1h: float = Field(default=0.0, ge=0)
    distance_from_home_km: float = Field(default=0.0, ge=0)
    failed_attempts_24h: int = Field(default=0, ge=0)
    device_changed: int = Field(default=0, ge=0, le=1)
    avg_amount_30d: float = Field(default=50.0, ge=0)
    merchant_category: str = Field(default="grocery")

    class Config:
        extra = "ignore"  # tolerate extra client fields


class BatchRequest(BaseModel):
    csv: str = Field(..., description="CSV text with a header row")


@app.get("/health")
def health() -> dict[str, Any]:
    try:
        predictor = get_predictor()
        return {
            "status": "ok",
            "model": "xgboost-compact",
            "n_trees": predictor.n_trees,
            "n_features": len(get_feature_names()),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/metrics")
def metrics() -> dict[str, Any]:
    try:
        return get_metrics()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not load metrics: {exc}")


@app.get("/demo")
def demo() -> dict[str, Any]:
    """Seed transactions so the Live Check page works out of the box."""
    return {"high_risk": demo_transaction(), "low_risk": legit_transaction()}


@app.post("/predict")
def predict(txn: Transaction) -> dict[str, Any]:
    try:
        return score_transaction(txn.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}")


@app.post("/batch")
def batch(req: BatchRequest) -> dict[str, Any]:
    try:
        reader = csv.DictReader(io.StringIO(req.csv))
        if not reader.fieldnames:
            raise ValueError("Empty CSV or missing header row.")
        rows = [dict(r) for r in reader]
        if len(rows) > 5000:
            raise ValueError("Batch limit is 5000 rows per request.")

        predictor = get_predictor()
        threshold = float(get_metrics()["final"]["threshold"])
        matrix = np.stack([transaction_to_vector(r) for r in rows])
        probas = predictor.predict_proba(matrix)

        results = []
        for i, (row, p) in enumerate(zip(rows, probas)):
            p = float(p)
            results.append(
                {
                    "row": i + 1,
                    "fraud_probability": p,
                    "fraud_score": round(p * 100, 2),
                    "risk": risk_label(p),
                    "flagged": p >= threshold,
                    "input": {k: row.get(k) for k in RAW_FIELDS if k in row},
                }
            )
        return {
            "count": len(results),
            "flagged": sum(1 for r in results if r["flagged"]),
            "threshold": threshold,
            "results": results,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Batch scoring failed: {exc}")
