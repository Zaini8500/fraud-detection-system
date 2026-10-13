# Sentinel — Fraud Detection System

Real-time fraud detection for financial transactions: an **XGBoost** classifier
trained on highly imbalanced data (**0.5% fraud**), with **exact SHAP**
explanations at training time and a lightweight SHAP-consistent approximation at
serving time. A dark-fintech **React dashboard** (KPIs, PR curve, confusion
matrix, live scoring with explanations, batch CSV scoring) talks to a **Python
API** that runs both locally (**FastAPI**) and on **Vercel serverless functions**
— one repo, one Vercel project.

## Architecture

```
┌──────────────────────────┐
│  data/generate.py        │  synthetic transactions (60k rows, 0.5% fraud)
└────────────┬─────────────┘
             ▼
┌──────────────────────────┐
│  models/train.py         │  XGBoost + scale_pos_weight, threshold tuning on
│                          │  the PR curve, baselines (RF / IsolationForest /
│                          │  SMOTE), exact SHAP via TreeExplainer
└────────────┬─────────────┘
             │  exports compact booster JSON (108 KB) + medians + metrics
             ▼
┌──────────────────────────┐      ┌──────────────────────────────────┐
│  backend/app.py          │      │  api/*.py  (Vercel serverless)   │
│  FastAPI (local dev)     │      │  predict · batch · metrics ·health│
└────────────┬─────────────┘      └───────────────┬──────────────────┘
             │  both import api/_model.py + api/_xgb_runtime.py
             │  (vendored pure-numpy XGBoost inference — no xgboost needed)
             ▼                                    ▼
┌──────────────────────────────────────────────────────────┐
│  frontend/  React + Vite + recharts (dark fintech theme) │
│  Dashboard · Live Check · Batch Scoring · Model Info      │
└──────────────────────────────────────────────────────────┘
```

**Serving design choice (XGBoost export, not sklearn):** the real XGBoost
booster is dumped to JSON, converted to compact per-tree numpy arrays
(`api/artifacts/booster_compact.json`), and scored by a ~100-line pure-numpy
predictor (`api/_xgb_runtime.py`) that is **bit-identical** to
`booster.predict()` (verified by `tests/test_parity.py`). Serverless functions
therefore need only `numpy` — no xgboost/shap/pandas. The local FastAPI app
imports the same vendored code, so predictions match everywhere.

## Quickstart

```bash
# 1. Training environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Generate data (60k rows, 0.5% fraud)
python data/generate.py --n 60000 --fraud-rate 0.005 --seed 42

# 3. Train + export (writes models/artifacts/ and mirrors to api/artifacts/)
python models/train.py --data data/transactions.csv --artifacts models/artifacts

# 4. Parity check: vendored predictor vs real XGBoost
python tests/test_parity.py

# 5. Run the API locally
uvicorn backend.app:app --reload --port 8000
# -> http://localhost:8000/health  /metrics  /predict  /batch  (/docs for Swagger)

# 6. Run the dashboard locally
cd frontend && npm install && npm run dev
# -> http://localhost:5173  (API calls proxied to :8000)
```

**Try it immediately:** open Live Check → "Load fraud-like demo" → Check transaction.
Or Batch Scoring → "Use sample CSV".

## API reference

| Endpoint | Method | Body | Returns |
|---|---|---|---|
| `/health` | GET | — | status, model, tree count |
| `/metrics` | GET | — | precision/recall/F1/AUC-PR, confusion matrix, PR curve, SHAP importance, model comparison |
| `/predict` | POST | transaction JSON | fraud probability, risk label, threshold, top contributing features |
| `/batch` | POST | `{"csv": "<csv text>"}` | scored rows (max 5,000), flagged count |
| `/demo` | GET | — | (local FastAPI only) seed transactions |

On Vercel the same endpoints live under `/api/*` (e.g. `/api/predict`).

Transaction fields: `amount, hour, day_of_week, card_present, is_international,
txn_velocity_1h, amount_velocity_1h, distance_from_home_km, failed_attempts_24h,
device_changed, avg_amount_30d, merchant_category`.

## How the imbalance is handled

- **Stratified** train/val/test splits keep the 0.5% fraud rate everywhere.
- Final model: XGBoost with `scale_pos_weight = neg/pos ≈ 199` — no resampling.
- **SMOTE** tried as an experiment; class weighting won on AUC-PR.
- **Threshold tuned to maximise F1 on the validation PR curve**, frozen, then
  evaluated once on the held-out test set. Accuracy is never reported alone.
- Baselines for context: stratified dummy, balanced RandomForest, unsupervised
  IsolationForest (trained on legitimate txns only).

## SHAP: exact at training, approximated at serving

- **Training** (`models/train.py`): `shap.TreeExplainer` computes **exact** SHAP
  values on a test sample → mean|SHAP| importance chart + summary plot
  (`frontend/public/shap_summary.png`).
- **Serving** (`api/_model.py`): shap/xgboost can't run in a tiny serverless
  function, so each prediction is explained with **baseline ablation**:
  `contributionᵢ = P(x) − P(x with featureᵢ := training median)`. This is a
  SHAP-consistent approximation (same sign convention), **not** exact SHAP —
  the UI and docs say so explicitly.

## Deploy on Vercel

One repo → one Vercel project. The React app builds to static files; `api/*.py`
become serverless functions. Exact settings are in [DEPLOY_CHECKLIST.md](DEPLOY_CHECKLIST.md):

- **Build Command:** `cd frontend && npm install && npm run build`
- **Output Directory:** `frontend/dist`
- **Install Command:** `cd frontend && npm install`
- No environment variables required. Python deps for `api/` come from
  `api/requirements.txt` (numpy only).

## Optional: real Kaggle data

```bash
python data/generate.py --kaggle   # needs the `kaggle` CLI authenticated
```

This downloads the `mlg-ulb/creditcardfraud` dataset into `data/kaggle/`. Its
features are PCA-anonymised (`V1..V28`); adapt the column handling in
`models/train.py` if you want to train on it.

## Honest limitations

1. **Synthetic data.** Real fraud data is proprietary and messier; the generator
   makes fraud *cleaner* than reality, so test metrics are optimistic. Treat them
   as a demonstration of the pipeline, not production performance.
2. **Serving explanations are approximate** (baseline ablation), not exact SHAP.
3. **Batch ≠ streaming.** `/batch` scores CSV uploads; a production system would
   use a streaming feature store and sub-100ms SLAs.
4. **No drift monitoring / retraining loop** — fraud patterns evolve; this is a
   portfolio build, not a production MLOps system.
5. **No authentication** on the API — add it before exposing publicly.

## Repo layout

```
├── api/                  # Vercel serverless functions (predict/batch/metrics/health)
│   ├── _model.py         # shared serving logic (also used by FastAPI)
│   ├── _xgb_runtime.py   # vendored pure-numpy XGBoost predictor
│   ├── artifacts/        # exported model: booster_compact.json, medians, metrics
│   └── requirements.txt  # numpy only
├── backend/app.py        # FastAPI dev server (same model, same predictions)
├── data/generate.py      # synthetic data generator (+ --kaggle path)
├── models/train.py       # training, threshold tuning, SHAP, export
├── frontend/             # React + Vite dashboard (recharts, plain CSS)
├── tests/test_parity.py  # vendored predictor == XGBoost (asserted)
├── vercel.json, DEPLOY_CHECKLIST.md, requirements.txt, LICENSE
```

## License

MIT — see [LICENSE](LICENSE).
