# Vercel Deploy Checklist — Fraud Detection System

One repo → **one** Vercel project. The React frontend builds to static files;
`api/*.py` become Python serverless functions automatically.

## Project settings (Vercel dashboard → Project → Settings)

| Setting | Value |
|---|---|
| Framework Preset | `Other` (or Vite — then override the commands below) |
| Root Directory | repo root (`./`) — do **not** set to `frontend/` |
| Install Command | `cd frontend && npm install` |
| Build Command | `cd frontend && npm run build` |
| Output Directory | `frontend/dist` |
| Environment Variables | none required |

> Why these: Vercel auto-detects `api/*.py` as Python serverless functions and
> installs `api/requirements.txt` (numpy only) for them. The frontend must be
> built from `frontend/` but the functions must stay at the repo-root `api/`,
> so the project root stays at `./` with custom commands.

## Pre-deploy verification (do all of these locally first)

- [ ] `python data/generate.py` → `data/transactions.csv` created
- [ ] `python models/train.py` → `api/artifacts/` contains
      `booster_compact.json`, `feature_names.json`, `feature_medians.json`, `metrics.json`
- [ ] `python tests/test_parity.py` → **PARITY TEST PASSED**
- [ ] `uvicorn backend.app:app --port 8000` → `GET /health` returns `{"status":"ok",...}`
- [ ] `cd frontend && npm install && npm run build` → succeeds, `frontend/dist/` created
- [ ] `git add -A` includes `api/artifacts/*.json` (the model **must** be committed —
      Vercel bundles whatever is in git) and `frontend/public/shap_summary.png`

## After deploy — smoke test

```bash
BASE=https://<your-project>.vercel.app
curl $BASE/api/health
# {"status":"ok","model":"xgboost-compact","n_trees":295,"n_features":20}

curl -X POST $BASE/api/predict -H 'Content-Type: application/json' -d '{
  "amount": 1849.99, "hour": 3, "day_of_week": 6, "card_present": 0,
  "is_international": 1, "txn_velocity_1h": 6, "amount_velocity_1h": 9200,
  "distance_from_home_km": 1240.5, "failed_attempts_24h": 3,
  "device_changed": 1, "avg_amount_30d": 64.2, "merchant_category": "electronics"
}'
# expect: "risk": "high", "flagged": true, "top_features": [...]

curl $BASE/api/metrics | head -c 300
# expect: {"final": {"precision": ..., "recall": ..., ...}, ...}
```

Then open `$BASE/` in a browser: Dashboard charts → Live Check demo → Batch sample CSV.

## Notes & gotchas

- **Python version:** Vercel picks the runtime from `api/requirements.txt`
  automatically (currently Python 3.12). No `runtime` pin needed.
- **Cold starts:** the compact model is ~108 KB JSON; the function loads it once
  per instance (`lru_cache`) — first request warms it.
- **`maxDuration: 30`** is set in `vercel.json` for batch scoring headroom.
  (30s needs a Pro plan on some Vercel tiers; on Hobby, functions cap lower —
  batch is still fine for a few thousand rows, it takes ~seconds.)
- **No env vars** — the frontend calls same-origin `/api/*`, so no
  `VITE_API_URL` is needed in production. For local dev the Vite proxy sends
  `/api/*` to `localhost:8000`.
- **Do not** put `xgboost`, `shap`, or `pandas` in `api/requirements.txt` —
  they exceed comfortable serverless size and are unnecessary: inference is
  pure numpy via `api/_xgb_runtime.py`.
- If `/api/*` returns 404 after deploy: check that `api/` is at the **repo
  root** in git (not nested under `frontend/`), and that the Root Directory
  setting is `./`.
- Model retraining = re-run `models/train.py`, commit the new
  `api/artifacts/*.json`, push → Vercel redeploys with the new model.
