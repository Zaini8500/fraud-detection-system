import { useEffect, useState } from 'react';
import { apiGet } from '../lib';

export default function ModelInfo({ api }) {
  const [metrics, setMetrics] = useState(null);

  useEffect(() => {
    apiGet(api, '/metrics').then(setMetrics).catch(() => {});
  }, [api]);

  const meta = metrics?.meta || {};
  const fin = metrics?.final || {};

  return (
    <div className="doc">
      <h1 className="page-title">How the model works</h1>
      <p className="page-sub">
        A plain-English tour of the fraud detector: the imbalance problem, how it was handled,
        and how predictions are explained.
      </p>

      <div className="card">
        <h2>🎯 The problem</h2>
        <p>
          Only about <b>0.5% of transactions are fraud</b> — roughly 1 in 200. A lazy model that
          always answers "legitimate" would be 99.5% accurate and completely useless. That is why
          this project <b>never reports accuracy alone</b>: every decision is driven by{' '}
          <b>precision</b> (of the flagged transactions, how many are truly fraud),{' '}
          <b>recall</b> (of all fraud, how much we catch), <b>F1</b>, and <b>AUC-PR</b>
          (area under the precision–recall curve).
        </p>
      </div>

      <div className="card" style={{ marginTop: 16 }}>
        <h2>⚖️ Handling the imbalance</h2>
        <ul>
          <li><b>Stratified splits</b> — train/validation/test splits preserve the 0.5% fraud rate, so evaluation is honest.</li>
          <li><b><code>scale_pos_weight</code></b> — the final XGBoost model weights each fraud case ~200× more than a
            legitimate one, so the optimiser cannot ignore the minority class. No resampling needed.</li>
          <li><b>SMOTE experiment</b> — synthetic minority oversampling was tried as an alternative (see the
            comparison table on the Dashboard). Class weighting won on AUC-PR with a far smaller training cost.</li>
          <li><b>Balanced RandomForest</b> and an <b>IsolationForest</b> anomaly baseline are trained for comparison —
            the unsupervised anomaly detector catches far less fraud, which is why this is a supervised problem.</li>
          <li><b>Threshold tuning on the PR curve</b> — instead of the default 0.5 cutoff, the decision threshold
            ({fin.threshold ? fin.threshold.toFixed(4) : '…'}) is chosen to <b>maximise F1 on the validation set</b>,
            then frozen and evaluated once on the held-out test set.</li>
        </ul>
        <div className="callout">
          Business trade-off: false negatives (missed fraud) cost money directly; false positives (blocking a real
          customer) cost trust. The tuned threshold balances them via F1 — and it is exposed in the API so the
          business can move it.
        </div>
      </div>

      <div className="card" style={{ marginTop: 16 }}>
        <h2>🔍 Explaining predictions with SHAP</h2>
        <p>
          During training, <b>exact SHAP values</b> are computed with <code>shap.TreeExplainer</code> on a test
          sample. SHAP (SHapley Additive exPlanations) fairly attributes the model's output to each feature —
          e.g. "3 AM transaction: +18 fraud points; card present: −6 points".
        </p>
        <p>
          SHAP needs the full <code>shap</code> + <code>xgboost</code> stack, which is too heavy for a serverless
          function. So at <b>serving time</b> the API uses a lightweight, dependency-free approximation:{' '}
          <b>baseline ablation</b> — <code>contributionᵢ = P(x) − P(x with feature i replaced by its training median)</code>.
          It is SHAP-consistent in spirit (positive pushes toward fraud, negative away), runs on pure numpy, and
          produces the bar chart you see on the Live Check page.
        </p>
        {meta.n_estimators && (
          <div style={{ marginTop: 12 }}>
            <div className="stat-row"><span className="k">Algorithm</span><span className="v">XGBoost (binary:logistic)</span></div>
            <div className="stat-row"><span className="k">Trees / max depth</span><span className="v">{meta.n_estimators} / {meta.max_depth}</span></div>
            <div className="stat-row"><span className="k">scale_pos_weight</span><span className="v">{Number(meta.scale_pos_weight).toFixed(1)}</span></div>
            <div className="stat-row"><span className="k">Decision threshold</span><span className="v">{Number(meta.threshold).toFixed(4)}</span></div>
            <div className="stat-row"><span className="k">Training rows</span><span className="v">{Number(meta.train_rows).toLocaleString()}</span></div>
          </div>
        )}
        <p className="note" style={{ marginTop: 14 }}>
          <img src="/shap_summary.png" alt="SHAP summary plot" style={{ width: '100%', borderRadius: 10, border: '1px solid var(--card-border)' }} />
          SHAP summary plot generated during training (beeswarm of exact SHAP values per feature).
        </p>
      </div>

      <div className="card" style={{ marginTop: 16 }}>
        <h2>🏗️ Deployment architecture</h2>
        <p>
          The model is trained once with the heavy stack (XGBoost + SHAP). The booster is then exported to a
          compact JSON of tree arrays plus a <b>~100-line pure-numpy predictor</b> vendored in <code>api/</code>.
          Vercel serverless functions need only <code>numpy</code> — no xgboost, no pandas, no shap.
          The local FastAPI backend imports the same vendored code, so predictions match exactly.
        </p>
        <p className="note">
          Honest limitations: the data is synthetic (real fraud data is proprietary); the serving-time
          explanation is an approximation, not exact SHAP; and a batch endpoint is not a streaming system —
          see the README for the full list.
        </p>
      </div>
    </div>
  );
}
