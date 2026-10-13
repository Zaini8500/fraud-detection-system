import { useEffect, useState } from 'react';
import {
  Bar, BarChart, CartesianGrid, Cell, Line, LineChart,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts';
import { apiGet, featureLabel, fmtPct } from '../lib';

function Kpi({ label, value, sub, tone }) {
  return (
    <div className={'card ' + (tone ? `kpi-${tone}` : '')}>
      <div className="kpi-value">{value}</div>
      <div className="kpi-label">{label}</div>
      {sub && <div className="kpi-delta note">{sub}</div>}
    </div>
  );
}

function ConfusionMatrix({ cm }) {
  const cells = [
    { cls: 'cm-head', n: '', l: '' },
    { cls: 'cm-head', n: 'Predicted legit', l: '' },
    { cls: 'cm-head', n: 'Predicted fraud', l: '' },
    { cls: 'cm-head', n: 'Actual legit', l: '' },
    { cls: 'cm-cell cm-tn', n: cm.tn.toLocaleString(), l: 'True negatives' },
    { cls: 'cm-cell cm-fp', n: cm.fp.toLocaleString(), l: 'False positives' },
    { cls: 'cm-head', n: 'Actual fraud', l: '' },
    { cls: 'cm-cell cm-fn', n: cm.fn.toLocaleString(), l: 'False negatives (missed fraud)' },
    { cls: 'cm-cell cm-tp', n: cm.tp.toLocaleString(), l: 'True positives (caught)' },
  ];
  return (
    <div className="cm-grid">
      {cells.map((c, i) => (
        <div key={i} className={c.cls}>
          <div className="n">{c.n}</div>
          <div className="l">{c.l}</div>
        </div>
      ))}
    </div>
  );
}

export default function Dashboard({ api }) {
  const [metrics, setMetrics] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    apiGet(api, '/metrics').then(setMetrics).catch((e) => setError(e.message));
  }, [api]);

  if (error) return <div className="alert alert-error">Could not load metrics: {error}. Is the API running?</div>;
  if (!metrics) return <div className="spinner" />;

  const fin = metrics.final;
  const cm = fin.confusion_matrix;
  const meta = metrics.meta || {};
  const prCurve = (fin.pr_curve || []).map((p) => ({
    recall: +p.recall.toFixed(3),
    precision: +p.precision.toFixed(3),
  }));
  const shapTop = (fin.shap_importance || []).slice(0, 10).map((d) => ({
    feature: featureLabel(d.feature),
    importance: +d.mean_abs_shap.toFixed(4),
  })).reverse();

  const fraudRate = meta.fraud_rate_train;
  const imbalance = [
    { name: 'Legitimate', count: Math.round((1 - fraudRate) * 1000), fill: '#22c55e' },
    { name: 'Fraud', count: Math.round(fraudRate * 1000), fill: '#ef4444' },
  ];

  return (
    <div>
      <h1 className="page-title">Fraud overview</h1>
      <p className="page-sub">
        XGBoost classifier on {meta.train_rows?.toLocaleString()} transactions.
        Decision threshold <code>{fin.threshold.toFixed(4)}</code> tuned to maximise F1 on the validation PR curve.
      </p>

      <div className="grid grid-4">
        <Kpi label="Precision" value={fmtPct(fin.precision)} tone="accent" sub="of flagged txns are truly fraud" />
        <Kpi label="Recall" value={fmtPct(fin.recall)} tone="green" sub="of fraud actually caught" />
        <Kpi label="F1 score" value={fin.f1.toFixed(3)} tone="amber" sub="harmonic mean of the two" />
        <Kpi label="AUC-PR" value={fin.auc_pr.toFixed(3)} tone="red" sub="area under precision-recall curve" />
      </div>

      <div className="grid grid-2" style={{ marginTop: 16 }}>
        <div className="card">
          <h3>Precision–Recall curve</h3>
          <p className="hint">Thresholds sweep the trade-off. The dot marks the tuned operating point.</p>
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={prCurve} margin={{ top: 8, right: 12, bottom: 8, left: 0 }}>
              <CartesianGrid stroke="#1e2a47" strokeDasharray="3 3" />
              <XAxis dataKey="recall" stroke="#8b96b3" tick={{ fontSize: 12 }} />
              <YAxis stroke="#8b96b3" tick={{ fontSize: 12 }} domain={[0, 1]} />
              <Tooltip
                contentStyle={{ background: '#0f1524', border: '1px solid #1e2a47', borderRadius: 8 }}
                formatter={(v, name) => [fmtPct(v, 1), name]}
              />
              <Line type="monotone" dataKey="precision" stroke="#4f7cff" strokeWidth={2.5} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="card">
          <h3>Confusion matrix — held-out test set</h3>
          <p className="hint">At the tuned threshold of {fin.threshold.toFixed(4)}.</p>
          <ConfusionMatrix cm={cm} />
          <p className="note" style={{ marginTop: 12 }}>
            False negatives (missed fraud) cost the business; false positives annoy customers.
            The threshold balances the two via F1.
          </p>
        </div>
      </div>

      <div className="grid grid-2" style={{ marginTop: 16 }}>
        <div className="card">
          <h3>Class imbalance</h3>
          <p className="hint">Only {fmtPct(fraudRate, 2)} of transactions are fraud — accuracy alone would lie.</p>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={imbalance} layout="vertical" margin={{ left: 20 }}>
              <CartesianGrid stroke="#1e2a47" strokeDasharray="3 3" />
              <XAxis type="number" stroke="#8b96b3" tick={{ fontSize: 12 }} />
              <YAxis type="category" dataKey="name" stroke="#8b96b3" tick={{ fontSize: 13 }} width={90} />
              <Tooltip contentStyle={{ background: '#0f1524', border: '1px solid #1e2a47', borderRadius: 8 }} />
              <Bar dataKey="count" radius={[0, 6, 6, 0]}>
                {imbalance.map((d, i) => <Cell key={i} fill={d.fill} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <p className="note">Per 1,000 transactions: {imbalance[0].count.toLocaleString()} legit vs {imbalance[1].count} fraud.</p>
        </div>

        <div className="card">
          <h3>What drives fraud? — mean |SHAP|</h3>
          <p className="hint">Exact SHAP values from TreeExplainer on a test sample.</p>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={shapTop} layout="vertical" margin={{ left: 20 }}>
              <CartesianGrid stroke="#1e2a47" strokeDasharray="3 3" />
              <XAxis type="number" stroke="#8b96b3" tick={{ fontSize: 12 }} />
              <YAxis type="category" dataKey="feature" stroke="#8b96b3" tick={{ fontSize: 12 }} width={150} />
              <Tooltip contentStyle={{ background: '#0f1524', border: '1px solid #1e2a47', borderRadius: 8 }} />
              <Bar dataKey="importance" fill="#7c5cff" radius={[0, 6, 6, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="card" style={{ marginTop: 16 }}>
        <h3>Model comparison</h3>
        <p className="hint">Same test set, each method's own tuned threshold. This is why imbalance handling matters.</p>
        <div className="table-wrap">
          <table>
            <thead>
              <tr><th>Model</th><th>Precision</th><th>Recall</th><th>F1</th><th>AUC-PR</th><th>Missed fraud (FN)</th></tr>
            </thead>
            <tbody>
              {Object.entries(metrics.comparison || {}).map(([name, m]) => (
                <tr key={name} className={name.includes('FINAL') ? 'flagged' : ''}>
                  <td>{name.replace(/_/g, ' ')}</td>
                  <td>{fmtPct(m.precision)}</td>
                  <td>{fmtPct(m.recall)}</td>
                  <td>{m.f1.toFixed(3)}</td>
                  <td>{m.auc_pr.toFixed(3)}</td>
                  <td>{m.confusion_matrix.fn.toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
