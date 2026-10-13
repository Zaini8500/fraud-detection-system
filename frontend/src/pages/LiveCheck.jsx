import { useEffect, useRef, useState } from 'react';
import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { apiGet, apiPost, featureLabel, fmtPct, riskClass } from '../lib';

const FIELDS = [
  { key: 'amount', label: 'Amount ($)', type: 'number', step: '0.01' },
  { key: 'hour', label: 'Hour of day (0–23)', type: 'number' },
  { key: 'day_of_week', label: 'Day of week (0=Mon)', type: 'number' },
  { key: 'card_present', label: 'Card present', type: 'select', options: [[1, 'Yes'], [0, 'No']] },
  { key: 'is_international', label: 'International', type: 'select', options: [[1, 'Yes'], [0, 'No']] },
  { key: 'txn_velocity_1h', label: 'Txns in last hour', type: 'number' },
  { key: 'amount_velocity_1h', label: 'Amount in last hour ($)', type: 'number', step: '0.01' },
  { key: 'distance_from_home_km', label: 'Distance from home (km)', type: 'number', step: '0.1' },
  { key: 'failed_attempts_24h', label: 'Failed attempts (24h)', type: 'number' },
  { key: 'device_changed', label: 'Device changed', type: 'select', options: [[1, 'Yes'], [0, 'No']] },
  { key: 'avg_amount_30d', label: 'Customer avg amount 30d ($)', type: 'number', step: '0.01' },
  {
    key: 'merchant_category', label: 'Merchant category', type: 'select',
    options: [['grocery', 'Grocery'], ['fuel', 'Fuel'], ['restaurant', 'Restaurant'], ['travel', 'Travel'],
              ['electronics', 'Electronics'], ['fashion', 'Fashion'], ['atm', 'ATM'], ['online', 'Online']],
  },
];

// Animated SVG arc gauge (0-100 fraud score).
function Gauge({ score }) {
  const [display, setDisplay] = useState(0);
  const raf = useRef(null);

  useEffect(() => {
    const start = display;
    const target = score;
    const t0 = performance.now();
    const dur = 900;
    cancelAnimationFrame(raf.current);
    const tick = (t) => {
      const k = Math.min(1, (t - t0) / dur);
      const eased = 1 - Math.pow(1 - k, 3);
      setDisplay(start + (target - start) * eased);
      if (k < 1) raf.current = requestAnimationFrame(tick);
    };
    raf.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [score]);

  const angle = (display / 100) * 180;
  const color = display >= 70 ? '#ef4444' : display >= 30 ? '#f59e0b' : '#22c55e';
  const r = 80;
  const cx = 100, cy = 100;
  const rad = (deg) => ((180 - deg) * Math.PI) / 180;
  const px = cx + r * Math.cos(rad(angle));
  const py = cy - r * Math.sin(rad(angle));
  const bgArc = `M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`;

  return (
    <div className="gauge-wrap">
      <svg width="220" height="130" viewBox="0 0 200 115">
        <path d={bgArc} fill="none" stroke="#1e2a47" strokeWidth="14" strokeLinecap="round" />
        <path d={bgArc} fill="none" stroke={color} strokeWidth="14" strokeLinecap="round"
              strokeDasharray={`${(angle / 180) * Math.PI * r} ${Math.PI * r}`}
              style={{ transition: 'stroke 0.3s' }} />
        <line x1={cx} y1={cy} x2={px} y2={py} stroke={color} strokeWidth="3" strokeLinecap="round" />
        <circle cx={cx} cy={cy} r="6" fill={color} />
      </svg>
      <div className="gauge-score" style={{ color }}>{display.toFixed(1)}</div>
      <div className="gauge-label">Fraud score / 100</div>
    </div>
  );
}

export default function LiveCheck({ api }) {
  const [form, setForm] = useState({
    amount: 1849.99, hour: 3, day_of_week: 6, card_present: 0, is_international: 1,
    txn_velocity_1h: 6, amount_velocity_1h: 9200, distance_from_home_km: 1240.5,
    failed_attempts_24h: 3, device_changed: 1, avg_amount_30d: 64.2,
    merchant_category: 'electronics',
  });
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const set = (key) => (e) => {
    const v = e.target.value;
    setForm((f) => ({ ...f, [key]: FIELDS.find((x) => x.key === key).type === 'select' && key !== 'merchant_category' ? Number(v) : key === 'merchant_category' ? v : Number(v) }));
  };

  async function loadDemo(kind) {
    try {
      const demo = await apiGet(api, '/demo');
      const d = demo[kind];
      setForm({
        amount: d.amount, hour: d.hour, day_of_week: d.day_of_week,
        card_present: d.card_present, is_international: d.is_international,
        txn_velocity_1h: d.txn_velocity_1h, amount_velocity_1h: d.amount_velocity_1h,
        distance_from_home_km: d.distance_from_home_km,
        failed_attempts_24h: d.failed_attempts_24h, device_changed: d.device_changed,
        avg_amount_30d: d.avg_amount_30d, merchant_category: d.merchant_category,
      });
    } catch {
      // /demo only exists on the FastAPI dev server; on Vercel keep the form.
    }
  }

  async function check(e) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const data = await apiPost(api, '/predict', form);
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  // Contributions are in log-odds (margin) space, like SHAP values.
  const contribs = (result?.top_features || []).map((d) => ({
    feature: featureLabel(d.feature),
    contribution: +d.contribution.toFixed(2),
  })).reverse();

  return (
    <div>
      <h1 className="page-title">Live transaction check</h1>
      <p className="page-sub">
        Enter a transaction and get an instant fraud score with a plain-English breakdown of
        which features pushed the score up or down.
      </p>

      <div className="grid grid-2">
        <div className="card">
          <h3>Transaction details</h3>
          <p className="hint">Try the demo presets, or type your own values.</p>
          <form onSubmit={check}>
            <div className="form-grid">
              {FIELDS.map((f) => (
                <div className="field" key={f.key}>
                  <label>{f.label}</label>
                  {f.type === 'select' ? (
                    <select value={form[f.key]} onChange={set(f.key)}>
                      {f.options.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
                    </select>
                  ) : (
                    <input type="number" step={f.step || '1'} value={form[f.key]} onChange={set(f.key)} />
                  )}
                </div>
              ))}
            </div>
            <div className="btn-row">
              <button className="btn" type="submit" disabled={loading}>
                {loading ? 'Scoring…' : '🔍 Check transaction'}
              </button>
              <button type="button" className="btn btn-ghost" onClick={() => loadDemo('high_risk')}>Load fraud-like demo</button>
              <button type="button" className="btn btn-ghost" onClick={() => loadDemo('low_risk')}>Load legit demo</button>
            </div>
          </form>
          {error && <div className="alert alert-error">{error}</div>}
        </div>

        <div className="card">
          <h3>Verdict</h3>
          <p className="hint">Score 0–100 · decision threshold shown below.</p>
          {!result && !loading && <p className="note">No check yet — submit the form to score a transaction.</p>}
          {loading && <div className="spinner" />}
          {result && (
            <div>
              <Gauge score={result.fraud_score} />
              <div style={{ textAlign: 'center', margin: '10px 0 4px' }}>
                <span className={riskClass(result.risk)}>{result.risk} risk</span>
              </div>
              <div className="stat-row"><span className="k">Fraud probability</span><span className="v">{fmtPct(result.fraud_probability, 2)}</span></div>
              <div className="stat-row"><span className="k">Decision threshold</span><span className="v">{fmtPct(result.threshold, 2)}</span></div>
              <div className="stat-row"><span className="k">Flagged for review</span><span className="v">{result.flagged ? '✅ Yes' : '— No'}</span></div>
            </div>
          )}
        </div>
      </div>

      {result && (
        <div className="card" style={{ marginTop: 16 }}>
          <h3>Why this score? — feature contributions (log-odds)</h3>
          <p className="hint">
            Positive bars pushed the score <b>toward fraud</b>; negative bars pushed it <b>toward legitimate</b>.
            Computed by baseline ablation in log-odds space (each feature vs its training median) — the same
            space SHAP values live in — a lightweight, SHAP-consistent approximation used at serving time.
          </p>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={contribs} layout="vertical" margin={{ left: 30, right: 30 }}>
              <CartesianGrid stroke="#1e2a47" strokeDasharray="3 3" />
              <XAxis type="number" stroke="#8b96b3" tick={{ fontSize: 12 }} />
              <YAxis type="category" dataKey="feature" stroke="#8b96b3" tick={{ fontSize: 12.5 }} width={190} />
              <Tooltip
                contentStyle={{ background: '#0f1524', border: '1px solid #1e2a47', borderRadius: 8 }}
                formatter={(v) => [`${v > 0 ? '+' : ''}${v} log-odds`, 'Contribution']}
              />
              <Bar dataKey="contribution" radius={[0, 6, 6, 0]}>
                {contribs.map((d, i) => (
                  <Cell key={i} fill={d.contribution >= 0 ? '#ef4444' : '#22c55e'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}
