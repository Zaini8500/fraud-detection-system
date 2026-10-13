import { useRef, useState } from 'react';
import { apiPost, fmtPct, riskClass } from '../lib';

const SAMPLE_CSV = `amount,hour,day_of_week,card_present,is_international,txn_velocity_1h,amount_velocity_1h,distance_from_home_km,failed_attempts_24h,device_changed,avg_amount_30d,merchant_category
24.50,13,2,1,0,1,24.50,3.2,0,0,58.0,grocery
1849.99,3,6,0,1,6,9200.00,1240.5,3,1,64.2,electronics
67.20,19,5,1,0,2,120.40,12.8,0,0,71.5,restaurant
499.00,1,0,0,1,5,2600.00,890.0,2,1,45.0,online
12.99,9,1,1,0,0,0.00,1.1,0,0,40.3,fuel`;

export default function Batch({ api }) {
  const [drag, setDrag] = useState(false);
  const [fileName, setFileName] = useState('');
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const fileRef = useRef(null);

  async function scoreText(csvText, name) {
    setLoading(true);
    setError(null);
    setResults(null);
    try {
      const data = await apiPost(api, '/batch', { csv: csvText });
      setResults({ ...data, fileName: name });
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  function handleFile(file) {
    if (!file) return;
    setFileName(file.name);
    const reader = new FileReader();
    reader.onload = () => scoreText(reader.result, file.name);
    reader.readAsText(file);
  }

  function downloadCsv() {
    if (!results) return;
    const header = 'row,fraud_probability,fraud_score,risk,flagged,amount,hour,merchant_category,card_present,is_international';
    const lines = results.results.map((r) =>
      [r.row, r.fraud_probability.toFixed(6), r.fraud_score, r.risk, r.flagged,
       r.input.amount ?? '', r.input.hour ?? '', r.input.merchant_category ?? '',
       r.input.card_present ?? '', r.input.is_international ?? ''].join(',')
    );
    const blob = new Blob([[header, ...lines].join('\n')], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'fraud_scores.csv';
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div>
      <h1 className="page-title">Batch scoring</h1>
      <p className="page-sub">
        Upload a CSV of transactions (same columns as the single-check form) and score up to
        5,000 rows in one request. Results can be downloaded as a scored CSV.
      </p>

      <div className="card">
        <div
          className={'drop-zone' + (drag ? ' drag' : '')}
          onClick={() => fileRef.current.click()}
          onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
          onDragLeave={() => setDrag(false)}
          onDrop={(e) => { e.preventDefault(); setDrag(false); handleFile(e.dataTransfer.files[0]); }}
        >
          <div style={{ fontSize: 34, marginBottom: 8 }}>📤</div>
          <div><b>{fileName || 'Drop a CSV here, or click to browse'}</b></div>
          <div className="note" style={{ marginTop: 6 }}>Header row required · max 5,000 rows</div>
          <input ref={fileRef} type="file" accept=".csv,text/csv" style={{ display: 'none' }}
                 onChange={(e) => handleFile(e.target.files[0])} />
        </div>
        <div className="btn-row">
          <button className="btn btn-ghost" onClick={() => scoreText(SAMPLE_CSV, 'sample_transactions.csv')}>
            Use sample CSV
          </button>
          <a href="/sample_transactions.csv" download className="note" style={{ alignSelf: 'center' }}>
            ⬇ Download the sample file
          </a>
        </div>
        {error && <div className="alert alert-error">{error}</div>}
        {loading && <div className="spinner" />}
      </div>

      {results && (
        <div className="card" style={{ marginTop: 16 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 10 }}>
            <div>
              <h3 style={{ margin: 0 }}>{results.fileName}</h3>
              <p className="hint" style={{ margin: '4px 0 0' }}>
                {results.count} scored · <b style={{ color: '#ef4444' }}>{results.flagged} flagged</b> at threshold {fmtPct(results.threshold, 1)}
              </p>
            </div>
            <button className="btn" onClick={downloadCsv}>⬇ Download scored CSV</button>
          </div>
          <div className="table-wrap" style={{ marginTop: 14 }}>
            <table>
              <thead>
                <tr><th>#</th><th>Risk</th><th>Score</th><th>Probability</th><th>Amount</th><th>Hour</th><th>Merchant</th><th>Card</th><th>Intl</th></tr>
              </thead>
              <tbody>
                {results.results.slice(0, 200).map((r) => (
                  <tr key={r.row} className={r.flagged ? 'flagged' : ''}>
                    <td>{r.row}</td>
                    <td><span className={'dot dot-' + r.risk} />{r.risk}</td>
                    <td><b>{r.fraud_score}</b></td>
                    <td>{fmtPct(r.fraud_probability, 2)}</td>
                    <td>${r.input.amount}</td>
                    <td>{r.input.hour}:00</td>
                    <td>{r.input.merchant_category}</td>
                    <td>{r.input.card_present === '1' || r.input.card_present === 1 ? 'present' : 'not present'}</td>
                    <td>{r.input.is_international === '1' || r.input.is_international === 1 ? 'yes' : 'no'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {results.count > 200 && <p className="note">Showing first 200 of {results.count} rows — download the CSV for the full list.</p>}
        </div>
      )}
    </div>
  );
}
