import { useEffect, useState } from 'react';
import { Link, NavLink, Route, Routes } from 'react-router-dom';
import Dashboard from './pages/Dashboard';
import LiveCheck from './pages/LiveCheck';
import Batch from './pages/Batch';
import ModelInfo from './pages/ModelInfo';

const API = import.meta.env.VITE_API_URL || '/api';

function Navbar() {
  return (
    <nav className="navbar">
      <Link to="/" className="brand">
        <span className="brand-mark">🛡️</span>
        <span>
          <div className="brand-name">Sentinel</div>
          <div className="brand-sub">Fraud Detection</div>
        </span>
      </Link>
      <div className="nav-links">
        <NavLink to="/" end className={({ isActive }) => 'nav-link' + (isActive ? ' active' : '')}>Dashboard</NavLink>
        <NavLink to="/live" className={({ isActive }) => 'nav-link' + (isActive ? ' active' : '')}>Live Check</NavLink>
        <NavLink to="/batch" className={({ isActive }) => 'nav-link' + (isActive ? ' active' : '')}>Batch Scoring</NavLink>
        <NavLink to="/model" className={({ isActive }) => 'nav-link' + (isActive ? ' active' : '')}>Model Info</NavLink>
      </div>
    </nav>
  );
}

function HealthPill() {
  const [health, setHealth] = useState(null);
  useEffect(() => {
    fetch(`${API}/health`).then((r) => r.json()).then(setHealth).catch(() => setHealth({ status: 'error' }));
  }, []);
  const ok = health && health.status === 'ok';
  return (
    <span className="note" style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
      <span className={'dot ' + (ok ? 'dot-low' : 'dot-high')} />
      {ok ? `API online · ${health.n_trees} trees` : health ? 'API unreachable' : 'Connecting…'}
    </span>
  );
}

export default function App() {
  return (
    <div className="app-shell">
      <Navbar />
      <Routes>
        <Route path="/" element={<Dashboard api={API} />} />
        <Route path="/live" element={<LiveCheck api={API} />} />
        <Route path="/batch" element={<Batch api={API} />} />
        <Route path="/model" element={<ModelInfo api={API} />} />
      </Routes>
      <footer className="site-footer">
        <span>Sentinel · XGBoost fraud detector · explanations via SHAP (training) + ablation (serving)</span>
        <HealthPill />
      </footer>
    </div>
  );
}
