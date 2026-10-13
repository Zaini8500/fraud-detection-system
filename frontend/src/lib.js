// Shared helpers: API base, formatting, feature display names.

export function fmtPct(x, digits = 1) {
  if (x == null || Number.isNaN(x)) return '—';
  return `${(x * 100).toFixed(digits)}%`;
}

// Human-friendly labels for model features.
const FEATURE_LABELS = {
  amount: 'Amount ($)',
  hour: 'Hour of day',
  day_of_week: 'Day of week',
  card_present: 'Card present',
  is_international: 'International',
  txn_velocity_1h: 'Txns last hour',
  amount_velocity_1h: 'Amount last hour ($)',
  distance_from_home_km: 'Distance from home (km)',
  failed_attempts_24h: 'Failed attempts (24h)',
  device_changed: 'Device changed',
  avg_amount_30d: 'Avg amount 30d ($)',
  amount_to_avg_ratio: 'Amount ÷ usual amount',
  mcc_grocery: 'Merchant: grocery',
  mcc_fuel: 'Merchant: fuel',
  mcc_restaurant: 'Merchant: restaurant',
  mcc_travel: 'Merchant: travel',
  mcc_electronics: 'Merchant: electronics',
  mcc_fashion: 'Merchant: fashion',
  mcc_atm: 'Merchant: ATM',
  mcc_online: 'Merchant: online',
};

export function featureLabel(name) {
  return FEATURE_LABELS[name] || name;
}

export function riskClass(risk) {
  return `risk-badge risk-${risk || 'low'}`;
}

export async function apiGet(api, path) {
  const res = await fetch(`${api}${path}`);
  if (!res.ok) throw new Error(`Request failed (${res.status})`);
  return res.json();
}

export async function apiPost(api, path, body) {
  const res = await fetch(`${api}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `Request failed (${res.status})`);
  return data;
}
