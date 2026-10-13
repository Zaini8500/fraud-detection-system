"""Quick end-to-end demo: score two transactions via the local API.

Run the backend first:
    uvicorn backend.app:app --port 8000
Then:
    python scripts/demo.py [--url http://localhost:8000]
"""

import argparse
import json
import sys
import urllib.request

FRAUD_LIKE = {
    "amount": 1849.99, "hour": 3, "day_of_week": 6, "card_present": 0,
    "is_international": 1, "txn_velocity_1h": 6, "amount_velocity_1h": 9200.0,
    "distance_from_home_km": 1240.5, "failed_attempts_24h": 3,
    "device_changed": 1, "avg_amount_30d": 64.2,
    "merchant_category": "electronics",
}

LEGIT = {
    "amount": 24.50, "hour": 13, "day_of_week": 2, "card_present": 1,
    "is_international": 0, "txn_velocity_1h": 1, "amount_velocity_1h": 24.5,
    "distance_from_home_km": 3.2, "failed_attempts_24h": 0,
    "device_changed": 0, "avg_amount_30d": 58.0,
    "merchant_category": "grocery",
}


def post(url: str, path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        url + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode())


def show(name: str, result: dict) -> None:
    print(f"\n=== {name} ===")
    print(f"Fraud probability: {result['fraud_probability']:.4f} "
          f"-> {result['risk'].upper()} risk "
          f"({'FLAGGED' if result['flagged'] else 'not flagged'})")
    print("Top contributing features (log-odds):")
    for c in result["top_features"][:5]:
        print(f"  {c['feature']:24s} = {c['value']!s:>12}  {c['contribution']:+.2f}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    args = parser.parse_args()
    try:
        show("Fraud-like transaction", post(args.url, "/predict", FRAUD_LIKE))
        show("Legitimate transaction", post(args.url, "/predict", LEGIT))
    except Exception as exc:
        print(f"Demo failed: {exc}\nIs the backend running? "
              f"uvicorn backend.app:app --port 8000", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
