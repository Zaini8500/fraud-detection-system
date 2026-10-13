"""
Synthetic transaction data generator for the Fraud Detection System.

Produces a highly imbalanced dataset (~0.5% fraud by default) with realistic
features. Fraudulent transactions are generated from different distributions
than legitimate ones, with overlap so the problem is learnable but non-trivial.

Usage:
    python data/generate.py --n 60000 --fraud-rate 0.005 --out data/transactions.csv --seed 42
    python data/generate.py --kaggle   # documented path: use the Kaggle credit-card dataset instead

The --kaggle flag is a documented optional path: if the `kaggle` CLI is
installed and authenticated, it downloads the well-known
"creditcardfraud" dataset (mlg-ulb/creditcardfraud) into data/kaggle/.
It is not required and is not exercised in this environment.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MERCHANT_CATEGORIES = [
    "grocery", "fuel", "restaurant", "travel",
    "electronics", "fashion", "atm", "online",
]

FEATURE_COLUMNS = [
    "amount",
    "hour",
    "day_of_week",
    "card_present",
    "is_international",
    "txn_velocity_1h",
    "amount_velocity_1h",
    "distance_from_home_km",
    "failed_attempts_24h",
    "device_changed",
    "avg_amount_30d",
    "amount_to_avg_ratio",
] + [f"mcc_{c}" for c in MERCHANT_CATEGORIES]

TARGET_COLUMN = "is_fraud"


def _legit_transactions(rng: np.random.Generator, n: int) -> pd.DataFrame:
    """Draw n legitimate transactions."""
    # Amount: log-normal, median ~ $28
    amount = rng.lognormal(mean=3.3, sigma=1.0, size=n)
    # Hour: busy during the day (mixture of two gaussians clipped to 0-23)
    hour = np.clip(
        np.where(
            rng.random(n) < 0.7,
            rng.normal(13, 4, n),
            rng.normal(20, 3, n),
        ),
        0, 23,
    ).astype(int)
    day_of_week = rng.integers(0, 7, n)
    card_present = rng.binomial(1, 0.88, n)
    is_international = rng.binomial(1, 0.03, n)
    txn_velocity_1h = rng.poisson(0.7, n)
    amount_velocity_1h = txn_velocity_1h * rng.lognormal(3.0, 0.8, n)
    distance_from_home_km = np.abs(rng.normal(8, 12, n))
    failed_attempts_24h = rng.poisson(0.15, n)
    device_changed = rng.binomial(1, 0.04, n)
    avg_amount_30d = rng.lognormal(3.4, 0.7, n)

    mcc = rng.choice(
        MERCHANT_CATEGORIES,
        size=n,
        p=[0.28, 0.12, 0.16, 0.04, 0.05, 0.10, 0.08, 0.17],
    )
    return pd.DataFrame(
        {
            "amount": amount,
            "hour": hour,
            "day_of_week": day_of_week,
            "card_present": card_present,
            "is_international": is_international,
            "txn_velocity_1h": txn_velocity_1h,
            "amount_velocity_1h": amount_velocity_1h,
            "distance_from_home_km": distance_from_home_km,
            "failed_attempts_24h": failed_attempts_24h,
            "device_changed": device_changed,
            "avg_amount_30d": avg_amount_30d,
            "merchant_category": mcc,
        }
    )


def _fraud_transactions(rng: np.random.Generator, n: int) -> pd.DataFrame:
    """Draw n fraudulent transactions (different, overlapping distributions)."""
    # Fraudsters go for bigger tickets
    amount = rng.lognormal(mean=5.3, sigma=0.9, size=n)
    # Odd hours
    hour = np.clip(
        np.where(
            rng.random(n) < 0.55,
            rng.normal(2.5, 2.0, n),
            rng.uniform(0, 24, n),
        ),
        0, 23,
    ).astype(int)
    day_of_week = rng.integers(0, 7, n)
    card_present = rng.binomial(1, 0.22, n)
    is_international = rng.binomial(1, 0.34, n)
    # Burst behaviour: many transactions in a short window
    txn_velocity_1h = rng.poisson(4.5, n)
    amount_velocity_1h = txn_velocity_1h * rng.lognormal(4.6, 0.8, n)
    distance_from_home_km = np.abs(rng.normal(320, 260, n))
    failed_attempts_24h = rng.poisson(2.2, n)
    device_changed = rng.binomial(1, 0.52, n)
    # Victims' own baselines are normal; the *ratio* is what spikes
    avg_amount_30d = rng.lognormal(3.4, 0.7, n)

    mcc = rng.choice(
        MERCHANT_CATEGORIES,
        size=n,
        p=[0.05, 0.03, 0.06, 0.14, 0.22, 0.12, 0.10, 0.28],
    )
    return pd.DataFrame(
        {
            "amount": amount,
            "hour": hour,
            "day_of_week": day_of_week,
            "card_present": card_present,
            "is_international": is_international,
            "txn_velocity_1h": txn_velocity_1h,
            "amount_velocity_1h": amount_velocity_1h,
            "distance_from_home_km": distance_from_home_km,
            "failed_attempts_24h": failed_attempts_24h,
            "device_changed": device_changed,
            "avg_amount_30d": avg_amount_30d,
            "merchant_category": mcc,
        }
    )


def generate(n: int, fraud_rate: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_fraud = max(1, int(round(n * fraud_rate)))
    n_legit = n - n_fraud

    legit = _legit_transactions(rng, n_legit)
    fraud = _fraud_transactions(rng, n_fraud)
    legit[TARGET_COLUMN] = 0
    fraud[TARGET_COLUMN] = 1

    df = pd.concat([legit, fraud], ignore_index=True)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Derived feature: how unusual is this amount for the customer?
    df["amount_to_avg_ratio"] = df["amount"] / (df["avg_amount_30d"] + 1.0)

    # One-hot encode the merchant category (kept explicit for serving parity)
    for cat in MERCHANT_CATEGORIES:
        df[f"mcc_{cat}"] = (df["merchant_category"] == cat).astype(int)
    df = df.drop(columns=["merchant_category"])

    # Round for a tidy CSV
    df["amount"] = df["amount"].round(2)
    df["amount_velocity_1h"] = df["amount_velocity_1h"].round(2)
    df["distance_from_home_km"] = df["distance_from_home_km"].round(1)
    df["avg_amount_30d"] = df["avg_amount_30d"].round(2)
    df["amount_to_avg_ratio"] = df["amount_to_avg_ratio"].round(3)

    return df[FEATURE_COLUMNS + [TARGET_COLUMN]]


def _kaggle_path(out_dir: Path) -> None:
    """Optional: download the real Kaggle credit-card fraud dataset.

    Requires the `kaggle` CLI authenticated (kaggle.json). Kept as a
    documented optional path only.
    """
    print("Downloading mlg-ulb/creditcardfraud via the kaggle CLI ...")
    import subprocess

    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", "mlg-ulb/creditcardfraud",
         "-p", str(out_dir), "--unzip"],
        check=True,
    )
    print(f"Done. Files are in {out_dir}. Note: its features are PCA-anonymised "
          f"(V1..V28); adapt models/train.py column mapping to use it.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic transaction data.")
    parser.add_argument("--n", type=int, default=60000, help="Total transactions.")
    parser.add_argument("--fraud-rate", type=float, default=0.005, help="Fraction of fraud.")
    parser.add_argument("--out", type=str, default="data/transactions.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--kaggle", action="store_true",
                        help="Download the Kaggle credit-card dataset instead.")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]

    if args.kaggle:
        _kaggle_path(repo_root / "data" / "kaggle")
        return 0

    df = generate(args.n, args.fraud_rate, args.seed)
    out = repo_root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    fraud_share = df[TARGET_COLUMN].mean()
    print(f"Wrote {len(df):,} rows -> {out}")
    print(f"Fraud rate: {fraud_share:.4%} ({int(df[TARGET_COLUMN].sum())} fraud cases)")
    print(f"Features: {len(FEATURE_COLUMNS)} ({', '.join(FEATURE_COLUMNS[:6])} ...)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
