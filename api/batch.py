"""
Vercel Python serverless function: POST /api/batch

Scores many transactions at once. To stay dependency-free (no pandas /
multipart parsing in serverless), the request is JSON:

    {"csv": "amount,hour,...,merchant_category\\n12.5,13,...,grocery\\n..."}

The CSV may use raw transaction columns (same names as /api/predict fields)
or the one-hot encoded model features. Response:

    {"count": 2, "flagged": 1, "results": [
        {"row": 1, "fraud_probability": 0.02, "risk": "low", "flagged": false,
         "top_features": [...]}, ...]}
"""

from http.server import BaseHTTPRequestHandler
import csv
import io
import json

import numpy as np

try:
    from _model import get_predictor, get_metrics, transaction_to_vector, risk_label
except ImportError:  # local development
    from api._model import get_predictor, get_metrics, transaction_to_vector, risk_label


def _rows_from_csv(text: str) -> list[dict]:
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise ValueError("Empty CSV or missing header row.")
    return [dict(r) for r in reader]


class handler(BaseHTTPRequestHandler):
    def _send_json(self, payload: dict, status: int = 200) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", 0) or 0)
            raw = self.rfile.read(length) if length else b"{}"
            body = json.loads(raw.decode("utf-8") or "{}")
            csv_text = body.get("csv", "")
            if not csv_text or not csv_text.strip():
                raise ValueError('Provide {"csv": "<csv text with header row>"}')
            rows = _rows_from_csv(csv_text)
            if len(rows) > 5000:
                raise ValueError("Batch limit is 5000 rows per request.")

            predictor = get_predictor()
            threshold = float(get_metrics()["final"]["threshold"])
            matrix = np.stack([transaction_to_vector(r) for r in rows])
            probas = predictor.predict_proba(matrix)

            results = []
            for i, (row, p) in enumerate(zip(rows, probas)):
                p = float(p)
                results.append(
                    {
                        "row": i + 1,
                        "fraud_probability": p,
                        "fraud_score": round(p * 100, 2),
                        "risk": risk_label(p),
                        "flagged": p >= threshold,
                        "input": {k: row.get(k) for k in
                                  ("amount", "hour", "merchant_category",
                                   "card_present", "is_international")},
                    }
                )
            flagged = sum(1 for r in results if r["flagged"])
            self._send_json(
                {"count": len(results), "flagged": flagged,
                 "threshold": threshold, "results": results}
            )
        except Exception as exc:
            self._send_json({"error": f"Batch scoring failed: {exc}"}, status=400)

    def do_GET(self) -> None:
        self._send_json(
            {"message": 'POST {"csv": "<csv text>"} to score a batch of transactions.'}
        )
