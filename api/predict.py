"""
Vercel Python serverless function: POST /api/predict

Scores a single transaction. Request body (JSON) uses raw transaction fields,
e.g.:
    {"amount": 1849.99, "hour": 3, "card_present": 0, "merchant_category": "electronics", ...}

Response:
    {"fraud_probability": 0.93, "fraud_score": 93.0, "risk": "high",
     "threshold": 0.42, "flagged": true, "top_features": [...]}
"""

from http.server import BaseHTTPRequestHandler
import json

try:
    from _model import score_transaction
except ImportError:  # local development
    from api._model import score_transaction


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
            txn = json.loads(raw.decode("utf-8") or "{}")
            if not isinstance(txn, dict):
                raise ValueError("Request body must be a JSON object.")
            self._send_json(score_transaction(txn))
        except Exception as exc:  # never leak a stack trace to clients
            self._send_json({"error": f"Prediction failed: {exc}"}, status=400)

    def do_GET(self) -> None:
        self._send_json(
            {"message": "POST a transaction JSON to this endpoint to score it."}
        )
