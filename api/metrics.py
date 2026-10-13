"""Vercel Python serverless function: GET /api/metrics -> model report."""

from http.server import BaseHTTPRequestHandler
import json

try:
    from _model import get_metrics
except ImportError:  # local development
    from api._model import get_metrics


class handler(BaseHTTPRequestHandler):
    def _send_json(self, payload: dict, status: int = 200) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        try:
            self._send_json(get_metrics())
        except Exception as exc:
            self._send_json({"error": f"Could not load metrics: {exc}"}, status=500)
