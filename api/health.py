"""Vercel Python serverless function: GET /api/health -> service status."""

from http.server import BaseHTTPRequestHandler
import json

try:
    from _model import get_predictor, get_feature_names
except ImportError:  # local development
    from api._model import get_predictor, get_feature_names


class handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        try:
            predictor = get_predictor()
            payload = {
                "status": "ok",
                "model": "xgboost-compact",
                "n_trees": predictor.n_trees,
                "n_features": len(get_feature_names()),
            }
            status = 200
        except Exception as exc:
            payload = {"status": "error", "detail": str(exc)}
            status = 500
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
