"""Production serving with the human gate and the learning loop attached.

  POST /v1/predict   {"inputs": {...}}  -> action/response, confidence, needs_approval, escalate, request_id
  POST /v1/feedback  {"request_id", "final_action", "outcome_success", "slice"?, "episode_id"?}
  GET  /health       GET /v1/model

Every prediction and every feedback event is appended to a JSONL log; `autoresearch ingest-live`
turns that log into episodes (accepted / overridden / failed) for the next training run.
stdlib only. Put it behind your own TLS / auth proxy for anything internet-facing: the built-in
bearer-token check is a baseline, not a security perimeter."""
from __future__ import annotations

import hmac
import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pandas as pd

from .gate import Deployment, GatePolicy

MAX_BODY = 1_000_000


class Service:
    def __init__(self, model, model_id: str, wf, policy: GatePolicy | None = None, log_path: str = "live.jsonl",
                 api_key: str | None = None):
        self.dep, self.model_id, self.wf = Deployment(model, model_id, policy), model_id, wf
        self.log_path, self.api_key = log_path, api_key
        self.pending, self.lock = {}, threading.Lock()

    def _log(self, rec: dict):
        with self.lock, open(self.log_path, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")

    def predict(self, inputs: dict) -> dict:
        missing = [c for c in self.wf.inputs if c not in inputs]
        if missing:
            raise ValueError(f"missing inputs: {missing}")
        df = pd.DataFrame([{c: inputs[c] for c in self.wf.inputs}])
        r = self.dep.recommend(df).iloc[0]
        rid = uuid.uuid4().hex
        out = {"request_id": rid, "action": str(r["action"]), "confidence": round(float(r["confidence"]), 4),
               "needs_approval": bool(r["needs_approval"]), "escalate": bool(r["escalate"]),
               "model": self.model_id}
        with self.lock:
            self.pending[rid] = {"inputs": inputs, "rec": out}
        self._log({"type": "prediction", "ts": time.time(), **out, "inputs": inputs})
        return out

    def feedback(self, body: dict) -> dict:
        rid = body.get("request_id")
        with self.lock:
            p = self.pending.pop(rid, None)
        if p is None:
            raise KeyError("unknown or already-used request_id")
        if "final_action" not in body or "outcome_success" not in body:
            raise ValueError("feedback needs final_action and outcome_success")
        final, ok = body["final_action"], bool(body["outcome_success"])
        status = "overridden" if final != p["rec"]["action"] else ("accepted" if ok else "failed")
        self._log({"type": "episode", "ts": time.time(), "request_id": rid, "episode_id": body.get("episode_id", rid),
                   "inputs": p["inputs"], "model_rec": p["rec"]["action"], "action": final, "outcome_success": ok,
                   "status": status, "slice": body.get("slice"), "model": self.model_id})
        return {"status": status}

    def handler(self):
        svc = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, obj):
                b = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

            def _auth(self) -> bool:
                if not svc.api_key:
                    return True
                tok = self.headers.get("Authorization", "").removeprefix("Bearer ")
                return hmac.compare_digest(tok, svc.api_key)

            def do_GET(self):
                if self.path == "/health":
                    return self._send(200, {"ok": True})
                if not self._auth():
                    return self._send(401, {"error": "unauthorized"})
                if self.path == "/v1/model":
                    return self._send(200, {"model": svc.model_id, "workflow": svc.wf.name, "inputs": svc.wf.inputs})
                self._send(404, {"error": "not found"})

            def do_POST(self):
                if not self._auth():
                    return self._send(401, {"error": "unauthorized"})
                n = int(self.headers.get("Content-Length", 0))
                if n > MAX_BODY:
                    return self._send(413, {"error": "body too large"})
                try:
                    body = json.loads(self.rfile.read(n) or b"{}")
                    if self.path == "/v1/predict":
                        return self._send(200, svc.predict(body.get("inputs", {})))
                    if self.path == "/v1/feedback":
                        return self._send(200, svc.feedback(body))
                    self._send(404, {"error": "not found"})
                except (ValueError, KeyError) as e:
                    self._send(400, {"error": str(e)})
                except Exception as e:
                    self._send(500, {"error": f"{type(e).__name__}"})
        return H

    def make_server(self, host="127.0.0.1", port=8080) -> ThreadingHTTPServer:
        return ThreadingHTTPServer((host, port), self.handler())


def load_live_log(path: str, wf) -> pd.DataFrame:
    """JSONL log -> episode table ready for `ingest`/retraining."""
    rows = []
    for line in open(path):
        r = json.loads(line)
        if r.get("type") != "episode":
            continue
        row = dict(r["inputs"])
        row.update({"episode_id": r["episode_id"], "action": r["action"], "outcome_success": r["outcome_success"],
                    "status": r["status"], "model_rec": r["model_rec"], "source": "live",
                    wf.slice_by: r.get("slice") or "unknown"})
        row.setdefault(wf.group_by, row.get(wf.group_by, "live"))
        rows.append(row)
    return pd.DataFrame(rows)
