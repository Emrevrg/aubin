"""Bağımlılıksız HTTP sunucu — her yere tak-çalıştır entegrasyon.

  POST /decide                {"state": ..., "questions": {...}}            → kalibre olasılıklar (+ bellek/güven bilgisi)
  POST /learn                 {"state": ..., "questions": {...}, "answers": {qid: anahtar}}  → anında öğrenme (ms)
  POST /v1/chat/completions   OpenAI uyumlu: son kullanıcı mesajı JSON {"state","questions"} → yanıt içeriği JSON karar
  GET  /health
Model AubinLearning ile sarılıysa /learn etkin; değilse /decide düz modelle çalışır.
"""
import json, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading


def serve(model, port=8009):
    lock = threading.Lock()

    class H(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            data = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)

        def do_GET(self):
            if self.path.rstrip("/") in ("/health", ""):
                self._send(200, {"ok": True, "model": type(model).__name__, "learning": hasattr(model, "learn")})
            else:
                self.send_error(404)

        def do_POST(self):
            path = self.path.rstrip("/")
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                with lock:                            # tek GPU: istekler sırayla
                    if path == "/decide":
                        out = model.decide(body["state"], body["questions"])
                    elif path in ("/v1/systemone", "/v1/systemone/batch", "/predict"):
                        # Laya/Jev uyumlu: {"state", "questions"} (ya da batch: {"items": [...]}) → {"answers", "routing"}
                        from .engine import AubinEngine
                        eng = model if isinstance(model, AubinEngine) else AubinEngine([("aubin", model, 0.0)])
                        if path.endswith("/batch"):
                            out = {"results": [eng.predict(it["state"], it["questions"], it.get("min_confidence")) for it in body["items"]]}
                        else:
                            out = eng.predict(body["state"], body["questions"], body.get("min_confidence"))
                    elif path == "/learn":
                        if not hasattr(model, "learn"):
                            return self._send(400, {"error": "model AubinLearning ile sarılı değil"})
                        out = {"learned_ms": model.learn(body["state"], body["questions"], body["answers"])}
                    elif path == "/v1/chat/completions":
                        msg = [m for m in body.get("messages", []) if m.get("role") == "user"][-1]["content"]
                        req = json.loads(msg) if isinstance(msg, str) else msg
                        dec = model.decide(req["state"], req["questions"])
                        out = {"id": f"aubin-{int(time.time() * 1000)}", "object": "chat.completion", "created": int(time.time()),
                               "model": body.get("model", "aubin"),
                               "choices": [{"index": 0, "finish_reason": "stop",
                                            "message": {"role": "assistant", "content": json.dumps(dec, ensure_ascii=False)}}]}
                    else:
                        return self.send_error(404)
                self._send(200, out)
            except Exception as e:
                self._send(400, {"error": str(e)})

        def log_message(self, *a):
            pass

    print(f"AUBIN hazır: http://127.0.0.1:{port}  (/decide, /learn, /v1/systemone[/batch] Laya/Jev-uyumlu, /v1/chat/completions, /health)", flush=True)
    ThreadingHTTPServer(("0.0.0.0", port), H).serve_forever()
