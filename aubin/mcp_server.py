"""AUBIN MCP sunucusu (stdio, bağımlılıksız JSON-RPC 2.0) — başka bir yapay zekâ AUBIN'i araç olarak çağırır.

Araçlar:
  decide(state, questions)                 → kalibre olasılıklar (tipli sorular)
  act(observation, actions, command?)      → anlık hamle + güven + gecikme (AubinController)
  loop(task, numeric?)                     → üretici + karar döngüsü (AubinLoop, Gemma üretici)

İstemci yapılandırması (ör. Claude Desktop / Claude Code):
  {"mcpServers": {"aubin": {"command": "python", "args": ["-m", "aubin.mcp_server", "--model", "emrevrg/AUBIN-12B"]}}}
Model ilk araç çağrısında bir kez yüklenir; istekler sırayla işlenir (tek GPU).
"""
from __future__ import annotations
import argparse, json, sys, threading, traceback

PROTOCOL = "2025-06-18"
TOOLS = [
    {"name": "decide", "description": "Typed decision: state + questions (choice / noul / score) -> calibrated probabilities.",
     "inputSchema": {"type": "object", "properties": {"state": {}, "questions": {"type": "object"}}, "required": ["state", "questions"]}},
    {"name": "act", "description": "Real-time control: observation + allowed actions {name: description} (+ optional command) "
                                   "-> one action with confidence and latency. Low confidence falls back to 'fallback' if given.",
     "inputSchema": {"type": "object", "properties": {"observation": {}, "actions": {"type": "object"}, "command": {"type": "string"},
                                                      "min_conf": {"type": "number"}, "fallback": {"type": "string"}},
                     "required": ["observation", "actions"]}},
    {"name": "loop", "description": "Generator + decider loop: candidates are generated, AUBIN picks with calibrated confidence, "
                                    "regenerates when unsure.",
     "inputSchema": {"type": "object", "properties": {"task": {"type": "string"}, "numeric": {"type": "boolean"}}, "required": ["task"]}},
]


class Server:
    def __init__(self, model_spec, think):
        self.spec, self.think, self.m, self.lock = model_spec, think, None, threading.Lock()

    def model(self):
        if self.m is None:
            from .core import Aubin, AubinEnsemble
            specs = [x.rsplit(":", 1) if ":" in x and not x.startswith("hf:") else (x, "1") for x in self.spec.split(",")]
            ms = [(Aubin(r, think_margin=self.think), float(w)) for r, w in specs]
            self.m = ms[0][0] if len(ms) == 1 else AubinEnsemble(ms)
        return self.m

    def call(self, name, args):
        with self.lock:
            m = self.model()
            if name == "decide":
                return m.decide(args["state"], args["questions"])
            if name == "act":
                from .control import AubinController
                ctl = AubinController(m, args["actions"], min_conf=args.get("min_conf", 0.0), fallback=args.get("fallback"))
                return ctl.act(args["observation"], args.get("command"))
            if name == "loop":
                from .loop import AubinLoop, GemmaGenerator, last_number, final_line
                base = m if hasattr(m, "tok") else m.members[0][0]
                lp = AubinLoop(m, GemmaGenerator(base), extract=last_number if args.get("numeric") else final_line, mode="aubin+vote")
                return lp.solve(args["task"])
            raise ValueError(f"bilinmeyen araç: {name}")

    def handle(self, msg):
        mid, meth, p = msg.get("id"), msg.get("method"), msg.get("params") or {}
        if meth == "initialize":
            return {"protocolVersion": p.get("protocolVersion", PROTOCOL), "capabilities": {"tools": {}},
                    "serverInfo": {"name": "aubin", "version": "0.1.0"}}
        if meth == "tools/list":
            return {"tools": TOOLS}
        if meth == "tools/call":
            try:
                out = self.call(p["name"], p.get("arguments") or {})
                return {"content": [{"type": "text", "text": json.dumps(out, ensure_ascii=False)}], "isError": False}
            except Exception as e:
                return {"content": [{"type": "text", "text": f"{type(e).__name__}: {e}"}], "isError": True}
        if meth == "ping":
            return {}
        if mid is None:                               # bildirim (ör. notifications/initialized): cevap yok
            return None
        raise KeyError(meth)


def main():
    ap = argparse.ArgumentParser(prog="aubin-mcp")
    ap.add_argument("--model", default="emrevrg/AUBIN-12B"); ap.add_argument("--think", type=float, default=None)
    a = ap.parse_args()
    S = Server(a.model, a.think)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        try:
            res = S.handle(msg)
            if msg.get("id") is None:
                continue
            reply = {"jsonrpc": "2.0", "id": msg["id"], "result": res}
        except KeyError as e:
            reply = {"jsonrpc": "2.0", "id": msg.get("id"), "error": {"code": -32601, "message": f"method not found: {e}"}}
        except Exception as e:
            print(traceback.format_exc(), file=sys.stderr)
            reply = {"jsonrpc": "2.0", "id": msg.get("id"), "error": {"code": -32603, "message": str(e)}}
        sys.stdout.write(json.dumps(reply, ensure_ascii=False) + "\n"); sys.stdout.flush()


if __name__ == "__main__":
    main()
