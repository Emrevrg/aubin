"""AUBIN çerçeve bağlantıları (ağır bağımlılıklar yalnız çağrılınca yüklenir).

    from aubin.integrations import langchain_tool, llamaindex_selector, crewai_router
    tool = langchain_tool(engine)                 # LangChain / LangGraph aracı: girdi {"state", "questions"} → karar JSON
    sel = llamaindex_selector(engine, choices)    # LlamaIndex: sorguya göre seçeneklerden birini seçer
    route = crewai_router(engine, {"billing": "...", "tech": "..."})   # CrewAI: görevi doğru ajana yönlendirir

`engine`: AubinEngine, Aubin, AubinEnsemble ya da AubinLearning (hepsi .decide sunar); uzak sunucu için RemoteAubin.
"""
from __future__ import annotations

import json
import urllib.request


class RemoteAubin:
    """Çalışan `aubin serve` sunucusuna bağlanır (Python'da model yüklemeden): .decide / .predict / .learn."""

    def __init__(self, base_url="http://127.0.0.1:8009", api_key=None, timeout=60):
        self.base, self.key, self.timeout = base_url.rstrip("/"), api_key, timeout

    def _post(self, path, body):
        h = {"Content-Type": "application/json"}
        if self.key:
            h["Authorization"] = f"Bearer {self.key}"
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(), headers=h)
        return json.loads(urllib.request.urlopen(req, timeout=self.timeout).read())

    def decide(self, state, questions):
        return self._post("/decide", {"state": state, "questions": questions})

    def predict(self, state, questions, min_confidence=None):
        return self._post("/v1/systemone", {"state": state, "questions": questions, "min_confidence": min_confidence})

    def learn(self, state, questions, answers):
        return self._post("/learn", {"state": state, "questions": questions, "answers": answers})


def _predict(engine, state, questions):
    return engine.predict(state, questions) if hasattr(engine, "predict") else {"answers": engine.decide(state, questions)}


def langchain_tool(engine, name="aubin_decide", description=None):
    from langchain_core.tools import StructuredTool

    def run(state: str, questions: dict) -> str:
        return json.dumps(_predict(engine, state, questions), ensure_ascii=False)
    return StructuredTool.from_function(run, name=name, description=description or
                                        "Typed, calibrated decision over a text state: choice / score / yes-no questions.")


def llamaindex_selector(engine, choices):
    """choices: [açıklama, ...] → sorgu için seçilen indeks + güven (LlamaIndex RouterQueryEngine'e uyarlanabilir)."""
    crit = {str(i): c for i, c in enumerate(choices)}

    def select(query):
        r = _predict(engine, query, {"route": {"type": "choice", "instructions": "Which option best handles this query?", "criteria": crit}})
        a = r["answers"]["route"]
        return int(a.get("choice") or 0), a.get("confidence")
    return select


def crewai_router(engine, agents):
    """agents: {ad: görev tanımı} → görev metnini en uygun ajana yönlendiren fonksiyon."""
    def route(task_text):
        r = _predict(engine, task_text, {"agent": {"type": "choice", "instructions": "Which agent should handle this task?", "criteria": agents}})
        return r["answers"]["agent"].get("choice"), r["answers"]["agent"].get("confidence")
    return route
