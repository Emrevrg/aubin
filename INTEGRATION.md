# Use AUBIN anywhere

AUBIN answers typed decisions — yes/no, multiple choice, scores — with calibrated probabilities, and **AUBIN-Learn** lets it
learn from your feedback instantly (no retraining). Pick the integration you need:

## 1. Python (two lines)
```python
from aubin import Aubin, AubinLearning
agent = AubinLearning(Aubin("emrevrg/AUBIN-12B"))         # memory + self-calibration on top of AUBIN
out = agent.decide(state, {"intent": {"type": "choice", "instructions": "Which banking intent?",
                                       "criteria": {"refund": "...", "card_lost": "...", "transfer": "..."}}})
agent.learn(state, questions, {"intent": "refund"})        # correct answer -> learned immediately
agent.save("memory.npz")                                   # persistent experience
```

## 2. HTTP server (no extra dependencies)
```bash
aubin serve --model emrevrg/AUBIN-12B --port 8009 --learn memory.npz
```
* `POST /decide` — `{"state": ..., "questions": {...}}` → `{qid: {answer, confidence, probabilities, memory}}`
* `POST /learn` — `{"state", "questions", "answers": {qid: key}}` → `{"learned_ms": ...}`
* `GET /health`

## 3. OpenAI-compatible endpoint (drop-in for existing clients)
```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8009/v1", api_key="local")
r = client.chat.completions.create(model="aubin", messages=[{"role": "user",
        "content": json.dumps({"state": state, "questions": questions})}])
decision = json.loads(r.choices[0].message.content)
```

## 4. MCP (Claude, IDE agents, other AIs)
`aubin/mcp_server.py` exposes AUBIN as an MCP tool so any MCP-capable assistant can ask it for a calibrated decision.

## 5. Agent loop / LLM pairing
`aubin loop "task" --generator openai:gpt-4o-mini` — a generator proposes candidates, AUBIN picks and asks for more when unsure.

Everything runs locally on your own GPU (one T4 is enough for AUBIN-12B in 4-bit): open weights, private data, zero per-call cost.

## AUBIN Engine: tiered fast/strong decisions, Laya/Jev-compatible

Each request is answered at the cheapest tier whose calibrated confidence clears that tier's threshold. Tiers are tried in this order:
1. Learned skills and memory: milliseconds.
2. Fast model, for example AUBIN-E4B.
3. Strong model, for example AUBIN-12B or 31B, which can reason when needed.

With `abstain=True` the engine returns "not sure" when no tier is confident, instead of a confident wrong answer.

``python
from aubin import Aubin, AubinEngine
eng = AubinEngine([("fast", Aubin("emrevrg/AUBIN-E4B-Control"), 0.85), ("strong", Aubin("emrevrg/AUBIN-12B"), 0.0)])
eng.predict(state, questions)   # {"answers": {qid: {"choice"|"score"|"noul", "confidence", "tier"}}, "routing": {...}}
``

The HTTP server also accepts Laya/Jev-style requests, so an existing client can switch by changing only the base URL:
- `POST /v1/systemone` takes `{"state", "questions", "min_confidence"?}`.
- `POST /v1/systemone/batch` takes `{"items": [...]}`.

## Frameworks, TypeScript, Docker
- **Remote client:** `from aubin.integrations import RemoteAubin`. It talks to a running `aubin serve` without loading a model.
- **LangChain / LangGraph:** `langchain_tool(engine)`
- **LlamaIndex:** `llamaindex_selector(engine, choices)`
- **CrewAI:** `crewai_router(engine, {agent: role})`
- **TypeScript, Node.js and browser:** `code/aubin-ts/index.ts`, the `Aubin` class with `predict`, `batch` and `learn`.
- **Docker:** `code/Dockerfile`. Run it with `docker run --gpus all -p 8009:8009 -e AUBIN_MODEL=emrevrg/AUBIN-12B aubin`.

## AUBIN Omni: one model, every ability

The base model is loaded **once**. Ability adapters are attached on top of it and switched per request in milliseconds.

``python
from aubin import AubinOmni
omni = AubinOmni.preset("E4B")          # also "12B", "31B"; see omni.abilities()
omni.decide(state, questions)           # typed, calibrated decisions
omni.click(screenshot, "open settings") # click point (x, y) on a 0-1000 scale
omni.web_step(task, {"n1": "<input placeholder='From'>", ...})   # web agent step
omni.act(observation, "go to the red key", actions, allowed=safe_moves)   # real-time control / games
omni.enable_learning()                  # AubinLearning: learn / acquire_skill on top
``

**Smoke test on a free T4, E4B in 4-bit, one model with all four adapters** (`code/omni_smoke.py`):
- **decide:** billing at 0.91, churn yes at 0.87.
- **web_step:** picked the "From" input with TYPE, at 0.99.
- **click:** 70% correct on 30 ScreenSpot samples, about 2.6 s per click.
- **act:** the safety shield redirected a move into lava to a safe move.
- **Switching back to decide:** answered "technical" correctly.

Abilities available per size today:

| size | abilities |
|---|---|
| E4B | decide, web, screen, control |
| 12B | decide, web, control (screen training running) |
| 31B | decide (web training running) |
