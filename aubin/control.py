"""AUBIN Control — anlık hamle seçimi (oyun / arayüz / robot): gözlem + (isteğe bağlı) komut + olası hamleler → hamle.

Tek ileri geçiş (düşünme kapalı), kalibre güven. Güven eşiğin altındaysa güvenli yedek hamle döner — hata payı buradan
düşer: model emin değilken rastgele davranmaz. Her çağrı gecikmeyi (ms) döndürür.

    from aubin import Aubin
    from aubin.control import AubinController
    ctl = AubinController(Aubin("emrevrg/AUBIN-12B"), actions={"up": "move up", "down": "move down",
                                                               "left": "move left", "right": "move right"},
                          min_conf=0.5, fallback="wait")
    ctl.act(observation="#####\n#A..K#\n#####", command="go to the key K")
    → {"action": "right", "confidence": 0.97, "fallback_used": False, "latency_ms": 180.3, "probabilities": {...}}
"""
from __future__ import annotations
import time


class AubinController:
    def __init__(self, model, actions, min_conf=0.0, fallback=None, realtime=True, instructions=None):
        """actions: {ad: açıklama}; fallback: güven düşükse dönecek hamle adı (ör. "wait"/"noop"; None = en olası hamle);
        realtime: düşünme kapalı (her hamle tek ileri geçiş)."""
        self.m, self.actions, self.min_conf, self.fallback = model, dict(actions), float(min_conf), fallback
        self.instructions = instructions or "Choose the single best next action for the current observation."
        if realtime:
            for mem in getattr(model, "members", None) and [x for x, _ in model.members] or [model]:
                if hasattr(mem, "think_margin"):
                    mem.think_margin = 0.0
        self.history = []

    def reset(self):
        """Yeni bölüm/görev: hamle geçmişi temizlenir (eğitimdeki biçimle aynı — geçmiş bölüm içindir)."""
        self.history = []

    def act(self, observation, command=None, actions=None, context=None, allowed=None):
        """allowed: güvenlik kalkanı — izinli hamle kümesi (ör. lav/duvara gitmeyenler). Modelin seçimi izinli değilse
        izinliler içinde en olası hamle seçilir (fallback_used=True)."""
        acts = dict(actions or self.actions)
        state = {"observation": observation}
        if context:
            state["context"] = context
        if self.history:
            state["recent_actions"] = [h["action"] for h in self.history[-5:]]
        q = {"type": "choice", "criteria": acts,
             "instructions": (f"Command: {command}. " if command else "") + self.instructions}
        t0 = time.perf_counter()
        res = self.m.decide(state, {"action": q})["action"]
        ms = (time.perf_counter() - t0) * 1000
        act, conf = res["answer"], float(res["confidence"])
        used = False
        if conf < self.min_conf and self.fallback is not None:
            act, used = self.fallback, True
        if allowed is not None and act not in allowed and any(a in allowed for a in res["probabilities"]):
            act = max((a for a in res["probabilities"] if a in allowed), key=lambda a: res["probabilities"][a])
            conf, used = float(res["probabilities"][act]), True
        out = {"action": act, "confidence": round(conf, 4), "fallback_used": used, "latency_ms": round(ms, 1),
               "probabilities": res["probabilities"]}
        self.history.append(out)
        return out
