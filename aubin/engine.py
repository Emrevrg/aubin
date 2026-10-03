"""AUBIN Engine — katmanlı Sistem-1 / Sistem-2 karar motoru.

Laya tek katmanlıdır (her istek küçük bir kodlayıcıya). AUBIN Engine isteği EN UCUZ yeterli katmanda bitirir:

  K0  öğrenilmiş beceriler + bellek (AubinLearning: acquire_skill / learn) — milisaniye, yalnız kanıtlanmış soru tiplerinde
  K1  hızlı model (ör. AUBIN-E4B, ~0.25 s/karar)
  K2  güçlü model (ör. AUBIN-12B / 31B; isteğe bağlı akıl yürütme geçişi)

Her katmanın kalibre güveni o katmanın eşiğini geçerse karar orada verilir; geçmezse bir üst katmana çıkılır. Son katmanda
da eşik tutmazsa ve `abstain=True` ise cevap "çekimser" (null) döner — yanlış ama emin cevap yerine "emin değilim".
Yanıt biçimi Laya/Jev istemcileriyle uyumludur: {"answers": {qid: {"choice"|"score"|"noul", "confidence", ...}}, "routing": {...}}.

    eng = AubinEngine([("fast", Aubin("emrevrg/AUBIN-E4B-Control"), 0.85), ("strong", Aubin("emrevrg/AUBIN-12B"), 0.0)])
    eng.predict(state, questions)
"""
from __future__ import annotations

import time

from .core import options


def _answer_field(q, key, probs):
    """Laya/Jev alan adları: choice → "choice", score → "score" (seviye indeksi), noul → "noul" (evet olasılığı)."""
    t = q.get("type", "choice")
    if t == "noul":
        return {"noul": round(float(probs.get("true", 0.0)), 4)}
    if t == "score":
        keys = list(probs)
        return {"score": keys.index(key) if key in keys else None, "label": key}
    return {"choice": key}


class AubinEngine:
    def __init__(self, tiers, abstain=False, learner=None):
        """tiers: [(ad, model, eşik)] ucuzdan pahalıya; model .decide(state, questions) sunar (Aubin, AubinEnsemble,
        AubinLearning). eşik: o katmanda kararı kabul etmek için gereken kalibre güven (son katmanda genelde 0).
        learner: isteğe bağlı AubinLearning — K0 olarak en başta denenir (öğrenilmiş beceri/bellek)."""
        self.tiers = list(tiers)
        if learner is not None:
            self.tiers.insert(0, ("learned", learner, getattr(learner, "engine_threshold", 0.9)))
        self.abstain = abstain
        self.stats = {name: 0 for name, _, _ in self.tiers}
        self.stats["abstained"] = 0

    def predict(self, state, questions, min_confidence=None):
        t0 = time.perf_counter()
        pending, answers, route, tier_ms = dict(questions), {}, {}, {}
        for li, (name, model, thr) in enumerate(self.tiers):
            if not pending:
                break
            last = li == len(self.tiers) - 1
            thr = max(thr, min_confidence or 0.0) if not last or self.abstain else thr
            t1 = time.perf_counter()
            res = model.decide(state, pending)
            tier_ms[name] = round((time.perf_counter() - t1) * 1e3, 1)
            for qid, r in res.items():
                conf = float(r["confidence"])
                if conf >= thr or (last and not self.abstain):
                    answers[qid] = {**_answer_field(questions[qid], r["answer"], r["probabilities"]),
                                    "confidence": round(conf, 4), "probabilities": r["probabilities"], "tier": name}
                    route[qid] = name; self.stats[name] += 1
                    pending.pop(qid)
            if last:
                for qid in list(pending):            # son katmanda da eşik tutmadı → çekimser
                    keys, _ = options(questions[qid])
                    answers[qid] = {"choice": None, "abstained": True, "tier": name, "confidence": None}
                    route[qid] = "abstained"; self.stats["abstained"] += 1
                    pending.pop(qid)
        return {"answers": answers, "routing": {"per_question": route, "tier_ms": tier_ms,
                                                 "total_ms": round((time.perf_counter() - t0) * 1e3, 1)}}

    def learn(self, state, questions, answers):
        """Geri bildirim K0'a (varsa) gider: bir sonraki benzer istek pahalı katmana hiç çıkmadan çözülebilir."""
        for name, model, _ in self.tiers:
            if hasattr(model, "learn"):
                return model.learn(state, questions, answers)
        return None
