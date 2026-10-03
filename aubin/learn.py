"""AUBIN-Learn — Norovox öz-öğrenme çekirdeğinin karar modeline entegrasyonu.

Norovox Alpha Mini ilkesi: bilgi ağırlıkta değil, DIŞ BELLEKTE tutulur; model düşünür, bellek hatırlar.
Burada bu ilke AUBIN'in hızlı karar sürecine taşınır:

  learn(...)   doğrulanmış bir vakayı belleğe yazar — gradyan yok, eğitim yok, ~milisaniye; bir sonraki soruda etkili.
  recall(...)  aynı soru tipindeki (kaynak + seçenek kümesi) en benzer çözülmüş vakalardan seçeneklere oy dağılımı.
  fuse(...)    modelin log-olasılıkları + bellek oyları (log-doğrusal); ağırlık kaynak-başına, YALNIZ dev'de ayarlanır.
  AubinLearner modeli sarar: emin değilse belleğe bakar; bellekte karşılığı yoksa araştırma kancası (Norovox
               Retrieval: yerel bilgi tabanı → web) kanıt getirir; geri bildirim gelince learn() ile eksiğini kapatır.

Gömme (embedding) takılabilir: varsayılan sözcük+karakter n-gram karma vektörü (bağımlılıksız, CPU'da µs-ms),
istenirse cümle gömme modeli (ör. BAAI/bge-small-en-v1.5) `embed_fn` olarak verilir.
"""
from __future__ import annotations

import hashlib
import math
import re
import time
from collections import defaultdict

import numpy as np

DIM = int(__import__("os").environ.get("AUBIN_HASH_DIM", 2 ** 13))


def hash_embed(texts, dim: int = DIM) -> np.ndarray:
    """Sözcük (1-2 gram) + karakter 4-gram karma (hashing) vektörü, L2-normalize. Model indirmeden çalışır."""
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for r, t in enumerate(texts):
        t = re.sub(r"\s+", " ", str(t).lower())
        ws = re.findall(r"[a-z0-9çğıöşü']+", t)
        feats = ws + [a + "_" + b for a, b in zip(ws, ws[1:])] + [t[i:i + 4] for i in range(0, max(0, len(t) - 3), 2)]
        for f in feats:
            h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "little")
            out[r, h % dim] += 1.0 if (h >> 63) & 1 else -1.0
        n = np.linalg.norm(out[r])
        if n:
            out[r] /= n
    return out


def sig(src, keys) -> str:
    """Bellek bölümü. ≤26 seçenek: kaynak + seçenek kümesinin tamamı. Geniş sorularda (banking77: 77–78 seçenek, testte
    seçenek kümesi değişebiliyor) yalnız kaynak — oylar etiket ADIYLA mevcut seçeneklere eşlenir."""
    if len(keys) > 26:
        return f"{src}|wide"
    return f"{src}|{len(keys)}|" + hashlib.md5("\x1f".join(map(str, keys)).encode()).hexdigest()[:10]


class DecisionMemory:
    """Soru tipine göre bölümlenmiş vaka belleği. Ekleme O(1); arama bölüm-içi kosinüs (matris çarpımı)."""

    def __init__(self, embed_fn=hash_embed):
        self.embed_fn = embed_fn
        self.vecs: dict[str, list] = defaultdict(list)      # imza → vektör listesi (yeni eklenenler)
        self.mats: dict[str, np.ndarray] = {}               # imza → donmuş matris (toplu arama için)
        self.labels: dict[str, list] = defaultdict(list)    # imza → etiket ADI listesi (seçenek kümesinden bağımsız)
        self.n = 0

    def _freeze(self, s):
        if self.vecs[s]:
            new = np.stack(self.vecs[s])
            self.mats[s] = new if s not in self.mats else np.concatenate([self.mats[s], new])
            self.vecs[s] = []

    def learn(self, text, src, keys, y, vec=None):
        """Tek vaka öğren (anında etkili). Dönüş: geçen süre (ms)."""
        t0 = time.perf_counter()
        s = sig(src, keys)
        v = vec if vec is not None else self.embed_fn([text])[0]
        self.vecs[s].append(v.astype(np.float32)); self.labels[s].append(str(keys[int(y)])); self.n += 1
        return (time.perf_counter() - t0) * 1e3

    def learn_many(self, texts, srcs, keyss, ys, batch=2048):
        for i in range(0, len(texts), batch):
            self.learn_vectors(self.embed_fn(texts[i:i + batch]), srcs[i:i + batch], keyss[i:i + batch], ys[i:i + batch])

    def learn_vectors(self, V, srcs, keyss, ys):
        """Önceden gömülmüş vakaları toplu öğren (gömme bir kez hesaplanır, farklı bellekler için yeniden kullanılır)."""
        for v, sr, ks, y in zip(V, srcs, keyss, ys):
            s = sig(sr, ks); self.vecs[s].append(np.asarray(v, dtype=np.float32)); self.labels[s].append(str(ks[int(y)])); self.n += 1
        for s in list(self.vecs):
            self._freeze(s)

    def recall(self, text, src, keys, k=16, tau=0.05, vec=None, min_sim=0.0):
        """Seçenekler üzerinde bellek dağılımı (benzerlik-ağırlıklı oy) + en yüksek benzerlik. Bellek boşsa (None, 0).
        min_sim: bu benzerliğin altındaki komşular oy vermez (bellek küçükken alakasız tek vakanın her soruya oy vermesini önler)."""
        s = sig(src, keys)
        self._freeze(s)
        M = self.mats.get(s)
        if M is None or not len(M):
            return None, 0.0
        q = vec if vec is not None else self.embed_fn([text])[0]
        sims = M @ q
        kk = min(k, len(sims))
        top = np.argpartition(-sims, kk - 1)[:kk]
        top = top[sims[top] >= min_sim]
        if not len(top):
            return None, float(sims.max())
        w = np.exp((sims[top] - sims[top].max()) / tau)
        p = vote([self.labels[s][i] for i in top], w, keys)
        return p, (float(sims[top].max()) if p is not None else 0.0)


def vote(labels, weights, keys):
    """Komşu etiket adlarını mevcut seçeneklere eşle (seçenek kümesi değişse de çalışır). Eşleşen yoksa None."""
    pos = {str(k): i for i, k in enumerate(keys)}
    p = np.zeros(len(keys), dtype=np.float64)
    for lab, wi in zip(labels, weights):
        j = pos.get(lab)
        if j is not None:
            p[j] += wi
    return p / p.sum() if p.sum() > 0 else None


def fuse(lp_model, p_mem, w, T=1.0, eps=1e-3):
    """Log-doğrusal birleştirme: log p_model/T + w·log(p_mem + ε). p_mem yoksa model aynen kalır."""
    lp = np.asarray(lp_model, dtype=np.float64) / T
    if p_mem is None or w == 0:
        return lp - np.logaddexp.reduce(lp)
    z = lp + w * np.log(np.asarray(p_mem) + eps)
    return z - np.logaddexp.reduce(z)


class SelfCalibrator:
    """Öz-kalibrasyon (Hedge / çarpımsal ağırlıklar): her kaynak için uzmanlar — model, bellek, birleşik — arasında güveni
    geri bildirimle kendisi ayarlar. Kayıp = log-kayıp; π_e ← π_e·exp(−η·kayıp_e). Pişmanlık sınırı: zamanla en iyi uzmana yaklaşır,
    yani bellek bir kaynakta işe yaramıyorsa ona olan güven kendiliğinden söner (kendi eksiğini kendisi kapatır)."""

    EXPERTS = ("model", "memory", "fused")

    def __init__(self, eta=1.0, prior=(0.6, 0.1, 0.3)):
        self.eta, self.prior = eta, np.log(np.asarray(prior, dtype=np.float64))
        self.logw = defaultdict(lambda: self.prior.copy())

    @staticmethod
    def experts(lp_model, p_mem, w_fuse=1.0):
        pm = np.exp(np.asarray(lp_model, dtype=np.float64) - np.logaddexp.reduce(lp_model))
        pk = pm if p_mem is None else 0.98 * np.asarray(p_mem) + 0.02 / len(pm)
        pf = np.exp(fuse(lp_model, p_mem, w_fuse))
        return np.stack([pm, pk, pf])

    def predict(self, src, E):
        pi = np.exp(self.logw[src] - np.logaddexp.reduce(self.logw[src]))
        return pi @ E, pi

    def update(self, src, E, y):
        self.logw[src] = self.logw[src] - self.eta * (-np.log(E[:, y] + 1e-12))


class Probe:
    """Bağımlılıksız çok sınıflı lojistik regresyon (numpy, L2, Adam, tam yığın) — beceri sınıflandırıcısı."""

    def __init__(self, C=10.0, iters=300, lr=0.05):
        self.C, self.iters, self.lr = C, iters, lr

    def fit(self, X, y, classes):
        self.classes = list(classes); idx = {c: i for i, c in enumerate(self.classes)}
        X = np.asarray(X, dtype=np.float32); n, d = X.shape; k = len(self.classes)
        Y = np.zeros((n, k), dtype=np.float32); Y[np.arange(n), [idx[c] for c in y]] = 1
        W, b = np.zeros((d, k), np.float32), np.zeros(k, np.float32)
        st = [np.zeros_like(W), np.zeros_like(W), np.zeros_like(b), np.zeros_like(b)]
        lam = 1.0 / (self.C * n)
        for t in range(1, self.iters + 1):
            Z = X @ W + b; Z -= Z.max(1, keepdims=True); P = np.exp(Z); P /= P.sum(1, keepdims=True)
            G = (P - Y) / n
            for g, m, v, p in ((X.T @ G + lam * W, st[0], st[1], W), (G.sum(0), st[2], st[3], b)):
                m *= 0.9; m += 0.1 * g; v *= 0.999; v += 0.001 * g * g
                p -= self.lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
        self.W, self.b = W, b
        return self

    def proba(self, x, keys):
        z = np.asarray(x, np.float32) @ self.W + self.b; z = np.exp(z - z.max()); z /= z.sum()
        pos = {c: j for j, c in enumerate(self.classes)}
        p = np.array([z[pos[str(k)]] if str(k) in pos else 0.0 for k in keys], dtype=np.float64)
        return p / p.sum() if p.sum() > 0 else None


class SkillLibrary:
    """Kendine yetenek ekleme: soru tipi (qid) başına veriden öğrenilen beceri + KENDİNİ SINAMA.

    acquire(...) örneklerin bir kısmını ayırır, beceriyi kalanla öğrenir, ayrılan kısımda model-tek vs model+beceri karşılaştırır;
    beceri yalnız doğruluk ≥ min_gain soru artar VE log-kayıp düşerse AÇILIR (bozmadan deneyerek öğrenme). Rapor döner."""

    WGRID = (0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)

    def __init__(self, embed_fn=hash_embed):
        self.embed, self.skills = embed_fn, {}            # qid -> {"probe", "w", "report"}

    def probs(self, qid, text, keys):
        s = self.skills.get(qid)
        if not s or s["w"] <= 0:
            return None, 0.0
        return s["probe"].proba(self.embed([text])[0], keys), s["w"]

    def acquire(self, qid, texts, answers, keys_list, model_lp, holdout=0.25, min_gain=2, T=1.0, seed=0):
        """texts/answers/keys_list: örnekler; model_lp(i) → i. örnek için modelin log-olasılıkları (yalnız ayrılan kısımda çağrılır)."""
        n = len(texts); rng = np.random.default_rng(seed); order = rng.permutation(n)
        h = max(10, int(n * holdout)); te, tr = order[:h], order[h:]
        X = self.embed([texts[i] for i in range(n)])
        classes = sorted({str(a) for a in answers})
        t0 = time.perf_counter()
        probe = Probe().fit(X[tr], [str(answers[i]) for i in tr], classes)
        fit_ms = (time.perf_counter() - t0) * 1e3
        rows = []
        for i in te:
            keys = [str(k) for k in keys_list[i]]
            if str(answers[i]) not in keys:
                continue
            rows.append((np.asarray(model_lp(i), dtype=np.float64), probe.proba(X[i], keys), keys.index(str(answers[i]))))

        def ev(w):
            nl = ac = 0
            for lp, p, y in rows:
                f = fuse(lp, p, w, T); nl -= f[y]; ac += int(np.argmax(f) == y)
            return nl / max(1, len(rows)), ac
        (n0, a0) = ev(0.0)
        w = min(self.WGRID, key=lambda w: ev(w)[0]); (n1, a1) = ev(w)
        on = a1 >= a0 + min_gain and n1 < n0
        rep = {"qid": qid, "train": len(tr), "self_test": len(rows), "fit_ms": round(fit_ms, 1),
               "self_test_acc": [round(a0 / max(1, len(rows)), 4), round(a1 / max(1, len(rows)), 4)],
               "self_test_nll": [round(float(n0), 4), round(float(n1), 4)], "weight": float(w) if on else 0.0, "enabled": bool(on)}
        if on:                                            # kanıtlandı → tüm veriyle yeniden öğren, aç
            probe = Probe().fit(X, [str(a) for a in answers], classes)
        self.skills[qid] = {"probe": probe, "w": w if on else 0.0, "report": rep}
        return rep


class AubinLearning:
    """Ürün katmanı: herhangi bir AUBIN modelini (Aubin / AubinEnsemble — .logprobs(state, questions) sunan) öz-öğrenen hale getirir.

        smart = AubinLearning(Aubin("emrevrg/AUBIN-12B"))
        out = smart.decide(state, questions)              # bellek + öz-kalibrasyonlu karar
        smart.learn(state, questions, {"q1": "refund"})   # doğru cevap → anında öğrenilir (eğitim yok)
        smart.save("mem.npz"); smart.load("mem.npz")      # bellek kalıcı
    Bellek bölümü = soru kimliği + seçenek kümesi (geniş sorularda etiket ADI eşlemesi)."""

    def __init__(self, model, memory=None, calibrator=None, k=16, tau=0.05, w_fuse=1.0, min_sim=0.5):
        self.min_sim = min_sim                         # yalnız yeterince benzer geçmiş vakalar oy verir
        self.model, self.mem = model, (memory or DecisionMemory())
        self.cal = calibrator or SelfCalibrator()
        self.k, self.tau, self.w = k, tau, w_fuse
        self.skills = SkillLibrary(self.mem.embed_fn)
        self._last = {}

    def acquire_skill(self, qid, examples, holdout=0.25, min_gain=2):
        """Yeni yetenek: examples = [(state, questions, {qid: doğru anahtar}), ...] aynı soru tipinden.
        Model yalnız kendini sınama kısmında çağrılır; beceri ancak kazanç kanıtlanırsa açılır. Dönüş: rapor."""
        from .core import options
        texts, answers, keys_list, items = [], [], [], []
        for st, qs, ans in examples:
            keys, _ = options(qs[qid]); texts.append(self._text(st, qs[qid])); answers.append(str(ans[qid]))
            keys_list.append([str(k) for k in keys]); items.append((st, qs))

        def model_lp(i):
            st, qs = items[i]
            lp = self.model.logprobs(st, {qid: qs[qid]})[qid][1]
            return lp.detach().cpu().numpy() if hasattr(lp, "detach") else lp
        return self.skills.acquire(qid, texts, answers, keys_list, model_lp, holdout=holdout, min_gain=min_gain)

    @staticmethod
    def _text(state, q):
        import json as _j
        st = state if isinstance(state, str) else _j.dumps(state, ensure_ascii=False, sort_keys=True)
        return f"{st} || {(q or {}).get('instructions', '')}"

    def decide(self, state, questions):
        lps = self.model.logprobs(state, questions)
        out = {}
        for qid, (keys, lp) in lps.items():
            lp = np.asarray(lp.detach().cpu() if hasattr(lp, "detach") else lp, dtype=np.float64)
            ps, ws = self.skills.probs(qid, self._text(state, questions[qid]), [str(k) for k in keys])
            if ps is not None:                             # kanıtlanmış beceri → modele katılır (log-doğrusal)
                lp = fuse(lp, ps, ws)
            p, smax = self.mem.recall(self._text(state, questions[qid]), qid, keys, self.k, self.tau, min_sim=self.min_sim)
            E = SelfCalibrator.experts(lp, p, self.w)
            mix, pi = self.cal.predict(qid, E)
            self._last[(qid, self._text(state, questions[qid]))] = E
            j = int(np.argmax(mix))
            out[qid] = {"answer": keys[j], "confidence": round(float(mix[j]), 4),
                        "probabilities": {str(k): round(float(v), 4) for k, v in zip(keys, mix)},
                        "memory": {"similar_case": round(smax, 3), "trust_model_memory_fused": [round(float(x), 3) for x in pi]}}
        return out

    def learn(self, state, questions, answers):
        """answers: {qid: doğru seçenek anahtarı}. Belleğe yazar + (son decide varsa) güveni günceller. Dönüş: ms."""
        from .core import options
        t0 = time.perf_counter()
        for qid, ans in answers.items():
            keys, _ = options(questions[qid])
            keys = [str(k) for k in keys]
            if str(ans) not in keys:
                continue
            y = keys.index(str(ans)); txt = self._text(state, questions[qid])
            E = self._last.pop((qid, txt), None)
            if E is not None:
                self.cal.update(qid, E, y)
            self.mem.learn(txt, qid, keys, y)
        return round((time.perf_counter() - t0) * 1e3, 3)

    def save(self, path):
        for s in list(self.mem.vecs):
            self.mem._freeze(s)
        np.savez_compressed(path, **{f"M::{s}": m for s, m in self.mem.mats.items()},
                            **{f"L::{s}": np.asarray(self.mem.labels[s]) for s in self.mem.mats})

    def load(self, path):
        Z = np.load(path, allow_pickle=False)
        for f in Z.files:
            if f.startswith("M::"):
                s = f[3:]; self.mem.mats[s] = Z[f]; self.mem.labels[s] = [str(x) for x in Z["L::" + s]]
        self.mem.n = sum(len(v) for v in self.mem.labels.values())


class AubinLearner:
    """Karar modeli + öz-öğrenme belleği + araştırma kancası.

    score_fn(item) → seçenek log-olasılıkları (ör. kev_llm.Scorer.score); item: {"state","q","keys","texts","src"}.
    research_fn(query) → kanıt metinleri listesi (ör. norovox_mini.retrieval.Retrieval().fetch).
    """

    def __init__(self, score_fn, memory=None, weights=None, margin=99.0, research_fn=None, k=16, tau=0.05, calibrator=None):
        self.score_fn, self.mem = score_fn, (memory or DecisionMemory())
        self.weights = weights or {}            # kaynak → bellek birleştirme ağırlığı (dev'de ayarlanır); yoksa "*" ya da 0.5
        self.margin, self.research_fn, self.k, self.tau = margin, research_fn, k, tau
        self.cal = calibrator if calibrator is not None else SelfCalibrator()
        self.log, self._last = [], {}

    @staticmethod
    def text(it):
        return f"{it['state']} || {it['q']}"

    def decide(self, it):
        t0 = time.perf_counter()
        lp = np.asarray(self.score_fn(it), dtype=np.float64)
        srt = np.sort(lp)[::-1]
        unsure = len(lp) > 1 and (srt[0] - srt[1]) < self.margin
        p, smax, evidence = self.mem.recall(self.text(it), it["src"], it["keys"], self.k, self.tau) + ([],)
        if p is None and unsure and self.research_fn is not None:     # bellekte karşılığı yok: dışarıdan kanıt (Norovox Retrieval)
            try:
                evidence = self.research_fn(f"{it['q']} {str(it['state'])[:300]}")[:3]
            except Exception:
                evidence = []
        E = SelfCalibrator.experts(lp, p, self.weights.get(it["src"], self.weights.get("*", 0.5)))
        mix, pi = self.cal.predict(it["src"], E)
        self._last[id(it)] = E
        ms = (time.perf_counter() - t0) * 1e3
        self.log.append({"src": it["src"], "unsure": bool(unsure), "memory_sim": round(smax, 3), "trust": [round(float(x), 3) for x in pi],
                         "evidence": len(evidence), "ms": round(ms, 2)})
        return int(np.argmax(mix)), np.log(mix + 1e-12), evidence

    def feedback(self, it, y):
        """Doğru cevap gelince: (1) belleğe yaz — bilgi eksiği anında kapanır; (2) öz-kalibrasyon — bu kaynakta model mi bellek mi
        daha isabetli, güven kendiliğinden kayar. Dönüş: ms."""
        t0 = time.perf_counter()
        E = self._last.pop(id(it), None)
        if E is not None:
            self.cal.update(it["src"], E, int(y))
        self.mem.learn(self.text(it), it["src"], it["keys"], y)
        return (time.perf_counter() - t0) * 1e3
