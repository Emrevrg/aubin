"""AUBIN-Learn hızlı beceri (Norovox SkillLibrary'nin karar modeli karşılığı): bellekteki vakalardan kaynak-başına saniyeler içinde
öğrenilen doğrusal sınıflandırıcı (gömme → etiket ADI). Model + beceri log-doğrusal birleşir; ağırlık ve düzenlileştirme
(C) YALNIZ kev_dev'de seçilir, kev_test / kev_transfer_test bir kez raporlanır. Bellek = Kev train + ek veri (test asla girmez).

    python learn_skill.py --pack model_lp.json.gz --emb emb_bge.npz --out learn_skill.json
"""
import argparse, glob, gzip, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import kevdata
from kev_llm import items
from aubin.learn import fuse

WGRID = [0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]


def lsm(x):
    x = np.asarray(x, dtype=np.float64); return x - np.logaddexp.reduce(x)


class SoftmaxProbe:
    """Bağımlılıksız çok sınıflı lojistik regresyon (numpy, L2 = 1/C, Adam, tam yığın). sklearn arayüzünün alt kümesi."""

    def __init__(self, C=1.0, max_iter=300, lr=0.05):
        self.C, self.max_iter, self.lr = C, max_iter, lr

    def fit(self, X, y):
        self.classes_ = np.array(sorted(set(y)))
        idx = {c: i for i, c in enumerate(self.classes_)}
        Y = np.zeros((len(y), len(self.classes_)), dtype=np.float32); Y[np.arange(len(y)), [idx[c] for c in y]] = 1
        X = np.asarray(X, dtype=np.float32); n, d = X.shape
        W = np.zeros((d, len(self.classes_)), dtype=np.float32); b = np.zeros(len(self.classes_), dtype=np.float32)
        mW, vW, mb, vb = np.zeros_like(W), np.zeros_like(W), np.zeros_like(b), np.zeros_like(b)
        lam = 1.0 / (self.C * n)
        for t in range(1, self.max_iter + 1):
            Z = X @ W + b; Z -= Z.max(1, keepdims=True); P = np.exp(Z); P /= P.sum(1, keepdims=True)
            G = (P - Y) / n
            gW, gb = X.T @ G + lam * W, G.sum(0)
            for g, m, v, p in ((gW, mW, vW, W), (gb, mb, vb, b)):
                m *= 0.9; m += 0.1 * g; v *= 0.999; v += 0.001 * g * g
                p -= self.lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
        self.W, self.b = W, b
        return self

    def predict_proba(self, X):
        Z = np.asarray(X, dtype=np.float32) @ self.W + self.b; Z -= Z.max(1, keepdims=True); P = np.exp(Z)
        return P / P.sum(1, keepdims=True)


def fit_skills(tr, VP, C):
    """Kaynak-başına çok sınıflı lojistik regresyon (etiket adları). Dönüş: {kaynak: (model, sınıflar, süre_ms)}."""
    LogisticRegression = SoftmaxProbe
    by = {}
    for i, x in enumerate(tr):
        by.setdefault(x["src"], []).append(i)
    out = {}
    for s, idx in by.items():
        y = [str(tr[i]["keys"][tr[i]["y"]]) for i in idx]
        if len(set(y)) < 2 or len(idx) < 50:
            continue
        t0 = time.perf_counter()
        m = LogisticRegression(C=C)
        m.fit(VP[idx], y)
        out[s] = (m, list(m.classes_), (time.perf_counter() - t0) * 1e3)
    return out


def skill_probs(skills, its, V):
    P = []
    for it, v in zip(its, V):
        sk = skills.get(it["src"])
        if sk is None:
            P.append(None); continue
        m, cls, _ = sk
        pr = m.predict_proba(v[None])[0]
        pos = {c: j for j, c in enumerate(cls)}
        p = np.array([pr[pos[str(k)]] if str(k) in pos else 0.0 for k in it["keys"]], dtype=np.float64)
        P.append(p / p.sum() if p.sum() > 0 else None)
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/tmp/kev"); ap.add_argument("--emb", default="")
    ap.add_argument("--pack", required=True); ap.add_argument("--out", default="learn_skill.json")
    a = ap.parse_args()
    emb = a.emb or (glob.glob("/kaggle/input/**/" + os.environ.get("AUBIN_EMB_FILE", "emb_bge.npz"), recursive=True) or [""])[0]
    Z = np.load(emb)
    K = kevdata.load(a.work); tr = items(K["kev_train"])
    VP = Z["VP"].astype(np.float32)
    assert len(VP) == len(tr), f"gömme/bellek boyu uyuşmuyor {len(VP)} != {len(tr)}"
    pack = json.load(gzip.open(a.pack, "rt", encoding="utf-8"))
    ev = {s: items(K[s]) for s in ("kev_dev", "kev_test", "kev_transfer_test")}
    import random as _r
    _c = items(K["kev_cal"]); _r.Random(0).shuffle(_c); ev["cal300"] = _c[:300]      # kev_dev'i eksik üyeler için ayar kümesi
    VE = {s: Z["VE_" + s].astype(np.float32) for s in ev}
    R = {"protocol": __doc__.split("    python")[0].strip(), "memory_items": len(tr), "C": {}, "runs": {}}
    SK = {}
    for C in (0.3, 1.0, 3.0, 10.0):
        t0 = time.time(); sk = fit_skills(tr, VP, C)
        SK[C] = {s: skill_probs(sk, ev[s], VE[s]) for s in ev}
        R["C"][str(C)] = {"fit_seconds_total": round(time.time() - t0, 1),
                          "fit_ms_per_source": {s: round(v[2], 1) for s, v in sk.items()},
                          "skill_only_dev_acc": round(float(np.mean([int(np.argmax(p)) == it["y"] for p, it in zip(SK[C]["kev_dev"], ev["kev_dev"]) if p is not None])), 4)}
        print("C", C, R["C"][str(C)]["skill_only_dev_acc"], R["C"][str(C)]["fit_seconds_total"], "s", flush=True)
    for rn, rp in pack.items():
        D = "kev_dev" if "kev_dev" in rp["suites"] else ("cal300" if "cal300" in rp["suites"] else None)
        if D is None:
            continue
        T = rp.get("T", 1.0)
        L = {s: [np.asarray(x) for x in rp["suites"][s]["lp"]] for s in ev if s in rp["suites"]}
        def rows(s, C, wsrc):
            return [(fuse(lp, p, wsrc.get(it["src"], wsrc.get("*", 0.0)), T), it["y"], it["src"])
                    for lp, p, it in zip(L[s], SK[C][s], ev[s])]
        nll = lambda rr: float(np.mean([-r[0][r[1]] for r in rr]))
        acc = lambda rr: float(np.mean([int(np.argmax(r[0])) == r[1] for r in rr]))
        bestC, bestw = min(((C, w) for C in SK for w in WGRID), key=lambda cw: nll(rows(D, cw[0], {"*": cw[1]})))
        gated = os.environ.get("AUBIN_SKILL_GATE", "1") == "1"
        wsrc = {"*": 0.0 if gated else bestw}       # temkinli: kanıtlanmayan kaynakta beceri KAPALI
        for s in sorted({it["src"] for it in ev[D]}):
            idx = [i for i, it in enumerate(ev[D]) if it["src"] == s]
            if len(idx) < 30:
                continue
            def sub(w):
                rr = rows(D, bestC, {"*": w})
                return float(np.mean([-rr[i][0][rr[i][1]] for i in idx])), sum(int(np.argmax(rr[i][0])) == rr[i][1] for i in idx)
            w_nll = min(WGRID, key=lambda w: (sub(w)[0], abs(w - bestw)))
            if gated:                                   # kapı: dev'de doğruluk ≥ +2 soru VE log-kayıp düşmeli; yoksa 0
                (n0, a0), (n1, a1) = sub(0.0), sub(w_nll)
                wsrc[s] = w_nll if (a1 >= a0 + 2 and n1 < n0) else 0.0
            else:
                wsrc[s] = w_nll
        out = {"selected_on_dev": {"C": bestC, "w": wsrc, "gated": gated, "dev_split": D}}
        for s in ("kev_dev", "kev_test", "kev_transfer_test"):
            if s not in L:
                continue
            base, fu = rows(s, bestC, {"*": 0.0}), rows(s, bestC, wsrc)
            bys = {}
            for (b, f) in zip(base, fu):
                bys.setdefault(b[2], [[], []]); bys[b[2]][0].append(int(np.argmax(b[0])) == b[1]); bys[b[2]][1].append(int(np.argmax(f[0])) == f[1])
            sk_only = [int(np.argmax(p)) == it["y"] for p, it in zip(SK[bestC][s], ev[s]) if p is not None]
            out[s] = {"model": round(acc(base), 4), "fused": round(acc(fu), 4), "model_nll": round(nll(base), 4), "fused_nll": round(nll(fu), 4),
                      "skill_only_when_available": round(float(np.mean(sk_only)), 4) if sk_only else None, "n": len(base),
                      "by_source": {k: [len(v[0]), round(float(np.mean(v[0])), 4), round(float(np.mean(v[1])), 4)] for k, v in sorted(bys.items())}}
        R["runs"][rn] = out
        print(rn, {s: (out[s]["model"], out[s]["fused"]) for s in ("kev_test", "kev_transfer_test") if s in out}, flush=True)
        json.dump(R, open(a.out, "w"), indent=1)
    print("BITTI", a.out, flush=True)


if __name__ == "__main__":
    main()
