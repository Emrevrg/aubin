"""AUBIN-Learn çevrimiçi deney: soru akışında her cevaptan sonra doğru etiket gelir (geri bildirim), sistem anında öğrenir.

İki akış:
  * kev_transfer_test — Kev'in de AUBIN'in de HİÇ eğitilmediği kaynaklar (mmlu, emotion, tweet_offensive, qnli, paws, sciq);
    bellek BOŞ başlar, yalnız geri bildirimle dolar → "bilmediğini anında öğrenme" ölçümü.
  * kev_test — bellek Kev train (+ek veri) ile başlar.
Yöntemler: model (sabit), AUBIN-Learn = bellek + öz-kalibrasyon (Hedge: model/bellek/birleşik uzmanlarına güven kaynak-başına
geri bildirimle kendiliğinden ayarlanır). Aynı maddelerde, aynı sırada; ikinci yarı ayrıca raporlanır (öğrenme etkisi).
Not: Bu, statik kilitli test skoru DEĞİLDİR (akışta etiket görülür); statik skorlar learn_eval.py raporundadır.

    python learn_online.py --emb emb_bge.npz --pack model_lp.json.gz --out learn_online.json
"""
import argparse, glob, gzip, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import kevdata
from kev_llm import items
from aubin.learn import DecisionMemory, SelfCalibrator


def run_stream(its, V, lps, T, mem, cal, k=16, tau=0.05, wf=1.0):
    hm, hl, src, hk = [], [], [], {}
    for j, it in enumerate(its):
        lp = np.asarray(lps[j], dtype=np.float64) / T
        p, _ = mem.recall(None, it["src"], it["keys"], k, tau, vec=V[j])
        if p is not None:
            hk.setdefault(it["src"], []).append(int(np.argmax(p)) == it["y"])
        E = SelfCalibrator.experts(lp, p, wf)
        mix, _ = cal.predict(it["src"], E)
        hm.append(int(np.argmax(lp)) == it["y"]); hl.append(int(np.argmax(mix)) == it["y"]); src.append(it["src"])
        cal.update(it["src"], E, it["y"])
        mem.learn(None, it["src"], it["keys"], it["y"], vec=V[j])
    half = len(its) // 2
    by = {}
    for s, a, b in zip(src, hm, hl):
        by.setdefault(s, [[], []]); by[s][0].append(a); by[s][1].append(b)
    r = lambda x: round(float(np.mean(x)), 4)
    return {"n": len(its), "model": r(hm), "aubin_learn": r(hl), "model_2nd_half": r(hm[half:]), "aubin_learn_2nd_half": r(hl[half:]),
            "by_source": {s: {"n": len(v[0]), "model": r(v[0]), "aubin_learn": r(v[1]),
                              "memory_only_when_available": (r(hk[s]) if hk.get(s) else None), "memory_available": len(hk.get(s, [])),
                              "model_2nd_half": r(v[0][len(v[0]) // 2:]), "aubin_learn_2nd_half": r(v[1][len(v[1]) // 2:])} for s, v in sorted(by.items())}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/tmp/kev"); ap.add_argument("--emb", default="")
    ap.add_argument("--pack", required=True); ap.add_argument("--out", default="learn_online.json")
    a = ap.parse_args()
    emb = a.emb or (glob.glob("/kaggle/input/**/emb_bge.npz", recursive=True) or [""])[0]
    Z = np.load(emb)
    K = kevdata.load(a.work); tr = items(K["kev_train"])
    VP = Z["VP"].astype(np.float32)
    assert len(VP) == len(tr), f"gömme/bellek boyu uyuşmuyor {len(VP)} != {len(tr)} (KEV_EXTRA_TRAIN aynı mı?)"
    pack = json.load(gzip.open(a.pack, "rt", encoding="utf-8"))
    R = {"protocol": __doc__.split("Not:")[0].strip(), "note": "Statik kilitli test skoru değildir; statik skorlar learn_eval raporunda.",
         "runs": {}}
    def stream(rp, s, warm, eta, prior):
        its = items(K[s]); V = Z["VE_" + s].astype(np.float32); L = rp["suites"][s]
        assert all(int(y) == it["y"] for y, it in zip(L["y"], its))
        mem = DecisionMemory()
        if warm:
            mem.learn_vectors(VP, [x["src"] for x in tr], [x["keys"] for x in tr], [x["y"] for x in tr])
        return run_stream(its, V, L["lp"], rp.get("T", 1.0), mem, SelfCalibrator(eta, prior))

    GRID = [(eta, pr) for eta in (0.3, 1.0, 3.0) for pr in ((0.6, 0.1, 0.3), (0.8, 0.05, 0.15), (0.34, 0.33, 0.33))]
    for rn, rp in pack.items():
        out = {}
        # öz-kalibrasyon hiperparametreleri (η, başlangıç güveni) YALNIZ dev akışlarında seçilir; test akışlarına bir kez uygulanır
        devs = [(s, w) for s, w in (("kev_transfer_dev", False), ("kev_dev", True)) if s in rp["suites"] and "VE_" + s in Z.files]
        best = (1.0, (0.6, 0.1, 0.3))
        if devs:
            best = max(GRID, key=lambda g: np.mean([stream(rp, s, w, *g)["aubin_learn"] for s, w in devs]))
        out["selected_on_dev"] = {"eta": best[0], "prior_model_memory_fused": best[1], "dev_streams": [s for s, _ in devs]}
        for s, warm in (("kev_transfer_test", False), ("kev_test", True)):
            if s not in rp["suites"]:
                continue
            out[s] = stream(rp, s, warm, *best)
            print(rn, s, best, {k: out[s][k] for k in ("model", "aubin_learn", "model_2nd_half", "aubin_learn_2nd_half")}, flush=True)
        R["runs"][rn] = out
        json.dump(R, open(a.out, "w"), indent=1)
    print("BITTI", a.out, flush=True)


if __name__ == "__main__":
    main()
