"""typed-decisions için AUBIN-Learn hızlı becerisi: soru-anahtarı başına bge gömmesi → softmax sınıflandırıcı (saniyeler, CPU).
train için NIVEN ile AYNI 5 kat (random.Random(7) karıştırma, i::5) → örnek-dışı (OOF) tahmin; test için tüm train'le eğitilir.
Girdi yalnız durum + soru metni ('factors' YOK). Çıktı td_fuse.py'nin üçüncü uzmanı.

    python td_skill.py --out /kaggle/working/td_skill.json
"""
import argparse, json, os, random, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from td_eval import load_td
from kev_llm import items
from learn_skill import SoftmaxProbe
from learn_eval import bge_embed_factory


def text(it):
    return f"{it['state']} || {it['q']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="td_skill.json"); ap.add_argument("--work", default="")
    ap.add_argument("--C", type=float, default=1.0)
    a = ap.parse_args()
    tr, te = items(load_td("train")), items(load_td("test"))
    emb = bge_embed_factory()
    Vtr, Vte = emb([text(x) for x in tr]), emb([text(x) for x in te])
    cid = lambda it: it["key"].split("/", 1)[1].rsplit("/", 1)[0]
    qk = lambda it: f"{it['src']}/{it['key'].rsplit('/', 1)[1]}"
    order = list(dict.fromkeys(cid(x) for x in tr))                 # veri seti sırası (NIVEN: range(len(tr)) karıştırılır)
    perm = list(range(len(order))); random.Random(7).shuffle(perm)
    fold = {order[perm[k]]: k % 5 for k in range(len(order))}       # NIVEN: folds[f] = ids[f::5]
    out = {"oof": {}, "test": {}, "C": a.C}
    for q in sorted({qk(x) for x in tr}):
        I = [i for i, x in enumerate(tr) if qk(x) == q]; J = [j for j, x in enumerate(te) if qk(x) == q]
        lab = lambda it: str(it["keys"][it["y"]])
        for f in range(5):
            trn = [i for i in I if fold[cid(tr[i])] != f]; val = [i for i in I if fold[cid(tr[i])] == f]
            m = SoftmaxProbe(C=a.C).fit(Vtr[trn], [lab(tr[i]) for i in trn])
            P = m.predict_proba(Vtr[val])
            for i, p in zip(val, P):
                out["oof"].setdefault(cid(tr[i]), {})[q] = {str(c): float(v) for c, v in zip(m.classes_, p)}
        m = SoftmaxProbe(C=a.C).fit(Vtr[I], [lab(tr[i]) for i in I])
        for j, p in zip(J, m.predict_proba(Vte[J])):
            out["test"].setdefault(cid(te[j]), {})[q] = {str(c): float(v) for c, v in zip(m.classes_, p)}
        print(q, len(I), len(J), flush=True)
    def acc(split, its):
        h = []
        for it in its:
            d = out[split].get(cid(it), {}).get(qk(it))
            if d:
                h.append(max(d, key=d.get) == str(it["keys"][it["y"]]))
        return round(float(np.mean(h)), 4)
    out["acc"] = {"train_oof": acc("oof", tr), "test": acc("test", te)}
    json.dump(out, open(a.out, "w"))
    print("TD_SKILL", out["acc"], flush=True)


if __name__ == "__main__":
    main()
