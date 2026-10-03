"""AUBIN-Learn Beceri-3 (dışbükey, çökmez): güçlü cümle gömmesi (dondurulmuş) + (kaynak, soru) başına çok sınıflı lojistik
regresyon, kaynakların AÇIK eğitim bölümünden (Kev train + kev_augment ek verisi). Ayarlar önceden sabit; kev_dev yalnız
birleştirme seçiminde, testler yalnız rapor. Çıktı biçimi learn_skill2.py ile aynı (fuse_skill2.py okur).

    KEV_EXTRA_TRAIN=kev_extra.jsonl python learn_skill3.py --work /tmp/kev --model BAAI/bge-large-en-v1.5 --out skill3.json
"""
import argparse, json, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import kevdata
from kev_llm import items
from learn_skill2 import gkey, text, SUITES


def embed(tok, enc, texts, prefix, max_len, bs=128):
    out = []
    order = np.argsort([len(t) for t in texts])
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            idx = order[i: i + bs]
            b = tok([prefix + texts[j] for j in idx], truncation=True, max_length=max_len, padding=True, return_tensors="pt").to("cuda")
            with torch.autocast("cuda", dtype=torch.float16):
                h = enc(**b).last_hidden_state
            if prefix:                                        # e5: ortalama havuzlama
                m = b["attention_mask"].unsqueeze(-1).float(); v = (h.float() * m).sum(1) / m.sum(1)
            else:                                             # bge: CLS
                v = h[:, 0].float()
            out.append((idx, torch.nn.functional.normalize(v, dim=-1).cpu().numpy()))
    E = np.zeros((len(texts), out[0][1].shape[1]), dtype=np.float32)
    for idx, v in out:
        E[idx] = v
    return E


def fit(X, y, n_cls, C, iters=400):
    X = torch.tensor(X, device="cuda"); y = torch.tensor(y, device="cuda")
    W = torch.zeros(X.shape[1], n_cls, device="cuda", requires_grad=True); b = torch.zeros(n_cls, device="cuda", requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1, max_iter=iters, history_size=20, line_search_fn="strong_wolfe")
    lam = 1.0 / (C * len(y))

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(X @ W * 20 + b, y) + lam * (W * W).sum()
        loss.backward(); return loss
    opt.step(closure)
    return W.detach(), b.detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/tmp/kev"); ap.add_argument("--model", default="BAAI/bge-large-en-v1.5")
    ap.add_argument("--max_len", type=int, default=256); ap.add_argument("--C", type=float, default=10.0)
    ap.add_argument("--out", default="skill3.json")
    a = ap.parse_args()
    prefix = "query: " if "e5" in a.model.lower() else ""
    K = kevdata.load(a.work)
    tr = items(K["kev_train"])
    G, cnt = {}, {}
    for it in tr:
        G.setdefault(gkey(it), set()).add(str(it["keys"][it["y"]])); cnt[gkey(it)] = cnt.get(gkey(it), 0) + 1
    G = {g: sorted(v) for g, v in G.items() if len(v) >= 2 and cnt[g] >= 50}
    tr = [it for it in tr if gkey(it) in G]
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(a.model); enc = AutoModel.from_pretrained(a.model, dtype=torch.float32).cuda().eval()
    t0 = time.time()
    Xtr = embed(tok, enc, [text(x) for x in tr], prefix, a.max_len)
    print({"train_items": len(tr), "groups": len(G), "embed_min": round((time.time() - t0) / 60, 1)}, flush=True)
    Wb = {}
    for g in sorted(G):
        idx = [i for i, x in enumerate(tr) if gkey(x) == g]
        pos = {c: j for j, c in enumerate(G[g])}
        Wb[g] = fit(Xtr[idx], [pos[str(tr[i]["keys"][tr[i]["y"]])] for i in idx], len(G[g]), a.C)
    R = {"protocol": __doc__.split("    KEV")[0].strip(), "model": a.model, "C": a.C, "max_len": a.max_len, "train_items": len(tr),
         "minutes": round((time.time() - t0) / 60, 1), "suites": {}}
    for sname in SUITES:
        if sname not in K:
            continue
        ev = items(K[sname])
        X = embed(tok, enc, [text(x) for x in ev], prefix, a.max_len)
        P = []
        for x, v in zip(ev, X):
            g = gkey(x)
            if g not in G:
                P.append(None); continue
            W, b = Wb[g]
            pr = torch.softmax(torch.tensor(v, device="cuda") @ W * 20 + b, -1).cpu().numpy()
            pos = {c: j for j, c in enumerate(G[g])}
            p = np.array([pr[pos[str(k)]] if str(k) in pos else 0.0 for k in x["keys"]], dtype=np.float64)
            P.append((p / p.sum()).round(5).tolist() if p.sum() > 0 else None)
        acc = [int(np.argmax(p)) == x["y"] for p, x in zip(P, ev) if p is not None]
        bys = {}
        for p, x in zip(P, ev):
            if p is not None:
                bys.setdefault(x["src"], []).append(int(np.argmax(p)) == x["y"])
        R["suites"][sname] = {"keys": [x["key"] for x in ev], "p": P, "covered": len(acc), "n": len(ev),
                              "skill_acc_covered": round(float(np.mean(acc)), 4) if acc else None,
                              "by_source": {k: [len(v), round(float(np.mean(v)), 4)] for k, v in sorted(bys.items())}}
        print(sname, R["suites"][sname]["skill_acc_covered"], R["suites"][sname]["by_source"], flush=True)
        json.dump(R, open(a.out, "w"))
    print("BITTI", a.out, flush=True)


if __name__ == "__main__":
    main()
