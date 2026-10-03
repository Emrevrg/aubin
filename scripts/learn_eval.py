"""AUBIN-Learn ölçümü (CPU yeter): öz-öğrenme belleği + kayıtlı model tahminleri → kilitli Kev testleri.

Protokol (dürüst):
  * Bellek = Kev decision-v7 TRAIN (+ kev_augment ek verisi). Test/dev/cal/transfer kümeleri belleğe ASLA girmez.
  * Model log-olasılıkları önceden ölçülmüş koşulardan (aynı madde sırası; y eşleşmesi doğrulanır).
  * Birleştirme ağırlığı (kaynak-başına, log-kayıpla), komşu sayısı ve sıcaklık YALNIZ dev'de (kev_dev; yoksa cal-300)
    seçilir; kev_test / kev_transfer_test bir kez raporlanır.
  * Ek: öğrenme eğrisi (bellek %0→%100), gecikme (learn/recall ms), çevrimiçi geri bildirim deneyi (ayrı protokol).

    python learn_eval.py --work /tmp/kev --pack model_lp.json.gz --embed bge,hash --out learn_report.json
"""
import argparse, gzip, json, math, os, random, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import kevdata
from kev_llm import items
from aubin.learn import DecisionMemory, hash_embed, fuse, sig, vote

SUITES = ("kev_dev", "kev_test", "kev_transfer_dev", "kev_transfer_test")
WGRID = [0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]


def bge_embed_factory(name="BAAI/bge-small-en-v1.5"):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "-q", "install", "sentence-transformers"], check=False)
        from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(name, device="cpu"); m.max_seq_length = int(os.environ.get("AUBIN_EMB_LEN", 128))
    return lambda texts: m.encode(list(texts), batch_size=64, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)


def text(it):
    return f"{it['state']} || {it['q']}"


def acc(rows):
    return float(np.mean([int(np.argmax(lp)) == y for lp, y in rows])) if rows else float("nan")


def nll(rows):
    return float(np.mean([-lp[y] for lp, y in rows])) if rows else float("nan")


def neighbours(mem, its, V, K=64):
    out = []
    for it, v in zip(its, V):
        s = sig(it["src"], it["keys"]); mem._freeze(s)
        M = mem.mats.get(s)
        if M is None or not len(M):
            out.append(None); continue
        sims = M @ v; kk = min(K, len(sims)); top = np.argpartition(-sims, kk - 1)[:kk]
        top = top[np.argsort(-sims[top])]
        out.append((sims[top].astype(np.float32), np.asarray([mem.labels[s][i] for i in top])))
    return out


def pmem(nb, keys, k, tau):
    if nb is None:
        return None
    sims, labs = nb[0][:k], nb[1][:k]
    return vote(list(labs), np.exp((sims - sims.max()) / tau), keys)


def combine(lps, ys, nbs, srcs, kss, k, tau, wsrc, T):
    return [(fuse(lp, pmem(nb, ks, k, tau), wsrc.get(s, wsrc.get("*", 0.0)), T), y) for lp, y, nb, s, ks in zip(lps, ys, nbs, srcs, kss)]


def tune(dev, T):
    """Global (k, τ, w) dev log-kaybıyla; sonra kaynak-başına w (≥30 madde) yine log-kayıpla (doğruluktan daha az gürültülü)."""
    lps, ys, nbs, srcs, ns = dev
    best = None
    for k in (8, 16, 32, 64):
        for tau in (0.02, 0.05, 0.1, 0.2):
            for w in WGRID:
                v = nll(combine(lps, ys, nbs, srcs, ns, k, tau, {"*": w}, T))
                if best is None or v < best[0] - 1e-9:
                    best = (v, k, tau, w)
    _, k, tau, wg = best
    wsrc = {"*": wg}
    for s in sorted(set(srcs)):
        idx = [i for i, x in enumerate(srcs) if x == s]
        if len(idx) < 30:
            continue
        sub = [lps[i] for i in idx], [ys[i] for i in idx], [nbs[i] for i in idx], [srcs[i] for i in idx], [ns[i] for i in idx]
        wsrc[s] = min(WGRID, key=lambda w: (nll(combine(*sub, k, tau, {"*": w}, T)), abs(w - wg)))
    full = combine(lps, ys, nbs, srcs, ns, k, tau, wsrc, T)
    return {"k": k, "tau": tau, "w": wsrc, "dev_nll_global": round(best[0], 4), "dev_acc_fused": round(acc(full), 4)}


def by_source(rows, srcs):
    d = {}
    for (lp, y), s in zip(rows, srcs):
        d.setdefault(s, []).append(int(np.argmax(lp)) == y)
    return {s: [round(float(np.mean(v)), 4), len(v)] for s, v in sorted(d.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/tmp/kev"); ap.add_argument("--pack", required=True)
    ap.add_argument("--embed", default="bge,hash"); ap.add_argument("--out", default="learn_report.json")
    a = ap.parse_args()
    K = kevdata.load(a.work)
    tr = items(K["kev_train"])
    ev = {s: items(K[s]) for s in SUITES}
    cal = items(K["kev_cal"]); random.Random(0).shuffle(cal); ev["cal300"] = cal[:300]
    pack = json.load(gzip.open(a.pack, "rt", encoding="utf-8"))
    is_extra = np.array([x["key"].startswith("kev_extra/") for x in tr])
    R = {"protocol": __doc__.split("Protokol (dürüst):")[1].split("python learn_eval")[0].strip(),
         "memory": {"kev_train_items": int((~is_extra).sum()), "extra_items": int(is_extra.sum())},
         "reference": {"Kev-9B kev_test": 0.874, "Kev-27B kev_test": 0.870, "Kev-27B kev_transfer_test": 0.896}, "embeds": {}}
    for emb in a.embed.split(","):
        t0 = time.time()
        fn = hash_embed if emb == "hash" else bge_embed_factory({"bge": "BAAI/bge-small-en-v1.5", "bge-base": "BAAI/bge-base-en-v1.5", "e5-base": "intfloat/e5-base-v2"}[emb])
        VP = np.concatenate([fn([text(x) for x in tr[i:i + 4096]]) for i in range(0, len(tr), 4096)])
        VE = {s: fn([text(x) for x in ev[s]]) for s in ev}
        print(emb, "gömme bitti", VP.shape, round(time.time() - t0), "s", flush=True)
        if emb != "hash":                                  # sonraki koşular için gömmeleri sakla (float16)
            np.savez_compressed(os.path.join(os.path.dirname(os.path.abspath(a.out)), f"emb_{emb}.npz"), VP=VP.astype(np.float16),
                                **{"VE_" + s: VE[s].astype(np.float16) for s in VE})
        out = {}
        for mem_name, mask in (("kev+extra", np.ones(len(tr), bool)), ("kev_only", ~is_extra)):
            if mem_name == "kev+extra" and not is_extra.any():
                continue
            idx = np.where(mask)[0]
            mem = DecisionMemory(fn)
            mem.learn_vectors(VP[idx], [tr[i]["src"] for i in idx], [tr[i]["keys"] for i in idx], [tr[i]["y"] for i in idx])
            NB = {s: neighbours(mem, ev[s], VE[s]) for s in ev}
            res = {"embed_seconds": round(time.time() - t0, 1), "memory_items": int(len(idx)), "runs": {}}
            it0 = ev["kev_test"][0]
            ms_l = [mem.learn(text(it0), "latency_probe", it0["keys"], it0["y"]) for _ in range(20)]
            t1 = time.perf_counter(); [mem.recall(text(x), x["src"], x["keys"]) for x in ev["kev_test"][:100]]
            res["latency_ms"] = {"learn_median": round(float(np.median(ms_l)), 3), "recall_mean_incl_embed": round((time.perf_counter() - t1) * 1e3 / 100, 3)}
            for rn, rp in pack.items():
                T = rp.get("T", 1.0); have = [s for s in ev if s in rp["suites"]]
                def prep(s, nbs=None):
                    its = ev[s]; L = rp["suites"][s]
                    assert len(L["y"]) == len(its) and all(int(y) == it["y"] for y, it in zip(L["y"], its)), f"{rn}/{s} sıra uyuşmuyor"
                    return [np.asarray(x) for x in L["lp"]], L["y"], (nbs or NB[s]), [it["src"] for it in its], [it["keys"] for it in its]
                devname = "kev_dev" if "kev_dev" in have else ("cal300" if "cal300" in have else None)
                if devname is None:
                    continue
                prm = tune(prep(devname), T)
                rr = {"dev_used": devname, "params": prm}
                for s in ("kev_test", "kev_transfer_test", "kev_dev", "kev_transfer_dev"):
                    if s not in have:
                        continue
                    P = prep(s)
                    base = combine(*P, prm["k"], prm["tau"], {"*": 0.0}, T)
                    fu = combine(*P, prm["k"], prm["tau"], prm["w"], T)
                    memonly = [((np.log(pm + 1e-9) if pm is not None else lp), y) for lp, y, pm in
                               zip(P[0], P[1], [pmem(nb, ks, prm["k"], prm["tau"]) for nb, ks in zip(P[2], P[4])])]
                    rr[s] = {"model": round(acc(base), 4), "memory_only": round(acc(memonly), 4), "fused": round(acc(fu), 4),
                             "model_nll": round(nll(base), 4), "fused_nll": round(nll(fu), 4), "n": len(base),
                             "by_source_model": by_source(base, P[3]), "by_source_fused": by_source(fu, P[3])}
                if "kev_test" in have and mem_name == "kev+extra":     # öğrenme eğrisi: bellek büyüdükçe test doğruluğu
                    curve = {}
                    rng = np.random.default_rng(1); u = rng.random(len(idx))
                    for f in (0.0, 0.05, 0.2, 0.5, 1.0):
                        sel = idx[u < f]; sub = DecisionMemory(fn)
                        if len(sel):
                            sub.learn_vectors(VP[sel], [tr[i]["src"] for i in sel], [tr[i]["keys"] for i in sel], [tr[i]["y"] for i in sel])
                        P = prep("kev_test", neighbours(sub, ev["kev_test"], VE["kev_test"]))
                        curve[str(f)] = {"memory_items": int(len(sel)), "kev_test_fused": round(acc(combine(*P, prm["k"], prm["tau"], prm["w"], T)), 4)}
                    rr["learning_curve"] = curve
                if "kev_test" in have:                                 # çevrimiçi geri bildirim: test akışında her cevaptan sonra doğru etiket belleğe
                    on = DecisionMemory(fn)
                    on.learn_vectors(VP[idx], [tr[i]["src"] for i in idx], [tr[i]["keys"] for i in idx], [tr[i]["y"] for i in idx])
                    hits = []
                    for j, it in enumerate(ev["kev_test"]):
                        nb = neighbours(on, [it], VE["kev_test"][j:j + 1])[0]
                        lp = fuse(np.asarray(rp["suites"]["kev_test"]["lp"][j]), pmem(nb, it["keys"], prm["k"], prm["tau"]),
                                  prm["w"].get(it["src"], prm["w"]["*"]), T)
                        hits.append(int(np.argmax(lp)) == it["y"])
                        on.learn_vectors(VE["kev_test"][j:j + 1], [it["src"]], [it["keys"]], [it["y"]])
                    rr["online_feedback_kev_test"] = {"note": "ayrı protokol: her cevaptan sonra doğru etiket belleğe yazılır (statik test değil)",
                                                      "accuracy": round(float(np.mean(hits)), 4),
                                                      "second_half": round(float(np.mean(hits[len(hits) // 2:])), 4)}
                res["runs"][rn] = rr
                print(emb, mem_name, rn, "TEST fused", {s: rr[s]["fused"] for s in ("kev_test", "kev_transfer_test") if s in rr},
                      "model", {s: rr[s]["model"] for s in ("kev_test", "kev_transfer_test") if s in rr}, flush=True)
            out[mem_name] = res
            R["embeds"][emb] = out
            json.dump(R, open(a.out, "w"), indent=1)
    print("BITTI", a.out, flush=True)


if __name__ == "__main__":
    main()
