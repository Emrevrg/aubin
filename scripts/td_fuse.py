"""AUBIN-Learn @ typed-decisions (Laya / Jev / meraGPT kümesi): AUBIN LLM + hızlı beceriler (NIVEN-Ağaç, bge-softmax) birleşimi.

Uzmanlar (hepsinin train tahmini ÖRNEK-DIŞI):
  * AUBIN (td_eval.py): soru-başı log-olasılık; AUBIN bu kümede hiç eğitilmedi.
  * NIVEN-Ağaç (hukum3/gbdt.py, yalnız durum JSON'u; 'factors' girdisi YOK): train 5-katlı OOF, test tam-train.
  * bge-beceri (td_skill.py): soru-başı gömme → softmax, aynı 5 kat.
Birleştirme: soru-anahtarı başına log-doğrusal ağırlıklar + AUBIN sıcaklığı YALNIZ train'de (log-kayıp) seçilir; test bir kez.

    python td_fuse.py --gbdt gbdt_preds.json [--aubin td_aubin12.json] [--skill td_skill.json] --out td_fused.json
"""
import argparse, glob, itertools, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from td_eval import load_td
from kev_llm import items

REF = {"meraGPT Decider 1 (leaderboard #1)": 0.768, "Laya (same test, our run)": 0.7665, "TypeSafe Jev 1.13.0": 0.727,
       "NIVEN (ours, 26 Sep)": 0.773, "teacher self-agreement": 0.735}
GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]


def lsm(x):
    x = np.asarray(x, dtype=np.float64); return x - np.logaddexp.reduce(x)


def raw_rows(split):
    try:
        from datasets import load_dataset
        rows = load_dataset("LocalLLaMA/typed-decisions", "all", split=split)
    except Exception:
        import pyarrow.parquet as pq
        rows = pq.read_table(glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/datasets--LocalLLaMA--typed-decisions/snapshots/*/all/{split}-*.parquet"))[0]).to_pylist()
    return [{"id": r["id"], "workflow": r["workflow"], "gold_full": json.loads(r["gold"]) if isinstance(r["gold"], str) else r["gold"]} for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gbdt", required=True); ap.add_argument("--aubin", default=""); ap.add_argument("--skill", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    G = json.load(open(a.gbdt)); A = json.load(open(a.aubin)) if a.aubin else None; S = json.load(open(a.skill)) if a.skill else None
    opts = {}
    for r in raw_rows("train"):                         # hukum.build_schema ile aynı seçenek sırası
        for q, g in r["gold_full"].items():
            opts.setdefault(f"{r['workflow']}/{q}", list(g["probabilities"].keys()))
    names = ["NIVEN-Ağaç"] + (["AUBIN"] if A else []) + (["bge-beceri"] if S else [])
    data = {}
    for split, gk in (("train", "oof"), ("test", "test")):
        its = items(load_td(split))
        la = {r["key"]: r for r in A[split]["items"]} if A else {}
        rows = []
        for it in its:
            cid = it["key"].split("/", 1)[1].rsplit("/", 1)[0]; q = it["key"].rsplit("/", 1)[1]; qk = f"{it['src']}/{q}"
            g = G[gk].get(cid, {}).get(qk)
            if g is None or (A and it["key"] not in la):
                continue
            pos = {o: j for j, o in enumerate(opts[qk])}
            pg = np.array([g[pos[k]] if k in pos else 0.0 for k in it["keys"]])
            E = {"NIVEN-Ağaç": np.log(pg / pg.sum() + 1e-6)}
            if A:
                E["AUBIN"] = np.asarray(la[it["key"]]["lp"])
            if S:
                d = S[gk].get(cid, {}).get(qk, {})
                ps = np.array([d.get(str(k), 0.0) for k in it["keys"]]); ps = ps / ps.sum() if ps.sum() > 0 else np.full(len(ps), 1 / len(ps))
                E["bge-beceri"] = np.log(ps + 1e-6)
            rows.append({"qk": qk, "y": it["y"], "E": E})
        data[split] = rows
    tr, te = data["train"], data["test"]
    acc = lambda rr, f: float(np.mean([int(np.argmax(f(r))) == r["y"] for r in rr]))
    nll = lambda rr, f: float(np.mean([-lsm(f(r))[r["y"]] for r in rr]))
    T = 1.0
    if A:
        T = min([0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0], key=lambda t: nll(tr, lambda r: r["E"]["AUBIN"] / t))
    sc = lambda r, n: r["E"][n] / (T if n == "AUBIN" else 1.0)
    W = {}
    for qk in sorted({r["qk"] for r in tr}):
        sub = [r for r in tr if r["qk"] == qk]
        W[qk] = min((w for w in itertools.product(GRID, repeat=len(names)) if sum(w) > 0),
                    key=lambda w: nll(sub, lambda r: sum(wi * sc(r, n) for wi, n in zip(w, names))))
    fused = lambda r: sum(wi * sc(r, n) for wi, n in zip(W[r["qk"]], names))
    label = "AUBIN-Learn (" + " + ".join(names) + ")"
    R = {"benchmark": "LocalLLaMA/typed-decisions test (400 cases x 5 = 2000 decisions); label = gold argmax",
         "protocol": __doc__.split("    python")[0].strip(), "reference": REF, "experts": names, "aubin_temperature": T,
         "n": {"train": len(tr), "test": len(te)},
         "test": {**{n: round(acc(te, lambda r, n=n: r["E"][n]), 4) for n in names}, label: round(acc(te, fused), 4)},
         "train_oof": {**{n: round(acc(tr, lambda r, n=n: r["E"][n]), 4) for n in names}, "fused": round(acc(tr, fused), 4)},
         "test_nll_fused": round(nll(te, fused), 4),
         "weights_per_question": {k: dict(zip(names, v)) for k, v in W.items()}}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(R, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(json.dumps(R["test"], ensure_ascii=False), "| train", json.dumps(R["train_oof"], ensure_ascii=False))


if __name__ == "__main__":
    main()
