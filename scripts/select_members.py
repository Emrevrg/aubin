"""AUBIN ansambl üye seçimi — DÜRÜST protokol: seçim yalnız dev bölümlerinde (kev_dev + kev_transfer_dev), test yalnız raporlanır.

Açgözlü ileri seçim (tekrarlı üye = ağırlık artışı): her adımda dev log-kaybını (NLL, sıcaklık cal'da) en çok düşüren üye eklenir;
iyileşme yoksa durur. Sonra seçilen ağırlıklarla kev_test / kev_transfer_test (ve dev) raporlanır.

    python select_members.py out.json e12a.json e12b.json e31.json ... [--max 8] [--metric nll|acc]
"""
import argparse, json
import numpy as np
from ensemble import metrics

DEV = ("kev_dev", "kev_transfer_dev")
TEST = ("kev_test", "kev_transfer_test")


def runs(r):
    d = dict(r["runs"] if "runs" in r else r)
    if r.get("runs_cal_items") and "cal300" not in d:      # kalibrasyon bölümü de seçim kümesi olabilir (--dev cal300)
        d["cal300"] = {"items": [{"src": "cal", "y": it["y"], "lp": it["lp"]} for it in r["runs_cal_items"]],
                       "all": {"accuracy": float(np.mean([int(np.argmax(it["lp"])) == it["y"] for it in r["runs_cal_items"]]))}}
    return d


def _n(lp, r):
    """Üyeyi kendi kalibrasyon sıcaklığıyla normalize et (farklı tabanların ham logit ölçekleri farklı)."""
    x = np.asarray(lp, dtype=np.float64) / float(r.get("runs_temperature", 1.0))
    return x - np.logaddexp.reduce(x)


def combo(R, w, s):
    its = [runs(r)[s]["items"] for r in R]
    rows = []
    for k in range(len(its[0])):
        lp = sum(wi * _n(x[k]["lp"], r) for wi, x, r in zip(w, its, R) if wi)
        rows.append((its[0][k]["src"], its[0][k]["y"], lp))
    return rows


def temp(R, w):
    cal = [r.get("runs_cal_items") for r in R]
    if not all(cal[i] for i in range(len(R)) if w[i]):
        return 1.0
    sel = [(wi, ci, r) for wi, ci, r in zip(w, cal, R) if wi]
    comb = [(0, sel[0][1][k]["y"], sum(wi * _n(ci[k]["lp"], r) for wi, ci, r in sel)) for k in range(len(sel[0][1]))]
    grid = np.concatenate([np.arange(0.3, 4.0, 0.05), np.arange(4.0, 20.01, 0.25)])
    return float(min(grid, key=lambda t: metrics(comb, t)["all"]["nll"]))


def score(R, cnt, metric):
    w = np.array(cnt, dtype=float); w = w / w.sum()
    T = temp(R, w)
    rows = [x for s in DEV for x in combo(R, w, s)]
    m = metrics(rows, T)["all"]
    return (-m["nll"] if metric == "nll" else m["accuracy"]), w, T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("files", nargs="+")
    ap.add_argument("--max", type=int, default=8); ap.add_argument("--metric", default="nll", choices=["nll", "acc"])
    ap.add_argument("--dev", default="kev_dev,kev_transfer_dev")
    a = ap.parse_args()
    global DEV
    DEV = tuple(a.dev.split(","))
    R = [json.load(open(f)) for f in a.files]
    for f, r in zip(a.files, R):
        miss = [s for s in DEV + ("kev_test",) if s not in runs(r)]
        assert not miss, f"{f}: eksik küme {miss}"
    cnt = [0] * len(R); best = None; path = []
    for _ in range(a.max):
        cands = []
        for i in range(len(R)):
            c = list(cnt); c[i] += 1
            cands.append((score(R, c, a.metric)[0], i))
        sc, i = max(cands)
        if best is not None and sc <= best + 1e-6:
            break
        best = sc; cnt[i] += 1; path.append((a.files[i], round(sc, 5)))
        print("ekle", a.files[i], "dev", a.metric, round(sc, 5), flush=True)
    _, w, T = score(R, cnt, a.metric)
    out = {"protocol": f"üye seçimi + ağırlık yalnız dev ({'+'.join(DEV)}); sıcaklık cal; test yalnız rapor",
           "members": [{"file": f, "weight": round(float(x), 4), "model": r.get("model"), "lora": r.get("lora")}
                       for f, r, x in zip(a.files, R, w) if x],
           "temperature": round(T, 2), "selection_path": path}
    for s in DEV + TEST:
        if not all(s in runs(r) for r, x in zip(R, w) if x):
            continue
        out[s] = metrics(combo(R, w, s), T)
        print(s, json.dumps(out[s]["all"]), flush=True)
    for f, r in zip(a.files, R):                       # tek üyeler (karşılaştırma için)
        out.setdefault("single", {})[f] = {s: runs(r)[s]["all"]["accuracy"] for s in DEV + TEST if s in runs(r)}
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
