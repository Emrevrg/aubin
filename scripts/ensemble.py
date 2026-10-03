"""AUBIN ansambl + kalibrasyon (GPU'suz): kev_llm.py çıktılarındaki soru-başı log-olasılıkları birleştirir.

    python ensemble.py out.json run1.json run2.json [--tag runs] [--weights 1,1]
Birleştirme: ağırlıklı log-olasılık ortalaması; sıcaklık YALNIZ calibration (decision-v7 cal) üzerinde seçilir.
"""
import argparse, json, math
import numpy as np


def softmax(x):
    x = np.asarray(x, dtype=np.float64); x = x - x.max(); e = np.exp(x); return e / e.sum()


def metrics(rows, T):
    by, allr = {}, []
    for src, y, lp in rows:
        p = softmax(np.asarray(lp) / T); oh = np.zeros(len(p)); oh[y] = 1
        r = (int(p.argmax()) == y, float(((p - oh) ** 2).sum()), float(p.max()), -math.log(max(p[y], 1e-12)))
        allr.append(r); by.setdefault(src, []).append(r)
    def m(v):
        ece = 0.0
        for j in range(15):
            bb = [x for x in v if j / 15 < x[2] <= (j + 1) / 15]
            if bb:
                ece += len(bb) / len(v) * abs(np.mean([x[2] for x in bb]) - np.mean([x[0] for x in bb]))
        return {"n": len(v), "accuracy": round(float(np.mean([x[0] for x in v])), 4), "brier": round(float(np.mean([x[1] for x in v])), 4),
                "nll": round(float(np.mean([x[3] for x in v])), 4), "ece15": round(float(ece), 4)}
    return {"all": m(allr), "by_source": {k: m(v) for k, v in sorted(by.items())}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("runs", nargs="+"); ap.add_argument("--tag", default="runs")
    ap.add_argument("--weights", default="")
    a = ap.parse_args()
    R = [json.load(open(f)) for f in a.runs]
    w = [float(x) for x in a.weights.split(",")] if a.weights else [1.0] * len(R)
    w = np.array(w) / sum(w)
    cal = [r.get(a.tag + "_cal_items") for r in R]
    out = {"members": [{"file": f, "model": r.get("model"), "lora": r.get("lora")} for f, r in zip(a.runs, R)], "weights": w.tolist()}
    T = 1.0
    if all(cal):
        comb = [(None, c[0]["y"], sum(wi * np.asarray(ci[k]["lp"]) for wi, ci in zip(w, cal))) for k, c in enumerate(zip(*cal))]
        T = min(np.concatenate([np.arange(0.3, 4.0, 0.05), np.arange(4.0, 20.01, 0.25)]), key=lambda t: metrics([(0, y, lp) for _, y, lp in comb], t)["all"]["nll"])
    out["temperature"] = round(float(T), 2)
    suites = set.intersection(*[set(r[a.tag].keys()) if a.tag in r else set(r["runs"].keys()) for r in R])
    for s in sorted(suites):
        its = [(r[a.tag] if a.tag in r else r["runs"])[s]["items"] for r in R]
        assert all(len(x) == len(its[0]) for x in its), "üyelerin madde sayısı farklı"
        rows = []
        for k in range(len(its[0])):
            ys = {x[k]["y"] for x in its}; assert len(ys) == 1, "madde sırası uyuşmuyor"
            rows.append((its[0][k]["src"], its[0][k]["y"], sum(wi * np.asarray(x[k]["lp"]) for wi, x in zip(w, its))))
        out[s] = metrics(rows, T)
        print(s, json.dumps(out[s]["all"]))
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
