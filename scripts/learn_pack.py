"""Kayıtlı koşuların soru-başı log-olasılıklarını AUBIN-Learn ölçümü için tek sıkıştırılmış pakete toplar (+ dev'de eşit
ağırlıklı ansambl). Çıktı: {ad: {"T": sıcaklık, "suites": {küme: {"lp": [...], "y": [...]}}}}

    python learn_pack.py out.json.gz runs/r12l/a12_b.json runs/f31l2/a31lora2.json ...
"""
import gzip, json, os, sys
import numpy as np


def lsm(x):
    x = np.asarray(x, dtype=np.float64); return x - np.logaddexp.reduce(x)


def main():
    out, files = sys.argv[1], sys.argv[2:]
    P = {}
    for f in files:
        j = json.load(open(f, encoding="utf-8")); r = j["runs"]
        name = os.path.splitext(os.path.basename(f))[0]
        S = {s: {"lp": [[round(float(v), 3) for v in it["lp"]] for it in r[s]["items"]], "y": [it["y"] for it in r[s]["items"]]}
             for s in r if isinstance(r[s], dict) and "items" in r[s]}
        if j.get("runs_cal_items"):
            S["cal300"] = {"lp": [[round(float(v), 3) for v in it["lp"]] for it in j["runs_cal_items"]], "y": [it["y"] for it in j["runs_cal_items"]]}
        P[name] = {"T": float(j.get("runs_temperature", 1.0)), "model": j.get("model"), "suites": S}
    mem = [n for n in P if "kev_dev" in P[n]["suites"] and "kev_test" in P[n]["suites"]]
    if len(mem) >= 2:                                  # eşit ağırlıklı ansambl (sıcaklık-ölçekli, normalize)
        S = {}
        for s in set.intersection(*[set(P[n]["suites"]) for n in mem]):
            lps = [[lsm(np.asarray(x) / P[n]["T"]) for x in P[n]["suites"][s]["lp"]] for n in mem]
            S[s] = {"lp": [[round(float(v), 3) for v in np.mean([l[i] for l in lps], 0)] for i in range(len(lps[0]))],
                    "y": P[mem[0]]["suites"][s]["y"]}
        P["ens_" + "+".join(mem)] = {"T": 1.0, "model": "ensemble", "suites": S}
    sel = os.environ.get("AUBIN_SEL", "")             # select_members.py çıktısı: dev'de seçilmiş ağırlıklı ansambl
    if sel:
        Sj = json.load(open(sel, encoding="utf-8"))
        ms = [(os.path.splitext(os.path.basename(m["file"]))[0], m["weight"]) for m in Sj["members"]]
        S = {}
        for s in set.intersection(*[set(P[n]["suites"]) for n, _ in ms]):
            lps = [([lsm(np.asarray(x) / P[n]["T"]) for x in P[n]["suites"][s]["lp"]], w) for n, w in ms]
            S[s] = {"lp": [[round(float(v), 3) for v in sum(w * l[i] for l, w in lps)] for i in range(len(lps[0][0]))],
                    "y": P[ms[0][0]]["suites"][s]["y"]}
        P["sel_" + "+".join(n for n, _ in ms)] = {"T": float(Sj.get("temperature", 1.0)), "model": "ensemble(dev-selected)", "suites": S}
    json.dump(P, gzip.open(out, "wt", encoding="utf-8"))
    print(out, os.path.getsize(out), {n: sorted(P[n]["suites"]) for n in P})


if __name__ == "__main__":
    main()
