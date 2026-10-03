"""AUBIN-Learn çoklu beceri birleştirmesi (kaynak-başı, YALNIZ kev_dev'de seçilir; kev_test / kev_transfer_test yalnız rapor).

Ansambl = select_members.py çıktısı (cal300'de seçilmiş üyeler/ağırlıklar/sıcaklık); kev_dev'i ayrı ölçülen üyeler için
birleşik dosyalar (merge_dev.py). Beceriler (learn_skill2/3 çıktıları) Kaggle kernel çıktısından BELLEĞE okunur (disk yazılmaz).
Her kaynak için adaylar: her beceri tek başına + hepsinin ortalaması × ağırlık ızgarası; kapı: dev doğruluğu ≥ +2 soru VE
dev log-kaybı düşmeli, yoksa beceri o kaynakta kapalı.

    python fuse_multi.py --sel runs/sat/sel_final_nll.json --member e12b=runs/close/m/e12b.json --member e12d=runs/close/m/e12d.json \
        --skill hesap:kullanıcı/kernel:dosya.json ... [--out_hf emrevrg/AUBIN-12B:reports/x.json]
"""
import argparse, io, json, os, sys
import numpy as np
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ensemble import metrics
from select_members import _n
from aubin.learn import fuse

W = [0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 1000.0]   # 1000 ≈ yalnız beceri (kaynak becerisi modele baskın)
SU = ("kev_dev", "kev_test", "kev_transfer_test")


def kernel_files(cfg, user, slug):
    """Kernel çıktı dosyaları {ad: url}. kaggle.json varsa REST (temel kimlik), yoksa OAuth (kagglesdk)."""
    p = os.path.expanduser(f"~/.kaggle/{cfg}/kaggle.json")
    if os.path.exists(p):
        c = json.load(open(p))
        j = requests.get("https://www.kaggle.com/api/v1/kernels/output", params={"userName": user, "kernelSlug": slug},
                         auth=(c["username"], c["key"]), timeout=60).json()
        return {f["fileName"]: f["url"] for f in j["files"]}
    os.environ["KAGGLE_CONFIG_DIR"] = os.path.expanduser(f"~/.kaggle/{cfg}")
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest as R
    api = KaggleApi(); api.authenticate()
    with api.build_kaggle_client() as k:
        r = R(); r.user_name = user; r.kernel_slug = slug
        return {f.file_name: f.url for f in k.kernels.kernels_api_client.list_kernel_session_output(r).files}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sel", required=True); ap.add_argument("--member", action="append", default=[])
    ap.add_argument("--skill", action="append", default=[]); ap.add_argument("--out_hf", default="")
    ap.add_argument("--pool", action="append", default=[])
    ap.add_argument("--sel_cal", action="store_true")      # seçim kümesine cal300 eklenir
    ap.add_argument("--crit", default="nll", choices=["nll", "acc", "acc_strict"])   # kaynak-başı beceri seçim ölçütü (yalnız kev_dev)   # kaynak-başı üye seçimi için aday üyeler (kev_dev'li dosyalar)
    a = ap.parse_args()
    S = json.load(open(a.sel, encoding="utf-8"))
    mp = dict(x.split("=", 1) for x in a.member)
    M = []
    for m in S["members"]:
        nm = os.path.splitext(os.path.basename(m["file"].replace("\\", "/")))[0]
        M.append((m["weight"], json.load(open(mp.get(nm, m["file"].replace("\\", "/")), encoding="utf-8"))))
    T = S["temperature"]
    SK, cache = {}, {}
    for s in a.skill:                                   # hesap:kullanıcı/kernel:dosya
        cfg, ks, fn = s.split(":")
        if ks not in cache:
            cache[ks] = kernel_files(cfg, *ks.split("/"))
        SK[f"{ks.split('/')[1]}/{fn}"] = requests.get(cache[ks][fn], timeout=300).json()["suites"]
    print("beceriler", list(SK), flush=True)

    def ens(s, MM=None):
        MM = MM or M
        its = [r["runs"][s]["items"] for _, r in MM]
        return [(its[0][i]["src"], its[0][i]["y"], sum(w * _n(x[i]["lp"], r) for (w, r), x in zip(MM, its))) for i in range(len(its[0]))]
    E = {s: ens(s) for s in SU}
    R_ps = {}
    if a.pool:                                         # kaynak-başı üye karışımı (YALNIZ kev_dev'de seçilir, kapılı)
        pool = [(os.path.splitext(os.path.basename(f))[0], json.load(open(f, encoding="utf-8"))) for f in a.pool]
        cands = {"base": None}
        for nm, r in pool:
            cands["only:" + nm] = [(1.0, r)]
        for j in range(len(M)):
            cands[f"drop:{j}"] = [(1.0 / (len(M) - 1), r) for k, (_, r) in enumerate(M) if k != j]
        cands["all"] = [(1.0 / len(pool), r) for _, r in pool]
        CE = {c: {s: (E[s] if v is None else ens(s, v)) for s in SU} for c, v in cands.items()}
        pick = {}
        for src in sorted({x[0] for x in E["kev_dev"]}):
            idx = [i for i, x in enumerate(E["kev_dev"]) if x[0] == src]

            def evc(c):
                rows = [CE[c]["kev_dev"][i] for i in idx]
                f = [fuse(lp, None, 0.0, T) for _, _, lp in rows]
                return float(np.mean([-fi[y] for fi, (_, y, _) in zip(f, rows)])), sum(int(np.argmax(fi) == y) for fi, (_, y, _) in zip(f, rows))
            n0, a0 = evc("base"); c = min(cands, key=lambda c: evc(c)[0]); n1, a1 = evc(c)
            pick[src] = c if (a1 >= a0 + 2 and n1 < n0) else "base"
            R_ps[src] = {"pick": pick[src], "dev_acc": [round(a0 / len(idx), 4), round(a1 / len(idx), 4)]}
            print("üye", src, R_ps[src], flush=True)
        for s in SU:
            E[s] = [CE[pick.get(E[s][i][0], "base")][s][i] for i in range(len(E[s]))]
    for s in ("kev_dev", "kev_test"):
        for n in SK:
            for i, (_, _, lp) in enumerate(E[s]):
                p = SK[n][s]["p"][i]
                assert p is None or len(p) == len(lp), f"hizalama {n} {s} #{i}"
    names = list(SK) + ["mean"]
    SELN = "kev_dev"
    if a.sel_cal:                                      # seçim kümesi = kev_dev + cal300 (kalibrasyon bölümü; testle çakışmaz)
        import random as _r
        keys = next(iter(SK.values()))["kev_cal"]["keys"]
        perm = list(range(len(keys))); _r.Random(0).shuffle(perm); cidx = perm[:300]   # learn_skill/select_members ile aynı cal300
        cal = [r["runs_cal_items"] for _, r in M]
        crow = [(keys[j].split("/")[1], cal[0][k]["y"], sum(w * _n(c[k]["lp"], r) for (w, r), c in zip(M, cal))) for k, j in enumerate(cidx)]
        for n in SK:
            for (_, _, lp), j in zip(crow, cidx):
                p = SK[n]["kev_cal"]["p"][j]
                assert p is None or len(p) == len(lp), f"cal hizalama {n}"
            SK[n]["SEL"] = {"p": SK[n]["kev_dev"]["p"] + [SK[n]["kev_cal"]["p"][j] for j in cidx]}
        E["SEL"] = E["kev_dev"] + crow; SELN = "SEL"
        print("seçim kümesi kev_dev+cal300:", len(E["SEL"]), flush=True)

    def P(s, name, i):
        if name is None:
            return None
        if name == "mean":
            v = [np.asarray(SK[n][s]["p"][i]) for n in SK if s in SK[n] and SK[n][s]["p"][i] is not None]
            return np.mean(v, 0) if v else None
        p = SK[name].get(s, {}).get("p", [None] * len(E[s]))[i]
        return None if p is None else np.asarray(p)

    choice, gate = {}, {}
    for src in sorted({x[0] for x in E["kev_dev"]}):
        idx = [i for i, x in enumerate(E[SELN]) if x[0] == src]

        def ev(n, w):
            nl = ac = 0
            for i in idx:
                _, y, lp = E[SELN][i]; f = fuse(lp, P(SELN, n, i), w, T); nl -= f[y]; ac += int(np.argmax(f) == y)
            return nl / len(idx), ac
        n0, a0 = ev(None, 0.0)
        allc = [(ev(n, w), n, w) for n in names for w in W]
        if a.crit == "nll":
            (n1, a1), bn, bw = min(allc, key=lambda z: z[0][0])
        else:                                          # acc: log-kaybı düşürenler içinde en yüksek dev doğruluğu (eşitlikte log-kayıp)
            ok = allc if a.crit == "acc_strict" else ([z for z in allc if z[0][0] < n0] or allc)
            (n1, a1), bn, bw = min(ok, key=lambda z: (-z[0][1], z[0][0]))
        on = (a1 >= a0 + max(4, 0.03 * len(idx))) if a.crit == "acc_strict" else (a1 >= a0 + 2 and n1 < n0)   # acc_strict: ≥4 soru ve ≥%3
        choice[src] = (bn, bw) if on else (None, 0.0)
        gate[src] = {"n": len(idx), "dev_acc": [round(a0 / len(idx), 4), round(a1 / len(idx), 4)], "dev_nll": [round(n0, 4), round(n1, 4)],
                     "skill": bn, "w": bw, "on": on}
        print(src, gate[src], flush=True)
    R = {"protocol": __doc__.split("    python")[0].strip(), "ensemble": S["members"], "temperature": T, "skills": list(SK),
         "choice_on_kev_dev": choice, "gate": gate}
    FU = {s: [(src, y, fuse(lp, P(s, choice.get(src, (None, 0))[0], i), choice.get(src, (None, 0))[1], T))
              for i, (src, y, lp) in enumerate(E[s])] for s in SU + ((SELN,) if SELN != "kev_dev" else ())}
    # kaynak-başı sıcaklık (YALNIZ kev_dev): beceri açılan kaynaklarda birleşik çıktının güveni yeniden ölçeklenir — doğruluk değişmez
    TG = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 40.0, 80.0, 150.0, 300.0, 600.0, 1200.0]
    Ts = {}
    for src, (nm, _) in choice.items():
        if nm is None:
            continue
        rows = [r for r in FU[SELN] if r[0] == src]
        Ts[src] = min(TG, key=lambda t: float(np.mean([-(lp / t - np.logaddexp.reduce(lp / t))[y] for _, y, lp in rows])))
    R["source_temperature_on_kev_dev"] = Ts
    for s in SU:
        base = [(src, y, np.asarray(lp) / T) for src, y, lp in E[s]]
        fu = [(src, y, lp / Ts.get(src, 1.0)) for src, y, lp in FU[s]]
        mb, mf = metrics(base, 1.0), metrics(fu, 1.0)
        R[s] = {"ensemble": mb["all"], "aubin_learn": mf["all"],
                "by_source": {k: [v["n"], mb["by_source"][k]["accuracy"], v["accuracy"]] for k, v in mf["by_source"].items()}}
        print(s, "ansambl", mb["all"]["accuracy"], "-> AUBIN-Learn", mf["all"]["accuracy"], "| nll", mb["all"]["nll"], "->", mf["all"]["nll"], flush=True)
    print(json.dumps(R["kev_test"]["by_source"]))
    if a.out_hf:                                       # sonuç dosyası diske değil doğrudan HF'ye
        repo, path = a.out_hf.split(":", 1)
        from huggingface_hub import HfApi
        HfApi(token=open(os.path.join(HERE, "..", ".hf_token")).read().strip()).upload_file(
            path_or_fileobj=io.BytesIO(json.dumps(R, indent=1).encode()), path_in_repo=path, repo_id=repo, commit_message="AUBIN-Learn skills fusion result")
        print("HF", repo, path)


if __name__ == "__main__":
    main()
