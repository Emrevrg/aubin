"""Kev eğitim kaynaklarından EK eğitim verisi: aynı HF veri setlerinin eğitim bölümünden, Kev'in HİÇBİR kümesiyle çakışmayan satırlar.

Kev decision-v7/train her kaynaktan 1000 satır kullanır (_meta: repo, revision, split, row). Bu betik, her kaynak için
  (1) durum metninin orijinal satırdan nasıl kurulduğunu, (2) orijinal etiket -> soru etiketi eşlemesini,
  (3) soru metnine giren satıra-özgü alanları (MNLI hipotezi, BoolQ sorusu)
Kev'in kendi satırlarından otomatik öğrenir; sonra yeni satırları rastgele bir Kev şablonuyla aynı biçime çevirir.
Sızıntı yok: Kev train/dev/cal/test/transfer kümelerindeki tüm durum metinleri (normalize hash) ve kullanılan satır numaraları dışlanır.

    python kev_augment.py --work /tmp/kev --per_source 4000 --out /content/kev_extra.jsonl
    KEV_EXTRA_TRAIN=/content/kev_extra.jsonl python kev_llm.py ... --train N
"""
import argparse, collections, copy, hashlib, json, os, random, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import kevdata

# hata oranına göre ağırlık (AUBIN Duo'nun dev'de en zayıf olduğu kaynaklar daha çok örnek alır)
WEIGHT = {"sst5": 2.0, "yelp": 2.0, "amazon": 2.0, "banking77": 1.5, "trec": 1.0, "agnews": 1.0, "mnli": 1.0,
          "imdb": 0.75, "dbpedia14": 0.75, "boolq": 1.0}


def clean(s):
    return re.sub(r"<br\s*/?>", " ", str(s))


def norm(s):
    return re.sub(r"\s+", " ", clean(s)).strip().lower()


def h(s):
    return hashlib.sha256(norm(s).encode()).hexdigest()


def state_text(st):
    return json.dumps(st, ensure_ascii=False, sort_keys=True) if isinstance(st, (dict, list)) else str(st)


def builders(row):
    """Bir veri seti satırından aday durum kurucular: tek alan ve iki alanlı birleşimler."""
    strs = {k: v for k, v in row.items() if isinstance(v, str) and v.strip()}
    out = {f"f:{k}": (lambda r, k=k: r[k]) for k in strs}
    ks = list(strs)
    for a in ks:
        for b in ks:
            if a != b:
                for sep in (" ", "\n", "\n\n", ". ", ": ", " - "):
                    out[f"j:{a}|{b}|{sep}"] = (lambda r, a=a, b=b, sep=sep: r[a] + sep + r[b])
    return out


def learn_builder(recs, ds):
    """Durum = alan (temizlenmiş) ya da onun başı (Kev uzun metinleri kısaltmış olabilir). Dönüş: (ad, eşleşme, azami uzunluk)."""
    votes, maxlen = collections.Counter(), 0
    for r in recs[:200]:
        row = ds[int(r["_meta"]["row"])]
        st = r["state"]
        if not isinstance(st, str):
            continue
        maxlen = max(maxlen, len(st))
        for name, f in builders(row).items():
            try:
                full, s = norm(f(row)), norm(st)
                if full == s or (len(s) >= 200 and full.startswith(s[: max(1, len(s) - 3)].rstrip(". "))):
                    votes[name] += 1
            except Exception:
                pass
    if not votes:
        return None, 0, 0
    name, n = votes.most_common(1)[0]
    return name, n, maxlen


def make_builder(name, maxlen=0):
    if name.startswith("f:"):
        k = name[2:]
        raw = lambda r: r[k]
    else:
        a, b, sep = name[2:].split("|", 2)
        raw = lambda r: r[a] + sep + r[b]

    def f(r):
        s = re.sub(r"[ \t]+", " ", clean(raw(r))).strip()
        if maxlen and len(s) > maxlen:                    # Kev'in en uzun durumu kadar, kelime sınırında kes
            s = s[:maxlen].rsplit(" ", 1)[0]
        return s
    return f


def q_text(q):
    ins = q.get("instructions")
    return ins.get("question", "") if isinstance(ins, dict) else (ins or "")


def set_q_text(q, old, new):
    ins = q.get("instructions")
    if isinstance(ins, dict):
        ins = dict(ins); ins["question"] = ins.get("question", "").replace(old, new); q["instructions"] = ins
    else:
        q["instructions"] = (ins or "").replace(old, new)


def learn_maps(recs, ds):
    """Her soru anahtarı için: (orijinal etiket alanı, eşleme sözlüğü) ve soru metnine giren alan."""
    maps, qfield = {}, {}
    qkeys = collections.Counter(k for r in recs for k in r["questions"])
    for qk, _ in qkeys.items():
        rows = [(r, ds[int(r["_meta"]["row"])]) for r in recs if qk in r["questions"]][:600]
        best = None
        n_lab = len({json.dumps(r["questions"][qk]["label"]) for r, _ in rows})
        cand = [k for k, v in rows[0][1].items() if isinstance(v, (int, bool)) or (isinstance(v, str) and len(v) < 40)]
        for f in cand:
            n_val = len({json.dumps(row.get(f)) for _, row in rows})
            if n_val > max(30, 3 * n_lab) or n_val > 0.5 * len(rows):     # kimlik/metin gibi her satırda farklı alanlar etiket olamaz
                continue
            m, ok, bad = {}, 0, 0
            for r, row in rows:
                o, lab = row.get(f), r["questions"][qk]["label"]
                key = json.dumps(o)
                if key in m and m[key] != lab:
                    bad += 1
                else:
                    m[key] = lab; ok += 1
            score = ok - 5 * bad
            if bad <= 0.01 * len(rows) and (best is None or score > best[0]):
                best = (score, f, m)
        if best:
            maps[qk] = (best[1], best[2])
        # satıra özgü soru metni (ör. MNLI hipotezi, BoolQ sorusu)
        for f, v in rows[0][1].items():
            if isinstance(v, str) and len(v) > 8 and all(
                    norm(row.get(f, "")) and norm(row.get(f, "")) in norm(q_text(r["questions"][qk])) for r, row in rows[:50]):
                qfield[qk] = f
                break
    return maps, qfield


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/tmp/kev"); ap.add_argument("--per_source", type=int, default=4000)
    ap.add_argument("--out", default="kev_extra.jsonl"); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from datasets import load_dataset
    paths = kevdata.fetch(a.work)
    raw = {k: [json.loads(l) for l in open(p, encoding="utf-8")] for k, p in paths.items()}
    excl = {h(state_text(r["state"])) for rs in raw.values() for r in rs}
    used = collections.defaultdict(set)
    for rs in raw.values():
        for r in rs:
            m = r.get("_meta", {})
            if m.get("repo") and m.get("row") is not None:
                used[(m["repo"], m.get("split"))].add(int(m["row"]))
    by_src = collections.defaultdict(list)
    for r in raw["kev_train"]:
        m = r["_meta"]
        if m.get("repo") and m.get("split") == "train" and m.get("row") is not None and isinstance(r["state"], str):
            by_src[m["source"]].append(r)
    rng = random.Random(a.seed); out, report = [], {}
    for src, recs in sorted(by_src.items()):
        m0 = recs[0]["_meta"]
        try:
            ds = load_dataset(m0["repo"], split="train", revision=m0.get("revision"))
        except Exception as e:
            report[src] = f"veri seti yüklenemedi: {e}"[:160]; continue
        bname, bn, maxlen = learn_builder(recs, ds)
        if not bname or bn < 0.85 * min(200, len(recs)):
            report[src] = f"durum kurucu bulunamadı ({bname}, {bn})"; continue
        build = make_builder(bname, maxlen)
        maps, qfield = learn_maps(recs, ds)
        # doğrulama: eşleme Kev'in TÜM satırlarında doğru etiketi veriyor mu (≥ %98 değilse o soru kullanılmaz)
        verify = {}
        for qk, (f, mp) in list(maps.items()):
            rs = [r for r in recs if qk in r["questions"]]
            hit = sum(mp.get(json.dumps(ds[int(r["_meta"]["row"])].get(f))) == r["questions"][qk]["label"] for r in rs)
            verify[qk] = round(hit / max(1, len(rs)), 4)
            if verify[qk] < 0.98:
                maps.pop(qk)
        if not maps:
            report[src] = {"etiket eşlemesi doğrulanmadı": verify}; continue
        n_want = int(a.per_source * WEIGHT.get(src, 1.0))
        idx = [i for i in range(len(ds)) if i not in used[(m0["repo"], "train")]]
        rng.shuffle(idx); made = 0
        for i in idx:
            if made >= n_want:
                break
            row = ds[i]
            try:
                st = build(row)
            except Exception:
                continue
            if not isinstance(st, str) or len(st) < 3 or h(st) in excl:
                continue
            tpl = rng.choice(recs)
            qs = {}
            for qk, q in tpl["questions"].items():
                if qk not in maps:
                    continue
                f, mp = maps[qk]
                key = json.dumps(row.get(f))
                if key not in mp:
                    continue
                q2 = copy.deepcopy(q); q2["label"] = mp[key]
                if qk in qfield:
                    old = tpl_row_val = None
                    trow = ds[int(tpl["_meta"]["row"])]
                    old = trow.get(qfield[qk]); new = row.get(qfield[qk])
                    if not old or not new or norm(old) not in norm(q_text(q2)):
                        continue
                    set_q_text(q2, old, new)
                qs[qk] = q2
            if not qs:
                continue
            excl.add(h(st))
            out.append({"state": st, "questions": qs,
                        "_meta": {"source": src, "repo": m0["repo"], "split": "train", "row": i, "id": f"aug/{src}/{i}",
                                  "variant": "aug", "template": tpl["_meta"].get("id")}})
            made += 1
        report[src] = {"made": made, "builder": bname, "maxlen": maxlen, "verify": verify, "questions": {k: v[0] for k, v in maps.items()},
                       "qfield": qfield, "pool": len(idx)}
        print(src, json.dumps(report[src], ensure_ascii=False)[:300], flush=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    json.dump(report, open(a.out + ".report.json", "w"), indent=1, ensure_ascii=False)
    print("YAZILDI", a.out, len(out), flush=True)


if __name__ == "__main__":
    main()
