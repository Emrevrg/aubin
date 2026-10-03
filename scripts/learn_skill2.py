"""AUBIN-Learn Beceri-2: kaynakların AÇIK eğitim bölümünden (Kev train + kev_augment ek verisi, Kev'in hiçbir kümesiyle
çakışmaz) öğrenilen ince-ayarlı kodlayıcı becerisi. Tek kodlayıcı + (kaynak, soru) başına çıkış dilimi; girdi = soru + seçenek
metinleri + durum. Eğitim ayarları ÖNCEDEN sabit (dev/test'e bakılmaz); kev_dev yalnız sonra birleştirme ağırlığı seçiminde,
kev_test / kev_transfer_test yalnız rapor. Çıktı: her küme için soru sırasına hizalı olasılıklar (seçenek ADIyla eşlenmiş).

    KEV_EXTRA_TRAIN=kev_extra.jsonl python learn_skill2.py --work /tmp/kev --model microsoft/deberta-v3-base --out skill2.json
"""
import argparse, json, math, os, random, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import kevdata
from kev_llm import items

SUITES = ("kev_dev", "kev_test", "kev_transfer_test", "kev_cal")


def gkey(it):
    return it["src"] + "|" + it["key"].rsplit("/", 1)[1]


def text(it):
    # soru (kısa; MNLI hipotezi / BoolQ sorusu burada) → durum → seçenekler: kesilme yalnız seçenek listesinden olur
    # (ilk sürümde seçenekler öndeydi; banking77'de 77 seçenek durumu tamamen kesiyordu)
    return f"{it['q']}\n\n{it['state']}\n\nOptions: " + " ; ".join(it["texts"][:26])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/tmp/kev"); ap.add_argument("--model", default="microsoft/deberta-v3-base")
    ap.add_argument("--epochs", type=float, default=3); ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5); ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default="skill2.json")
    ap.add_argument("--micro", type=int, default=0)    # >0: mikro-yığın boyu (gradyan biriktirme + checkpointing)
    a = ap.parse_args()
    random.seed(a.seed); torch.manual_seed(a.seed)
    K = kevdata.load(a.work)
    tr = items(K["kev_train"])
    G = {}
    for it in tr:
        G.setdefault(gkey(it), set()).add(str(it["keys"][it["y"]]))
    cnt = {}
    for it in tr:
        cnt[gkey(it)] = cnt.get(gkey(it), 0) + 1
    G = {g: sorted(v) for g, v in G.items() if len(v) >= 2 and cnt[g] >= 50}
    off, tot = {}, 0
    for g in sorted(G):
        off[g] = tot; tot += len(G[g])
    tr = [it for it in tr if gkey(it) in G]
    print({"train_items": len(tr), "groups": len(G), "slots": tot}, flush=True)

    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(a.model)
    enc = AutoModel.from_pretrained(a.model, dtype=torch.float32).cuda().float()   # ana ağırlıklar fp32 (AMP fp16 yalnız hesapta)
    if a.micro:
        enc.gradient_checkpointing_enable()
    head = torch.nn.Linear(enc.config.hidden_size, tot).cuda()
    params = list(enc.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW([{"params": enc.parameters(), "lr": a.lr}, {"params": head.parameters(), "lr": a.lr * 10}], weight_decay=0.01)
    steps = int(math.ceil(len(tr) / a.bs) * a.epochs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / max(1, int(0.06 * steps))) * max(0.0, (steps - s) / steps))
    scaler = torch.cuda.amp.GradScaler()

    def batch(its):
        b = tok([text(x) for x in its], truncation=True, max_length=a.max_len, padding=True, return_tensors="pt")
        return {k: v.cuda() for k, v in b.items()}

    def logits(b, its):
        h = enc(**b).last_hidden_state[:, 0]
        z = head(h).float()
        m = torch.full_like(z, -1e4)
        for i, x in enumerate(its):
            g = gkey(x); m[i, off[g]: off[g] + len(G[g])] = 0
        return z + m

    t0 = time.time(); s = 0; enc.train()
    while s < steps:
        order = list(range(len(tr))); random.shuffle(order)
        for i in range(0, len(order), a.bs):
            if s >= steps:
                break
            its = [tr[j] for j in order[i: i + a.bs]]
            opt.zero_grad(set_to_none=True)
            for j in range(0, len(its), a.micro or len(its)):         # gradyan biriktirme (büyük kodlayıcılar T4'e sığsın)
                mb = its[j: j + (a.micro or len(its))]
                y = torch.tensor([off[gkey(x)] + G[gkey(x)].index(str(x["keys"][x["y"]])) for x in mb]).cuda()
                with torch.autocast("cuda", dtype=torch.float16):
                    z = logits(batch(mb), mb)
                loss = torch.nn.functional.cross_entropy(z, y) * len(mb) / len(its)
                scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt); scaler.update(); sched.step(); s += 1
            if s % 200 == 0:
                print(f"adım {s}/{steps} kayıp {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)

    enc.eval(); R = {"protocol": __doc__.split("    KEV")[0].strip(), "model": a.model, "epochs": a.epochs, "lr": a.lr,
                     "max_len": a.max_len, "train_items": len(tr), "train_minutes": round((time.time() - t0) / 60, 1), "suites": {}}
    for sname in SUITES:
        if sname not in K:
            continue
        ev = items(K[sname]); P = []
        with torch.no_grad():
            for i in range(0, len(ev), 64):
                its = ev[i: i + 64]
                ok = [x for x in its if gkey(x) in G]
                pr = {}
                if ok:
                    with torch.autocast("cuda", dtype=torch.float16):
                        z = logits(batch(ok), ok)
                    for x, row in zip(ok, torch.log_softmax(z, -1).exp().cpu().numpy()):
                        g = gkey(x); pos = {c: j for j, c in enumerate(G[g])}
                        p = np.array([row[off[g] + pos[str(k)]] if str(k) in pos else 0.0 for k in x["keys"]], dtype=np.float64)
                        pr[x["key"]] = (p / p.sum()).round(5).tolist() if p.sum() > 0 else None
                P += [pr.get(x["key"]) for x in its]
        acc = [int(np.argmax(p)) == x["y"] for p, x in zip(P, ev) if p is not None]
        bys = {}
        for p, x in zip(P, ev):
            if p is not None:
                bys.setdefault(x["src"], []).append(int(np.argmax(p)) == x["y"])
        R["suites"][sname] = {"keys": [x["key"] for x in ev], "p": P, "covered": len(acc), "n": len(ev),
                              "skill_acc_covered": round(float(np.mean(acc)), 4) if acc else None,
                              "by_source": {k: [len(v), round(float(np.mean(v)), 4)] for k, v in sorted(bys.items())}}
        print(sname, R["suites"][sname]["covered"], R["suites"][sname]["skill_acc_covered"], R["suites"][sname]["by_source"], flush=True)
        json.dump(R, open(a.out, "w"))
    print("BITTI", a.out, flush=True)


if __name__ == "__main__":
    main()
