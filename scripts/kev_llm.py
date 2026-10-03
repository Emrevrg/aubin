"""AUBIN by Norovox — açık-tabanlı karar modeli, Kev'in AÇIK paketleriyle AYNI protokolde ölçülür.

Tipli soru → tek ileri geçiş → seçenek olasılıkları (metin üretimi yok). Seçenekler harflerle listelenir, cevap
konumundaki harf logitleri okunur (>26 seçenekte 26'lık gruplar + final turu). Sıcaklık decision-v7 calibration'da.
Eğitim (--train): LoRA, kayıp = harf logitleri üzerinde çapraz-entropi (Kev'in pointer-head'inin LM-başlı karşılığı).
Kev'in transfer kaynakları (mmlu, sciq, qnli, paws, emotion, tweet_offensive...) EĞİTİMDE ASLA kullanılmaz.

    python kev_llm.py --model google/gemma-4-12B-it --suites kev_transfer_test,kev_test --out r.json [--train N --adapter out/]
"""
import argparse, json, math, os, random, sys, time
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kevdata

L = [chr(65 + i) for i in range(26)]


def options(q):
    t, c = q["type"], q.get("criteria")
    if t == "noul":
        keys = ["false", "true"]
        if isinstance(c, dict):
            return keys, [f"false — {c.get('false', 'no')}", f"true — {c.get('true', 'yes')}"]
        return keys, ["false — no", "true — yes"]
    if t == "choice":
        keys = list(c.keys())
        return keys, [f"{k}: {v}" for k, v in c.items()]
    if isinstance(c, list):
        return [str(i) for i in range(len(c))], [str(v) for v in c]
    keys = list(c.keys())
    return keys, [f"{k}: {v}" for k, v in c.items()]


def items(cases):
    out = []
    for c in cases:
        st = json.dumps(c["state"], ensure_ascii=False)
        for qid, q in c["questions"].items():
            keys, texts = options(q)
            lab = str(c["gold"][qid]["label"])
            if lab not in keys:
                continue
            out.append({"key": f"{c['id']}/{qid}", "src": c["source"], "state": st, "q": q.get("instructions") or qid, "type": q["type"],
                        "keys": keys, "texts": texts, "y": keys.index(lab)})
    return out


class Scorer:
    def __init__(self, model_id, four_bit=True, max_state=6000, device_map=""):
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.dev = "cuda:0" if torch.cuda.is_available() else "cpu"
        kw = dict(device_map=device_map or self.dev, dtype=torch.float16 if self.dev != "cpu" else torch.float32)
        if device_map == "auto" and torch.cuda.device_count() > 1:     # 31B 4-bit iki T4'e bölünür
            kw["max_memory"] = {i: "13GiB" for i in range(torch.cuda.device_count())}
        if four_bit:
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                           bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
        try:
            self.m = AutoModelForCausalLM.from_pretrained(model_id, **kw)
        except Exception as e:
            print("[AutoModelForCausalLM olmadı]", str(e)[:200], flush=True)
            from transformers import AutoModelForImageTextToText
            self.m = AutoModelForImageTextToText.from_pretrained(model_id, **kw)
        self.m.eval()
        self.max_state = max_state
        self.letter_ids = []
        for a in L:
            ids = {self.tok.encode(a, add_special_tokens=False)[0], self.tok.encode(" " + a, add_special_tokens=False)[-1]}
            self.letter_ids.append(sorted(ids))

    def prompt(self, it, idx):
        st = it["state"]
        if len(st) > self.max_state:
            st = st[: self.max_state] + " …"
        opts = "\n".join(f"{L[j]}) {it['texts'][i]}" for j, i in enumerate(idx))
        u = (f"Decide based ONLY on the state below.\n\nSTATE:\n{st}\n\nQUESTION: {it['q']}\n\nOPTIONS:\n{opts}\n\n"
             f"Reply with the single letter of the correct option.")
        msgs = [{"role": "user", "content": u}]
        try:
            p = self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        except Exception:
            p = self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        return p + "Answer: "

    def think_prompt(self, it):
        """Eminsiz sorular için: kısa akıl yürütme, sonra 'Answer: <harf>'."""
        st = it["state"] if len(it["state"]) <= self.max_state else it["state"][: self.max_state] + " …"
        opts = "\n".join(f"{L[j]}) {t}" for j, t in enumerate(it["texts"]))
        u = (f"Decide based ONLY on the state below.\n\nSTATE:\n{st}\n\nQUESTION: {it['q']}\n\nOPTIONS:\n{opts}\n\n"
             f"Think briefly step by step (at most 6 short sentences), weighing the most plausible options. "
             f"Then write the final line exactly as 'Answer: <letter>'.")
        msgs = [{"role": "user", "content": u}]
        try:
            return self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        except Exception:
            return self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)

    @torch.no_grad()
    def think_scores(self, its, max_new=256, bs=8, samples=1, temp=0.7):
        """Toplu akıl yürütme üretimi → 'Answer: ' konumunda harf logitleri. Dönüş: soru-başı log-olasılık listesi.
        samples>1: self-consistency — açgözlü yol + (samples-1) örneklenmiş akıl yürütme; cevap olasılıkları ortalanır."""
        out = []
        for i in range(0, len(its), bs):
            batch = its[i:i + bs]
            ps = [self.think_prompt(it) for it in batch]
            enc = self.tok(ps, return_tensors="pt", padding=True, add_special_tokens=False).to(self.dev)
            runs = [self.m.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=self.tok.pad_token_id)]
            for _ in range(samples - 1):
                runs.append(self.m.generate(**enc, max_new_tokens=max_new, do_sample=True, temperature=temp, top_p=0.95,
                                            pad_token_id=self.tok.pad_token_id))
            probs = [None] * len(batch)
            for gen in runs:
                texts = self.tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
                conts = []
                for p, t in zip(ps, texts):
                    k = t.find("Answer:")
                    reason = (t[:k] if k >= 0 else t).rstrip()
                    conts.append(p + reason + "\nAnswer: ")
                lg = self.letter_logits(conts)
                for b, it in enumerate(batch):
                    pr = torch.softmax(lg[b, :len(it["keys"])].float().cpu(), -1)
                    probs[b] = pr if probs[b] is None else probs[b] + pr
            for b in range(len(batch)):
                out.append(torch.log(probs[b] / len(runs) + 1e-12))
        return out

    def letter_logits(self, prompts, grad=False):
        enc = self.tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.dev)
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            try:                                         # yalnız son konumun logiti (262k sözlükte tüm konumlar = GB'larca bellek)
                out = self.m(**enc, logits_to_keep=1)
            except TypeError:
                out = self.m(**enc)
            last = out.logits[:, -1].float()
        return torch.stack([last[:, ids].logsumexp(-1) for ids in self.letter_ids], -1)   # [B, 26]

    def score(self, it, perms=1):
        n = len(it["keys"])
        idx = list(range(n))
        if n <= 26:
            # konum yanlılığına karşı: P döngüsel kaydırmada sorulur, seçenek-başı log-olasılıkların ortalaması
            acc = torch.zeros(n)
            shifts = sorted({(k * n) // perms for k in range(perms)})
            for s in shifts:
                order = idx[s:] + idx[:s]
                lg = torch.log_softmax(self.letter_logits([self.prompt(it, order)])[0, :n].cpu(), -1)
                back = torch.empty(n); back[torch.tensor(order)] = lg
                acc += back
            return acc / len(shifts)
        # >26 seçenek: 26'lık gruplar, her grubun en iyisi finalde yarışır; olasılık final dağılımından
        winners, full = [], torch.full((n,), -1e4)
        for s in range(0, n, 26):
            g = idx[s:s + 26]
            lg = self.letter_logits([self.prompt(it, g)])[0, :len(g)].cpu()
            full[torch.tensor(g)] = lg
            winners.append(g[int(lg.argmax())])
        lg = self.letter_logits([self.prompt(it, winners)])[0, :len(winners)].cpu()
        full[torch.tensor(winners)] = lg + 50.0
        return full


def metrics(rows, T=1.0):
    by, allr = {}, []
    for it, lg in rows:
        p = F.softmax(lg / T, -1).numpy()
        ok = int(p.argmax()) == it["y"]; oh = np.zeros(len(p)); oh[it["y"]] = 1
        r = (ok, float(((p - oh) ** 2).sum()), float(p.max()), float(-math.log(max(p[it["y"]], 1e-12))))
        allr.append(r); by.setdefault(it["src"], []).append(r)
    def m(v):
        ece = 0.0
        for j in range(15):
            bb = [x for x in v if j / 15 < x[2] <= (j + 1) / 15]
            if bb:
                ece += len(bb) / len(v) * abs(np.mean([x[2] for x in bb]) - np.mean([x[0] for x in bb]))
        return {"n": len(v), "accuracy": round(float(np.mean([x[0] for x in v])), 4), "brier": round(float(np.mean([x[1] for x in v])), 4),
                "nll": round(float(np.mean([x[3] for x in v])), 4), "ece15": round(float(ece), 4)}
    return {"all": m(allr), "by_source": {k: m(v) for k, v in sorted(by.items())}}


def resolve_adapter(path):
    """Adaptör dizinini bul: verilen yol yoksa /kaggle/input altında aynı adlı (adapter_config.json içeren) dizini ara."""
    import glob
    if not path or os.path.exists(os.path.join(path, "adapter_config.json")):
        return path
    if path.startswith("hf:"):                        # herkese açık HF deposundaki adaptör (hesaplar arası paylaşım)
        from huggingface_hub import snapshot_download
        repo, _, sub = path[3:].partition("//")
        pre = sub + "/" if sub else ""
        d = snapshot_download(repo, allow_patterns=[pre + "adapter_*", pre + "aubin.json"])   # aubin.json: taban + sıcaklık
        return os.path.join(d, sub) if sub else d
    base = os.path.basename(path.rstrip("/"))
    hits = [os.path.dirname(h) for h in glob.glob("/kaggle/input/**/adapter_config.json", recursive=True)]
    named = [h for h in hits if os.path.basename(h) == base]
    found = (named or hits or [path])[0]
    print("[adaptör yolu]", path, "->", found, "| adaylar:", hits[:6], flush=True)
    return found


def train_lora(S, tr, steps, lr, bs, out_dir, seed=0, init_adapter="", mine=0, mine_thr=0.9, dev=None, eval_every=100,
               teacher=None, alpha=0.5, data_value=False, save_every=0, resume="", wide=False):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    # 4-bit olmayan dev tablolar (Gemma-4 E2B/E4B katman-başı gömmeleri ~2-3B param) fp32'ye çevrilirse T4 taşar →
    # o durumda hafif hazırlık: tüm ağırlıklar donuk, gradyan kontrol noktası + girdi gradyanı
    big = sum(p.numel() for p in S.m.parameters() if p.dtype in (torch.float16, torch.bfloat16, torch.float32))
    if big > 1.5e9 or os.environ.get("AUBIN_LIGHT_PREP") == "1":   # 31B: gömme tablosu 1,41e9 → fp32'ye çevrilirse 5,25 GB OOM
        for p in S.m.parameters():
            p.requires_grad_(False)
            if p.dtype == torch.float16 and p.numel() < 5e7:   # norm vb. küçük parçalar fp32 (kararlılık), dev tablolar fp16 kalır
                p.data = p.data.float()
        S.m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        S.m.enable_input_require_grads()
        print({"light_kbit_prep": True, "unquantized_params": big}, flush=True)
    else:
        S.m = prepare_model_for_kbit_training(S.m, use_gradient_checkpointing=True)
    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                     # yalnız dil katmanları (görsel/ses kulesindeki özel Linear türleri hariç)
                     # Phi-3/4 ailesi birleşik katmanlar kullanır (qkv_proj, gate_up_proj)
                     target_modules=r"^(?!.*(vision|audio|siglip|embed_vision|embed_audio)).*\.(q_proj|k_proj|v_proj|qkv_proj|o_proj|gate_proj|up_proj|gate_up_proj|down_proj)$")
    if init_adapter:                                  # önceki turun adaptöründen devam
        from peft import PeftModel
        S.m = PeftModel.from_pretrained(S.m, init_adapter, is_trainable=True)
    else:
        S.m = get_peft_model(S.m, cfg)
    S.m.print_trainable_parameters()
    opt = torch.optim.AdamW([p for p in S.m.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
    rng = random.Random(seed); pool = [x for x in tr if len(x["keys"]) <= 26 or wide]   # wide: >26 seçenekli sorular da (banking77)
    if teacher:                                       # damıtmada yalnız öğretmenin etiketlediği sorular (yumuşak etiket her örnekte)
        pool = [x for x in pool if x["key"] in teacher or x["key"].startswith("kev_extra/")]   # ek veri: sert etiketle
        print({"distill_pool": len(pool)}, flush=True)
    if mine:                                          # zor-örnek madenciliği: model neyi bilmiyorsa onu çalış
        S.m.eval(); cand = rng.sample(pool, min(mine, len(pool))); hard, easy = [], []
        t1 = time.time()
        for i, it in enumerate(cand):
            pc = float(F.softmax(S.score(it), -1)[it["y"]])
            (hard if pc < mine_thr else easy).append(it)
            if i % 500 == 0:
                print({"mine": i, "hard": len(hard), "min": round((time.time() - t1) / 60, 1)}, flush=True)
        rng.shuffle(easy)
        pool = hard + easy[: max(1, len(hard) // 4)]        # zorların yanında %20 kolay: unutmayı önler
        print({"mined": len(cand), "hard": len(hard), "pool": len(pool)}, flush=True)
    # uzunluğa göre kovalar: batch içindeki dolgu (boşa hesap) en aza iner
    plen = lambda it: len(it["state"]) + sum(len(t) for t in it["texts"]) + len(it["q"])
    srt = sorted(pool, key=plen); B = max(bs * 8, 32)
    buckets = [srt[i:i + B] for i in range(0, len(srt), B)]
    best = (-1.0, 0, None)
    if dev:                                           # başlangıç (eğitimsiz) dev doğruluğu: taban çizgisi
        S.m.eval()
        init_state = None
        if init_adapter:                              # devam eğitimi: iyileşme olmazsa BAŞLANGIÇ adaptörü geri yüklenir (sıfırlanmaz)
            from peft import get_peft_model_state_dict
            init_state = {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(S.m).items()}
        hits0 = [int(S.score(it).argmax()) == it["y"] for it in dev]
        best = (float(np.mean(hits0)), 0, init_state)
        print({"dev_step": 0, "dev_acc": round(best[0], 4)}, flush=True)
        if data_value:                                # veri değeri: hesap, modelin seçim bölümünde zayıf olduğu kaynaklara kayar
            err = {}
            for it, h in zip(dev, hits0):
                err.setdefault(it["src"], []).append(1 - h)
            w = {s: (float(np.mean(v)) + 0.03) for s, v in err.items()}
            mw = float(np.mean(list(w.values())))
            w = {s: min(4.0, max(0.25, v / mw)) for s, v in w.items()}
            pool = [x for x in pool for _ in range(max(1, round(w.get(x["src"], 1.0) * 2)))]
            srt = sorted(pool, key=plen); buckets = [srt[i:i + B] for i in range(0, len(srt), B)]
            print({"data_value_weights": {s: round(v, 2) for s, v in sorted(w.items(), key=lambda kv: -kv[1])}, "pool": len(pool)}, flush=True)
    start = 0
    if resume and os.path.exists(os.path.join(resume, "state.json")):   # oturum sınırı kalkar: önceki oturumun kontrol noktasından devam
        stt = json.load(open(os.path.join(resume, "state.json")))
        start = int(stt["step"])
        opt.load_state_dict(torch.load(os.path.join(resume, "opt.pt"), map_location="cpu"))
        if os.path.exists(os.path.join(resume, "best.pt")):
            best = (float(stt["best_acc"]), int(stt["best_step"]), torch.load(os.path.join(resume, "best.pt"), map_location="cpu"))
        rng = random.Random(seed * 100003 + start)
        print({"resume_from_step": start, "best": stt.get("best_acc")}, flush=True)

    def checkpoint(step_done):
        ck = (out_dir or "lora_out") + "_ckpt"
        os.makedirs(ck, exist_ok=True)
        S.m.save_pretrained(os.path.join(ck, "adapter"))
        torch.save(opt.state_dict(), os.path.join(ck, "opt.pt"))
        if best[2] is not None:
            torch.save(best[2], os.path.join(ck, "best.pt"))
        json.dump({"step": step_done, "best_acc": best[0], "best_step": best[1]}, open(os.path.join(ck, "state.json"), "w"))
        print({"checkpoint": step_done, "dir": ck}, flush=True)

    t0 = time.time(); S.m.train(); ntok = 0
    for step in range(start, steps):
        bk = buckets[rng.randrange(len(buckets))]
        batch = [bk[rng.randrange(len(bk))] for _ in range(bs)]
        perms = []
        for it in batch:                                   # seçenek sırası karıştırılır: konum yanlılığı öğrenilmez
            n = len(it["keys"])
            if n <= 26:
                idx = list(range(n))
            else:                                          # geniş soru: doğru + rastgele çeldiriciler (≤26) — testteki 26'lık grup
                k = 26 if rng.random() < 0.75 else rng.randint(2, 6)    # turu (%75) ve az seçenekli final turu (%25) birebir çalışılır
                idx = rng.sample([j for j in range(n) if j != it["y"]], k - 1) + [it["y"]]
            rng.shuffle(idx); perms.append(idx)
        lg = S.letter_logits([S.prompt(it, idx) for it, idx in zip(batch, perms)], grad=True)
        loss = 0.0
        for b, (it, idx) in enumerate(zip(batch, perms)):
            ce = F.cross_entropy(lg[b, :len(idx)].unsqueeze(0), torch.tensor([idx.index(it["y"])], device=lg.device))
            if teacher and it["key"] in teacher and len(teacher[it["key"]]) == len(it["keys"]) and len(idx) == len(it["keys"]):   # damıtma: öğretmenin (AUBIN-31B) dağılımı, öğrencinin sırasına çevrilir
                tq = torch.softmax(torch.tensor(teacher[it["key"]], dtype=torch.float32), -1)[torch.tensor(idx)].to(lg.device)
                kd = -(tq * F.log_softmax(lg[b, :len(idx)], -1)).sum()
                ce = (1 - alpha) * ce + alpha * kd
            loss = loss + ce
        loss = loss / bs
        for g in opt.param_groups:
            g["lr"] = lr * min(1.0, (step + 1) / 30) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / steps)))
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in S.m.parameters() if p.requires_grad], 1.0); opt.step()
        if step % 20 == 0:
            print({"step": step, "loss": round(float(loss), 4), "min": round((time.time() - t0) / 60, 1)}, flush=True)
        if dev and ((step + 1) % eval_every == 0 or step == steps - 1):   # çöküşe karşı: en iyi dev adaptörü tutulur
            S.m.eval()
            acc = float(np.mean([int(S.score(it).argmax()) == it["y"] for it in dev]))
            S.m.train()
            print({"dev_step": step + 1, "dev_acc": round(acc, 4), "best": round(best[0], 4)}, flush=True)
            if acc > best[0]:
                from peft import get_peft_model_state_dict
                best = (acc, step + 1, {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(S.m).items()})
        if save_every and (step + 1) % save_every == 0 and step + 1 < steps:
            checkpoint(step + 1)
    S.m.eval()
    if dev and best[2] is not None:
        from peft import set_peft_model_state_dict
        set_peft_model_state_dict(S.m, best[2]); print({"best_dev_step": best[1], "best_dev_acc": round(best[0], 4)}, flush=True)
    elif dev:                                         # eğitim tabanı hiç geçemedi: adaptör etkisiz (lora_B = 0) → ham taban
        with torch.no_grad():
            for n_, p_ in S.m.named_parameters():
                if "lora_B" in n_:
                    p_.zero_()
        print({"best_dev_step": 0, "note": "eğitim tabanı geçemedi; adaptör sıfırlandı"}, flush=True)
    S.best_dev = best[:2] if dev else None
    if out_dir:
        S.m.save_pretrained(out_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-12B-it"); ap.add_argument("--work", default="/tmp/kev")
    ap.add_argument("--suites", default="kev_transfer_test,kev_test"); ap.add_argument("--out", default="aubin_kev.json")
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--no4bit", action="store_true")
    ap.add_argument("--train", type=int, default=0); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=4); ap.add_argument("--adapter", default="")
    ap.add_argument("--cal", type=int, default=300); ap.add_argument("--perms", type=int, default=1); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_before", action="store_true"); ap.add_argument("--device_map", default="")
    ap.add_argument("--init_adapter", default=""); ap.add_argument("--mine", type=int, default=0)
    ap.add_argument("--dev_n", type=int, default=0); ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--think_margin", type=float, default=0.0); ap.add_argument("--think_tokens", type=int, default=256)
    ap.add_argument("--think_bs", type=int, default=8)
    ap.add_argument("--think_samples", type=int, default=1); ap.add_argument("--think_temp", type=float, default=0.7)
    ap.add_argument("--label_train", type=int, default=0); ap.add_argument("--teacher", default=""); ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--dev_src", default="dev", choices=["dev", "cal"]); ap.add_argument("--data_value", action="store_true")
    ap.add_argument("--train_only", action="store_true"); ap.add_argument("--label_offset", type=int, default=0)
    ap.add_argument("--save_every", type=int, default=0); ap.add_argument("--resume", default="")
    ap.add_argument("--wide", action="store_true")      # >26 seçenekli soruları (banking77, 77 sınıf) da eğit ve seç
    ap.add_argument("--no_extra", action="store_true")  # karşılaştırma: KEV_EXTRA_TRAIN ek verisi bu süreçte kullanılmaz
    a = ap.parse_args()
    if a.no_extra:
        os.environ.pop("KEV_EXTRA_TRAIN", None)
    a.init_adapter = resolve_adapter(a.init_adapter)
    K = kevdata.load(a.work)
    assert not ({c["source"] for c in K["kev_train"]} & kevdata.HOLDOUT)
    S = Scorer(a.model, four_bit=not a.no4bit, device_map=a.device_map)
    S.tok.padding_side = "left"
    if S.tok.pad_token is None:
        S.tok.pad_token = S.tok.eos_token
    R = {"model": a.model, "protocol": "Kev açık paketleri; harf-logit tek geçiş; sıcaklık decision-v7 calibration", "runs": {}}
    R["perms"] = a.perms

    def think(rows):
        """Eminsiz (en iyi iki log-olasılık farkı < marj) sorular: akıl yürütmeli ikinci geçiş ile değiştirilir."""
        if not a.think_margin:
            return rows, 0
        def marg(lg):
            v = torch.sort(torch.log_softmax(lg.float(), -1), descending=True).values
            return float(v[0] - v[1]) if len(v) > 1 else 99.0
        sel = [i for i, (it, lg) in enumerate(rows) if len(it["keys"]) <= 26 and marg(lg) < a.think_margin]
        if not sel:
            return rows, 0
        new = S.think_scores([rows[i][0] for i in sel], a.think_tokens, a.think_bs, a.think_samples, a.think_temp)
        rows = list(rows)
        for i, lg in zip(sel, new):
            rows[i] = (rows[i][0], lg)
        return rows, len(sel)

    def eval_all(tag):
        rng = random.Random(0)
        cal = items(K["kev_cal"]); rng.shuffle(cal); cal = cal[: a.cal]
        cal_rows = [(it, S.score(it, a.perms)) for it in cal]
        cal_rows, nth = think(cal_rows); print(f"[{tag}] kalibrasyonda düşünülen soru:", nth, flush=True)
        T = min(np.concatenate([np.arange(0.3, 4.0, 0.05), np.arange(4.0, 20.01, 0.25)]), key=lambda t: metrics(cal_rows, t)["all"]["nll"])
        R[tag + "_temperature"] = round(float(T), 2)
        R[tag + "_cal_items"] = [{"y": it["y"], "lp": [round(float(x), 4) for x in torch.log_softmax(lg.float(), -1)]} for it, lg in cal_rows]
        print(f"[{tag} kalibrasyon] T =", R[tag + "_temperature"], flush=True)
        R.setdefault(tag, {})
        for s in a.suites.split(","):
            its = items(K[s])
            if a.limit:
                rng.shuffle(its); its = its[: a.limit]
            t0 = time.time(); rows = []
            for i, it in enumerate(its):
                rows.append((it, S.score(it, a.perms)))
                if i % 100 == 0:
                    print(tag, s, i, len(its), round(time.time() - t0), "s", flush=True)
            fast = rows
            t1 = time.time(); rows, nth = think(rows)
            r = metrics(rows, T); r["raw_T1"] = metrics(rows, 1.0)["all"]
            if nth:
                r["fast_only"] = metrics(fast, T)["all"]; r["thought"] = nth; r["think_seconds"] = round(time.time() - t1, 1)
                r["fast_items"] = [[round(float(x), 4) for x in torch.log_softmax(lg.float(), -1)] for _, lg in fast]
            # soru-başı ham log-olasılıklar (ansambl / sonradan kalibrasyon GPU'suz yapılabilsin)
            r["items"] = [{"src": it["src"], "y": it["y"], "lp": [round(float(x), 4) for x in torch.log_softmax(lg.float(), -1)]}
                          for it, lg in rows]
            r["seconds"] = round(time.time() - t0, 1)
            R[tag][s] = r
            print(tag, s, json.dumps(r["all"]), flush=True)
            json.dump(R, open(a.out, "w"), indent=1)

    if a.teacher and not os.path.exists(a.teacher):   # Kaggle kernel kaynağı yolu hesaba göre değişir
        import glob
        a.teacher = (glob.glob("/kaggle/input/**/" + os.path.basename(a.teacher), recursive=True) or [a.teacher])[0]
        print("[öğretmen yolu]", a.teacher, flush=True)
    teach = json.load(open(a.teacher)) if a.teacher else None
    if a.label_train:                                 # öğretmen modu: eğitim sorularının log-olasılıkları → dosya
        if a.init_adapter:
            from peft import PeftModel
            S.m = PeftModel.from_pretrained(S.m, a.init_adapter).eval()
        pool = [x for x in items(K["kev_train"]) if len(x["keys"]) <= 26]; random.Random(11).shuffle(pool)
        lab, t0 = {}, time.time()
        for i, it in enumerate(pool[a.label_offset: a.label_offset + a.label_train]):   # aralık: hesaplar arası bölüşüm
            lab[it["key"]] = [round(float(v), 4) for v in S.score(it)]
            if i % 200 == 0:
                print({"label": i, "min": round((time.time() - t0) / 60, 1)}, flush=True)
                json.dump(lab, open(a.out, "w"))
        json.dump(lab, open(a.out, "w")); print("ETIKETLEME BITTI", len(lab), flush=True)
        return
    if a.init_adapter and not a.train:              # yalnız değerlendirme: eğitilmiş adaptörle ölç
        from peft import PeftModel
        S.m = PeftModel.from_pretrained(S.m, a.init_adapter).eval()
        R["adapter"] = a.init_adapter
    if a.eval_before:
        eval_all("zeroshot")
    if a.train:
        if a.dev_src == "cal":                        # seçim kalibrasyon bölümünde: Jev'le kıyaslanan dev kümesine hiç dokunulmaz
            devs = items(K["kev_cal"]); random.Random(0).shuffle(devs)
            devs = devs[a.cal:] if len(devs) - a.cal >= a.dev_n else devs[::-1]
        else:
            devs = items(K["kev_dev"]); random.Random(5).shuffle(devs)
        train_lora(S, items(K["kev_train"]), a.train, a.lr, a.bs, a.adapter, seed=a.seed, init_adapter=a.init_adapter, mine=a.mine,
                   dev=[d for d in devs if len(d["keys"]) <= 26 or a.wide][: a.dev_n] if a.dev_n else None, eval_every=a.eval_every,
                   teacher=teach, alpha=a.alpha, data_value=a.data_value, save_every=a.save_every, resume=a.resume, wide=a.wide)
        R["best_dev"] = getattr(S, "best_dev", None)
        R["lora"] = {"steps": a.train, "bs": a.bs, "lr": a.lr, "r": 16, "seed": a.seed}
        if a.train_only:                              # ölçüm ayrı, temiz süreçte (eğitim sonrası parçalı bellek düşünme üretiminde OOM veriyor)
            json.dump(R, open(a.out, "w"), indent=1); print("EGITIM BITTI", R["best_dev"], flush=True)
            return
    eval_all("runs")
    R["temperature"] = R.get("runs_temperature")
    json.dump(R, open(a.out, "w"), indent=1)
    print("BITTI", flush=True)


if __name__ == "__main__":
    main()
