"""AUBIN görsel bilgisayar kullanımı — ScreenSpot (Cheng et al., 2024) tıklama konumlandırma ölçümü.

Her örnek: ekran görüntüsü (mobil/masaüstü/web) + talimat ("open settings") → model tek bir (x, y) noktası söyler (0-1000 ölçekli).
Başarı = nokta hedef öğenin kutusu içinde (resmi ölçüt). Metin/simge alt kırılımı ayrıca raporlanır.

    python screenspot_eval.py --model google/gemma-4-12B-it [--adapter <lora>] --n 0 --out ss.json
"""
import argparse, io, json, re, time
import torch

PROMPT = ("You are a GUI agent. In this screenshot, where should I click to: \"{ins}\"?\n"
          "Answer with ONLY the click point as (x, y), where x and y are integers from 0 to 1000 "
          "(0,0 = top-left corner, 1000,1000 = bottom-right corner).")


VISION = ("vision_tower", "embed_vision", "multi_modal_projector")


def fp16_clamp(m, lim=6.0e4):
    """12B + görüntü belirteçleri: fp16 artık akışı taşar (inf → NaN). Dil modeli katman/MLP/dikkat çıkışları ±lim'e kırpılır."""
    def fix(h):
        return torch.nan_to_num(h, nan=0.0, posinf=lim, neginf=-lim).clamp(-lim, lim)

    def hook(mod, inp, out):
        if isinstance(out, tuple) and out and torch.is_tensor(out[0]) and out[0].dtype == torch.float16:
            return (fix(out[0]),) + tuple(out[1:])
        if torch.is_tensor(out) and out.dtype == torch.float16:
            return fix(out)
        return out
    n = 0
    for name, mod in m.named_modules():
        cls = type(mod).__name__
        if "language_model" in name and (cls.endswith("DecoderLayer") or cls.endswith("MLP") or cls.endswith("Attention")):
            mod.register_forward_hook(hook); n += 1
    print("fp16_clamp", n, flush=True)
    return n


def bnb4(skip_vision=True, compute=torch.float16):
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=compute,
                              bnb_4bit_use_double_quant=True, llm_int8_skip_modules=list(VISION) + ["lm_head"] if skip_vision else None)


def vision_fp32(m):
    """12B: görsel kule fp16'da taşar (kayıp/çıktı NaN). Görsel modüller fp32 ağırlık + autocast kapalı koşar; dil modeli fp16 kalır."""
    n = 0
    for name, mod in m.named_modules():
        if name.split(".")[-1] in VISION and not any(p in VISION for p in name.split(".")[:-1]):
            mod.float()
            def fw(*a, _f=mod.forward, **k):
                c = lambda x: x.float() if torch.is_tensor(x) and x.is_floating_point() else x
                with torch.autocast("cuda", enabled=False):
                    return _f(*[c(x) for x in a], **{kk: c(v) for kk, v in k.items()})
            mod.forward = fw; n += 1
    print("vision_fp32", n, flush=True)
    return n


def load(n, shard=""):
    from datasets import load_dataset
    ds = load_dataset("rootsautomation/ScreenSpot", split="test")
    ds = ds.shuffle(seed=0)                                # sıra karışık: süre kesilirse ara sonuç da yansız alt küme
    ds = ds.select(range(n)) if n else ds
    if shard:                                              # "i/k": iki GPU'ya bölünmüş tam ölçüm (birleşimi = tam küme)
        i, k = map(int, shard.split("/"))
        ds = ds.shard(num_shards=k, index=i, contiguous=False)
    return ds


def parse(txt):
    m = re.findall(r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)", txt)
    if not m:
        return None
    x, y = float(m[0][0]), float(m[0][1])
    if x <= 1 and y <= 1:
        x, y = x * 1000, y * 1000
    return x / 1000, y / 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-12B-it"); ap.add_argument("--adapter", default="")
    ap.add_argument("--n", type=int, default=0); ap.add_argument("--out", default="screenspot.json")
    ap.add_argument("--no4bit", action="store_true"); ap.add_argument("--work", default="")
    ap.add_argument("--fp32vis", action="store_true")       # 12B: görsel kule fp32 (fp16 taşması → boş çıktı)
    ap.add_argument("--clamp", action="store_true")         # 12B: dil modeli fp16 taşma kırpması
    ap.add_argument("--fp32lm", action="store_true")        # 12B: dil modeli fp32 hesap (yavaş, taşmasız)
    ap.add_argument("--shard", default="")                  # "0/2", "1/2"
    a = ap.parse_args()
    from transformers import AutoProcessor, AutoModelForImageTextToText
    proc = AutoProcessor.from_pretrained(a.model)
    dt = torch.float32 if a.fp32lm else torch.float16
    kw = dict(device_map="auto", dtype=dt)
    if not a.no4bit:
        kw["quantization_config"] = bnb4(a.fp32vis, dt)
    m = AutoModelForImageTextToText.from_pretrained(a.model, **kw).eval()
    if a.fp32vis and not a.fp32lm:
        vision_fp32(m)
    if a.clamp:
        fp16_clamp(m)
    if a.adapter:
        from peft import PeftModel
        m = PeftModel.from_pretrained(m, a.adapter).eval()
    ds = load(a.n, a.shard)
    R = {"model": a.model, "adapter": a.adapter, "protocol": __doc__.split("    python")[0].strip(), "rows": []}
    t0 = time.time()
    for i, ex in enumerate(ds):
        img = ex["image"].convert("RGB"); W, H = img.size
        x1, y1, x2, y2 = [float(v) for v in ex["bbox"]]
        if max(x1, y1, x2, y2) > 1.5:                     # piksel kutusu → oran
            x1, x2, y1, y2 = x1 / W, x2 / W, y1 / H, y2 / H
        msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": PROMPT.format(ins=ex["instruction"])}]}]
        inp = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(m.device)
        t1 = time.time()
        with torch.no_grad():
            try:
                g = m.generate(**inp, max_new_tokens=16, do_sample=False)
            except RuntimeError as e:                      # bazı görsel kuleler fp32 kalır (12B): piksel girdisini fp32 yap
                if "Half" not in str(e):
                    raise
                for k in list(inp.keys()):
                    if k.startswith("pixel") and torch.is_floating_point(inp[k]):
                        inp[k] = inp[k].float()
                g = m.generate(**inp, max_new_tokens=16, do_sample=False)
        txt = proc.decode(g[0, inp["input_ids"].shape[1]:], skip_special_tokens=True)
        if i < 3:                                          # tanı: ham belirteçler (boş çıktı nedenini görmek için)
            print({"i": i, "ids": g[0, inp["input_ids"].shape[1]:].tolist()[:16], "txt": txt[:40], "len": inp["input_ids"].shape[1]}, flush=True)
        p = parse(txt)
        ok = bool(p and x1 <= p[0] <= x2 and y1 <= p[1] <= y2)
        R["rows"].append({"i": i, "type": ex.get("data_type"), "platform": ex.get("data_source"), "out": txt[:40], "ok": ok,
                          "ms": round((time.time() - t1) * 1e3)})
        if (i + 1) % 50 == 0:
            r = R["rows"]
            print({"n": len(r), "acc": round(sum(x["ok"] for x in r) / len(r), 4), "min": round((time.time() - t0) / 60, 1), "ex": txt[:30]}, flush=True)
            json.dump(R, open(a.out, "w"), indent=0)
    r = R["rows"]
    by = {}
    for x in r:
        by.setdefault(f"{x['platform']}/{x['type']}", []).append(x["ok"])
    R["summary"] = {"n": len(r), "accuracy": round(sum(x["ok"] for x in r) / len(r), 4),
                    "by": {k: [len(v), round(sum(v) / len(v), 4)] for k, v in sorted(by.items())},
                    "median_ms": sorted(x["ms"] for x in r)[len(r) // 2], "minutes": round((time.time() - t0) / 60, 1)}
    json.dump(R, open(a.out, "w"), indent=0)
    print("SONUC", json.dumps(R["summary"]), flush=True)


if __name__ == "__main__":
    main()
