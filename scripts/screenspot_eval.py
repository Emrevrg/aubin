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


def load(n):
    from datasets import load_dataset
    ds = load_dataset("rootsautomation/ScreenSpot", split="test")
    return ds.select(range(n)) if n else ds


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
    a = ap.parse_args()
    from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
    proc = AutoProcessor.from_pretrained(a.model)
    kw = dict(device_map="auto", dtype=torch.float16)
    if not a.no4bit:
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
                                                       bnb_4bit_use_double_quant=True)
    m = AutoModelForImageTextToText.from_pretrained(a.model, **kw).eval()
    if a.adapter:
        from peft import PeftModel
        m = PeftModel.from_pretrained(m, a.adapter).eval()
    ds = load(a.n)
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
