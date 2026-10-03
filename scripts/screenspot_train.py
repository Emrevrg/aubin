"""AUBIN görsel bilgisayar kullanımı eğitimi: ekran görüntüsü + talimat → tıklama noktası (x, y), 0-1000 ölçek.
Veri: agentsea/wave-ui (train). SIZINTI YOK: ScreenSpot'tan, GroundUI/agent_studio'dan (ScreenSpot içerir) ve Mind2Web test
bölümlerinden gelen örnekler dışlanır. Hedef = öğe kutusunun merkezi. LoRA (dil modeli katmanları), 4-bit taban.
Ölçüm ayrı: screenspot_eval.py --adapter <çıktı>.

    python screenspot_train.py --model google/gemma-4-E4B-it --n 8000 --out /kaggle/working/ss_lora
"""
import argparse, json, math, os, random, time
import torch

from screenspot_eval import PROMPT

EXCL = ("screenspot", "agent_studio", "mind2web_test")


def examples(n, seed, skip=0):
    from datasets import load_dataset
    ds = load_dataset("agentsea/wave-ui", split="train", streaming=True).shuffle(seed=seed, buffer_size=2000)
    k = -skip
    for ex in ds:
        src = str(ex.get("source") or "")
        if any(src.startswith(e) for e in EXCL) or not ex.get("instruction") or not ex.get("bbox"):
            continue
        img = ex["image"].convert("RGB"); W, H = img.size
        x1, y1, x2, y2 = [float(v) for v in ex["bbox"]]
        if max(x1, y1, x2, y2) <= 1.5:                 # oran kutusu
            x1, x2, y1, y2 = x1 * W, x2 * W, y1 * H, y2 * H
        if not (0 <= x1 < x2 <= W + 2 and 0 <= y1 < y2 <= H + 2):
            continue
        cx, cy = round((x1 + x2) / 2 / W * 1000), round((y1 + y2) / 2 / H * 1000)
        k += 1
        if k <= 0:
            continue
        yield img, ex["instruction"], f"({cx}, {cy})", src
        if k >= n:
            return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-E4B-it"); ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--r", type=int, default=16); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="ss_lora"); ap.add_argument("--work", default=""); ap.add_argument("--init", default="")
    ap.add_argument("--skip", type=int, default=0)     # akıştaki ilk N uygun örneği atla (önceki turda görülenler)
    a = ap.parse_args()
    random.seed(a.seed); torch.manual_seed(a.seed)
    from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model
    proc = AutoProcessor.from_pretrained(a.model)
    m = AutoModelForImageTextToText.from_pretrained(
        a.model, device_map={"": 0}, dtype=torch.float16,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
                                               bnb_4bit_use_double_quant=True))
    m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); m.enable_input_require_grads()
    tm = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
    if a.init:                                         # önceki turun adaptöründen devam (yeni örneklerle)
        from peft import PeftModel
        m = PeftModel.from_pretrained(m, a.init, is_trainable=True)
    else:
        m = get_peft_model(m, LoraConfig(r=a.r, lora_alpha=2 * a.r, lora_dropout=0.05, target_modules=tm, task_type="CAUSAL_LM"))
    for p in m.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    m.print_trainable_parameters()
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    steps = a.n // a.accum
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / max(1, int(0.05 * steps))) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    scaler = torch.amp.GradScaler("cuda")
    m.train(); t0 = time.time(); seen = 0; run = 0.0; srcs = {}
    for img, ins, ans, src in examples(a.n, a.seed, a.skip):
        user = {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": PROMPT.format(ins=ins)}]}
        full = proc.apply_chat_template([user, {"role": "assistant", "content": [{"type": "text", "text": ans}]}],
                                        tokenize=True, return_dict=True, return_tensors="pt")
        plen = proc.apply_chat_template([user], add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")["input_ids"].shape[1]
        full = {k: v.to("cuda") for k, v in full.items()}
        lab = full["input_ids"].clone(); lab[:, :plen] = -100
        with torch.autocast("cuda", dtype=torch.float16):
            loss = m(**full, labels=lab).loss / a.accum
        scaler.scale(loss).backward(); run += loss.item() * a.accum; seen += 1; srcs[src] = srcs.get(src, 0) + 1
        if seen % a.accum == 0:
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
        if seen % 200 == 0:
            print({"seen": seen, "loss": round(run / 200, 4), "min": round((time.time() - t0) / 60, 1)}, flush=True); run = 0.0
        if seen % 2000 == 0:
            m.save_pretrained(a.out)
    m.save_pretrained(a.out)
    json.dump({"model": a.model, "n": seen, "lr": a.lr, "accum": a.accum, "r": a.r, "sources": srcs, "excluded": EXCL,
               "minutes": round((time.time() - t0) / 60, 1)}, open(os.path.join(a.out, "train_info.json"), "w"), indent=1)
    print("BITTI", a.out, seen, flush=True)


if __name__ == "__main__":
    main()
