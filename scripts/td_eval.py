"""AUBIN'i Laya/Jev/meraGPT'nin kümesinde (LocalLLaMA/typed-decisions, Apache-2.0) ölç + AUBIN-Learn için soru-başı log-olasılıkları sakla.

Kurallar: etiket = gold[q]["label"] (öğretmen dağılımının argmax'ı; tablo 'Accuracy' ile aynı tanım). 'factors' (vakayı üreten gizli
etkenler) ASLA girdi olarak kullanılmaz. train bölümü yalnız bellek/beceri ve birleştirme ayarı (çapraz doğrulama) için; test bir kez.

    python td_eval.py --init_adapter hf:emrevrg/AUBIN-12B --out /kaggle/working/td_aubin12.json [--limit N]
"""
import argparse, json, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from kev_llm import Scorer, items, resolve_adapter


def load_td(split):
    """typed-decisions satırları → kev_llm vaka biçimi."""
    try:
        from datasets import load_dataset
        rows = load_dataset("LocalLLaMA/typed-decisions", "all", split=split)
    except Exception:
        import glob, pyarrow.parquet as pq
        p = glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/datasets--LocalLLaMA--typed-decisions/snapshots/*/all/{split}-*.parquet"))[0]
        rows = pq.read_table(p).to_pylist()
    cases = []
    for r in rows:
        st = r["state"]; qs = r["questions"]; gd = r["gold"]
        st = json.loads(st) if isinstance(st, str) else st
        qs = json.loads(qs) if isinstance(qs, str) else qs
        gd = json.loads(gd) if isinstance(gd, str) else gd
        cases.append({"id": f"td_{split}/{r['id']}", "source": r["workflow"], "state": st,
                      "questions": {q: {k: v[k] for k in ("type", "instructions", "criteria") if k in v} for q, v in qs.items()},
                      "gold": {q: {"label": str(gd[q]["label"]).lower() if isinstance(gd[q]["label"], bool) else str(gd[q]["label"])} for q in qs}})
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-12B-it"); ap.add_argument("--init_adapter", default="")
    ap.add_argument("--no4bit", action="store_true"); ap.add_argument("--device_map", default="")
    ap.add_argument("--work", default=""); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="td_eval.json")
    a = ap.parse_args()
    S = Scorer(a.model, four_bit=not a.no4bit, device_map=a.device_map)
    S.tok.padding_side = "left"
    if S.tok.pad_token is None:
        S.tok.pad_token = S.tok.eos_token
    ad = resolve_adapter(a.init_adapter)
    if ad:
        from peft import PeftModel
        S.m = PeftModel.from_pretrained(S.m, ad).eval()
    R = {"model": a.model, "adapter": a.init_adapter, "benchmark": "LocalLLaMA/typed-decisions (all)",
         "reference": {"meraGPT sd-1": 0.768, "Laya (our run)": 0.7665, "TypeSafe Jev 1.13.0": 0.727, "teacher self-agreement": 0.735}}
    for split in ("test", "train"):                       # test önce: süre biterse asıl sayı elde olsun
        its = items(load_td(split))
        if a.limit:
            its = its[: a.limit]
        t0 = time.time(); rows = []
        for i, it in enumerate(its):
            with torch.no_grad():
                lg = S.score(it)
            rows.append({"key": it["key"], "src": it["src"], "y": it["y"], "lp": [round(float(x), 4) for x in torch.log_softmax(lg.float(), -1)]})
            if i % 200 == 0:
                print(split, i, len(its), round(time.time() - t0), "s", flush=True)
        acc = float(np.mean([int(np.argmax(r["lp"])) == r["y"] for r in rows]))
        R[split] = {"n": len(rows), "accuracy_T1": round(acc, 4), "seconds": round(time.time() - t0, 1), "items": rows}
        print(split, "ACC", round(acc, 4), flush=True)
        json.dump(R, open(a.out, "w"))
    print("BITTI", a.out, flush=True)


if __name__ == "__main__":
    main()
