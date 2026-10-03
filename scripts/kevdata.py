"""Kev (jaredpalmer/kev) açık değerlendirme paketlerini NIVEN vaka biçimine çevirir — AYNI maddelerde kıyas için.

Kev kaydı: {"state": {...}, "questions": {qid: {type, instructions, criteria, label, src}}, "_meta": {...}}
NIVEN vakası: {"id", "source", "state", "questions": {qid: {type, instructions, criteria}}, "gold": {qid: {"label"}}}

Paketler (HF dataset jaredpalmer/kev-suites, Apache-2.0):
  v7/decision-v7  train / development / calibration / test   (Kev'in eğitildiği kaynaklar)
  v4/transfer-v4  test (764)                                 (Kev'in HİÇ eğitilmediği kaynaklar: mmlu, sciq, qnli,
                                                              paws, emotion, tweet_offensive, contrastive, composition)
Bu kaynaklar NIVEN eğitiminde de KULLANILMAZ (transfer-v4 yalnız değerlendirme).
"""
import json, os

REPO = "jaredpalmer/kev-suites"
FILES = {
    "kev_train": "v7/decision-v7/train.jsonl",
    "kev_dev": "v7/decision-v7/development.jsonl",
    "kev_cal": "v7/decision-v7/calibration.jsonl",
    "kev_test": "v7/decision-v7/test.jsonl",
    "kev_transfer_test": "v4/transfer-v4/test.jsonl",
    "kev_transfer_dev": "v4/transfer-v4/development.jsonl",   # Jev'in ölçüldüğü bölüm (Kev README: Jev yalnız development)
}
HOLDOUT = {"mmlu", "emotion", "tweet_offensive", "qnli", "paws", "sciq"}


def fetch(out_dir):
    from huggingface_hub import hf_hub_download
    os.makedirs(out_dir, exist_ok=True)
    return {k: hf_hub_download(REPO, p, repo_type="dataset", local_dir=out_dir) for k, p in FILES.items()}


def _label(v):
    return str(v).lower() if isinstance(v, bool) else str(v)


def convert(path, tag):
    cases = []
    for i, line in enumerate(open(path, encoding="utf-8")):
        r = json.loads(line)
        qs, gold, src = {}, {}, None
        for qid, q in r["questions"].items():
            src = q.get("src") or r.get("_meta", {}).get("source") or tag
            qs[qid] = {k: q[k] for k in ("type", "instructions", "criteria") if k in q}
            gold[qid] = {"label": _label(q["label"])}
        meta = r.get("_meta", {})
        cases.append({"id": f"{tag}/{meta.get('id', i)}/{i}", "source": src, "workflow": tag,
                      "state": r["state"], "questions": qs, "gold": gold})
    return cases


def load(out_dir):
    p = fetch(out_dir)
    K = {k: convert(v, k) for k, v in p.items()}
    extra = os.environ.get("KEV_EXTRA_TRAIN", "")         # ek eğitim verisi (kev_augment.py; Kev'in hiçbir kümesiyle çakışmaz)
    if extra and os.path.exists(extra):
        K["kev_train"] += convert(extra, "kev_extra")
        print({"kev_extra_train": len(K["kev_train"])}, flush=True)
    return K
