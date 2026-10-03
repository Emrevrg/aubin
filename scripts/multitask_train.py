"""AUBIN tek-model çok-görev eğitimi: TEK adaptörde tipli karar (Kev kaynakları + ek veri) + web ajanı (Mind2Web train)
+ gerçek zamanlı kontrol (ızgara oyunu). Görevler eşit örnek sayısıyla karıştırılır (büyük veri küçüğü ezmesin); seçim için
her görevden ayrı dev parçası; testler (kev_test, Mind2Web test_domain, kontrol tohum 0) yalnız sonra, ayrı betiklerle.

    python multitask_train.py --work /tmp/kev --init_adapter hf:emrevrg/AUBIN-12B --per_task 4000 --steps 2000 --adapter out/
"""
import argparse, json, os, random, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import kev_llm as KL
import kevdata


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/tmp/kev"); ap.add_argument("--model", default="google/gemma-4-12B-it")
    ap.add_argument("--init_adapter", default="hf:emrevrg/AUBIN-12B"); ap.add_argument("--adapter", default="lora_mt")
    ap.add_argument("--per_task", type=int, default=4000); ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=2e-5); ap.add_argument("--bs", type=int, default=4); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device_map", default="")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    # 1) tipli karar: Kev eğitim kaynakları + kev_augment ek verisi (KEV_EXTRA_TRAIN), geniş (77 seçenekli) sorular dahil
    K = kevdata.load(a.work)
    kev = KL.items(K["kev_train"]); rng.shuffle(kev)
    kdev = [x for x in KL.items(K["kev_dev"]) if len(x["keys"]) <= 26]; rng.shuffle(kdev)
    # 2) web ajanı: Mind2Web eğitim bölümü (MindAct çoktan-seçmeli öğeler)
    import mind2web_eval as MW
    S_ = MW.load_scores(); scores = S_.get("scores", S_) if isinstance(S_, dict) else S_
    web = MW.train_items(MW.load_split("train", 0), scores, 50, random.Random(a.seed + 1), per_action=3); rng.shuffle(web)
    wdev, web = web[:150], web[150:]
    # 3) gerçek zamanlı kontrol: ızgara oyunu (eğitim tohumları 1000+, dev 900+, ölçüm tohumu 0'a dokunulmaz)
    import control_train as CT
    ctl, teach = CT.examples(3000, 1000, 0); rng.shuffle(ctl)
    cdev, _ = CT.examples(60, 900, 0)
    tr = kev[:a.per_task] + web[:a.per_task] + ctl[:a.per_task]; rng.shuffle(tr)
    dev = kdev[:150] + wdev + cdev[:150]
    print({"train": len(tr), "kev": min(len(kev), a.per_task), "web": min(len(web), a.per_task), "control": min(len(ctl), a.per_task),
           "dev": len(dev)}, flush=True)
    S = KL.Scorer(a.model, four_bit=True, device_map=a.device_map)
    S.tok.padding_side = "left"
    if S.tok.pad_token is None:
        S.tok.pad_token = S.tok.eos_token
    KL.train_lora(S, tr, a.steps, a.lr, a.bs, a.adapter, seed=a.seed, init_adapter=KL.resolve_adapter(a.init_adapter),
                  dev=dev, eval_every=max(200, a.steps // 8), teacher=teach, alpha=0.5, save_every=500, wide=True)
    json.dump({"best_dev": getattr(S, "best_dev", None), "train": len(tr), "per_task": a.per_task, "steps": a.steps},
              open(a.adapter.rstrip("/") + "_train.json", "w"))
    print("EGITIM BITTI", getattr(S, "best_dev", None), flush=True)


if __name__ == "__main__":
    main()
