"""AUBIN Control eğitimi: komutlu ızgara oyunundan sınırsız, doğru etiketli örnek → LoRA (kev_llm.train_lora).

Örnekler control_bench'in AYNI biçiminden gelir ama FARKLI tohumlardan (eğitim 1000+, ölçüm 0) → ölçüm bölümleri eğitimde görülmez.
Her örnek: bir bölümde en kısa yol boyunca bir konum; doğru hamle(ler) BFS'ten; birden çok en-kısa hamle varsa yumuşak etiket
(eşit dağılım, `teacher` sözlüğü). Durum, AubinController.act ile birebir aynı biçimde kurulur (observation + recent_actions).
    python control_train.py --init_adapter aubin12-plain --episodes 3000 --train 1200 --adapter /kaggle/working/lora_ctl
"""
import argparse, json, math, os, random, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import control_bench as CB
import kev_llm as KL

INSTR = ("Choose the next move that follows the command along a shortest safe path "
         "(never step on lava, objects block movement).")


def examples(n_eps, seed, offpath=0):
    """offpath: bölüm başına en kısa yol DIŞINDAKİ rastgele erişilebilir konumlardan ek örnek (sapınca toparlanmayı öğretir;
    ölçümde model kendi hamleleriyle yoldan çıkabilir — yalnız yol üstü örnekler bu durumu hiç göstermez)."""
    rng = random.Random(seed); out, teacher = [], {}
    keys = list(CB.ACTIONS); texts = [f"{k}: {v}" for k, v in CB.ACTIONS.items()]
    for e in range(n_eps):
        ep = CB.make(rng); pos = ep["agent"]; tp, col, k = ep["target"]; block = {p for p, _, _ in ep["items"][1:]}
        free = [(r, c) for r in range(1, ep["n"] + 1) for c in range(1, ep["n"] + 1)
                if (r, c) not in ep["lava"] and (r, c) not in block and (r, c) != tp]
        for j, q in enumerate(rng.sample(free, min(offpath, len(free)))):
            d, best = CB.optimal(q, tp, ep["lava"], block, ep["n"])
            if not best:
                continue
            state = {"observation": CB.render(ep, q)}
            if rng.random() < 0.5:                                  # bazen geçmiş de ver (modelin kendi önceki hamleleri gibi)
                state["recent_actions"] = [rng.choice(keys) for _ in range(rng.randint(1, 5))]
            key = f"ctl/{seed}/{e}/off{j}"
            out.append({"key": key, "src": "control_grid_off", "state": json.dumps(state, ensure_ascii=False),
                        "q": f"Command: go to the {col} {k}. " + INSTR, "type": "choice", "keys": keys, "texts": texts,
                        "y": keys.index(sorted(best)[0])})
            teacher[key] = [0.0 if a in best else -30.0 for a in keys]
        hist = []
        while True:
            d, best = CB.optimal(pos, tp, ep["lava"], block, ep["n"])
            if not best:
                break
            state = {"observation": CB.render(ep, pos)}
            if hist:
                state["recent_actions"] = hist[-5:]
            key = f"ctl/{seed}/{e}/{len(hist)}"
            y = keys.index(sorted(best)[0])
            out.append({"key": key, "src": "control_grid", "state": json.dumps(state, ensure_ascii=False),
                        "q": f"Command: go to the {col} {k}. " + INSTR, "type": "choice", "keys": keys, "texts": texts, "y": y})
            teacher[key] = [0.0 if a in best else -30.0 for a in keys]        # eşit dağılım en kısa hamleler üzerinde
            a = rng.choice(sorted(best)); dr, dc = CB.MOVES[a]; pos = (pos[0] + dr, pos[1] + dc); hist.append(a)
            if pos == tp:
                break
    return out, teacher


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-12B-it"); ap.add_argument("--init_adapter", default="")
    ap.add_argument("--episodes", type=int, default=3000); ap.add_argument("--train", type=int, default=1200)
    ap.add_argument("--lr", type=float, default=5e-5); ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--adapter", default="lora_ctl"); ap.add_argument("--no4bit", action="store_true")
    ap.add_argument("--work", default=""); ap.add_argument("--save_every", type=int, default=400)
    ap.add_argument("--offpath", type=int, default=0)
    a = ap.parse_args()
    tr, teach = examples(a.episodes, 1000, a.offpath)
    dev, _ = examples(60, 900, a.offpath)                        # seçim: ayrı tohum (ölçüm tohumu 0'a hiç dokunulmaz)
    print({"train_examples": len(tr), "dev_examples": len(dev)}, flush=True)
    S = KL.Scorer(a.model, four_bit=not a.no4bit)
    S.tok.padding_side = "left"
    if S.tok.pad_token is None:
        S.tok.pad_token = S.tok.eos_token
    KL.train_lora(S, tr, a.train, a.lr, a.bs, a.adapter, seed=0, init_adapter=KL.resolve_adapter(a.init_adapter),
                  dev=dev[:300], eval_every=200, teacher=teach, alpha=0.5, save_every=a.save_every)
    json.dump({"best_dev": getattr(S, "best_dev", None), "train_examples": len(tr)}, open(a.adapter.rstrip("/") + "_train.json", "w"))
    print("EGITIM BITTI", getattr(S, "best_dev", None), flush=True)


if __name__ == "__main__":
    main()
