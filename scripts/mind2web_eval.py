"""AUBIN bilgisayar kullanımı (web ajanı) ölçümü — Mind2Web (Deng et al., 2023), MindAct çoktan-seçmeli protokolü.

Her adım: görev + önceki eylemler + resmi sıralayıcının (scores_all_data.pkl) ilk-K adayı → AUBIN tipli kararlar:
  1) element: hangi öğe? (K seçenek; >26 ise 26'lık gruplar + final turu)
  2) op: CLICK / TYPE / SELECT
  3) value: TYPE/SELECT ise kısa üretim
Ölçüler (Mind2Web ile aynı): Element Accuracy, Operation F1 (op+değer token F1), Step Success Rate.
Pozitif öğe ilk-K'da değilse adım başarısız sayılır (MindAct ile aynı).

    python mind2web_eval.py --model google/gemma-4-12B-it --adapter <lora> --split test_task --steps 200 --out m2w.json
"""
import argparse, json, os, pickle, random, re, sys, time
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

OPS = ["CLICK", "TYPE", "SELECT"]
KEEP = ("aria_label", "aria-label", "placeholder", "title", "name", "type", "value", "alt", "role", "id", "input_value", "label")


def load_split(split, limit_files=0):
    from huggingface_hub import HfApi, hf_hub_download
    import pyarrow.parquet as pq
    fs = sorted(f for f in HfApi().list_repo_files("osunlp/Multimodal-Mind2Web", repo_type="dataset")
                if f.startswith(f"data/{split}-") and f.endswith(".parquet"))
    rows = []
    for f in fs[: limit_files or None]:
        p = hf_hub_download("osunlp/Multimodal-Mind2Web", f, repo_type="dataset")
        t = pq.read_table(p, columns=["action_uid", "cleaned_html", "operation", "pos_candidates", "neg_candidates",
                                      "annotation_id", "confirmed_task", "action_reprs", "target_action_index", "website"])
        rows += t.to_pylist()
    return rows


def load_scores():
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("osunlp/Mind2Web", "scores_all_data.pkl", repo_type="dataset")
    with open(p, "rb") as f:
        return pickle.load(f)


def describe(cand, html_index):
    """Aday öğenin MindAct benzeri kısa metni: <etiket öznitelikler> görünen metin </etiket> + üst öğe bağlamı."""
    c = json.loads(cand) if isinstance(cand, str) else cand
    attrs = json.loads(c["attributes"]) if isinstance(c.get("attributes"), str) else (c.get("attributes") or {})
    nid = str(attrs.get("backend_node_id"))
    bits = " ".join(f'{k}="{str(attrs[k])[:40]}"' for k in KEEP if attrs.get(k))
    own, parent = html_index.get(nid, ("", ""))
    s = f"<{c.get('tag', '?')} {bits}> {own[:90]} </{c.get('tag', '?')}>".replace("  ", " ")
    if parent and parent != own:
        s += f" (inside: {parent[:70]})"
    return nid, s[:260]


def html_text_index(cleaned_html):
    from lxml import html as lh
    idx = {}
    try:
        root = lh.fromstring(cleaned_html)
    except Exception:
        return idx
    for el in root.iter():
        nid = el.get("backend_node_id") if hasattr(el, "get") else None
        if nid:
            try:
                own = re.sub(r"\s+", " ", el.text_content()).strip()
                par = el.getparent()
                ptxt = re.sub(r"\s+", " ", par.text_content()).strip() if par is not None else ""
                idx[nid] = (own[:160], ptxt[:160])
            except Exception:
                pass
    return idx


def tournament(S, st, cands, group=5):
    """MindAct çoktan-seçmeli: 5'li gruplar + 'None of the above'; seçilenler bir sonraki tura; tek aday kalana dek."""
    alive, rounds = list(cands), 0
    while len(alive) > 1:
        nxt = []
        for i in range(0, len(alive), group):
            g = alive[i:i + group]
            it = {"state": st, "q": "Which page element should be acted on next to accomplish the task? Choose 'None of the above' if none fits.",
                  "keys": [c[0] for c in g] + ["none"], "texts": [c[1] for c in g] + ["None of the above"], "y": 0}
            with torch.no_grad():
                lp = S.score(it)
            j = int(lp.argmax())
            if j < len(g):
                nxt.append((g[j], float(torch.softmax(lp, -1)[j])))
        rounds += 1
        if not nxt:
            return None, 0.0, rounds
        alive = [x[0] for x in nxt]
        if len(alive) == 1:
            return alive[0], nxt[0][1], rounds
    return (alive[0], 1.0, rounds) if alive else (None, 0.0, rounds)


def train_items(rows, scores, topk, rng, per_action=2):
    """Mind2Web EĞİTİM bölümünden MindAct tarzı çoktan-seçmeli örnekler (kev_llm item biçimi):
    grup = 5 aday (+ 'None of the above'); pozitif içeren grupta cevap pozitif, içermeyende 'None'."""
    out = []
    for r in rows:
        key = f"{r['annotation_id']}_{r['action_uid']}"
        sc = scores.get(key) if isinstance(scores, dict) else None
        idx = html_text_index(r["cleaned_html"])
        pos = [describe(c, idx) for c in r["pos_candidates"]]
        neg = [describe(c, idx) for c in r["neg_candidates"]]
        if not pos or not neg:
            continue
        if sc:
            neg.sort(key=lambda x: -float(sc.get(x[0], -1e9)))
        neg = neg[:topk]
        hist = r["action_reprs"][: int(r["target_action_index"])]
        st = json.dumps({"task": r["confirmed_task"], "website": r["website"], "previous_actions": hist[-6:] or ["(none)"]}, ensure_ascii=False)
        q = "Which page element should be acted on next to accomplish the task? Choose 'None of the above' if none fits."
        n_pos = max(1, per_action - 1)                 # son grup hariç hepsi pozitif içerir (farklı negatiflerle)
        for k in range(per_action):
            g = rng.sample(neg, min(4 if k < n_pos else 5, len(neg)))
            if k < n_pos:
                g = g + [rng.choice(pos)]
            rng.shuffle(g)
            keys = [c[0] for c in g] + ["none"]; texts = [c[1] for c in g] + ["None of the above"]
            y = next((i for i, c in enumerate(g) if c in pos), len(g))
            out.append({"key": f"{key}/{k}", "src": "mind2web_train", "state": st, "q": q, "type": "choice",
                        "keys": keys, "texts": texts, "y": y})
        opr = json.loads(r["operation"]) if isinstance(r["operation"], str) else r["operation"]
        out.append({"key": f"{key}/op", "src": "mind2web_train", "state": st + f"\nChosen element: {pos[0][1]}",
                    "q": "Which operation should be performed on the chosen element?", "type": "choice",
                    "keys": OPS, "texts": ["CLICK: click it", "TYPE: type text into it", "SELECT: choose an option from it"],
                    "y": OPS.index(opr["op"]) if opr["op"] in OPS else 0})
    return out


def f1(pred, gold):
    p, g = pred.lower().split(), gold.lower().split()
    if not p and not g:
        return 1.0
    common = sum(min(p.count(w), g.count(w)) for w in set(p))
    if common == 0:
        return 0.0
    pr, rc = common / len(p), common / len(g)
    return 2 * pr * rc / (pr + rc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-12B-it"); ap.add_argument("--adapter", default="")
    ap.add_argument("--split", default="test_task"); ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--topk", type=int, default=50); ap.add_argument("--files", type=int, default=0)
    ap.add_argument("--no4bit", action="store_true"); ap.add_argument("--device_map", default="")
    ap.add_argument("--work", default=""); ap.add_argument("--out", default="m2w.json"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train_steps", type=int, default=0); ap.add_argument("--train_files", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-5); ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--save_adapter", default="lora_m2w"); ap.add_argument("--per_action", type=int, default=2)
    ap.add_argument("--value_fmt", type=int, default=1)   # 1: değer Mind2Web yazımında (eğitim verisindeki gibi ":" "," boşluk)
    ap.add_argument("--select", default="tournament", choices=["tournament", "flat"])  # flat: tek K-yollu seçim (26'lık gruplar + final)
    a = ap.parse_args()
    t0 = time.time()
    rows = load_split(a.split, a.files)
    S_ = load_scores()
    scores = S_.get("scores", S_) if isinstance(S_, dict) else S_
    print({"rows": len(rows), "score_keys_sample": list(scores)[:2] if isinstance(scores, dict) else type(scores).__name__,
           "load_s": round(time.time() - t0, 1)}, flush=True)
    rng = random.Random(a.seed)
    order = list(range(len(rows))); rng.shuffle(order)
    from kev_llm import Scorer, resolve_adapter
    S = Scorer(a.model, four_bit=not a.no4bit, device_map=a.device_map)
    train_info = None
    if a.train_steps:                                  # Mind2Web EĞİTİM bölümüyle ince ayar (MindAct gibi), sonra aynı test ölçümü
        from kev_llm import train_lora
        tr_rows = load_split("train", a.train_files)
        tr = train_items(tr_rows, scores, a.topk, random.Random(a.seed + 1), per_action=a.per_action)
        random.Random(a.seed + 2).shuffle(tr)
        dev, tr = tr[:150], tr[150:]
        print({"train_actions": len(tr_rows), "train_items": len(tr), "dev_items": len(dev)}, flush=True)
        tr_rows = None
        t1 = time.time()
        train_lora(S, tr, a.train_steps, a.lr, a.bs, a.save_adapter, seed=a.seed, init_adapter=resolve_adapter(a.adapter) if a.adapter else "",
                   dev=dev, eval_every=max(100, a.train_steps // 6))
        S.m.eval()
        train_info = {"steps": a.train_steps, "items": len(tr), "lr": a.lr, "bs": a.bs, "minutes": round((time.time() - t1) / 60, 1),
                      "best_dev": getattr(S, "best_dev", None), "adapter_dir": a.save_adapter}
        print(train_info, flush=True)
    elif a.adapter:
        from peft import PeftModel
        S.m = PeftModel.from_pretrained(S.m, resolve_adapter(a.adapter)).eval()
    R = {"train": train_info, "split": a.split, "select": a.select, "protocol": (f"MindAct multi-choice (groups of 5 + None, iterative), top-{a.topk} by official ranker" if a.select == "tournament" else f"single {a.topk}-way choice (26-way groups + final), top-{a.topk} by official ranker"), "steps": [], "model": a.model,
         "adapter": a.adapter}
    n_done = 0
    for i in order:
        if n_done >= a.steps:
            break
        r = rows[i]
        key = f"{r['annotation_id']}_{r['action_uid']}"
        sc = scores.get(key) if isinstance(scores, dict) else None
        idx = html_text_index(r["cleaned_html"])
        pos = [describe(c, idx) for c in r["pos_candidates"]]
        neg = [describe(c, idx) for c in r["neg_candidates"]]
        allc = pos + neg
        if not pos:
            continue
        if sc:
            allc.sort(key=lambda x: -float(sc.get(x[0], -1e9)))
        else:
            rng.shuffle(allc)
        cands = allc[: a.topk]
        pos_ids = {p[0] for p in pos}
        hist = r["action_reprs"][: int(r["target_action_index"])]
        state = {"task": r["confirmed_task"], "website": r["website"], "previous_actions": hist[-6:] or ["(none)"]}
        st = json.dumps(state, ensure_ascii=False)
        if a.select == "flat":
            item = {"state": st, "q": "Which page element should be acted on next to accomplish the task?",
                    "keys": [c[0] for c in cands], "texts": [c[1] for c in cands], "y": 0}
            with torch.no_grad():
                lpf = S.score(item)
            jf = int(lpf.argmax()); win, p_pick = cands[jf], float(torch.softmax(lpf, -1)[jf])
        else:
            win, p_pick, _r = tournament(S, st, cands)
        if win is None:                                # model "hiçbiri" dedi: adım başarısız, yine de ilk adayla devam
            win = cands[0]
        pick_id = win[0]
        pick = [c[0] for c in cands].index(pick_id)
        op_item = {"state": st + f"\nChosen element: {cands[pick][1]}", "q": "Which operation should be performed on the chosen element?",
                   "keys": OPS, "texts": ["CLICK: click it", "TYPE: type text into it", "SELECT: choose an option from it"], "y": 0}
        with torch.no_grad():
            op = OPS[int(S.score(op_item).argmax())]
        val = ""
        if op in ("TYPE", "SELECT"):
            msgs = [{"role": "user", "content": f"Task: {r['confirmed_task']}\nPrevious actions: {hist[-6:]}\nElement: {cands[pick][1]}\n"
                                                 f"Operation: {op}. Reply with ONLY the exact text to {'type' if op == 'TYPE' else 'select'}."}]
            p = S.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            enc = S.tok([p], return_tensors="pt", add_special_tokens=False).to(S.dev)
            with torch.no_grad():
                g = S.m.generate(**enc, max_new_tokens=24, do_sample=False, pad_token_id=S.tok.pad_token_id)
            val = S.tok.decode(g[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip().split("\n")[0].strip().strip('"')
            if a.value_fmt:
                val = re.sub(r"[:,]", " ", val).strip()
        opr = json.loads(r["operation"]) if isinstance(r["operation"], str) else r["operation"]
        gold_op = opr["op"]; gold_val = opr.get("value", "") or ""
        ele_ok = pick_id in pos_ids
        op_f1 = f1(f"{op} {val}".strip(), f"{gold_op} {gold_val}".strip())
        R["steps"].append({"key": key, "pos_in_topk": bool(pos_ids & {c[0] for c in cands}), "ele": ele_ok, "op": op, "val": val,
                           "gold_op": gold_op, "gold_val": gold_val, "op_f1": round(op_f1, 4),
                           "step_sr": bool(ele_ok and op == gold_op and (gold_op == "CLICK" or f1(val, gold_val) == 1.0)),
                           "p_pick": round(p_pick, 4)})
        n_done += 1
        if n_done % 20 == 0:
            s = R["steps"]
            print({"n": n_done, "ele_acc": round(sum(x["ele"] for x in s) / len(s), 4), "op_f1": round(sum(x["op_f1"] for x in s) / len(s), 4),
                   "step_sr": round(sum(x["step_sr"] for x in s) / len(s), 4), "min": round((time.time() - t0) / 60, 1)}, flush=True)
            json.dump(R, open(a.out, "w"), indent=1)
    s = R["steps"]
    R["summary"] = {"n": len(s), "element_accuracy": round(sum(x["ele"] for x in s) / len(s), 4),
                    "operation_f1": round(sum(x["op_f1"] for x in s) / len(s), 4),
                    "step_success_rate": round(sum(x["step_sr"] for x in s) / len(s), 4),
                    "pos_in_topk_rate": round(sum(x["pos_in_topk"] for x in s) / len(s), 4), "minutes": round((time.time() - t0) / 60, 1)}
    json.dump(R, open(a.out, "w"), indent=1)
    print("SONUC", json.dumps(R["summary"]), flush=True)


if __name__ == "__main__":
    main()
