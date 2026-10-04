"""AUBIN Omni son hali: 12B ekran + 31B web sonuçlarını manifestoya (omni.json), PRESETS'e, Omni kartına ve tüm depoların code/
klasörüne işler. Depo görünürlüğüne DOKUNMAZ (hepsi özel kalır).

    python finalize_omni.py --s12 0.683 [--w31 emrevrg/AUBIN-31B-Web --w31acc 45.0] [--dry]
"""
import argparse, io, json, os, re
from huggingface_hub import CommitOperationAdd, HfApi

HERE = os.path.dirname(os.path.abspath(__file__))
E4B = "google/gemma-4-E4B-it"
REPOS = ["AUBIN-31B", "AUBIN-12B", "AUBIN-Omni", "AUBIN-E4B-Screen", "AUBIN-12B-Web", "AUBIN-E4B-Web", "AUBIN-12B-Control",
         "AUBIN-E4B-Control", "AUBIN-12B-v3b", "AUBIN-12B-v3d", "AUBIN-E4B-v3"]
CODE = ["aubin/omni.py", "screenspot_eval.py", "screenspot_train.py"]


def comp(adapter):
    return {"base": E4B, "adapter": adapter}


def manifest(a):
    s12 = "base" if a.s12 else comp("emrevrg/AUBIN-E4B-Screen")
    w31 = a.w31 or comp("emrevrg/AUBIN-E4B-Web")
    return {
        "12B": {"base": "google/gemma-4-12B-it", "decide": "emrevrg/AUBIN-12B", "web": "emrevrg/AUBIN-12B-Web",
                "control": "emrevrg/AUBIN-12B-Control", "screen": s12},
        "31B": {"base": "google/gemma-4-31B-it", "decide": "emrevrg/AUBIN-31B", "web": w31,
                "screen": comp("emrevrg/AUBIN-E4B-Screen"), "control": comp("emrevrg/AUBIN-E4B-Control")},
        "E4B": {"base": E4B, "screen": "emrevrg/AUBIN-E4B-Screen", "web": "emrevrg/AUBIN-E4B-Web",
                "control": "emrevrg/AUBIN-E4B-Control", "decide": "emrevrg/AUBIN-E4B-v3"},
    }


def table(a):
    s12 = f"✅ ScreenSpot {a.s12 * 100:.1f} ¹" if a.s12 else "✅ via E4B-Screen ²"
    w31 = (f"✅ Mind2Web {a.w31acc} ³" if a.w31 else
           f"✅ via E4B-Web ² · AUBIN-31B-Web: Mind2Web {a.w31acc} ³" if a.w31acc else "✅ via E4B-Web ²")
    return "\n".join([
        "| ability | call | E4B (fast) | 12B | 31B |",
        "|---|---|---|---|---|",
        "| typed, calibrated decisions | `omni.decide(state, questions)` | ✅ | ✅ | ✅ |",
        f"| click on screenshots | `omni.click(image, \"open settings\")` | ✅ ScreenSpot 69.3 | {s12} | ✅ via E4B-Screen ² |",
        f"| web agent step | `omni.web_step(task, candidates)` | ✅ | ✅ Mind2Web 43.5 | {w31} |",
        "| real-time control and games | `omni.act(obs, command, actions, allowed=…)` | ✅ | ✅ 92% · 0 lava | ✅ via E4B-Control ² |",
        "| self-learning and new skills | `omni.enable_learning()` → `learn` · `acquire_skill` | ✅ | ✅ | ✅ |",
        "",
        ("¹ The Gemma-4-12B base itself (no AUBIN adapter), run with fp32 compute: in fp16, image tokens overflow the 12B language "
         "model and every answer comes out empty. Full ScreenSpot, 1,272 screenshots. AUBIN fine-tuning of the 12B screen ability "
         "is the next training run.  " if a.s12 else ""),
        ("² Companion model: this ability runs on a separate, lazily loaded E4B adapter (measured on its own card), "
         "so one `omni` object still offers every ability. `omni.companions` lists them.  "),
        (f"³ AUBIN-31B web adapter, Mind2Web MindAct protocol (cross-domain split, 200 steps)." if a.w31 else
         f"³ [AUBIN-31B-Web](https://huggingface.co/emrevrg/AUBIN-31B-Web), Mind2Web MindAct protocol (cross-domain split, 200 steps). "
         "Its weights are uploaded with the public release; until then Omni's 31B web step uses the E4B-Web companion." if a.w31acc else ""),
    ]).rstrip() + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s12", type=float, default=0.0); ap.add_argument("--w31", default=""); ap.add_argument("--w31acc", default="")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    man = manifest(a)
    # PRESETS (kod) = manifesto
    p = os.path.join(HERE, "aubin", "omni.py")
    src = open(p, encoding="utf-8").read()
    body = "PRESETS = " + json.dumps(man, indent=4).replace("\n", "\n") + "\n"
    src = re.sub(r"PRESETS = \{.*?\n\}\n", lambda _: body, src, count=1, flags=re.S)
    open(p, "w", encoding="utf-8").write(src)
    compile(src, p, "exec")
    # Omni kartı: yetenek tablosu
    cp = os.path.join(HERE, "release", "omni_card.md")
    card = open(cp, encoding="utf-8").read()
    start = card.index("| ability | call |"); end = card.index("The ability manifest lives in")
    card = card[:start] + table(a) + "\n" + card[end:]
    open(cp, "w", encoding="utf-8").write(card)
    full = {"name": "AUBIN Omni", "sizes": man,
            "note": "abilities are AUBIN LoRA adapters on one shared Gemma-4 base, loaded lazily on first use; "
                    "'base' = the base model itself (adapters off); {base, adapter} = a lazily loaded companion model"}
    print(json.dumps(full, indent=1)[:1500]); print(table(a))
    if a.dry:
        return
    api = HfApi(token=open(os.path.join(HERE, "..", ".hf_token")).read().strip())
    code = lambda: [CommitOperationAdd(path_in_repo="code/" + c, path_or_fileobj=os.path.join(HERE, c)) for c in CODE]
    api.create_commit("emrevrg/AUBIN-Omni", operations=code() + [
        CommitOperationAdd(path_in_repo="omni.json", path_or_fileobj=io.BytesIO(json.dumps(full, indent=1).encode("utf-8"))),
        CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=cp)],          # gövde; make_pro_cards.py banner/film/destek ekler
        commit_message="Omni: every ability in 12B and 31B (manifest + code)")
    for r in REPOS:
        if r != "AUBIN-Omni":
            api.create_commit("emrevrg/" + r, operations=code(), commit_message="Omni: 12B/31B abilities, fp32 vision path")
        print("OK", r, flush=True)


if __name__ == "__main__":
    main()
