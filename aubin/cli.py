"""aubin decide case.json | aubin serve --port 8009 | aubin loop "task" [--generator gemma|openai:gpt-4o-mini] [--numeric]

case.json: {"state": {...}, "questions": {id: {type, instructions, criteria}}}
  --think 4                       eminsiz sorularda kısa akıl yürütme (0 = kapalı, tek geçiş)
  --model "repoA:0.6,repoB:0.4"   farklı tabanlardaki üyelerin ansamblı (--ens_T birleşim sıcaklığı)
"""
import argparse, json, sys


def main():
    ap = argparse.ArgumentParser(prog="aubin")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("decide"); d.add_argument("case"); d.add_argument("--model", default="emrevrg/AUBIN-12B")
    s = sub.add_parser("serve"); s.add_argument("--model", default="emrevrg/AUBIN-12B"); s.add_argument("--port", type=int, default=8009)
    s.add_argument("--learn", default="", help="öz-öğrenme belleği dosyası (.npz); verilirse /learn etkin, bellek kalıcı")
    # aubin loop "görev" : üretici aday yazar → AUBIN seçer → güven düşükse yeniden üret (üretici: gemma | openai:<model>)
    lp = sub.add_parser("loop"); lp.add_argument("task"); lp.add_argument("--model", default="emrevrg/AUBIN-12B")
    lp.add_argument("--generator", default="gemma"); lp.add_argument("--base_url", default="https://api.openai.com/v1")
    lp.add_argument("--n", type=int, default=4); lp.add_argument("--rounds", type=int, default=3)
    lp.add_argument("--target", type=float, default=0.8); lp.add_argument("--numeric", action="store_true")
    for x in (d, s, lp):
        x.add_argument("--perms", type=int, default=1); x.add_argument("--think", type=float, default=None)
        x.add_argument("--think_mix", type=float, default=None)
        x.add_argument("--ens_T", type=float, default=0.8)
    a = ap.parse_args()
    from .core import Aubin, AubinEnsemble
    specs = [x.rsplit(":", 1) if ":" in x else (x, "1") for x in a.model.split(",")]
    ms = [(Aubin(r, perms=a.perms, think_margin=a.think, think_mix=a.think_mix), float(w)) for r, w in specs]
    m = ms[0][0] if len(ms) == 1 else AubinEnsemble(ms, a.ens_T)
    if a.cmd == "loop":
        from .loop import AubinLoop, GemmaGenerator, OpenAIGenerator, last_number, final_line
        base = ms[0][0]
        gen = GemmaGenerator(base) if a.generator == "gemma" else OpenAIGenerator(a.generator.split(":", 1)[1], a.base_url)
        loop = AubinLoop(m, gen, extract=last_number if a.numeric else final_line, target=a.target, max_rounds=a.rounds,
                         n=a.n, mode="aubin+vote")
        print(json.dumps(loop.solve(a.task), ensure_ascii=False, indent=1))
    elif a.cmd == "decide":
        case = json.load(open(a.case, encoding="utf-8")) if a.case != "-" else json.load(sys.stdin)
        print(json.dumps(m.decide(case["state"], case["questions"]), ensure_ascii=False, indent=1))
    else:
        from .server import serve
        if a.learn:                                  # AUBIN-Learn: bellek + öz-kalibrasyon; dosya varsa yüklenir, çıkışta saklanır
            import atexit, os
            from .learn import AubinLearning
            m = AubinLearning(m)
            if os.path.exists(a.learn):
                m.load(a.learn)
            atexit.register(lambda: m.save(a.learn))
        serve(m, a.port)


if __name__ == "__main__":
    main()
