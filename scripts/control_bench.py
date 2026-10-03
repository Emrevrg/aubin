"""AUBIN Control ölçümü: komutlu ızgara oyunu (anlık hamle, doğru cevap her adımda kesin bilinir).

Her bölüm: 7x7 ızgara, duvarlar, lav (~), renkli nesneler (engel), komut "go to the <renk> <nesne>".
Her adımda doğru hamleler = hedefe giden EN KISA yolların ilk adımları (BFS, lav ve nesnelerden kaçınarak).
Ölçüler: adım doğruluğu, bölüm başarısı, lava düşme, güven ≥ eşik iken doğruluk ve kapsama, karar başına gecikme (ms).
Karşılaştırma: rastgele ve açgözlü (Manhattan, engelleri görmez) politikalar — aynı bölümler.
    python control_bench.py --adapter aubin12-plain --episodes 100 --out control.json
"""
import argparse, collections, json, os, random, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
MOVES = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
ACTIONS = {"up": "move one cell up (row - 1)", "down": "move one cell down (row + 1)",
           "left": "move one cell left (column - 1)", "right": "move one cell right (column + 1)"}
COLORS, KINDS = ["red", "green", "blue", "yellow"], ["key", "ball", "box"]


def make(rng, n=7, lava=5, objs=4):
    while True:
        cells = [(r, c) for r in range(1, n + 1) for c in range(1, n + 1)]
        rng.shuffle(cells)
        agent, rest = cells[0], cells[1:]
        items = [(rest[i], rng.choice(COLORS), rng.choice(KINDS)) for i in range(objs)]
        names = {(c, k) for _, c, k in items}
        if len(names) < objs:
            continue
        lv = set(rest[objs: objs + lava])
        tgt = items[0]
        if optimal(agent, tgt[0], lv, {p for p, _, _ in items[1:]}, n)[0] is not None:
            return {"n": n, "agent": agent, "items": items, "lava": lv, "target": tgt}


def optimal(start, goal, lava, block, n):
    """BFS uzaklık haritası hedeften; dönüş: (uzaklık, en kısa yolun ilk adımları)."""
    ok = lambda p: 1 <= p[0] <= n and 1 <= p[1] <= n and p not in lava and p not in block
    dist, dq = {goal: 0}, collections.deque([goal])
    while dq:
        p = dq.popleft()
        for dr, dc in MOVES.values():
            q = (p[0] + dr, p[1] + dc)
            if q not in dist and (ok(q) or q == start):
                dist[q] = dist[p] + 1; dq.append(q)
    if start not in dist:
        return None, set()
    best = {a for a, (dr, dc) in MOVES.items() if dist.get((start[0] + dr, start[1] + dc), 1e9) == dist[start] - 1}
    return dist[start], best


def render(ep, pos):
    n = ep["n"]; g = [["#"] * (n + 2) for _ in range(n + 2)]
    for r in range(1, n + 1):
        for c in range(1, n + 1):
            g[r][c] = "."
    for p in ep["lava"]:
        g[p[0]][p[1]] = "~"
    for p, col, k in ep["items"]:
        g[p[0]][p[1]] = col[0].upper() if k == "key" else col[0]
    g[pos[0]][pos[1]] = "@"
    obj = [f"{col} {k} at row {p[0]}, column {p[1]}" for p, col, k in ep["items"]]
    return {"grid": "\n".join("".join(r) for r in g),
            "legend": "@ = you, # = wall, ~ = lava (deadly), letters = objects (they block movement)",
            "you": f"row {pos[0]}, column {pos[1]}", "objects": obj}


def run(policy, eps, max_extra=4):
    steps = hit = succ = dead = 0; lat, conf_rows = [], []
    for ep in eps:
        getattr(policy, "reset", lambda: None)()                 # yeni bölüm: hamle geçmişi sıfır (eğitim biçimiyle aynı)
        pos = ep["agent"]; tp, col, k = ep["target"]; block = {p for p, _, _ in ep["items"][1:]}
        d0, _ = optimal(pos, tp, ep["lava"], block, ep["n"])
        for _ in range(d0 + max_extra):
            d, best = optimal(pos, tp, ep["lava"], block, ep["n"])
            obs = render(ep, pos)
            # görünen haritadan güvenli hamleler (lav / duvar / nesneye gitmeyen; hedef hücre serbest) — güvenlik kalkanı için
            safe = {a for a, (dr, dc) in MOVES.items() if (q := (pos[0] + dr, pos[1] + dc)) == tp or
                    (1 <= q[0] <= ep["n"] and 1 <= q[1] <= ep["n"] and q not in ep["lava"] and q not in block)}
            t0 = time.perf_counter(); a, c = policy(obs, f"go to the {col} {k}", pos, tp, safe); lat.append((time.perf_counter() - t0) * 1000)
            steps += 1; ok = a in best; hit += ok; conf_rows.append((c, ok))
            dr, dc = MOVES.get(a, (0, 0)); q = (pos[0] + dr, pos[1] + dc)
            if q == tp:
                succ += 1; break
            if q in ep["lava"]:
                dead += 1; break
            if 1 <= q[0] <= ep["n"] and 1 <= q[1] <= ep["n"] and q not in block:
                pos = q
    out = {"episodes": len(eps), "decisions": steps, "step_accuracy": round(hit / steps, 4), "success": round(succ / len(eps), 4),
           "lava_deaths": dead, "ms_per_decision_median": round(sorted(lat)[len(lat) // 2], 1)}
    for th in (0.8, 0.9):
        sel = [ok for c, ok in conf_rows if c is not None and c >= th]
        if sel:
            out[f"acc_when_conf_ge_{th}"] = round(sum(sel) / len(sel), 4); out[f"coverage_conf_ge_{th}"] = round(len(sel) / len(conf_rows), 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="aubin12-plain"); ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--work", default=""); ap.add_argument("--out", default="control.json")
    ap.add_argument("--no_model", action="store_true"); ap.add_argument("--label", default="AUBIN-12B")
    a = ap.parse_args()
    rng = random.Random(a.seed); eps = [make(rng) for _ in range(a.episodes)]
    R = {"episodes": a.episodes, "seed": a.seed, "rows": {}}
    r2 = random.Random(1)
    R["rows"]["random"] = run(lambda o, cmd, p, t, s: (r2.choice(list(MOVES)), None), eps)
    def greedy(o, cmd, p, t, s=None):
        dr, dc = t[0] - p[0], t[1] - p[1]
        return ("down" if dr > 0 else "up") if abs(dr) >= abs(dc) and dr else ("right" if dc > 0 else "left"), None
    R["rows"]["greedy (ignores obstacles)"] = run(greedy, eps)
    # aynı güvenlik kalkanıyla kural tabanlı taban çizgileri (kalkanın tek başına katkısını ayırmak için)
    r3 = random.Random(2)
    R["rows"]["random + safety shield"] = run(lambda o, cmd, p, t, s: (r3.choice(sorted(s)) if s else "up", None), eps)
    def greedy_shield(o, cmd, p, t, s):
        dr, dc = t[0] - p[0], t[1] - p[1]
        pref = []                                                  # hedefe yaklaştıran hamleler önce (büyük eksen önce)
        vert = ("down" if dr > 0 else "up") if dr else None; hor = ("right" if dc > 0 else "left") if dc else None
        pref += [m for m in ((vert, hor) if abs(dr) >= abs(dc) else (hor, vert)) if m]
        for m in pref + sorted(s):
            if m in s:
                return m, None
        return "up", None
    R["rows"]["greedy + safety shield"] = run(greedy_shield, eps)
    if not a.no_model:
        from kev_llm import resolve_adapter
        from aubin.core import Aubin
        from aubin.control import AubinController
        A = Aubin(resolve_adapter(a.adapter) or None, think_margin=0.0)
        ctl = AubinController(A, ACTIONS, instructions="Choose the next move that follows the command along a shortest safe path "
                                                       "(never step on lava, objects block movement).")
        class Pol:                                               # bölüm başında ctl.reset; kalkan isteğe bağlı
            def __init__(self, shield): self.shield = shield
            def reset(self): ctl.reset()
            def __call__(self, o, cmd, p, t, s):
                r = ctl.act(o, command=cmd, allowed=s if self.shield else None); return r["action"], r["confidence"]
        R["rows"][f"{a.label} (one pass per move)"] = run(Pol(False), eps)
        R["rows"][f"{a.label} + safety shield (never steps on visible lava/walls)"] = run(Pol(True), eps)
    json.dump(R, open(a.out, "w"), indent=1)
    print("SONUC", json.dumps(R), flush=True)


if __name__ == "__main__":
    main()
