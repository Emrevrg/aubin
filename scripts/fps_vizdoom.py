"""AUBIN FPS ölçümü — ViZDoom (Kempka et al., 2016) "defend_the_center": canavarlar her yönden gelir, ajan döner ve ateş eder.
Ölçü = bölüm başına öldürme (kill). Algı: oyun motorunun etiket tamponu (nesne algılayıcısı rolü) → metin gözlem
("düşman, nişangâhın 12° solunda, yakın"); KARAR: AUBIN (AubinController, her hamle tek ileri geçiş). Karşılaştırma:
rastgele ajan ve kural ajanı (en yakın düşmana dön, nişandaysa ateş et) aynı tohumlarla.

    python fps_vizdoom.py --episodes 5 --baseline_episodes 20 --adapter emrevrg/AUBIN-12B --out fps.json
"""
import argparse, json, os, random, subprocess, sys, time


def ensure_vizdoom():
    try:
        import vizdoom  # noqa
    except Exception:
        subprocess.run([sys.executable, "-m", "pip", "-q", "install", "vizdoom"], check=False)
    import vizdoom
    return vizdoom


ACTIONS = {"turn_left": "turn the view left", "turn_right": "turn the view right", "attack": "fire the weapon at the crosshair"}


def make_game(vzd, seed):
    g = vzd.DoomGame()
    g.load_config(os.path.join(vzd.scenarios_path, "defend_the_center.cfg"))
    g.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
    g.set_labels_buffer_enabled(True); g.set_window_visible(False); g.set_seed(seed)
    g.set_mode(vzd.Mode.PLAYER); g.init()
    btn = [str(b).split(".")[-1] for b in g.get_available_buttons()]
    return g, btn


def observe(state, W=320, fov=90.0):
    """Etiketlerden düşmanlar: nişangâha açısal uzaklık (derece, -sol/+sağ), boy (yakınlık), nişanda mı."""
    enemies = []
    for l in state.labels:
        if l.object_name in ("DoomPlayer",) or "Puff" in l.object_name or "Blood" in l.object_name:
            continue
        cx = l.x + l.width / 2
        ang = (cx - W / 2) / W * fov
        enemies.append({"ang": round(ang, 1), "h": int(l.height), "on": l.x <= W / 2 <= l.x + l.width})
    enemies.sort(key=lambda e: (-e["h"], abs(e["ang"])))
    return enemies


def describe(enemies, ammo, health, coarse=False):
    if not enemies:
        s = "No enemy visible."
    else:
        parts = []
        for e in enemies[:4]:
            deg = (round(abs(e["ang"]) / 10) * 10) if coarse else round(abs(e["ang"]))   # kaba açı: bellek benzerliği için
            side = "ON the crosshair" if e["on"] else (f"{deg:.0f} degrees to the {'left' if e['ang'] < 0 else 'right'}")
            dist = "very close" if e["h"] > 90 else ("near" if e["h"] > 45 else "far")
            parts.append(f"enemy {side}, {dist}")
        s = "; ".join(parts) + "."
    return f"{s} Ammo {ammo}, health {health}."


def run(vzd, policy, seed, max_steps=700, mem=None):
    g, btn = make_game(vzd, seed)
    idx = {"turn_left": btn.index("TURN_LEFT"), "turn_right": btn.index("TURN_RIGHT"), "attack": btn.index("ATTACK")}
    g.new_episode(); steps = 0; ms = []
    if mem is not None:
        mem["kills"] = lambda: int(g.get_game_variable(vzd.GameVariable.KILLCOUNT))
    while not g.is_episode_finished() and steps < max_steps:
        st = g.get_state()
        en = observe(st)
        ammo = int(g.get_game_variable(vzd.GameVariable.AMMO2)); hp = int(g.get_game_variable(vzd.GameVariable.HEALTH))
        t0 = time.perf_counter(); a = policy(en, ammo, hp); ms.append((time.perf_counter() - t0) * 1e3)
        act = [0] * len(btn); act[idx[a]] = 1
        g.make_action(act, 4); steps += 1
    kills = int(g.get_game_variable(vzd.GameVariable.KILLCOUNT))
    g.close()
    return {"seed": seed, "kills": kills, "steps": steps, "median_ms": round(sorted(ms)[len(ms) // 2], 1) if ms else 0}


def rule_policy(en, ammo, hp):
    if not en:
        return "turn_left"
    e = en[0]
    if e["on"]:
        return "attack"
    return "turn_left" if e["ang"] < 0 else "turn_right"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=5); ap.add_argument("--baseline_episodes", type=int, default=20)
    ap.add_argument("--adapter", default="emrevrg/AUBIN-12B"); ap.add_argument("--out", default="fps.json")
    ap.add_argument("--work", default=""); ap.add_argument("--learn", action="store_true")
    a = ap.parse_args()
    vzd = ensure_vizdoom()
    R = {"protocol": __doc__.split("    python")[0].strip(), "scenario": "defend_the_center", "rows": {}}
    rng = random.Random(0)
    for name, pol in () if a.baseline_episodes <= 0 else (("random", lambda e, am, hp: rng.choice(list(ACTIONS))), ("rule: turn to nearest, fire when aimed", rule_policy)):
        eps = [run(vzd, pol, 1000 + i) for i in range(a.baseline_episodes)]
        R["rows"][name] = {"episodes": len(eps), "mean_kills": round(sum(e["kills"] for e in eps) / len(eps), 2),
                           "max_kills": max(e["kills"] for e in eps), "per_episode": eps}
        print(name, R["rows"][name]["mean_kills"], flush=True)
        json.dump(R, open(a.out, "w"), indent=1)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from aubin import Aubin, AubinController
    ctl = AubinController(Aubin(a.adapter), actions=ACTIONS,
                          instructions="You are playing a first-person shooter. Enemies approach from all sides. "
                                       "Turn toward the closest enemy and fire only when an enemy is ON the crosshair; do not waste ammo.")

    def aubin_policy(en, ammo, hp):
        return ctl.act(describe(en, ammo, hp), command="kill the enemies and survive")["action"]
    tag = f"AUBIN ({a.adapter})"
    if a.learn:                                        # deneyerek öğrenme: ilerleme getiren hamleler anında belleğe
        from aubin import AubinLearning
        smart = AubinLearning(ctl.m, min_sim=0.95)        # yalnız aynı oyun durumu oy verir
        # bellek anahtarı = oyunun ayrık durumu (metnin karma vektörü sol/sağ ayrımını kaçırıyordu → tek hamleye çöküyordu)
        smart._text = lambda state, q: state.get("situation", "")
        Q = {"action": {"type": "choice", "criteria": ACTIONS, "instructions": "Command: kill the enemies and survive. " + ctl.instructions}}
        mem = {"prev": None}

        def aubin_policy(en, ammo, hp):
            e0 = en[0] if en else None
            sit = ("none" if e0 is None else ("on_crosshair" if e0["on"] else ("left" if e0["ang"] < 0 else "right"))
                   + "_" + ("far" if e0["h"] <= 45 else "near")) + ("_noammo" if ammo <= 0 else "")
            st = {"observation": describe(en, ammo, hp), "situation": f"situation {sit}"}
            if mem["prev"] is not None:                   # önceki hamlenin sonucu: ilerleme varsa o hamle doğrudur
                pst, pa, pen, pk = mem["prev"]
                kills = mem["kills"]()
                near0 = abs(pen[0]["ang"]) if pen else None; near1 = abs(en[0]["ang"]) if en else None
                good = (pa == "attack" and kills > pk) or (pa != "attack" and near0 is not None and near1 is not None
                                                         and (near1 < near0 - 1 or (en and en[0]["on"])))
                if good:
                    smart.learn(pst, Q, {"action": pa})
            out = smart.decide(st, Q)["action"]["answer"]
            mem["prev"] = (st, out, en, mem["kills"]())
            return out
        tag += " + AUBIN-Learn (online, memory persists across episodes)"
    eps = []
    for i in range(a.episodes):
        ctl.reset(); mem_reset = locals().get("mem")
        if mem_reset is not None:
            mem_reset["prev"] = None
        eps.append(run(vzd, aubin_policy, 1000 + i, mem=locals().get("mem"))); print("AUBIN bölüm", eps[-1], flush=True)
        R["rows"][tag] = {"episodes": len(eps), "mean_kills": round(sum(e["kills"] for e in eps) / len(eps), 2),
                                             "max_kills": max(e["kills"] for e in eps), "per_episode": eps}
        json.dump(R, open(a.out, "w"), indent=1)
    print("SONUC", json.dumps({k: v["mean_kills"] for k, v in R["rows"].items()}), flush=True)


if __name__ == "__main__":
    main()
