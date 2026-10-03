"""AUBIN Omni duman testi: tek taban + tüm yetenek adaptörleri; her yetenek bir kez çağrılır, süre ve çıktı yazılır.
ScreenSpot'tan 30 örnekte tıklama doğruluğu da ölçülür (tam ölçüm screenspot_eval.py ile).

    python omni_smoke.py --size E4B --out omni.json
"""
import argparse, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--size", default="E4B"); ap.add_argument("--out", default="omni.json")
    ap.add_argument("--work", default=""); ap.add_argument("--n_click", type=int, default=30)
    a = ap.parse_args()
    from aubin import AubinOmni
    t0 = time.time(); omni = AubinOmni.preset(a.size); R = {"size": a.size, "abilities": omni.abilities(), "load_s": round(time.time() - t0, 1)}
    print(R, flush=True)
    t = time.time()
    R["decide"] = omni.decide("Hi, we were billed twice for March. Please refund the duplicate or we cancel.",
                              {"department": {"type": "choice", "instructions": "Which department should handle this?",
                                              "criteria": {"billing": "invoices, payments, refunds", "technical": "bugs, outages", "other": "everything else"}},
                               "churn": {"type": "noul", "instructions": "Does the user threaten to cancel?"}})
    R["decide_ms"] = round((time.time() - t) * 1e3); print("decide", R["decide"], flush=True)
    if "web" in omni.abilities():
        t = time.time()
        R["web"] = omni.web_step("Book a one-way flight from New York to Paris on June 5",
                                 {"n1": '<input placeholder="From"> </input>', "n2": '<a> Hotels </a>', "n3": '<button> Sign in </button>'})
        R["web_ms"] = round((time.time() - t) * 1e3); print("web", R["web"], flush=True)
    if "control" in omni.abilities():
        t = time.time()
        R["control"] = omni.act("You are at row 3, col 3. The red key is at row 1, col 3. Lava at row 2, col 3; free: left, right, down.",
                                "go to the red key", {"up": "row - 1", "down": "row + 1", "left": "col - 1", "right": "col + 1"},
                                allowed={"left", "right", "down"})
        R["control_ms"] = round((time.time() - t) * 1e3); print("control", R["control"], flush=True)
    if "screen" in omni.abilities():
        from datasets import load_dataset
        ds = load_dataset("rootsautomation/ScreenSpot", split="test").select(range(a.n_click))
        ok, ms = 0, []
        for ex in ds:
            img = ex["image"].convert("RGB"); W, H = img.size
            x1, y1, x2, y2 = [float(v) for v in ex["bbox"]]
            if max(x1, y1, x2, y2) > 1.5:
                x1, x2, y1, y2 = x1 / W, x2 / W, y1 / H, y2 / H
            t = time.time(); r = omni.click(img, ex["instruction"]); ms.append((time.time() - t) * 1e3)
            ok += int(r["x"] is not None and x1 <= r["x"] / 1000 <= x2 and y1 <= r["y"] / 1000 <= y2)
        R["screen"] = {"n": a.n_click, "accuracy": round(ok / a.n_click, 3), "median_ms": round(sorted(ms)[len(ms) // 2])}
        print("screen", R["screen"], flush=True)
    R["decide_again"] = omni.decide("The app crashes every time I open settings.",
                                    {"department": {"type": "choice", "instructions": "Which department should handle this?",
                                                    "criteria": {"billing": "invoices, payments, refunds", "technical": "bugs, outages", "other": "everything else"}}})
    print("decide (adaptör geri geçişi)", R["decide_again"], flush=True)
    json.dump(R, open(a.out, "w"), indent=1, default=str)
    print("BITTI", flush=True)


if __name__ == "__main__":
    main()
