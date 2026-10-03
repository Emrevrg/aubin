"""AUBIN Loop — üretici + karar verici döngüsü (generate → decide → refine).

Bir üretici (herhangi bir LLM) aday cevaplar yazar; AUBIN adaylar arasından KALİBRE olasılıkla seçer; güven hedefin
altındaysa üreticiden, önceki adaylar geri bildirim olarak verilip yeni adaylar istenir.

Gemma'da tek ağırlık: GemmaGenerator aynı tabanı AUBIN adaptörü KAPALI kullanır (ek bellek yok), karar AUBIN adaptörüyle.
Dış üretici: OpenAIGenerator (ChatGPT, vLLM, Ollama… OpenAI-uyumlu uç). Anahtar yalnız ortam değişkeninden okunur.

    from aubin import Aubin
    from aubin.loop import AubinLoop, GemmaGenerator, last_number
    A = Aubin("emrevrg/AUBIN-12B")
    loop = AubinLoop(A, GemmaGenerator(A), extract=last_number, target=0.8)
    loop.solve("Natalia sold clips to 48 friends in April, and half as many in May. How many in total?")
    → {"answer": "72", "confidence": 0.97, "rounds": 1, "candidates": [...], "trace": [...]}
"""
from __future__ import annotations
import contextlib, json, os, re, urllib.request
import torch

SOLVE = ("Solve the task below. Reason briefly, then end with a last line exactly of the form 'Final answer: <answer>'.\n\n"
         "TASK:\n{task}{context}")


def last_number(text):
    """Sayısal cevap: 'Final answer:' satırı, yoksa metindeki son sayı (virgül/para birimi temizlenir)."""
    m = re.search(r"final answer\s*[:：]\s*(.+)", text, flags=re.I)
    s = m.group(1) if m else text
    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", s)
    if not nums:
        return None
    v = nums[0 if m else -1].replace(",", "").rstrip(".")
    try:
        f = float(v)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return v


def final_line(text):
    m = re.search(r"final answer\s*[:：]\s*(.+)", text, flags=re.I)
    return (m.group(1) if m else text.strip().splitlines()[-1] if text.strip() else "").strip() or None


class GemmaGenerator:
    """Aday üretici: AUBIN'in kendi tabanı, adaptör kapalı (tek ağırlık seti)."""

    def __init__(self, aubin, max_new=320, temperature=0.8, top_p=0.95):
        self.A, self.max_new, self.temp, self.top_p = aubin, max_new, temperature, top_p

    def _prompt(self, text):
        msgs = [{"role": "user", "content": text}]
        try:
            return self.A.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        except Exception:
            return self.A.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)

    @torch.no_grad()
    def batch(self, texts, greedy=False):
        """Birden çok istem tek toplu üretimde (döngüde ve ölçümde verim için)."""
        tok, m = self.A.tok, self.A.m
        enc = tok([self._prompt(t) for t in texts], return_tensors="pt", padding=True, add_special_tokens=False).to(self.A.dev)
        ctx = m.disable_adapter() if getattr(self.A, "adapters", None) and hasattr(m, "disable_adapter") else contextlib.nullcontext()
        kw = dict(do_sample=False) if greedy else dict(do_sample=True, temperature=self.temp, top_p=self.top_p)
        with ctx:
            gen = m.generate(**enc, max_new_tokens=self.max_new, pad_token_id=tok.pad_token_id, **kw)
        return tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    def __call__(self, prompt, n, greedy_first=False):
        out = self.batch([prompt], greedy=True) if greedy_first else []
        return out + (self.batch([prompt] * (n - len(out))) if n > len(out) else [])


class OpenAIGenerator:
    """OpenAI-uyumlu sohbet ucu. Anahtar koda/istem yazılmaz: api_key_env ortam değişkeninden okunur."""

    def __init__(self, model="gpt-4o-mini", base_url="https://api.openai.com/v1", api_key_env="OPENAI_API_KEY",
                 temperature=0.8, max_tokens=512):
        self.model, self.url, self.env, self.temp, self.max_tokens = model, base_url.rstrip("/"), api_key_env, temperature, max_tokens

    def __call__(self, prompt, n, greedy_first=False):
        body = {"model": self.model, "messages": [{"role": "user", "content": prompt}], "n": n,
                "temperature": self.temp, "max_tokens": self.max_tokens}
        req = urllib.request.Request(self.url + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json",
                                              "Authorization": "Bearer " + os.environ.get(self.env, "")})
        with urllib.request.urlopen(req, timeout=120) as r:
            d = json.load(r)
        return [c["message"]["content"] for c in d["choices"]]


class AubinLoop:
    """generate → decide → refine. decider: Aubin / AubinEnsemble (decide(state, questions) arayüzü)."""

    def __init__(self, decider, generator, extract=final_line, target=0.8, max_rounds=3, n=4, none_option=True,
                 mode="aubin", rationale_chars=600):
        self.D, self.G, self.extract = decider, generator, extract
        self.target, self.max_rounds, self.n, self.none = target, max_rounds, n, none_option
        self.mode, self.rc = mode, rationale_chars              # mode: "aubin" | "aubin+vote" (AUBIN olasılığı × oy payı)

    def question(self, task, cands):
        crit = {f"c{i}": f"{c['answer']} — reasoning: {c['text'][-self.rc:].strip()}" for i, c in enumerate(cands)}
        if self.none:
            crit["none"] = "none of the proposed answers is correct"
        return {"type": "choice", "instructions": "Which proposed answer to the task is correct? Check the reasoning.",
                "criteria": crit}

    def decide(self, task, context, cands):
        res = self.D.decide({"task": task, "context": context or ""}, {"pick": self.question(task, cands)})["pick"]
        probs = dict(res["probabilities"])
        if self.mode == "aubin+vote":                        # az örnekte oy gürültülü: +1 düzeltmeli oy payıyla çarpılır
            tot = sum(c["votes"] for c in cands) + len(cands)
            for i, c in enumerate(cands):
                probs[f"c{i}"] = probs.get(f"c{i}", 0.0) * (c["votes"] + 1) / tot
            z = sum(probs.values()) or 1.0
            probs = {k: v / z for k, v in probs.items()}
        best = max(probs, key=probs.get)
        return best, probs[best], probs

    def solve(self, task, context=None, prior=None):
        """prior: önceden üretilmiş aday metinleri (ör. ölçümde toplu üretilenler) — ilk turda üretici çağrılmaz."""
        cands, trace, feedback = [], [], ""
        for r in range(self.max_rounds):
            outs = prior if (r == 0 and prior is not None) else self.G(SOLVE.format(task=task, context=(
                "\n\nCONTEXT:\n" + context if context else "")) + feedback, self.n)
            for o in outs:
                a = self.extract(o)
                if a is None:
                    continue
                hit = next((c for c in cands if c["answer"] == a), None)
                if hit:
                    hit["votes"] += 1
                else:
                    cands.append({"answer": a, "text": o, "votes": 1})
            if not cands:
                continue
            best, conf, probs = self.decide(task, context, cands)
            trace.append({"round": r + 1, "candidates": [(c["answer"], c["votes"]) for c in cands], "pick": best,
                          "confidence": round(conf, 4)})
            if best != "none" and conf >= self.target:
                break
            feedback = ("\n\nEarlier proposed answers were judged unreliable: " + ", ".join(c["answer"] for c in cands) +
                        ". Solve again carefully from scratch.")
        if not cands:
            return {"answer": None, "confidence": 0.0, "rounds": len(trace), "candidates": [], "trace": trace}
        real = {k: v for k, v in probs.items() if k != "none"}
        k = best if best != "none" else max(real, key=real.get)
        c = cands[int(k[1:])]
        return {"answer": c["answer"], "confidence": round(float(probs[k]), 4), "rounds": len(trace),
                "candidates": [(x["answer"], x["votes"]) for x in cands], "trace": trace,
                "rejected_all": best == "none"}
