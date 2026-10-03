"""AUBIN çekirdeği: tipli durum + tipli sorular → kalibre olasılıklar (tek ileri geçiş, metin üretimi yok).

Soru biçimi Jev/TypeSafe ve Kev ile uyumludur:
    questions = {"iade": {"type": "choice", "instructions": "...", "criteria": {"a": "...", "b": "..."}},
                 "acil": {"type": "noul", "instructions": "..."},
                 "puan": {"type": "score", "instructions": "...", "criteria": ["kötü", "orta", "iyi"]}}
    Aubin("emrevrg/AUBIN-12B").decide(state, questions)
    → {"iade": {"answer": "a", "confidence": 0.93, "probabilities": {"a": 0.93, "b": 0.07}}, ...}
"""
from __future__ import annotations
import json, os
import torch
import torch.nn.functional as F

L = [chr(65 + i) for i in range(26)]


def options(q):
    t, c = q["type"], q.get("criteria")
    if t == "noul":
        keys = ["false", "true"]
        if isinstance(c, dict):
            return keys, [f"false — {c.get('false', 'no')}", f"true — {c.get('true', 'yes')}"]
        return keys, ["false — no", "true — yes"]
    if t == "choice":
        return list(c.keys()), [f"{k}: {v}" for k, v in c.items()]
    if isinstance(c, list):
        return [str(i) for i in range(len(c))], [str(v) for v in c]
    return list(c.keys()), [f"{k}: {v}" for k, v in c.items()]


class Aubin:
    """AUBIN karar modeli. base: açık taban (Apache-2.0), adapter: AUBIN LoRA'sı (yerel dizin veya HF repo)."""

    def __init__(self, adapter="emrevrg/AUBIN-12B", base=None, four_bit=True, device_map=None, temperature=None,
                 perms=1, max_state=6000, cache_state=False, think_margin=None, think_tokens=256, think_mix=None,
                 think_samples=1, think_temp=0.7):
        # önbellek: Gemma-4 GPU'da birebir değil (fark 0.355) → varsayılan KAPALI
        # think_margin: en iyi iki seçeneğin log-olasılık farkı bunun altındaysa kısa akıl yürütme + yeniden puan
        # ("eminsizse düşün"; Kev dev'de 31B: transfer .852→.864, ~%14 soruda üretim yapılır)
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        meta = self._meta(adapter[0] if isinstance(adapter, (list, tuple)) else adapter)
        base = base or meta.get("base_model", "google/gemma-4-12B-it")
        self.T = float(temperature or meta.get("temperature", 1.0))
        self.perms, self.max_state = perms, max_state
        self.think_margin = float(think_margin if think_margin is not None else meta.get("think_margin", 0.0))
        self.think_tokens = think_tokens
        # think_mix: eminsiz soruda düşünen yolun payı (1.0 = yalnız düşünen yol; Duo tarifi 0.5 = hızlı + düşünen yarı yarıya)
        self.think_mix = float(think_mix if think_mix is not None else meta.get("think_mix", 1.0))
        # think_samples>1: self-consistency — açgözlü akıl yürütme + (n-1) örneklenmiş; cevap olasılıkları ortalanır
        self.think_samples, self.think_temp = int(think_samples), float(think_temp)
        self.cache_state, self._pc = cache_state, None
        self.tok = AutoTokenizer.from_pretrained(base)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        # AUBIN_DEVICE=xla: Kaggle/Cloud TPU (torch_xla) — bf16, 4-bit yok, uzunluk 256'nın katına dolgu (derleme tekrar kullanılır)
        self.xla = os.environ.get("AUBIN_DEVICE", "") == "xla"
        if self.xla:
            import torch_xla.core.xla_model as xm
            if os.environ.get("AUBIN_XLA_SPMD", "") == "1":
                import torch_xla.runtime as xr
                xr.use_spmd()                        # cihaz alınmadan önce açılmalı
            dev = xm.xla_device()
            kw = dict(dtype=torch.bfloat16, low_cpu_mem_usage=True)
            four_bit = False
        else:
            dev = "cuda:0" if torch.cuda.is_available() else "cpu"
            device_map = device_map or meta.get("device_map")
            kw = dict(device_map=device_map or dev, dtype=torch.float16 if dev != "cpu" else torch.float32)
        if four_bit and dev != "cpu":
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                           bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
        try:
            m = AutoModelForCausalLM.from_pretrained(base, **kw)
        except Exception:
            from transformers import AutoModelForImageTextToText
            m = AutoModelForImageTextToText.from_pretrained(base, **kw)
        self.adapters = []
        ads = adapter if isinstance(adapter, (list, tuple)) else ([adapter] if adapter else [])
        if meta.get("no_adapter"):                   # sürüm yalnız protokol + kalibrasyon (ağırlık değişikliği yok)
            ads = []
        if ads:                                      # birden çok adaptör = ansambl (log-olasılık ortalaması)
            from peft import PeftModel
            m = PeftModel.from_pretrained(m, ads[0], adapter_name="a0")
            for i, ad in enumerate(ads[1:], 1):
                m.load_adapter(ad, adapter_name=f"a{i}")
            self.adapters = [f"a{i}" for i in range(len(ads))]
        if self.xla:
            m = self._to_xla(m, dev)
        self.m = m.eval()
        self.dev = dev
        self.letter_ids = [sorted({self.tok.encode(a, add_special_tokens=False)[0],
                                   self.tok.encode(" " + a, add_special_tokens=False)[-1]}) for a in L]

    @classmethod
    def attach(cls, model, tok, adapter=None, temperature=None, think_margin=None):
        """Zaten yüklü bir modele (ör. AubinOmni'nin çok-adaptörlü tabanı) Aubin puanlayıcısı bağlar — yeniden yükleme yok.
        Etkin adaptör dışarıdan seçilir (set_adapter); kalibrasyon sıcaklığı/düşünme eşiği adaptörün aubin.json'undan."""
        a = cls.__new__(cls)
        meta = cls._meta(adapter) if adapter else {}
        a.T = float(temperature or meta.get("temperature", 1.0))
        a.perms, a.max_state = 1, 6000
        a.think_margin = float(think_margin if think_margin is not None else meta.get("think_margin", 0.0))
        a.think_tokens, a.think_mix, a.think_samples, a.think_temp = 256, float(meta.get("think_mix", 1.0)), 1, 0.7
        a.cache_state, a._pc, a.xla, a.adapters = False, None, False, []
        a.tok = tok
        a.tok.padding_side = "left"
        if a.tok.pad_token is None:
            a.tok.pad_token = a.tok.eos_token
        a.m = model
        a.dev = model.device if hasattr(model, "device") else next(model.parameters()).device
        a.letter_ids = [sorted({tok.encode(x, add_special_tokens=False)[0], tok.encode(" " + x, add_special_tokens=False)[-1]}) for x in L]
        return a

    @staticmethod
    def _to_xla(m, dev):
        """TPU'ya taşı. AUBIN_XLA_SPMD=1: tüm çiplere (v5e-8 = 8 x 16 GB) satır-bölmeli SPMD — 12B/31B bf16 tek çipe sığmaz.
        Parametreler tek tek taşınıp hemen bölünür; tam kopya hiçbir çipte birikmez."""
        if os.environ.get("AUBIN_XLA_SPMD", "") != "1":
            return m.to(dev)
        import numpy as np
        import torch_xla.runtime as xr
        import torch_xla.distributed.spmd as xs
        n =xr.global_runtime_device_count()
        mesh = xs.Mesh(np.arange(n), (1, n), ("data", "model"))
        for mod in m.modules():
            for name, p in list(mod.named_parameters(recurse=False)) + list(mod.named_buffers(recurse=False)):
                t = p.data.to(dev)
                if t.dim() == 2 and t.shape[0] % n == 0:
                    xs.mark_sharding(t, mesh, ("model", None))
                if isinstance(p, torch.nn.Parameter):
                    p.data = t
                else:
                    mod._buffers[name] = t
        return m

    @staticmethod
    def _meta(adapter):
        try:
            if adapter and os.path.isdir(adapter):
                return json.load(open(os.path.join(adapter, "aubin.json")))
            if adapter:
                from huggingface_hub import hf_hub_download
                return json.load(open(hf_hub_download(adapter, "aubin.json")))
        except Exception:
            pass
        return {}

    def _prompt(self, state_text, q, texts, order):
        st = state_text if len(state_text) <= self.max_state else state_text[: self.max_state] + " …"
        opts = "\n".join(f"{L[j]}) {texts[i]}" for j, i in enumerate(order))
        u = (f"Decide based ONLY on the state below.\n\nSTATE:\n{st}\n\nQUESTION: {q}\n\nOPTIONS:\n{opts}\n\n"
             f"Reply with the single letter of the correct option.")
        msgs = [{"role": "user", "content": u}]
        try:
            p = self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        except Exception:
            p = self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        return p + "Answer: "

    @torch.no_grad()
    def _letters(self, prompts):
        if self.cache_state and len(prompts) == 1 and "\n\nQUESTION:" in prompts[0]:
            return self._letters_cached(prompts[0])
        enc = self.tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False,
                       pad_to_multiple_of=256 if self.xla else None).to(self.dev)
        try:
            out = self.m(**enc, logits_to_keep=1)
        except TypeError:
            out = self.m(**enc)
        last = out.logits[:, -1].float()
        r = torch.stack([last[:, ids].logsumexp(-1) for ids in self.letter_ids], -1)
        return r.cpu() if self.xla else r

    @torch.no_grad()
    def _letters_cached(self, prompt):
        """Durum önbelleği: aynı duruma sorulan her soruda durum kısmı (önek) BİR kez işlenir, KV-önbelleği
        kopyalanıp yalnız soru+seçenekler (sonek) hesaplanır → çok soruda büyük hız kazancı."""
        import copy
        full = self.tok(prompt, add_special_tokens=False).input_ids
        pre_ids = self.tok(prompt[: prompt.index("\n\nQUESTION:")], add_special_tokens=False).input_ids
        k = 0                                         # token düzeyinde ortak önek: bölme sınırı birebir aynı tokenlarla
        while k < min(len(full), len(pre_ids)) - 1 and full[k] == pre_ids[k]:
            k += 1
        key = (tuple(full[:k]), getattr(self.m, "active_adapter", None))
        if self._pc is None or self._pc[0] != key:
            out = self.m(input_ids=torch.tensor([full[:k]], device=self.dev), use_cache=True)
            self._pc = (key, k, out.past_key_values)
        n_pre, pkv = self._pc[1], copy.deepcopy(self._pc[2])
        sids = torch.tensor([full[k:]], device=self.dev)
        cp = torch.arange(n_pre, n_pre + sids.shape[1], device=self.dev)
        pos = cp.unsqueeze(0)
        # cache_position açıkça verilir: kayan pencereli katmanların önbelleği kırpılmış tutulur, get_seq_length() önek
        # uzunluğunu değil pencere boyunu döndürebilir → maske/konum kayar (Gemma-4 12B'de görülen 0.355 fark)
        try:
            out = self.m(input_ids=sids, past_key_values=pkv, position_ids=pos, cache_position=cp, use_cache=True, logits_to_keep=1)
        except TypeError:
            out = self.m(input_ids=sids, past_key_values=pkv, position_ids=pos, cache_position=cp, use_cache=True)
        last = out.logits[:, -1].float()
        return torch.stack([last[:, ids].logsumexp(-1) for ids in self.letter_ids], -1)

    def _think(self, state_text, q, texts):
        """Kısa akıl yürütme üret, 'Answer: ' konumunda harf log-olasılıkları (kev_llm.think_scores ile aynı)."""
        st = state_text if len(state_text) <= self.max_state else state_text[: self.max_state] + " …"
        opts = "\n".join(f"{L[j]}) {t}" for j, t in enumerate(texts))
        u = (f"Decide based ONLY on the state below.\n\nSTATE:\n{st}\n\nQUESTION: {q}\n\nOPTIONS:\n{opts}\n\n"
             f"Think briefly step by step (at most 6 short sentences), weighing the most plausible options. "
             f"Then write the final line exactly as 'Answer: <letter>'.")
        msgs = [{"role": "user", "content": u}]
        try:
            p = self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        except Exception:
            p = self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        enc = self.tok([p], return_tensors="pt", add_special_tokens=False).to(self.dev)
        probs, reasons = None, []
        for s in range(max(1, self.think_samples)):
            kw = dict(do_sample=False) if s == 0 else dict(do_sample=True, temperature=self.think_temp, top_p=0.95)
            with torch.no_grad():
                gen = self.m.generate(**enc, max_new_tokens=self.think_tokens, pad_token_id=self.tok.pad_token_id, **kw)
            t = self.tok.decode(gen[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            k = t.find("Answer:")
            reasons.append((t[:k] if k >= 0 else t).strip())
            lg = self._letters([p + (t[:k] if k >= 0 else t).rstrip() + "\nAnswer: "])[0, :len(texts)].cpu()
            pr = torch.softmax(lg.float(), -1)
            probs = pr if probs is None else probs + pr
        self.last_reasoning = reasons[0]
        return torch.log(probs / max(1, self.think_samples) + 1e-12)

    def _maybe_think(self, lp, state_text, q, texts):
        if not self.think_margin or len(texts) > 26 or len(texts) < 2:
            return lp, False
        v = torch.sort(lp, descending=True).values
        if float(v[0] - v[1]) >= self.think_margin:
            return lp, False
        return self._think(state_text, q, texts), True

    def _mixed(self, fast, state_text, q, texts):
        """Sıcaklıkla ölçeklenmiş log-olasılık; eminsizse düşünen yol think_mix oranında karıştırılır
        (jev_compare.combine ile birebir aynı hesap)."""
        z = torch.log_softmax(fast / self.T, -1)
        think, did = self._maybe_think(fast, state_text, q, texts)
        if did and self.think_mix > 0:
            z = (1 - self.think_mix) * z + self.think_mix * torch.log_softmax(think / self.T, -1)
        return z

    def _score(self, state_text, q, texts):
        """Sıcaklık UYGULANMIŞ log-olasılıklar."""
        if len(self.adapters) > 1:
            outs = []
            for name in self.adapters:
                self.m.set_adapter(name)
                outs.append(self._mixed(self._score_one(state_text, q, texts), state_text, q, texts))
            return torch.log_softmax(torch.stack(outs).mean(0), -1)
        return self._mixed(self._score_one(state_text, q, texts), state_text, q, texts)

    def _score_one(self, state_text, q, texts):
        n = len(texts); idx = list(range(n))
        if n > 26:                                   # 26'lık gruplar + final turu
            full, winners = torch.full((n,), -1e4), []
            for s in range(0, n, 26):
                g = idx[s:s + 26]
                lg = self._letters([self._prompt(state_text, q, texts, g)])[0, :len(g)].cpu()
                full[torch.tensor(g)] = lg; winners.append(g[int(lg.argmax())])
            lg = self._letters([self._prompt(state_text, q, texts, winners)])[0, :len(winners)].cpu()
            full[torch.tensor(winners)] = lg + 50.0
            return torch.log_softmax(full, -1)
        acc = torch.zeros(n)
        shifts = sorted({(k * n) // self.perms for k in range(self.perms)})
        for s in shifts:                             # seçenek sırası döndürülür: konum yanlılığı ortalanır
            order = idx[s:] + idx[:s]
            lg = torch.log_softmax(self._letters([self._prompt(state_text, q, texts, order)])[0, :n].cpu(), -1)
            back = torch.empty(n); back[torch.tensor(order)] = lg; acc += back
        return acc / len(shifts)

    def logprobs(self, state, questions):
        """Her soru için sıcaklıkla ölçeklenmiş log-olasılıklar {id: (keys, tensor)} — ansambl bunları birleştirir."""
        st = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
        out = {}
        for qid, q in questions.items():
            keys, texts = options(q)
            out[qid] = (keys, torch.log_softmax(self._score(st, q.get("instructions") or qid, texts), -1))
        return out

    def decide(self, state, questions):
        """state: dict/str; questions: {id: {type, instructions, criteria}} → {id: {answer, confidence, probabilities}}"""
        return _package(self.logprobs(state, questions), 1.0)


def _package(lps, T):
    out = {}
    for qid, (keys, lp) in lps.items():
        p = F.softmax(lp / T, -1)
        j = int(p.argmax())
        out[qid] = {"answer": keys[j], "confidence": round(float(p[j]), 4),
                    "probabilities": {k: round(float(v), 4) for k, v in zip(keys, p)}}
    return out


class AubinEnsemble:
    """Farklı tabanlardaki AUBIN üyelerinin ağırlıklı log-olasılık birleşimi (ör. 31B + 12B).
    members: [(Aubin, ağırlık), ...]; temperature: birleşim sonrası sıcaklık (Kev dev'de ölçülen tarif: T=0.8)."""

    def __init__(self, members, temperature=0.8):
        s = sum(w for _, w in members)
        self.members = [(m, w / s) for m, w in members]
        self.T = temperature

    def decide(self, state, questions):
        acc = None
        for m, w in self.members:
            lp = m.logprobs(state, questions)
            acc = {k: (v[0], w * v[1]) for k, v in lp.items()} if acc is None else \
                {k: (acc[k][0], acc[k][1] + w * lp[k][1]) for k in acc}
        return _package(acc, self.T)
