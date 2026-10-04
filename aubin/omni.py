"""AUBIN Omni — tek model, tüm yetenekler: taban (Gemma-4 12B veya 31B) BİR KEZ yüklenir, yetenek adaptörleri üstüne takılır,
istek türüne göre milisaniyede geçilir (PEFT set_adapter). Tek arayüz:

    omni = AubinOmni.preset("12B")                 # ya da "31B"
    omni.decide(state, questions)                  # tipli, kalibre karar (Kev / Laya / Jev tipi)
    omni.click(screenshot, "open settings")        # ekran görüntüsünde tıklama noktası (x, y) 0-1000
    omni.web_step(task, candidates, history)       # web ajanı: hangi öğe + işlem
    omni.act(observation, command, actions)        # gerçek zamanlı kontrol / oyun hamlesi
    omni.learning                                  # AubinLearning: learn / acquire_skill (öz-öğrenme, yetenek ekleme)

Bir yetenek o boyutta henüz eğitilmediyse (ör. 31B ekran), aynı ailenin en yakın adaptörü ile değil, açıkça hata ile döner.
"""
from __future__ import annotations

import json
import re

PRESETS = {   # yetenek → adaptör deposu (yalnız ölçülmüş, yayımlanmış adaptörler)
    "12B": {"base": "google/gemma-4-12B-it", "decide": "emrevrg/AUBIN-12B", "web": "emrevrg/AUBIN-12B-Web",
            "control": "emrevrg/AUBIN-12B-Control"},
    "31B": {"base": "google/gemma-4-31B-it", "decide": "emrevrg/AUBIN-31B"},
    "E4B": {"base": "google/gemma-4-E4B-it", "screen": "emrevrg/AUBIN-E4B-Screen", "web": "emrevrg/AUBIN-E4B-Web",
            "control": "emrevrg/AUBIN-E4B-Control", "decide": "emrevrg/AUBIN-E4B-v3"},
}
CLICK_PROMPT = ('You are a GUI agent. In this screenshot, where should I click to: "{ins}"?\n'
                "Answer with ONLY the click point as (x, y), where x and y are integers from 0 to 1000 "
                "(0,0 = top-left corner, 1000,1000 = bottom-right corner).")


class AubinOmni:
    def __init__(self, base, adapters, four_bit=True, device_map="auto", lazy=True):
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
        from peft import PeftModel
        kw = dict(device_map=device_map, dtype=torch.float16)
        if four_bit:
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                           bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
        self.proc = AutoProcessor.from_pretrained(base)
        self.tok = getattr(self.proc, "tokenizer", self.proc)
        m = AutoModelForImageTextToText.from_pretrained(base, **kw)
        names = list(adapters)
        first = "decide" if "decide" in adapters else names[0]
        self.model = PeftModel.from_pretrained(m, adapters[first], adapter_name=first)
        self._loaded = {first}
        if not lazy:                                   # lazy=True (varsayılan): diğer yetenekler ilk kullanımda yüklenir → hızlı açılış
            for n in names:
                if n not in self._loaded:
                    self.model.load_adapter(adapters[n], adapter_name=n); self._loaded.add(n)
        self.model.eval()
        self.base, self.adapters, self._active = base, dict(adapters), first
        self._scorers = {}
        self.learning = None

    @classmethod
    def from_hub(cls, repo="emrevrg/AUBIN-Omni", size="E4B", **kw):
        """Yetenek manifestosunu (omni.json) Hub'dan okur: {boyut: {"base": ..., yetenek: adaptör}}. Yeni yetenekler eklendikçe
        kod değişmeden gelir."""
        from huggingface_hub import hf_hub_download
        man = json.load(open(hf_hub_download(repo, "omni.json"), encoding="utf-8"))
        p = dict(man["sizes"][size]); base = p.pop("base")
        return cls(base, p, **kw)

    @classmethod
    def preset(cls, size="12B", **kw):
        p = dict(PRESETS[size]); base = p.pop("base")
        return cls(base, p, **kw)

    def abilities(self):
        return sorted(self.adapters)

    def use(self, ability):
        if ability not in self.adapters:
            raise ValueError(f"'{ability}' bu boyutta henüz yok (mevcut: {self.abilities()})")
        if ability not in self._loaded:                # tembel yükleme: yalnız ilk kullanımda (~saniyeler)
            self.model.load_adapter(self.adapters[ability], adapter_name=ability); self._loaded.add(ability)
            self.model.eval()
        if self._active != ability:
            self.model.set_adapter(ability); self._active = ability
        return self

    # --- tipli karar: aubin.core.Aubin ile aynı harf-logit puanlaması, aynı modeli paylaşır ---
    def _aubin(self):
        """Etkin yeteneğin puanlayıcısı (aynı model; adaptörün kendi kalibrasyon sıcaklığı/düşünme eşiği)."""
        ab = self._active or "decide"
        if ab not in self._scorers:
            from .core import Aubin
            self._scorers[ab] = Aubin.attach(self.model, self.tok, self.adapters.get(ab))
        return self._scorers[ab]

    def decide(self, state, questions):
        self.use("decide")
        out = self._aubin().decide(state, questions)
        return out

    def enable_learning(self, **kw):
        from .learn import AubinLearning
        omni = self

        class _M:                                         # AubinLearning'e "decide adaptörü açık" model görünümü
            def logprobs(self, state, questions):
                omni.use("decide"); return omni._aubin().logprobs(state, questions)
        self.learning = AubinLearning(_M(), **kw)
        return self.learning

    # --- ekran: tıklama noktası ---
    def click(self, image, instruction, max_new_tokens=16):
        import torch
        self.use("screen")
        msgs = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": CLICK_PROMPT.format(ins=instruction)}]}]
        inp = self.proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(self.model.device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
            g = self.model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
        txt = self.proc.decode(g[0, inp["input_ids"].shape[1]:], skip_special_tokens=True)
        m = re.findall(r"(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)", txt)
        return {"x": float(m[0][0]), "y": float(m[0][1]), "raw": txt} if m else {"x": None, "y": None, "raw": txt}

    # --- web ajanı: aday öğelerden seçim (Mind2Web/MindAct biçimi) ---
    def web_step(self, task, candidates, history=(), website=""):
        """candidates: {öğe_kimliği: kısa HTML açıklaması}. Dönüş: seçilen öğe + işlem (CLICK/TYPE/SELECT) + güven."""
        self.use("web")
        st = json.dumps({"task": task, "website": website, "previous_actions": list(history)[-6:] or ["(none)"]}, ensure_ascii=False)
        crit = dict(candidates); crit["none"] = "None of the above"
        el = self._aubin().decide(st, {"el": {"type": "choice", "criteria": crit,
                                             "instructions": "Which page element should be acted on next to accomplish the task? Choose 'None of the above' if none fits."}})["el"]
        op = self._aubin().decide(st + f"\nChosen element: {candidates.get(el['answer'], '')}",
                                  {"op": {"type": "choice", "instructions": "Which operation should be performed on the chosen element?",
                                          "criteria": {"CLICK": "click it", "TYPE": "type text into it", "SELECT": "choose an option from it"}}})["op"]
        return {"element": el["answer"], "element_confidence": el["confidence"], "operation": op["answer"], "operation_confidence": op["confidence"]}

    # --- kontrol / oyun ---
    def act(self, observation, command, actions, allowed=None):
        self.use("control")
        q = {"type": "choice", "criteria": dict(actions), "instructions": f"Command: {command}. Choose the single best next action for the current observation."}
        r = self._aubin().decide({"observation": observation}, {"action": q})["action"]
        a = r["answer"]
        if allowed is not None and a not in allowed:
            a = max((k for k in r["probabilities"] if k in allowed), key=lambda k: r["probabilities"][k], default=a)
        return {"action": a, "confidence": r["confidence"], "probabilities": r["probabilities"]}
