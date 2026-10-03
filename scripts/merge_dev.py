"""Ayrı koşuda ölçülen kev_dev'i (aynı adaptör) üye dosyasına ekler: python merge_dev.py üye.json dev.json çıktı.json"""
import json, sys

m, d, out = sys.argv[1:4]
M = json.load(open(m, encoding="utf-8")); D = json.load(open(d, encoding="utf-8"))
assert abs(M.get("runs_temperature", 1) - D.get("runs_temperature", 1)) < 1e-6, "sıcaklık farklı: aynı adaptör değil?"
M["runs"]["kev_dev"] = D["runs"]["kev_dev"]
json.dump(M, open(out, "w", encoding="utf-8"))
print(out, M["runs_temperature"], D["runs"]["kev_dev"]["all"]["accuracy"])
