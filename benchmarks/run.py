"""Reproduce the model card numbers. Expects archive directories under data/archives/{1m,10m,100m}
(tokens.u32 + meta.json + a question bank), which are not distributed with this repository.
    python benchmarks/run.py [--tiers 1M,10M,100M]
"""
import sys, os, json, time, random, pathlib
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "shadow_runtime"))
from shadow_runtime.retriever import load_archive, Inverted, enc, _dec
from shadow_runtime.answer_engine import Engine
from shadow_runtime.prompt import normalize

TIERS = {"1M": "1m", "10M": "10m", "100M": "100m"}
sel = "1M,10M,100M"
if "--tiers" in sys.argv: sel = sys.argv[sys.argv.index("--tiers") + 1]
out = {}
for tier in sel.split(","):
    path = ROOT / "data" / "archives" / TIERS[tier]
    tok, meta, bank = load_archive(str(path))
    t0 = time.time(); inv = Inverted(tok)
    print(f"[{tier}] index {len(tok)//64} blocks in {time.time()-t0:.0f}s", flush=True)
    eng = Engine(tok, inv)
    per = {}
    for b in bank:
        a, how = eng.answer(b["question"])
        ok = normalize(a) == normalize(str(b["answer"]))
        d = per.setdefault(b["task"], [0, 0]); d[0] += ok; d[1] += 1
    tot = sum(v[0] for v in per.values()); n = sum(v[1] for v in per.values())
    out[tier] = {t: f"{v[0]}/{v[1]}" for t, v in sorted(per.items())}; out[tier]["ALL"] = round(tot / n, 3)
    print(f"[{tier}] " + "  ".join(f"{t} {v[0]}/{v[1]}" for t, v in sorted(per.items())) + f"  ALL {tot/n:.2f}", flush=True)
json.dump(out, open(HERE / "my_results.json", "w"), indent=1)
print("wrote benchmarks/my_results.json")
