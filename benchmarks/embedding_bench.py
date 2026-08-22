"""Benchmark the frozen binary vocabulary on word similarity (WordSim-353).

Every token has a fixed 512-bit code. If the codes carry meaning, similar words should
have nearby codes. This script scores each human-rated word pair by Hamming similarity
between the two words' codes and reports the Spearman correlation with the human ratings.
Random codes score about 0. Runs offline on the files in this repo.

    python benchmarks/embedding_bench.py
"""
import csv, pathlib, sys
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from shadow_runtime.retriever import enc

fp = np.unpackbits(np.load(ROOT / "fp131072.npy"), axis=1)[:, :512]
rows = [l.split(",") for l in open(HERE / "wordsim353.csv", encoding="utf-8").read().splitlines() if l]
rng = np.random.default_rng(0); rand = rng.integers(0, 2, size=fp.shape).astype(np.uint8)

def score(table):
    xs, ys = [], []
    for w1, w2, human in rows:
        i1, i2 = enc(" " + w1.lower()), enc(" " + w2.lower())
        if len(i1) != 1 or len(i2) != 1: continue
        xs.append(1 - np.mean(table[i1[0]] != table[i2[0]])); ys.append(float(human))
    from scipy.stats import spearmanr
    return spearmanr(xs, ys).statistic, len(xs)

r, n = score(fp); r0, _ = score(rand)
print(f"WordSim-353, single-token pairs (n={n})")
print(f"  frozen vocabulary codes : Spearman {r:.3f}")
print(f"  random codes (baseline) : Spearman {r0:.3f}")
