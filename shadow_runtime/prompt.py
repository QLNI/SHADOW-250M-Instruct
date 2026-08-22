"""Prompt assembly for retrieval QA. Retrieved chunks are framed inside the user turn:

  <bos><start_of_turn>user\n  [ <unused0> pos=N \n <chunk tokens> <unused1> \n ] x K
  \nUsing only the retrieved passages above, answer the question. If the answer is not in them, reply exactly "NOT IN CONTEXT".
  Question: {q}<end_of_turn>\n<start_of_turn>model\n
cold mask = True on every token inside a chunk frame (open..close), False elsewhere (hot window).
Chunks = retrieved 64-token blocks expanded to (b, b+1) and merged when adjacent, ordered by archive position.
"""
import sys, json, pathlib, re, string
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent; PKG = HERE.parent
sys.path.insert(0, str(HERE))
from retriever import enc, BLK
BOS, EOS, SOT, EOT = 2, 1, 8, 9; COPEN, CCLOSE = 6, 7
ABSTAIN = "NOT IN CONTEXT"
INSTR = f"\nUsing only the retrieved passages above, answer the question. If the answer is not in them, reply exactly \"{ABSTAIN}\".\nQuestion: "
_NL = None
def nl():
    global _NL
    if _NL is None: _NL = enc("\n")
    return _NL

def chunks_from_blocks(tok, blocks, expand=1, max_chunks=16):
    """retrieved block ids -> list of (start_block, token list): (b, b+expand) spans, merged when overlapping, by position."""
    spans = sorted({(int(b), int(b) + expand + 1) for b in blocks[:max_chunks]})
    merged = []
    for a, e in spans:
        if merged and a <= merged[-1][1]: merged[-1][1] = max(merged[-1][1], e)
        else: merged.append([a, e])
    nb = len(tok) // BLK
    return [(a, [int(t) for t in tok[a * BLK:min(e, nb) * BLK]]) for a, e in merged if a < nb]

def build_prompt(chunks, question):
    """chunks: [(pos_block, ids)] -> (ids, cold_mask) for the user turn + model turn opener."""
    ids = [BOS, SOT] + enc("user\n"); cold = [False] * len(ids)
    for pos, ct in chunks:
        fr = [COPEN] + enc(f"pos={pos}\n"); body = ct; close = [CCLOSE] + nl()
        seg = fr + body + close; ids += seg; cold += [True] * len(seg)
    tail = enc(INSTR + question) + [EOT] + nl() + [SOT] + enc("model\n")
    ids += tail; cold += [False] * len(tail)
    return ids, cold

def normalize(s):
    s = s.lower().strip(); s = "".join(ch for ch in s if ch not in set(string.punctuation)); s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())
def final_answer(text):
    """trace-aware: the answer is the text after the last '->' (quote-then-answer format), else the whole text."""
    return text.rsplit("->", 1)[-1] if "->" in text else text
def reward(answer_text, gold, task):
    """verifiable reward: 1 if exact match after normalisation (numbers: exact digits), else 0. T5: must be the abstain marker."""
    a = normalize(final_answer(answer_text)); g = normalize(str(gold))
    if task == "T5" or g == normalize(ABSTAIN): return 1.0 if a == normalize(ABSTAIN) else 0.0
    if a == normalize(ABSTAIN): return 0.0
    if a == g: return 1.0
    # short free-form gold (hotpot/squad): token-F1 >= 0.8 counts
    at, gt = a.split(), g.split()
    if not at or not gt: return 0.0
    common_ = sum(min(at.count(w), gt.count(w)) for w in set(at));
    if common_ == 0: return 0.0
    p, r = common_ / len(at), common_ / len(gt); f1 = 2 * p * r / (p + r)
    return 1.0 if f1 >= 0.8 else 0.0
