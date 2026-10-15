"""Archive retriever: exact lexical inverted index over 64-token blocks, with 2-hop follow-up.

An archive is a directory with tokens.u32 (uint32 token stream) and meta.json. Blocks of 64 tokens
are indexed by their content features (stop-filtered unigrams plus hashed bigrams and trigrams,
idf-weighted). A query returns the top-k blocks; a second hop follows identifiers named near the
match, so alias chains ("the record for K is stored under reference K2") resolve.

Also holds the tokenizer used by the runtime (sentencepiece + id remap).
"""
import os, sys, json, pathlib, hashlib, re
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent; PKG = HERE.parent
BLK = 64

def load_archive(path):
    tok = np.memmap(os.path.join(path, "tokens.u32"), np.uint32, "r")
    meta = json.load(open(os.path.join(path, "meta.json")))
    bp = os.path.join(path, "bank_valid.jsonl"); bp = bp if os.path.exists(bp) else os.path.join(path, "bank.jsonl")   # validated bank if present
    bank = [json.loads(l) for l in open(bp, encoding="utf-8")]
    return tok, meta, bank

_STOP = None
def stop_ids(cache=None):
    """500 most frequent ids, fixed once from a held-out fineweb shard (deploy/stop500.npy)."""
    global _STOP
    if _STOP is None:
        _STOP = np.load(PKG / "tokenizer" / "stop500.npy")
    return _STOP

NBG = 1 << 20                               # hashed bigram feature space

def _feats(ids):
    """content features of a token sequence: unique unigrams (stop-filtered) + hashed bigrams of ALL adjacent
    tokens (digits/punctuation are frequent as unigrams but their sequences -- 'Vega-713' -- are the signal)."""
    ids = np.asarray(ids, np.int64); st = stop_ids()
    uni = np.unique(ids); uni = uni[(~np.isin(uni, st)) & (uni > 9)]
    if len(ids) >= 2:
        bg = (ids[:-1] * np.int64(1000003) + ids[1:] * np.int64(7919)) % NBG
        bg = bg[(ids[:-1] > 9) & (ids[1:] > 9)]
        if len(ids) >= 3:                      # trigrams too: 'Cygnus','-','9' / '9','2','2' make the key number discriminative
            tg = (ids[:-2] * np.int64(1000003) + ids[1:-1] * np.int64(7919) + ids[2:] * np.int64(104729) + np.int64(17)) % NBG
            bg = np.concatenate([bg, tg[(ids[:-2] > 9) & (ids[1:-1] > 9) & (ids[2:] > 9)]])
        bg = np.unique(bg) + (1 << 17)
    else: bg = np.zeros(0, np.int64)
    return uni, bg

def _content(ids):
    u, g = _feats(ids); return np.concatenate([u, g])


# ---------------- exact lexical inverted index (BM25-style, unigram + bigram features) ----------------
# The sketch retrievers above lose the rare-key signal inside 64-token blocks; an inverted index keeps it exactly.
# Deploy cost at 100M tokens: ~60M postings (~250 MB) next to the 32 GB 1-bit KV archive; query = 3-10 posting lists.
class Inverted:
    def __init__(s, tok, k1=1.2, b=0.75):
        import scipy.sparse as sp
        nb = len(tok) // BLK; rows = []; cols = []
        for bi in range(nb):
            u, g = _feats(tok[bi * BLK:(bi + 1) * BLK]); f = np.concatenate([u, g]); rows.append(f); cols.append(np.full(len(f), bi, np.int64))
        r = np.concatenate(rows); c = np.concatenate(cols)
        s.M = sp.csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=((1 << 17) + NBG, nb))   # feature x block
        df = np.asarray(s.M.sum(1)).ravel(); s.idf = np.log(1 + (nb - df + 0.5) / (df + 0.5)).astype(np.float32)
        s.nb = nb; s.tok = tok
    def topk(s, ids, k):
        f = _content(ids)
        if len(f) == 0: return np.arange(min(k, s.nb)), np.zeros(min(k, s.nb))
        sc = np.asarray((s.idf[f][None, :] @ s.M[f]).todense()).ravel() if False else s.idf[f] @ s.M[f]
        sc = np.asarray(sc).ravel()
        idx = np.argpartition(-sc, min(k, s.nb - 1))[:k]; idx = idx[np.argsort(-sc[idx], kind="stable")]
        return idx, sc[idx]
    def topk_hops(s, ids, k, rounds=2, k1=8, idf_min=5.0, win=14):
        """multi-hop: round 1 = top-k1 direct hits (kept). Then EACH round-1 block gets its own follow-up query made of
        the rare features within +-win tokens (over the stream) of where the question's tokens matched inside it --
        'The record for K is stored under reference K2.' -> K2's n-grams -- and contributes its best new block.
        Per-source queries (not a union) so one alias chain cannot be drowned by the others' noise."""
        idx1, sc1 = s.topk(ids, k1); out = [int(i) for i in idx1]; qset = set(_content(ids).tolist()); qtext = _dec(np.asarray(ids, np.int64))
        qtok = set(np.asarray(ids, np.int64).tolist()) - set(stop_ids().tolist()); src = out[:]
        # template features = present in >=2 of the round-1 blocks ('stored under reference', 'access code for vault'):
        # they would pull in every block of the same template; only features UNIQUE to a source block are followed.
        cnt = {}
        for bi in src:
            for f in set(_content(np.asarray(s.tok[bi * BLK:(bi + 1) * BLK], np.int64)).tolist()): cnt[f] = cnt.get(f, 0) + 1
        template = {f for f, c in cnt.items() if c >= 2}
        for r in range(rounds - 1):
            new = []
            for bi in src:
                if len(out) + len(new) >= k: break
                blk = np.asarray(s.tok[bi * BLK:(bi + 1) * BLK], np.int64)
                pos = [i for i, t in enumerate(blk) if t in qtok]
                if not pos: continue
                lo = max(0, bi * BLK + min(pos) - 8); hi = min(len(s.tok), bi * BLK + max(pos) + win + 1)
                w = np.asarray(s.tok[lo:hi], np.int64)
                # follow-up query: entity-like identifiers mentioned near the match (Name-123 style, the shape of archive
                # keys) other than the question's own; fall back to the rare non-template features of the window.
                ents = [e for e in set(re.findall(r"[A-Z][a-z]+-\d+", _dec(w))) if e not in qtext]
                if ents: fb = np.unique(np.concatenate([_content(enc(" " + e)) for e in ents]))
                else:
                    fb = _content(w); fb = fb[(s.idf[fb] >= idf_min) & ~np.isin(fb, list(qset | template))]
                if len(fb) == 0: continue
                sc = np.asarray(s.idf[fb] @ s.M[fb]).ravel(); sc[out + new] = -1e9
                for j in np.argsort(-sc)[:2]:                       # best new block of this chain (skip the source's own neighbourhood)
                    j = int(j)
                    if abs(j - bi) > 1 and j not in new: new.append(j); break
            out += new; src = new
            if not new: break
        return np.array(out[:k]), np.zeros(min(k, len(out)))


# ---------------- tokenizer (sentencepiece + id remap; expansion for out-of-subset ids computed lazily) ----
_sp = None; _o2n = None; _n2o = None; _exp_cache = {}
def _load_tok():
    global _sp, _o2n, _n2o
    if _sp is None:
        import sentencepiece as spm
        _sp = spm.SentencePieceProcessor(model_file=str(PKG / "tokenizer" / "tokenizer.model"))
        _n2o = np.fromfile(PKG / "tokenizer" / "new2old.u32", np.uint32).astype(np.int64)
        _o2n = np.full(262144, -1, np.int64); _o2n[_n2o] = np.arange(len(_n2o))
def _expand(old_id):
    """out-of-subset token -> its byte-fallback sequence in the 131k id space (lossless)."""
    seq = _exp_cache.get(old_id)
    if seq is None:
        piece = _sp.id_to_piece(int(old_id)).replace("▁", " ")
        raw = _sp.decode([int(old_id)]).encode("utf-8") if not piece.startswith("<0x") else bytes([int(piece[3:5], 16)])
        seq = [int(_o2n[_sp.piece_to_id(f"<0x{b:02X}>")]) for b in raw]
        _exp_cache[old_id] = seq
    return seq
def enc(text):
    _load_tok()
    out = []
    for i in _sp.encode(text):
        n = _o2n[i]
        if n >= 0: out.append(int(n))
        else: out.extend(_expand(i))
    return out
def _dec(ids):
    _load_tok()
    return _sp.decode(_n2o[np.asarray(ids, np.int64)].tolist())
