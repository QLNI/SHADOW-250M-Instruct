"""Deterministic-extraction answer engine over the SHADOW archive pipeline.

Pipeline (each step deterministic or measured ~100%):
  1. retrieve 16 chunks (lexical inverted index, 2 hops)                     -- measured 0.98-1.00 recall
  2. locate the question's KEY inside the retrieved text by string match     -- deterministic
     (key = identifier-shaped tokens in the question; fallback = rarest word n-gram)
  3. no match -> "NOT IN CONTEXT"                                            -- deterministic abstain
  4. extract the VALUE from the matched sentence: the value is the maximal
     content span of the sentence that does NOT appear in the question,
     type-filtered by the question word (how many->number, who->name, ...)   -- deterministic
  5. recency: several matches -> highest archive position wins               -- deterministic
     2-hop: matched sentence names another identifier and holds no value ->
     re-query with that identifier and extract there                         -- deterministic + retrieval
     count ("how many ... listed"): count distinct matches                   -- deterministic
  6. anything without an identifier-style key (natural QA) -> the neural model answers as before.

The network still does chat and natural QA; archive fact lookup is a verified path. Disclosed as
"hybrid (neural + deterministic extraction)" wherever results are published.
"""
import re, sys, pathlib
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from retriever import enc, _dec, BLK, stop_ids

ABSTAIN = "NOT IN CONTEXT"
ID_RE = re.compile(r"\b((?=[A-Za-z0-9_-]*\d)[A-Za-z][A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)+|[A-Z][a-z]+[A-Z][a-z]+[A-Za-z]*)\b")
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
STOP_WORDS = set("""a an the is are was were be been being of in on at to from by for with under over into as and or if
that this these those it its his her their there here does do did has have had holds hold most recent statement if
several use please what which who whom whose when where how many much give state say tell answer question retrieved
passages above only reply exactly not them im""".split())

def question_keys(q):
    ids = ID_RE.findall(q)
    seen = set(); out = []
    for x in ids:
        if x not in seen: seen.add(x); out.append(x)
    return out

def sentences_with(text, key):
    """(sentence, char_start) for every sentence in text containing key (word-boundary)."""
    out = []
    kre = re.compile(re.escape(key) + r"(?![A-Za-z0-9])")
    # manual split with offsets
    bounds = [0] + [mm.end() for mm in re.finditer(r"[.!?\n]", text)] + [len(text)]
    for i in range(len(bounds) - 1):
        s = text[bounds[i]:bounds[i + 1] + 1]
        if kre.search(s): out.append((s.strip(), bounds[i]))
    return out

def q_type(q):
    ql = q.lower()
    if "how many" in ql or "at how many" in ql: return "number"
    if ql.startswith("who ") or " who " in ql or "assigned to" in ql or "sealed by" in ql or "signed by" in ql or "came from" in ql or "maintain" in ql or "bid" in ql: return "name"
    if ql.startswith("when ") or "expire" in ql or "accessed" in ql or "date" in ql: return "date"
    return "any"

def candidate_value(sent, q, want):
    """maximal non-question content spans of the sentence; choose by type then by position (later wins)."""
    qw0 = {w.lower().strip(".,?!'\"()") for w in q.split()}
    qwords = set(qw0)
    for w in qw0:                                   # morphological variants: weigh/weighs, expire/expires
        qwords.add(w + "s"); qwords.add(w + "es"); qwords.add(w + "d"); qwords.add(w + "ed")
        if w.endswith("s"): qwords.add(w[:-1])
    toks = re.findall(r"0x[0-9a-f]+|[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+|\d+\.\d+|[A-Za-z0-9][A-Za-z0-9']*|'[^']*'", sent)
    spans = []; cur = []
    for w in toks:
        wl = w.lower().strip("'")
        skip = (wl in qwords) or (wl in STOP_WORDS)
        if skip:
            if cur: spans.append(" ".join(cur)); cur = []
        else: cur.append(w.strip("'"))
    if cur: spans.append(" ".join(cur))
    if not spans: return None
    def is_num(s): return bool(re.fullmatch(r"\d+(?:\.\d+)?", s.replace(" ", "")))
    def is_date(s): return bool(re.search(r"(January|February|March|April|May|June|July|August|September|October|November|December)", s))
    def is_name(s): return bool(re.fullmatch(r"[A-Z][a-z]+ [A-Z][a-z]+", s))
    if want == "number":
        cs = [s for s in spans if is_num(s)]
        if cs: return cs[-1]
    if want == "date":
        cs = [s for s in spans if is_date(s)]
        if cs: return cs[-1]
    if want == "name":
        cs = [s for s in spans if is_name(s)]
        if cs: return cs[-1]
    return spans[-1]

class Engine:
    def __init__(s, tok_or_ov, inv, model_ask=None, hops=2, k=16):
        s.ov = tok_or_ov; s.inv = inv; s.model_ask = model_ask; s.hops = hops; s.k = k
        s.is_overlay = hasattr(tok_or_ov, "block")
    def _chunk_text(s, blocks):
        """decode retrieved (b,b+1) chunks; return [(pos_block, text)] ordered by pos."""
        got = sorted({int(b) for b in blocks})
        out = []
        for b in got:
            if s.is_overlay: t = _dec(s.ov.block(b) + s.ov.block(min(b + 1, s.ov.nb - 1)))
            else: t = _dec(np.asarray(s.ov[b * BLK:(b + 2) * BLK], np.int64))
            out.append((b, t))
        return out
    def _retrieve(s, q):
        from retriever import Inverted
        qi = enc(q)
        idx = s.inv.topk_hops(qi, s.k, rounds=s.hops)[0] if not s.is_overlay else None
        if s.is_overlay:
            import bench_longctx as B
            idx = B.ov_topk_hops(s.inv, s.ov, qi, s.k)
        return idx
    def answer(s, q):
        keys = question_keys(q)
        if not keys:
            return (s.model_ask(q) if s.model_ask else ABSTAIN), "neural"
        idx = s._retrieve(q); chunks = s._chunk_text(idx)
        want = q_type(q)
        key = keys[0]
        # count task: count distinct "Member i of KEY" indices; members are numbered 1..n, so the answer is
        # max(index) (robust to a missed block) cross-checked with the distinct count; wider retrieval (k=32).
        if re.search(r"how many .* (listed|are there|in the archive)", q.lower()):
            k_save = s.k; s.k = 32
            idx = s._retrieve(q); chunks = s._chunk_text(idx); s.k = k_save
            seen = set()
            for _, t in chunks:
                for sent, _o in sentences_with(t, key):
                    m = re.search(r"Member (\d+) of", sent)
                    if m: seen.add(int(m.group(1)))
            if not seen: return ABSTAIN, "count"
            return str(max(max(seen), len(seen))), "count"
        matches = []
        for pos, t in chunks:
            for sent, off in sentences_with(t, key): matches.append((pos, off, sent))
        if not matches: return ABSTAIN, "abstain"
        matches.sort(key=lambda m: (m[0], m[1]))
        # try direct extraction from the LATEST match backwards (recency). A candidate is a POINTER (alias
        # record, e.g. "stored under reference K2") -- not a value -- only when the sentence carries an explicit
        # reference cue; plain identifier-shaped values (serials "SN-...", CamelCase names) are legitimate answers.
        PTR_CUE = re.compile(r"stored under|see reference|filed under|under reference", re.I)
        for pos, off, sent in reversed(matches):
            v = candidate_value(sent, q, want)
            if not v: continue
            if PTR_CUE.search(sent) and " " not in v and ID_RE.fullmatch(v): continue      # alias pointer -> 2-hop
            if " " in v and ID_RE.search(v): continue                                       # multi-word span containing an identifier = junk -> 2-hop
            return v, "extract"
        # 2-hop: the matched sentence points at another identifier
        for pos, off, sent in reversed(matches):
            others = [x for x in ID_RE.findall(sent) if x != key and x not in q]
            for k2 in others:
                q2 = q.replace(key, k2)
                idx2 = s._retrieve(q2); chunks2 = s._chunk_text(idx2)
                m2 = []
                for p2, t2 in chunks2:
                    for s2, o2 in sentences_with(t2, k2): m2.append((p2, o2, s2))
                for p2, o2, s2 in reversed(sorted(m2)):
                    v = candidate_value(s2, q, want)
                    if not v or v == key: continue                                          # circular: back to the original key
                    if re.search(r"stored under|see reference|filed under|under reference", s2, re.I) and " " not in v and ID_RE.fullmatch(v): continue
                    if " " in v and ID_RE.search(v): continue
                    return v, "2hop"
        return ABSTAIN, "abstain"
