import os, sys, struct
os.environ.setdefault("SHADOW_D", "1536"); os.environ.setdefault("SHADOW_NL", "10")
os.environ.setdefault("SHADOW_NH", "24"); os.environ.setdefault("SHADOW_NKV", "2")
os.environ.setdefault("SHADOW_HD", "64"); os.environ.setdefault("SHADOW_FFNH", "4224")
os.environ.setdefault("SHADOW_KV_TWO_TIER", "1")
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from common import RVQ, requant
from model_250m import Shadow250M
from export_rvq import rvq_pack

CK, OUT = sys.argv[1], sys.argv[2]; COMPACT = "--compact" in sys.argv      
ck = torch.load(CK, map_location="cpu", weights_only=False)
V, FPD = ck.get("cfg", {}).get("V", 131072), 512
model = Shadow250M(torch.zeros(V, FPD), torch.zeros(V, FPD), V)
model.load_state_dict(ck["model"], strict=False); model.eval(); requant(model)

def tern_pack(w):
    wt = w.detach().float()
    sc = 1.0 / wt.abs().mean(dim=1, keepdim=True).clamp_(min=1e-5)          
    t = (wt * sc).round().clamp(-1, 1).to(torch.int8).numpy()
    rs = (1.0 / sc[:, 0]).numpy().astype(np.float32)                        
    w = wt.numpy()
    codes = (t + 1).astype(np.uint8)                              
    o, i = w.shape; assert i % 4 == 0
    c4 = codes.reshape(o, i // 4, 4)
    packed = (c4[:, :, 0] | (c4[:, :, 1] << 2) | (c4[:, :, 2] << 4) | (c4[:, :, 3] << 6)).astype(np.uint8)
    if COMPACT:                                                   
        pad = (-i) % 5; c5 = np.concatenate([codes, np.ones((o, pad), np.uint8)], 1).reshape(o, -1, 5).astype(np.uint16)
        packed = (c5[:, :, 0] + 3 * c5[:, :, 1] + 9 * c5[:, :, 2] + 27 * c5[:, :, 3] + 81 * c5[:, :, 4]).astype(np.uint8)
    return packed, rs, t.astype(np.float32) * rs[:, None]

class Wrap(torch.nn.Module):                                      
    def __init__(s, m):
        super().__init__(); s.emb = m.inp; s.b = m.b; s.step = m.struct; s.nf = m.nf; s.head = m.head; s.tb = m.tied_bias
wrap = Wrap(model)
recs = []; ntern = nrvq = 0; tern_bytes = rvq_bytes = dense_bytes = 0
rvq_ids = set()
for name, mod in wrap.named_modules():
    if isinstance(mod, RVQ):
        rvq_ids.add(id(mod))
        if mod.g == 32:                                           
            packed, rs, deq = tern_pack(mod.weight)
            recs.append((name, 4 if COMPACT else 3, (mod.o, mod.i, packed, rs))); ntern += 1; tern_bytes += packed.nbytes + rs.nbytes
        else:
            cbT, idx, scale = rvq_pack(mod)
            recs.append((name, 1, (mod.o, mod.i, mod.g, mod.st, cbT, idx, scale))); nrvq += 1
            rvq_bytes += cbT.nbytes + idx.nbytes + scale.nbytes
for name, p in wrap.named_parameters():
    owner = name.rsplit(".", 1)[0]
    mod = dict(wrap.named_modules()).get(owner)
    if mod is not None and id(mod) in rvq_ids: continue           
    a = p.detach().float().numpy()
    if COMPACT and a.ndim >= 1 and a.size >= 4096: recs.append((name, 5, a.astype(np.float16))); dense_bytes += a.size * 2
    else: recs.append((name, 0, a)); dense_bytes += a.nbytes
for name, b in wrap.named_buffers():
    if any(x in name for x in ("cent", "cb", "initialized", "sign", "mu", "ctv", "low", "high", "updates", "inv")): continue
    a = b.detach().float().numpy(); recs.append((name, 0, a)); dense_bytes += a.nbytes

with open(OUT, "wb") as f:
    f.write(b"SHDW"); f.write(struct.pack("<II", 1, len(recs)))
    for name, kind, pay in recs:
        nb = name.encode(); f.write(struct.pack("<I", len(nb))); f.write(nb); f.write(struct.pack("<I", kind))
        if kind == 0:
            a = np.ascontiguousarray(pay, np.float32); f.write(struct.pack("<I", a.ndim)); f.write(struct.pack("<" + "I" * a.ndim, *a.shape)); f.write(a.tobytes())
        elif kind == 1:
            o, i, g, st, cbT, idx, scale = pay
            f.write(struct.pack("<IIII", o, i, g, st)); f.write(np.ascontiguousarray(cbT, np.float32).tobytes())
            f.write(np.ascontiguousarray(idx, np.uint8).tobytes()); f.write(np.ascontiguousarray(scale, np.float32).tobytes())
        elif kind == 5:
            a = np.ascontiguousarray(pay, np.float16); f.write(struct.pack("<I", a.ndim)); f.write(struct.pack("<" + "I" * a.ndim, *a.shape)); f.write(a.tobytes())
        else:
            o, i, packed, rs = pay
            f.write(struct.pack("<II", o, i)); f.write(np.ascontiguousarray(packed, np.uint8).tobytes()); f.write(rs.tobytes())
size = os.path.getsize(OUT)
print(f"loaded checkpoint ({ck.get('step','finetuned')})")
print(f"ternary FFN modules {ntern}: {tern_bytes/1e6:.2f} MB | RVQ 1-bit modules {nrvq}: {rvq_bytes/1e6:.2f} MB | dense fp32: {dense_bytes/1e6:.2f} MB")
print(f".shdw = {size/1e6:.2f} MB  (+ 8.39 MB fp131072 table = {(size+8388736)/1e6:.2f} MB deploy)")

mod = model.b[0].up; w = mod.weight.detach().float()
sc = 1.0 / w.abs().mean(dim=1, keepdim=True).clamp_(min=1e-5); wq = (w * sc).round().clamp(-1, 1) / sc
_, _, deq = tern_pack(mod.weight); print("ternary round-trip max|err| vs trainer forward:", float((torch.tensor(deq) - wq).abs().max()))
