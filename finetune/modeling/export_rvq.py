import sys, numpy as np, torch

sys.path.insert(0, r"C:\Users\USER\Desktop\shadow_final\model_250m")
import common
from common import RVQ


def rvq_capture(m):
    W = m.weight.detach().float()
    sc = W.abs().mean(1, keepdim=True).clamp_min(1e-8)          
    r = (W / sc).reshape(-1, m.g).clone()                       
    idxs, rmss = [], []
    for t in range(m.st):
        rms = r.pow(2).mean().sqrt().clamp_min(1e-8)
        idx = torch.cdist(r / rms, m.cb[t]).argmin(1)
        r = r - m.cb[t][idx] * rms
        idxs.append(idx.cpu().numpy().astype(np.uint8)); rmss.append(float(rms))
    return sc.squeeze(1).cpu().numpy(), idxs, rmss


def rvq_pack(m):
    o, i, g, st = m.o, m.i, m.g, m.st
    G = i // g
    Npad = (o + 63) & ~63
    nch = Npad // 64
    sc, idxs, rmss = rvq_capture(m)

    cbT = np.zeros((st, g, 16), dtype=np.float32)
    for t in range(st):
        cb = m.cb[t].detach().cpu().numpy()                     
        cbT[t] = cb.T * rmss[t]                                 

    idx = np.zeros((st, nch, G, 32), dtype=np.uint8)
    for t in range(st):
        flat = idxs[t].reshape(o, G)                            
        pad = np.zeros((Npad, G), dtype=np.uint8)
        pad[:o] = flat
        for c in range(nch):
            lo = pad[c * 64: c * 64 + 32]                       
            hi = pad[c * 64 + 32: c * 64 + 64]                  
            idx[t, c] = (lo.T | (hi.T << 4))                    

    scale = np.zeros(Npad, dtype=np.float32)
    scale[:o] = sc
    return cbT, idx, scale


def rvq_unpack(cbT, idx, scale, o, i, g, st):
    G = i // g
    Npad = scale.shape[0]; nch = Npad // 64
    W = np.zeros((Npad, i), dtype=np.float32)
    for t in range(st):
        for c in range(nch):
            blk = idx[t, c]                                     
            lo = (blk & 0x0F).T                                 
            hi = (blk >> 4).T                                   
            for sub, rows in ((lo, range(c * 64, c * 64 + 32)),
                              (hi, range(c * 64 + 32, c * 64 + 64))):
                for bi, n in enumerate(rows):
                    codes = sub[bi]                             
                    W[n] += cbT[t][:, codes].T.reshape(-1)      
    return W[:o] * scale[:o, None]


if __name__ == "__main__":
    torch.manual_seed(0)
    print(f"{'module':>22s} {'shape':>14s} {'bits/w':>7s} {'max|err|':>10s} {'rel RMSE':>10s}  verdict")
    ok = True

    for name, (i, o, g, st) in {
        "proj 1-bit  (g8,st2)": (256, 256, 8, 2),
        "proj 1-bit  rect     ": (256, 128, 8, 2),
        "FFN 0.125   (g32,st1)": (256, 1024, 32, 1),
        "FFN 0.125   down     ": (1024, 256, 32, 1),
        "odd rows (pad test)  ": (256, 100, 8, 2),
    }.items():
        m = RVQ(i, o, g, st)
        m.weight.data = torch.randn(o, i) / (i ** 0.5)
        m.enc()                                                  
        ref = m._q.detach().cpu().numpy()

        cbT, idxp, scale = rvq_pack(m)
        got = rvq_unpack(cbT, idxp, scale, o, i, g, st)

        err = np.abs(ref - got).max()
        rel = np.sqrt(((ref - got) ** 2).mean()) / np.sqrt((ref ** 2).mean())
        good = err < 1e-5
        ok &= good
        print(f"{name:>22s} {f'{o}x{i}':>14s} {m.bits():7.3f} {err:10.2e} {rel:10.2e}  "
              f"{'PASS' if good else 'FAIL'}")

    print()
    print("ROUNDTRIP_OK" if ok else "ROUNDTRIP_FAIL")
