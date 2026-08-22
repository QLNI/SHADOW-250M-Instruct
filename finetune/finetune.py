"""Fine-tune SHADOW 250M Instruct on your own chat data, on a single GPU (8 GB is enough).

Data: a .jsonl file, one conversation per line:
    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

Run:
    python finetune.py --data my_data.jsonl --steps 300 --out my_model
    python export_model.py my_model/finetuned.pt my_model.shdw          # 52 MB deploy file
    ./shadow my_model.shdw fp131072.npy --chat                          # your model, on CPU

Defaults are safe for style and domain fine-tunes: low learning rate, loss only on assistant
tokens, quantisation kept in the loop so the exported model behaves like the trained one.
"""
import argparse, json, math, os, sys, time, random, pathlib
for k, v in {"SHADOW_D": "1536", "SHADOW_NL": "10", "SHADOW_NH": "24", "SHADOW_NKV": "2", "SHADOW_HD": "64",
             "SHADOW_FFNH": "4224", "SHADOW_FAST_ATTN": "1", "SHADOW_KV_BITS": "1", "SHADOW_KV_TWO_TIER": "1"}.items():
    os.environ.setdefault(k, v)
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "modeling")); sys.path.insert(0, str(HERE / "shadow_runtime"))
import numpy as np, torch, torch.nn.functional as F
import common
from common import requant
from model_250m import Shadow250M
from retriever import enc

BOS, EOS, SOT, EOT = 2, 1, 8, 9

def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="jsonl with {'messages': [...]} per line")
    ap.add_argument("--init", default=str(HERE / "shadow250m_instruct.pt"))
    ap.add_argument("--table", default=str(HERE / "fp131072.npy"))
    ap.add_argument("--out", default="finetuned")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--micro-batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()

def build_ids(messages):
    ids, msk = [BOS], [0]
    for m in messages:
        role = "user" if m["role"] != "assistant" else "model"
        head = [SOT] + enc(role + "\n"); ids += head; msk += [0] * len(head)
        body = enc(m["content"]) + [EOT] + enc("\n")
        ids += body
        msk += ([1] * (len(body) - 1) + [0]) if role == "model" else [0] * len(body)
    return ids, msk

class Packer:
    def __init__(s, path, ctx, rng, val_frac):
        s.ex = []
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line: continue
            ids, msk = build_ids(json.loads(line)["messages"])
            s.ex.append((np.asarray(ids, np.int64), np.asarray(msk, np.int64)))
        rng.shuffle(s.ex)
        if len(s.ex) < 2: raise SystemExit("need at least 2 conversations in the data file")
        nval = min(max(1, int(len(s.ex) * val_frac)), max(1, len(s.ex) // 5))
        s.val = s.ex[:nval]; s.train = s.ex[nval:]; s.ctx = ctx; s.rng = rng
        print(f"data: {len(s.train)} train / {len(s.val)} val conversations")
    def pack(s, B, val=False):
        pool = s.val if val else s.train
        X = np.zeros((B, s.ctx), np.int64); Y = np.full((B, s.ctx), -100, np.int64)
        for r in range(B):
            pos = 0
            while pos < s.ctx:
                ids, m = pool[s.rng.randrange(len(pool))]
                ids, m = ids[:s.ctx - pos], m[:s.ctx - pos]
                X[r, pos:pos + len(ids)] = ids
                tgt = np.full(len(ids), -100, np.int64); tgt[:-1] = np.where(m[1:] == 1, ids[1:], -100)
                Y[r, pos:pos + len(ids)] = tgt; pos += len(ids)
                if pos > s.ctx * 0.9: break
        return torch.tensor(X), torch.tensor(Y)

def main():
    a = get_args(); rng = random.Random(a.seed); torch.manual_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    _of = common.RVQ.forward
    def _tern(s, x):
        if s.g == 32:
            w = s.weight; sc = 1.0 / w.abs().mean(dim=1, keepdim=True).clamp_(min=1e-5)
            return F.linear(x, (w + ((w * sc).round().clamp(-1, 1) / sc - w).detach()).to(x.dtype))
        return _of(s, x)
    common.RVQ.forward = _tern
    _oenc = common.RVQ.enc
    def _enc2(s):
        if s.g == 32: return
        _oenc(s)
    common.RVQ.enc = _enc2
    fp = np.unpackbits(np.load(a.table), axis=1)[:, :512]
    cent = torch.tensor(fp.astype(np.float32) * 2 - 1, device=dev); cent_n = F.normalize(cent, dim=-1)
    model = Shadow250M(cent, cent_n, cent.shape[0]).to(dev)
    ck = torch.load(a.init, map_location=dev, weights_only=False)
    sd = {k: v.float() if v.is_floating_point() else v for k, v in ck["model"].items()}
    model.load_state_dict(sd); requant(model)
    for md in model.modules():
        if isinstance(md, common.KVCodec1): md.eval()
    print(f"loaded {a.init} on {dev}")
    data = Packer(a.data, a.ctx, rng, a.val_frac)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
    def loss_of(x, y):
        h, _ = model.trunk(x); ph = model.head(h).float().reshape(-1, 512)
        yf = y.reshape(-1); v = yf >= 0; ph = ph[v]; yf = yf[v]
        ce = 0.0
        for i in range(0, ph.shape[0], 8192):
            lg = ph[i:i + 8192] @ model.cent_n.T + model.tied_bias
            ce = ce + F.cross_entropy(lg, yf[i:i + 8192], reduction="sum")
        return ce / max(1, int(v.sum()))
    @torch.no_grad()
    def val():
        model.eval(); tot = 0.0
        for _ in range(4):
            x, y = data.pack(a.micro_batch, val=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                tot += float(loss_of(x.to(dev), y.to(dev)))
        model.train()
        for md in model.modules():
            if isinstance(md, common.KVCodec1): md.eval()
        return tot / 4
    v0 = val(); print(f"step 0  val loss {v0:.4f}")
    t0 = time.time()
    for step in range(1, a.steps + 1):
        lr = a.lr * min(1.0, step / a.warmup) * (0.5 * (1 + math.cos(math.pi * step / a.steps)))
        for g in opt.param_groups: g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        for _ in range(a.accum):
            x, y = data.pack(a.micro_batch)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                (loss_of(x.to(dev), y.to(dev)) / a.accum).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); requant(model)
        if step % a.log_every == 0:
            el = time.time() - t0
            print(f"step {step:>4}  lr {lr:.2e}  {el/step:.1f}s/step  eta {(a.steps-step)*el/step/60:.0f}min", flush=True)
    v1 = val()
    torch.save({"model": model.state_dict()}, out / "finetuned.pt")
    print(f"done  val loss {v0:.4f} -> {v1:.4f}  saved {out/'finetuned.pt'}")

if __name__ == "__main__":
    main()
