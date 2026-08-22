import math, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from contextlib import contextmanager
try:
    from paged_kv import PagedKVArchive, PagedKVView, ExactChunkCountView
except ImportError:
    from paged_kv import PagedKVArchive, PagedKVView, ExactChunkCountView


import os as _os
def _e(k,d): return int(_os.environ.get(k,d))

D=_e("SHADOW_D",1024); NL=_e("SHADOW_NL",20); NH=_e("SHADOW_NH",16)
NKV=_e("SHADOW_NKV",4); HD=_e("SHADOW_HD",64); FFNH=_e("SHADOW_FFNH",3072); FPD=512
FAST_ATTN=bool(int(__import__('os').environ.get('SHADOW_FAST_ATTN','0')))  
TRAIN_EXACT_POT=bool(int(__import__('os').environ.get(
    'SHADOW_TRAIN_EXACT_POT','1')))
TRAIN_EXACT_KV=bool(int(__import__('os').environ.get(
    'SHADOW_TRAIN_EXACT_KV','1')))
KV_BITS=_e("SHADOW_KV_BITS",1)             
KV_TWO_TIER=bool(int(_os.environ.get("SHADOW_KV_TWO_TIER","0")))   
KV_COLD_MASK=None                          


class _ExactSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,q):
        return q
    @staticmethod
    def backward(ctx,grad):
        return grad,None

def ste(x,q): return _ExactSTE.apply(x,q)
def pot(x):                                   
    m=x.abs().amax(-1,keepdim=True).clamp_min(1e-6); s=torch.exp2(torch.ceil(torch.log2(m/127.)))
    return ste(x,(x/s).round().clamp(-127,127)*s)

def walsh_hadamard(x):
    n=x.shape[-1]
    if n<1 or n&(n-1):
        raise ValueError(f"Walsh-Hadamard width must be a power of two, got {n}")
    shape=x.shape
    y=x
    h=1
    while h<n:
        y=y.reshape(*shape[:-1],n//(2*h),2,h)
        a,b=y[...,0,:],y[...,1,:]
        y=torch.cat((a+b,a-b),-1).reshape(shape)
        h*=2
    return y/math.sqrt(n)

def _kv2_levels(x):
    m=x.abs().amax(-1,keepdim=True).clamp_min(1e-6)
    scale=torch.exp2(torch.ceil(torch.log2(m/1.5)))
    return ((x/scale).round().clamp(-2,1)+0.5)*scale

def kv2(x):                                   
    rotated=walsh_hadamard(x)
    restored=walsh_hadamard(_kv2_levels(rotated))
    return ste(x,restored)

def kv2_pack(x):
    if x.shape[-1] % 4:
        raise ValueError(f"2-bit packing requires a multiple-of-4 last dimension, got {x.shape[-1]}")
    rotated=walsh_hadamard(x)
    m=rotated.abs().amax(-1,keepdim=True).clamp_min(1e-6)
    scale=torch.exp2(torch.ceil(torch.log2(m/1.5)))
    code=(rotated/scale).round().clamp(-2,1).to(torch.int16).add_(2).to(torch.uint8)
    z=code.reshape(*code.shape[:-1],code.shape[-1]//4,4)
    packed=z[...,0] | (z[...,1]<<2) | (z[...,2]<<4) | (z[...,3]<<6)
    return packed.contiguous(),scale.to(x.dtype).contiguous()

def kv2_unpack(packed,scale,dtype=None):
    parts=torch.stack((packed&3,(packed>>2)&3,(packed>>4)&3,(packed>>6)&3),-1)
    code=parts.reshape(*packed.shape[:-1],packed.shape[-1]*4)
    dt=dtype or scale.dtype
    rotated=(code.to(dt)-1.5)*scale.to(dt)
    return walsh_hadamard(rotated)


class KVCodec1(nn.Module):
    def __init__(s,heads,width,seed=0,momentum=0.01,decision_grid=256.0):
        super().__init__()
        if width<1 or width&(width-1):
            raise ValueError(f"KVCodec1 width must be a power of two, got {width}")
        g=torch.Generator().manual_seed(int(seed))
        sign=(torch.randint(0,2,(heads,width),generator=g,dtype=torch.int8)*2-1)
        s.heads,s.width,s.momentum=int(heads),int(width),float(momentum)
        s.decision_grid=float(decision_grid)
        s.register_buffer("sign",sign)
        s.register_buffer("mu",torch.zeros(heads,width))
        s.register_buffer("ctv",torch.zeros(heads,width))
        s.register_buffer("low",torch.full((heads,width),-1.0))
        s.register_buffer("high",torch.ones(heads,width))
        s.register_buffer("initialized",torch.tensor(False))
        s.register_buffer("updates",torch.tensor(0,dtype=torch.int64),persistent=False)

    def _check(s,x):
        if x.ndim!=4 or x.shape[1]!=s.heads or x.shape[-1]!=s.width:
            raise ValueError(
                f"KVCodec1 expects (B,{s.heads},T,{s.width}), got {tuple(x.shape)}")

    @torch.no_grad()
    def calibrate(s,x):
        s._check(x); xf=x.detach().float()
        mu=xf.mean((0,2))
        sg=s.sign.float()[None,:,None,:]
        z=walsh_hadamard((xf-mu[None,:,None,:])*sg)
        flat=z.permute(1,0,2,3).reshape(s.heads,-1,s.width)
        decision=(flat*s.decision_grid).round()/s.decision_grid
        ctv=decision.median(1).values
        hi=decision>ctv[:,None,:]; lo=~hi
        low=(flat*lo).sum(1)/lo.sum(1).clamp_min(1)
        high=(flat*hi).sum(1)/hi.sum(1).clamp_min(1)
        rate=1.0 if not bool(s.initialized) else s.momentum
        for dst,src in ((s.mu,mu),(s.ctv,ctv),(s.low,low),(s.high,high)):
            dst.lerp_(src.to(dst.dtype),rate)
        s.initialized.fill_(True); s.updates.add_(1)
        return s

    def _ensure(s,x,update):
        if update or not bool(s.initialized):
            s.calibrate(x)

    def transform(s,x):
        s._check(x)
        sg=s.sign.to(x.dtype)[None,:,None,:]
        return walsh_hadamard((x-s.mu.to(x.dtype)[None,:,None,:])*sg)

    def bits(s,x,update=False):
        s._ensure(x,update)
        decision=(s.transform(x)*s.decision_grid).round()/s.decision_grid
        threshold=(s.ctv.to(x.dtype)*s.decision_grid).round()/s.decision_grid
        return decision>threshold[None,:,None,:]

    def reconstruct_bits(s,bits,dtype=torch.float32):
        if bits.ndim!=4 or bits.shape[1]!=s.heads or bits.shape[-1]!=s.width:
            raise ValueError(f"bad 1-bit KV shape {tuple(bits.shape)}")
        low=s.low.to(dtype)[None,:,None,:]
        high=s.high.to(dtype)[None,:,None,:]
        rotated=torch.where(bits,high,low)
        sg=s.sign.to(dtype)[None,:,None,:]
        return s.mu.to(dtype)[None,:,None,:]+walsh_hadamard(rotated)*sg

    def forward(s,x):
        bits=s.bits(x,update=s.training)
        return ste(x,s.reconstruct_bits(bits,x.dtype))

    @torch.no_grad()
    def pack(s,x):
        bits=s.bits(x,update=False)
        b=bits.reshape(*bits.shape[:-1],s.width//8,8).to(torch.uint8)
        weights=torch.tensor((1,2,4,8,16,32,64,128),
                             dtype=torch.uint8,device=x.device)
        return (b*weights).sum(-1).to(torch.uint8).contiguous()

    def unpack(s,packed,dtype=None):
        if packed.ndim!=4 or packed.shape[1]!=s.heads or packed.shape[-1]*8!=s.width:
            raise ValueError(f"bad packed 1-bit KV shape {tuple(packed.shape)}")
        parts=torch.stack(tuple(((packed>>i)&1) for i in range(8)),-1)
        bits=parts.reshape(*packed.shape[:-1],s.width).bool()
        return s.reconstruct_bits(bits,dtype or s.mu.dtype)


@contextmanager
def freeze_codec_updates(module):
    codecs=[m for m in module.modules() if isinstance(m,KVCodec1)]
    states=[m.training for m in codecs]
    try:
        for m in codecs:
            m.training=False
        yield
    finally:
        for m,state in zip(codecs,states):
            m.training=state


def mem8_pack(x):
    m=x.abs().amax(-1,keepdim=True).clamp_min(1e-6)
    scale=torch.exp2(torch.ceil(torch.log2(m/127.0)))
    return (x/scale).round().clamp(-127,127).to(torch.int8),scale.to(x.dtype)


def mem8_unpack(packed,scale,dtype=None):
    dt=dtype or scale.dtype
    return packed.to(dt)*scale.to(dt)


def bitpack512(x):
    if x.shape[-1] != FPD:
        raise ValueError(f"Hamming keys require {FPD} features, got {x.shape[-1]}")
    bits=(x>=0).to(torch.uint8).reshape(*x.shape[:-1],FPD//8,8)
    weights=torch.tensor((1,2,4,8,16,32,64,128),dtype=torch.uint8,device=x.device)
    return (bits*weights).sum(-1).to(torch.uint8).contiguous()


class HammingHebbianMemory:
    def __init__(s,batch,dim,device,capacity=4096,match_threshold=0.90,
                 hebb_eta=0.25,retrieval_k=4):
        s.batch,s.dim,s.device=int(batch),int(dim),torch.device(device)
        s.capacity=int(capacity); s.match_threshold=float(match_threshold)
        s.hebb_eta=float(hebb_eta); s.retrieval_k=int(retrieval_k)
        s.keys=[[] for _ in range(s.batch)]
        s.vals=[[] for _ in range(s.batch)]
        s.scales=[[] for _ in range(s.batch)]
        s.strength=[[] for _ in range(s.batch)]
        s.age=[[] for _ in range(s.batch)]; s.clock=0
        s._lut=torch.tensor([int(i).bit_count() for i in range(256)],
                            dtype=torch.int16,device=s.device)

    def __len__(s):
        return sum(len(x) for x in s.keys)

    def counts(s):
        return [len(x) for x in s.keys]

    def _distances(s,key,b):
        if not s.keys[b]:
            return torch.empty(0,dtype=torch.int32,device=s.device)
        bank=torch.stack(s.keys[b])
        xor=torch.bitwise_xor(bank,key).to(torch.long)
        return s._lut[xor].sum(-1,dtype=torch.int32)

    @torch.no_grad()
    def write(s,values,key_features):
        if values.shape!=(s.batch,s.dim) or key_features.shape!=(s.batch,FPD):
            raise ValueError(f"write shapes must be {(s.batch,s.dim)} and "
                             f"{(s.batch,FPD)}, got {values.shape}, {key_features.shape}")
        packed_keys=bitpack512(key_features)
        events=[]
        for b in range(s.batch):
            s.clock+=1; key=packed_keys[b].detach(); value=values[b].detach()
            distances=s._distances(key,b)
            best=int(distances.argmin()) if len(distances) else -1
            similarity=(1.0-float(distances[best])/FPD) if best>=0 else -1.0
            if best>=0 and similarity>=s.match_threshold:
                old=mem8_unpack(s.vals[b][best],s.scales[b][best],value.dtype)
                rate=s.hebb_eta*similarity
                updated=old+rate*(value-old)
                vp,vs=mem8_pack(updated)
                s.vals[b][best]=vp; s.scales[b][best]=vs
                s.strength[b][best]+=similarity; s.age[b][best]=s.clock
                events.append({"kind":"reinforce","slot":best,
                               "similarity":similarity})
                continue
            vp,vs=mem8_pack(value)
            if len(s.keys[b])>=s.capacity:

                slot=min(range(len(s.keys[b])),
                         key=lambda i:(s.strength[b][i],s.age[b][i]))
                s.keys[b][slot]=key; s.vals[b][slot]=vp; s.scales[b][slot]=vs
                s.strength[b][slot]=1.0; s.age[b][slot]=s.clock
                kind="replace"
            else:
                slot=len(s.keys[b]); s.keys[b].append(key)
                s.vals[b].append(vp); s.scales[b].append(vs)
                s.strength[b].append(1.0); s.age[b].append(s.clock)
                kind="allocate"
            events.append({"kind":kind,"slot":slot,"similarity":similarity})
        return events

    @torch.no_grad()
    def retrieve(s,key_features,k=None):
        if key_features.shape!=(s.batch,FPD):
            raise ValueError(f"query shape must be {(s.batch,FPD)}, got {key_features.shape}")
        packed=bitpack512(key_features); outputs=[]; metadata=[]
        for b in range(s.batch):
            distances=s._distances(packed[b],b)
            if not len(distances):
                outputs.append(torch.zeros(s.dim,device=s.device,
                                           dtype=key_features.dtype))
                metadata.append({"found":False,"slot":-1,"similarity":0.0})
                continue
            take=min(int(k or s.retrieval_k),len(distances))
            d,idx=torch.topk(distances,take,largest=False)
            values=torch.stack([
                mem8_unpack(s.vals[b][int(i)],s.scales[b][int(i)],
                           key_features.dtype) for i in idx])
            similarity=1.0-d.to(torch.float32)/FPD
            weights=torch.softmax(similarity*32.0,-1).to(values.dtype)
            outputs.append((values*weights[:,None]).sum(0))
            metadata.append({"found":True,"slot":int(idx[0]),
                             "similarity":float(similarity[0]),
                             "distance":int(d[0])})
        return torch.stack(outputs),metadata


def make_memory_state(batch,dim,device,chunk_size=128,capacity=256,
                      match_threshold=0.90,hebb_eta=0.25,retrieval_k=4):
    if chunk_size<1:
        raise ValueError("memory chunk_size must be positive")
    return {
        "store":HammingHebbianMemory(
            batch,chunk_size*dim,device,capacity=capacity,
            match_threshold=match_threshold,hebb_eta=hebb_eta,
            retrieval_k=retrieval_k),
        "evicted":torch.empty(batch,0,dim,device=device),
        "chunk_size":int(chunk_size),
        "hidden_dim":int(dim),
        "recall_threshold":0.60,
        "writes":0,
        "last_retrieval":None,
    }


@torch.no_grad()
def memory_absorb_evicted(memory,evicted,head,norm):
    if memory is None or not evicted.shape[1]:
        return []
    memory["evicted"]=torch.cat((memory["evicted"],evicted.detach()),1)
    events=[]
    n=int(memory["chunk_size"])
    while memory["evicted"].shape[1]>=n:
        chunk=memory["evicted"][:,:n]
        memory["evicted"]=memory["evicted"][:,n:]
        pooled=chunk.mean(1)
        key_features=head(norm(pooled)).float()


        value=chunk.reshape(chunk.shape[0],-1)
        events.extend(memory["store"].write(value,key_features))
        memory["writes"]+=1
    return events


@torch.no_grad()
def memory_append_recall(memory,query_hidden,context,head,norm):
    if memory is None or not len(memory["store"]):
        return context
    query=head(norm(query_hidden)).float()
    recall,meta=memory["store"].retrieve(query)
    memory["last_retrieval"]=meta


    threshold=float(memory["recall_threshold"])
    if not any(m["found"] and m["similarity"]>=threshold for m in meta):
        return context
    gate=torch.tensor([m["similarity"] if m["found"] and
                       m["similarity"]>=threshold else 0.0 for m in meta],
                      device=recall.device,dtype=recall.dtype)[:,None,None]
    recalled_chunk=recall.reshape(
        recall.shape[0],memory["chunk_size"],memory["hidden_dim"])
    return torch.cat((context,recalled_chunk*gate),1)
def rope(x,cos,sin):
    a,b=x[...,0::2],x[...,1::2]; return torch.stack([a*cos-b*sin,a*sin+b*cos],-1).flatten(-2)

class RMS(nn.Module):
    def __init__(s,d,eps=1e-6): super().__init__(); s.w=nn.Parameter(torch.ones(d)); s.eps=eps
    def forward(s,x): n=torch.rsqrt(x.float().pow(2).mean(-1,keepdim=True)+s.eps).to(x.dtype); return x*n*s.w.to(x.dtype)

class RVQ(nn.Module):
    def __init__(s,i,o,g,st=1,codes=16):
        super().__init__(); s.i,s.o,s.g,s.st,s.c=i,o,g,st,codes
        s.weight=nn.Parameter(torch.randn(o,i)/math.sqrt(i))
        s.register_buffer("cb",torch.zeros(st,codes,g)); s.register_buffer("cb_init",torch.tensor(False)); s._q=None
    @torch.no_grad()
    def _km(s,X,k,it=12,samp=8192):
        S=X[torch.randperm(X.shape[0],device=X.device)[:min(X.shape[0],samp)]]
        C=S[torch.randperm(S.shape[0],device=X.device)[:min(k,S.shape[0])]].clone()
        if C.shape[0]<k: C=torch.cat([C,C[:k-C.shape[0]]])
        for _ in range(it):
            a=torch.cdist(S,C).argmin(1)
            for j in range(k):
                m=a==j
                if m.any(): C[j]=S[m].mean(0)
        return C
    @torch.no_grad()
    def _fit(s):
        sc=s.weight.abs().mean(1,keepdim=True).clamp_min(1e-8); r=(s.weight/sc).reshape(-1,s.g).clone()
        for t in range(s.st):
            rms=r.pow(2).mean().sqrt().clamp_min(1e-8); s.cb[t]=s._km(r/rms,s.c)
            idx=torch.cdist(r/rms,s.cb[t]).argmin(1); r=r-s.cb[t][idx]*rms
        s.cb_init.fill_(True)
    def enc(s):
        if not bool(s.cb_init): s._fit()
        with torch.no_grad():
            sc=s.weight.detach().abs().mean(1,keepdim=True).clamp_min(1e-8); r=(s.weight.detach()/sc).reshape(-1,s.g); acc=torch.zeros_like(r)
            for t in range(s.st):
                rms=r.pow(2).mean().sqrt().clamp_min(1e-8); idx=torch.cdist(r/rms,s.cb[t]).argmin(1)
                q=s.cb[t][idx]*rms; acc=acc+q; r=r-q
            quantized=acc.reshape(s.o,s.i)*sc



            if (s._q is not None and s._q.shape==quantized.shape and
                    s._q.device==quantized.device and
                    s._q.dtype==quantized.dtype):
                s._q.copy_(quantized)
            else:
                s._q=quantized
    def qw(s):

        if s._q is None or s._q.device!=s.weight.device: s.enc()
        return ste(s.weight,s._q)
    def forward(s,x): return F.linear(x,s.qw().to(x.dtype))
    def bits(s): return s.st*math.log2(s.c)/s.g
def requant(m):
    for md in m.modules():
        if isinstance(md,RVQ): md.enc()

def cs(Ts,dev):
    inv=1.0/(10000**(torch.arange(0,HD,2,device=dev).float()/HD)); f=torch.arange(Ts,device=dev).float()[:,None]*inv[None]
    return f.cos()[None,None].to(torch.bfloat16), f.sin()[None,None].to(torch.bfloat16)

def cs_at(start,Ts,dev):
    inv=1.0/(10000**(torch.arange(0,HD,2,device=dev).float()/HD))
    f=torch.arange(start,start+Ts,device=dev).float()[:,None]*inv[None]
    return f.cos()[None,None].to(torch.bfloat16),f.sin()[None,None].to(torch.bfloat16)

def shiftmax(dot,alpha,Ts,device):
    aq=ste(alpha,(alpha*4096).round()/4096)
    e=aq*dot                                                  
    e=ste(e,e.floor())
    cm=torch.ones(Ts,Ts,dtype=torch.bool,device=device).tril()
    e=e.masked_fill(~cm,-1e9)
    w=torch.exp2((e-e.amax(-1,keepdim=True)).clamp_min(-15))
    return w/w.sum(-1,keepdim=True)

class Block(nn.Module):
    def __init__(s,layer_idx=0):
        super().__init__(); s.n1=RMS(D); s.n2=RMS(D)
        s.layer_idx=int(layer_idx)
        s.q=RVQ(D,NH*HD,8,2); s.k=RVQ(D,NKV*HD,8,2); s.v=RVQ(D,NKV*HD,8,2); s.o=RVQ(NH*HD,D,8,2)    
        s.qn=RMS(HD); s.kn=RMS(HD); s.g=nn.Parameter(torch.zeros(NH*HD)); s.alpha=nn.Parameter(torch.full((1,NH,1,1),0.25))
        s.up=RVQ(D,FFNH,32,1); s.gt=RVQ(D,FFNH,32,1); s.dn=RVQ(FFNH,D,32,1)                          
        s.kv_bits=KV_BITS
        s.kcodec=KVCodec1(NKV,HD,seed=1000+int(layer_idx))
        s.vcodec=KVCodec1(NKV,HD,seed=2000+int(layer_idx))
    def forward(s,x,cos,sin):
        z=s.n1(x); Bs,Ts,_=z.shape
        q=s.qn(s.q(z).view(Bs,Ts,NH,HD)).transpose(1,2); k=s.kn(s.k(z).view(Bs,Ts,NKV,HD)).transpose(1,2); v=s.v(z).view(Bs,Ts,NKV,HD).transpose(1,2)
        q=rope(q,cos,sin); k=rope(k,cos,sin)
        if s.kv_bits==1 and (TRAIN_EXACT_KV or not s.training):
            if KV_TWO_TIER:



                m=KV_COLD_MASK
                if m is not None:
                    kc=s.kcodec(k); vc=s.vcodec(v); mm=m[:,None,:,None]
                    k=torch.where(mm,kc,k); v=torch.where(mm,vc,v)
            else:
                k=s.kcodec(k); v=s.vcodec(v)
        elif s.kv_bits!=1:
            k=kv2(k); v=kv2(v)
        k=k.repeat_interleave(NH//NKV,1); v=v.repeat_interleave(NH//NKV,1)
        if TRAIN_EXACT_POT or not s.training:
            q=pot(q).to(torch.bfloat16)
            k=pot(k).to(torch.bfloat16)
            v=pot(v).to(torch.bfloat16)
        else:


            q=q.to(torch.bfloat16)
            k=k.to(torch.bfloat16)
            v=v.to(torch.bfloat16)
        if FAST_ATTN:




            aq=ste(s.alpha,(s.alpha*4096).round()/4096)*0.6931471805599453
            y=F.scaled_dot_product_attention((q*aq).to(v.dtype),k,v,is_causal=True,scale=1.0)
            y=y.transpose(1,2).reshape(Bs,Ts,NH*HD)
        else:
            w=shiftmax(q@k.transpose(-1,-2),s.alpha,Ts,x.device)
            y=(w.to(v.dtype)@v).transpose(1,2).reshape(Bs,Ts,NH*HD)  

        x=x+s.o(y*torch.sigmoid(s.g)); h=s.n2(x); return x+s.dn(F.silu(s.gt(h))*s.up(h))

    def _qkv(s,z,cos,sin):
        Bs,Ts,_=z.shape
        q=s.qn(s.q(z).view(Bs,Ts,NH,HD)).transpose(1,2)
        k=s.kn(s.k(z).view(Bs,Ts,NKV,HD)).transpose(1,2)
        v=s.v(z).view(Bs,Ts,NKV,HD).transpose(1,2)
        return rope(q,cos,sin),rope(k,cos,sin),v

    def _finish_attention(s,x,q,k,v,causal,exact=True):
        kr=k.repeat_interleave(NH//NKV,1); vr=v.repeat_interleave(NH//NKV,1)
        q=pot(q).to(torch.bfloat16); kr=pot(kr).to(torch.bfloat16); vr=pot(vr).to(torch.bfloat16)
        dot=q@kr.transpose(-1,-2)
        aq=ste(s.alpha,(s.alpha*4096).round()/4096)
        raw=aq*dot
        e=ste(raw,raw.floor()) if exact else raw*0.6931471805599453
        if causal:
            tq,tk=e.shape[-2:]

            cm=torch.arange(tk,device=x.device)[None,:] <= (
                torch.arange(tq,device=x.device)[:,None]+tk-tq)
            e=e.masked_fill(~cm[None,None],-1e9)
        if exact:
            w=torch.exp2((e-e.amax(-1,keepdim=True)).clamp_min(-15))
            w=w/w.sum(-1,keepdim=True)
        else:
            w=torch.softmax(e,-1)
        y=(w.to(vr.dtype)@vr).transpose(1,2).reshape(x.shape[0],x.shape[1],NH*HD)
        x=x+s.o(y*torch.sigmoid(s.g)); h=s.n2(x)
        return x+s.dn(F.silu(s.gt(h))*s.up(h))

    @staticmethod
    def _popcount_bytes(x):
        lut=torch.tensor([int(i).bit_count() for i in range(256)],
                         dtype=torch.int16,device=x.device)
        return lut[x.long()].sum(-1,dtype=torch.int32)

    def _archive_overflow(s,cache,overflow):
        if overflow<=0 or not cache.get("cold_enabled") or s.kv_bits!=1:
            return
        ck=cache["k"][:,:,:overflow].detach()
        cv=cache["v"][:,:,:overflow].detach()
        start=int(cache["hot_start_position"])
        positions=torch.arange(start,start+overflow,dtype=torch.int64,device=ck.device)
        cache["cold_archive"].append(ck,cv,positions)
        cache["hot_start_position"]=start+overflow

    def _retrieve_cold(s,q,cache,dtype):
        archive=cache.get("cold_archive")
        if (not cache.get("cold_enabled") or archive is None or
                len(archive)==0 or s.kv_bits!=1):
            return None,None
        if q.shape[2]!=1:
            raise ValueError("cold KV retrieval requires single-token decode")
        B,_,_,width=q.shape
        groups=NH//NKV
        q_group=q.reshape(B,NKV,groups,1,width).mean(2)
        q_code=s.kcodec.pack(q_group)[:,:,0]                 
        started=time.perf_counter()
        n=len(archive)
        final_k=min(int(cache["cold_topk"]),n)
        wide_target=min(max(final_k*4,final_k),n)
        wide_indices,wide_distances=archive.exact_hamming_topk(q_code,wide_target)
        kp=archive.gather(wide_indices,"k")
        vp=archive.gather(wide_indices,"v")
        byte_width=archive.packed_width
        kval=s.kcodec.unpack(kp,dtype)
        q_score=pot(q.reshape(B,NKV,groups,width)).float()
        k_score=pot(kval).float()
        score=torch.einsum("bhgd,bhwd->bhgw",q_score,k_score).amax(2)
        selected=torch.topk(score,final_k,-1).indices
        sg=selected[...,None].expand(-1,-1,-1,byte_width)
        kp=torch.gather(kp,2,sg); vp=torch.gather(vp,2,sg)
        cache["cold_last_indices"]=torch.gather(wide_indices,2,selected)
        if cache.get("retrieval_trace_enabled"):
            selected_dist=torch.gather(wide_distances,2,selected)
            cache["cold_last_trace"]={
                "original_positions":archive.gather_positions(
                    cache["cold_last_indices"]).detach().cpu(),
                "hamming_distances":selected_dist.detach().cpu(),
                "shortlist_rank":selected.detach().cpu(),
                "rerank_scores":torch.gather(score,2,selected).detach().cpu(),
                "layer":s.layer_idx,
                "heads":tuple(range(NKV)),
                "cold_token_count":n,
                "hot_token_count":int(cache["k"].shape[2]),
                "latency_ms":(time.perf_counter()-started)*1000.0,
            }
        return s.kcodec.unpack(kp,dtype),s.vcodec.unpack(vp,dtype)

    @torch.no_grad()
    def prefill_cached(s,x,cos,sin,max_ctx=2048,exact=True,
                       archive_cold=False,cold_chunk=128,cold_topk=32,
                       retrieval_trace=False):
        z=s.n1(x); q,k0,v0=s._qkv(z,cos,sin)
        if s.kv_bits==1:
            kp=s.kcodec.pack(k0); vp=s.vcodec.pack(v0)
            k=s.kcodec.unpack(kp,z.dtype); v=s.vcodec.unpack(vp,z.dtype)
        else:
            kp,ks=kv2_pack(k0); vp,vs=kv2_pack(v0)
            k=kv2_unpack(kp,ks,z.dtype); v=kv2_unpack(vp,vs,z.dtype)
        out=s._finish_attention(x,q,k,v,causal=True,exact=exact)
        cache={"k":kp[:,:,-max_ctx:],"v":vp[:,:,-max_ctx:],
               "format":("random_walsh_ctv_1bit_v1" if s.kv_bits==1
                         else "walsh_hadamard_2bit_v1")}
        if archive_cold and s.kv_bits==1:
            archive=PagedKVArchive(
                kp.shape[0],kp.shape[1],kp.shape[-1],
                page_size=int(cold_chunk),device=kp.device)
            cache.update({
                "cold_enabled":True,"cold_archive":archive,
                "cold_k":PagedKVView(archive,"k"),
                "cold_v":PagedKVView(archive,"v"),
                "chunk_keys":ExactChunkCountView(archive,int(cold_chunk)),
                "cold_chunk":int(cold_chunk),"cold_topk":int(cold_topk),
                "cold_last_indices":None,"cold_last_trace":None,
                "retrieval_trace_enabled":bool(retrieval_trace),
                "hot_start_position":max(0,int(x.shape[1])-int(max_ctx)),
            })
        else:
            cache["cold_enabled"]=False
        if s.kv_bits==2:
            cache.update({"ks":ks[:,:,-max_ctx:],"vs":vs[:,:,-max_ctx:]})
        return out,cache

    @torch.no_grad()
    def decode_cached(s,x,position,cache,max_ctx=2048,exact=True,
                      retrieve_cold=True):
        expected=("random_walsh_ctv_1bit_v1" if s.kv_bits==1
                  else "walsh_hadamard_2bit_v1")
        if cache.get("format")!=expected:
            raise ValueError(f"legacy or unknown KV cache; expected {expected}")
        cos,sin=cs_at(position,x.shape[1],x.device)
        z=s.n1(x); q,k0,v0=s._qkv(z,cos,sin)
        if s.kv_bits==1:
            kp=s.kcodec.pack(k0); vp=s.vcodec.pack(v0)
        else:
            kp,ks=kv2_pack(k0); vp,vs=kv2_pack(v0)
        joined_k=torch.cat((cache["k"],kp),2)
        joined_v=torch.cat((cache["v"],vp),2)
        overflow=max(0,joined_k.shape[2]-max_ctx)
        cache["k"],cache["v"]=joined_k,joined_v
        s._archive_overflow(cache,overflow)
        cache["k"]=cache["k"][:,:,overflow:]
        cache["v"]=cache["v"][:,:,overflow:]
        if s.kv_bits==1:
            k=s.kcodec.unpack(cache["k"],z.dtype)
            v=s.vcodec.unpack(cache["v"],z.dtype)
            cold_k,cold_v=(s._retrieve_cold(q,cache,z.dtype)
                           if retrieve_cold else (None,None))
            if cold_k is not None:
                k=torch.cat((cold_k,k),2); v=torch.cat((cold_v,v),2)
        else:
            cache["ks"]=torch.cat((cache["ks"],ks),2)[:,:,-max_ctx:]
            cache["vs"]=torch.cat((cache["vs"],vs),2)[:,:,-max_ctx:]
            k=kv2_unpack(cache["k"],cache["ks"],z.dtype)
            v=kv2_unpack(cache["v"],cache["vs"],z.dtype)
        return s._finish_attention(x,q,k,v,causal=x.shape[1]>1,exact=exact),cache

class StructStep(nn.Module):
    def __init__(s):
        super().__init__(); s.Wq=RVQ(D,D,8,2); s.cin=RVQ(2*D,FFNH,8,2); s.cout=RVQ(FFNH,D,8,2)
        s.nf=RMS(D); s.verify=nn.Linear(D,1)                                          
    def forward(s,h):
        Bs,Ts,_=h.shape; q=s.Wq(h)
        att=(q@h.transpose(-1,-2))/math.sqrt(D)
        cm=torch.ones(Ts,Ts,dtype=torch.bool,device=h.device).tril()
        att=torch.softmax(att.masked_fill(~cm,-1e9),-1)
        r=att@h                                                                       
        h=s.nf(h+s.cout(F.silu(s.cin(torch.cat([h,r],-1)))))                          
        return h, torch.sigmoid(s.verify(h)).squeeze(-1)                              

    @torch.no_grad()
    def decode_cached(s,h_new,h_context):
        q=s.Wq(h_new)
        att=(q@h_context.transpose(-1,-2))/math.sqrt(D)
        if h_new.shape[1]>1:
            tq,tk=att.shape[-2:]
            cm=torch.arange(tk,device=h_new.device)[None,:] <= (
                torch.arange(tq,device=h_new.device)[:,None]+tk-tq)
            att=att.masked_fill(~cm[None],-1e9)
        w=torch.softmax(att,-1); r=w@h_context
        h=s.nf(h_new+s.cout(F.silu(s.cin(torch.cat([h_new,r],-1)))))
        return h,torch.sigmoid(s.verify(h)).squeeze(-1)
