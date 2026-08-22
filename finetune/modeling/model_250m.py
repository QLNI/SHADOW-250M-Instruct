import os, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import common
from common import D,NL,NH,NKV,HD,FFNH,FPD, RMS,RVQ,Block,StructStep, requant, cs


def load_vocab(fp_path,dev):
    fp=np.unpackbits(np.load(fp_path),axis=1)[:,:FPD]
    cent=torch.tensor(fp.astype(np.float32)*2-1,device=dev)      
    return cent, F.normalize(cent,dim=-1), cent.shape[0]

class Shadow250M(nn.Module):
    def __init__(s,cent,cent_n,V,use_memory=False):
        super().__init__()


        s.register_buffer("cent",cent,persistent=False)
        s.register_buffer("cent_n",cent_n,persistent=False); s.V=V
        s.inp=nn.Linear(FPD,D,bias=False)
        s.b=nn.ModuleList([Block(i) for i in range(NL)])         
        s.struct=StructStep()                                    
        s.nf=RMS(D); s.head=nn.Linear(D,FPD,bias=False)          
        s.tied_bias=nn.Parameter(torch.zeros(V))


        s.default_use_memory=bool(use_memory)
        inv=1.0/(10000**(torch.arange(0,HD,2).float()/HD)); s.register_buffer("inv",inv,persistent=False)
    def trunk(s,idx):
        cos,sin=cs(idx.shape[1],idx.device); x=s.inp(s.cent[idx])
        for blk in s.b: x=blk(x,cos,sin)
        x,conf=s.struct(x)                                       
        return s.nf(x), conf
    def logits(s,x): return (s.head(x).float()@s.cent_n.T)+s.tied_bias      
    def forward(s,idx,ys=None):
        x,conf=s.trunk(idx)
        if ys is None: return s.logits(x)

        ph=s.head(x).float().reshape(-1,FPD); yf=ys.reshape(-1); Nn=ph.shape[0]; CH=2048
        return sum(F.cross_entropy(ph[i:i+CH]@s.cent_n.T+s.tied_bias,yf[i:i+CH],reduction="sum") for i in range(0,Nn,CH))/Nn

    @torch.no_grad()
    def prefill_cached(s,idx,max_ctx=2048,exact_shiftmax=True,
                       use_memory=None,memory_chunk=128,memory_capacity=256,
                       retrieval_trace=False,stream_block=256):
        if use_memory is None: use_memory=s.default_use_memory
        if idx.shape[1]>max_ctx:
            if not use_memory:
                idx=idx[:,-max_ctx:]
            else:
                logits,state=s.prefill_cached(
                    idx[:,:max_ctx],max_ctx=max_ctx,
                     exact_shiftmax=exact_shiftmax,use_memory=use_memory,
                     memory_chunk=memory_chunk,memory_capacity=memory_capacity,
                     retrieval_trace=retrieval_trace,stream_block=stream_block)
                cursor=max_ctx
                final=idx.shape[1]-1
                while cursor<final:
                    stop=min(final,cursor+int(stream_block))
                    state=s.ingest_cached(idx[:,cursor:stop],state)
                    cursor=stop
                return s.decode_cached(idx[:,final:final+1],state)
        cos,sin=cs(idx.shape[1],idx.device); x=s.inp(s.cent[idx]); layers=[]
        for blk in s.b:
            x,cache=blk.prefill_cached(x,cos,sin,max_ctx=max_ctx,
                                       exact=exact_shiftmax,
                                       archive_cold=bool(use_memory),
                                       cold_chunk=memory_chunk,
                                       retrieval_trace=retrieval_trace); layers.append(cache)
        trunk=x[:,-max_ctx:]; y,_=s.struct(x); ph=s.head(s.nf(y[:,-1])).float()
        return ph@s.cent_n.T+s.tied_bias,{"layers":layers,"trunk":trunk,
            "position":idx.shape[1],"max_ctx":max_ctx,
            "exact_shiftmax":exact_shiftmax,
            "retrieval_trace":bool(retrieval_trace),
            "stream_block":int(stream_block),
            "memory":common.make_memory_state(
                idx.shape[0],D,idx.device,chunk_size=memory_chunk,
                capacity=memory_capacity) if use_memory else None}

    @torch.no_grad()
    def ingest_cached(s,idx_block,state):
        if idx_block.shape[1]==0:
            return state
        x=s.inp(s.cent[idx_block]); pos=int(state["position"])
        for i,blk in enumerate(s.b):
            x,state["layers"][i]=blk.decode_cached(
                x,pos,state["layers"][i],max_ctx=state["max_ctx"],
                exact=state["exact_shiftmax"],retrieve_cold=False)
        combined=torch.cat((state["trunk"],x),1)
        overflow=max(0,combined.shape[1]-state["max_ctx"])
        common.memory_absorb_evicted(
            state.get("memory"),combined[:,:overflow],s.head,s.nf)
        state["trunk"]=combined[:,overflow:]
        s.struct.decode_cached(x,state["trunk"])
        state["position"]=pos+idx_block.shape[1]
        return state

    @torch.no_grad()
    def decode_cached(s,idx_new,state):
        if state.get("memory") is not None and idx_new.shape[1]!=1:
            logits=None
            for j in range(idx_new.shape[1]):
                logits,state=s.decode_cached(idx_new[:,j:j+1],state)
            return logits,state
        x=s.inp(s.cent[idx_new]); pos=int(state["position"])
        for i,blk in enumerate(s.b):
            x,state["layers"][i]=blk.decode_cached(x,pos,state["layers"][i],
                                                   max_ctx=state["max_ctx"],
                                                   exact=state["exact_shiftmax"])
        combined=torch.cat((state["trunk"],x),1)
        overflow=max(0,combined.shape[1]-state["max_ctx"])
        common.memory_absorb_evicted(
            state.get("memory"),combined[:,:overflow],s.head,s.nf)
        state["trunk"]=combined[:,overflow:]
        context=common.memory_append_recall(
            state.get("memory"),x[:,-1],state["trunk"],s.head,s.nf)
        y,_=s.struct.decode_cached(x,context); state["position"]=pos+idx_new.shape[1]
        ph=s.head(s.nf(y[:,-1])).float(); return ph@s.cent_n.T+s.tied_bias,state

if __name__=="__main__":
    dev="cuda" if torch.cuda.is_available() else "cpu"
    FP=os.path.join(os.path.dirname(__file__),"dolma3_fp_512b.npy")
    if not os.path.exists(FP): FP=os.path.join(os.path.dirname(__file__),"..","..","shadow-o","phase0_tokenizer","dolma3_fp_512b.npy")
    cent,cent_n,V=load_vocab(FP,dev); m=Shadow250M(cent,cent_n,V).to(dev)
    body=sum(p.numel() for n,p in m.named_parameters() if "cent" not in n and "tied_bias" not in n)
    print(f"SHADOW-O 250M | body {body/1e6:.1f}M params | vocab {V} (0 params) | config d{D} L{NL} GQA{NH}/{NKV}")
    b=m.b[0]; print(f"  proj {b.q.bits():.3f} bit/w | FFN {b.up.bits():.3f} bit/w | struct step: 1 | Q4/loop: none")
    x=torch.randint(0,V,(1,64),device=dev); requant(m)
    with torch.autocast("cuda",dtype=torch.bfloat16) if dev=="cuda" else torch.no_grad():
        lg=m(x); print(f"  forward OK: logits {tuple(lg.shape)} finite={bool(torch.isfinite(lg).all())}")
