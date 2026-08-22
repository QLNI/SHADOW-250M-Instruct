import sys, struct, numpy as np
src, dst = sys.argv[1], sys.argv[2]; FP16 = "--fp16" in sys.argv     
f = open(src, "rb"); assert f.read(4) == b"SHDW"; ver, n = struct.unpack("<II", f.read(8))
out = open(dst, "wb"); out.write(b"SHDW"); out.write(struct.pack("<II", 1, n))
tern_in = tern_out = dense_in = dense_out = 0
for _ in range(n):
    nl, = struct.unpack("<I", f.read(4)); name = f.read(nl); kind, = struct.unpack("<I", f.read(4))
    out.write(struct.pack("<I", nl)); out.write(name)
    if kind == 0:
        nd, = struct.unpack("<I", f.read(4)); dims = struct.unpack("<" + "I" * nd, f.read(4 * nd)); c = int(np.prod(dims))
        a = np.frombuffer(f.read(4 * c), np.float32); dense_in += 4 * c
        if FP16 and c >= 4096: out.write(struct.pack("<I", 5)); out.write(struct.pack("<I", nd)); out.write(struct.pack("<" + "I" * nd, *dims)); out.write(a.astype(np.float16).tobytes()); dense_out += 2 * c
        else: out.write(struct.pack("<I", 0)); out.write(struct.pack("<I", nd)); out.write(struct.pack("<" + "I" * nd, *dims)); out.write(a.tobytes()); dense_out += 4 * c
    elif kind == 1:
        o, i, g, st = struct.unpack("<IIII", f.read(16)); G = i // g; Npad = (o + 63) & ~63; nch = Npad // 64
        cb = f.read(st * g * 16 * 4); idx = f.read(st * nch * G * 32); sc = f.read(Npad * 4)
        out.write(struct.pack("<I", 1)); out.write(struct.pack("<IIII", o, i, g, st)); out.write(cb); out.write(idx); out.write(sc)
    elif kind == 3:
        o, i = struct.unpack("<II", f.read(8)); packed = np.frombuffer(f.read(o * i // 4), np.uint8).reshape(o, i // 4); rs = f.read(o * 4)
        tern_in += packed.nbytes
        c4 = np.stack([(packed >> (2 * j)) & 3 for j in range(4)], -1).reshape(o, i)          
        pad = (-i) % 5; c5 = np.concatenate([c4, np.ones((o, pad), np.uint8)], 1).reshape(o, -1, 5).astype(np.uint16)
        p5 = (c5[:, :, 0] + 3 * c5[:, :, 1] + 9 * c5[:, :, 2] + 27 * c5[:, :, 3] + 81 * c5[:, :, 4]).astype(np.uint8)
        out.write(struct.pack("<I", 4)); out.write(struct.pack("<II", o, i)); out.write(p5.tobytes()); out.write(rs); tern_out += p5.nbytes
    else: raise SystemExit(f"unknown kind {kind}")
out.close()
import os
print(f"ternary {tern_in/1e6:.1f} -> {tern_out/1e6:.1f} MB | dense {dense_in/1e6:.1f} -> {dense_out/1e6:.1f} MB | file {os.path.getsize(src)/1e6:.1f} -> {os.path.getsize(dst)/1e6:.1f} MB")
