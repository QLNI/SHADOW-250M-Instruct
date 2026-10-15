"""Export a fine-tuned checkpoint to the 52 MB deploy file for the CPU runtime.
    python export_model.py my_model/finetuned.pt my_model.shdw
"""
import os, sys, pathlib
for k, v in {"SHADOW_D": "1536", "SHADOW_NL": "10", "SHADOW_NH": "24", "SHADOW_NKV": "2", "SHADOW_HD": "64",
             "SHADOW_FFNH": "4224", "SHADOW_FAST_ATTN": "1", "SHADOW_KV_BITS": "1", "SHADOW_KV_TWO_TIER": "1"}.items():
    os.environ.setdefault(k, v)
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "modeling"))
import subprocess
src, dst = sys.argv[1], sys.argv[2]
tmp = dst + ".full"
r = subprocess.run([sys.executable, str(HERE / "modeling" / "export_ternary.py"), src, tmp],
                   env={**os.environ, "PYTHONPATH": str(HERE / "modeling")})
if r.returncode: sys.exit(r.returncode)
r = subprocess.run([sys.executable, str(HERE / "modeling" / "repack_shdw.py"), tmp, dst, "--fp16"])
os.remove(tmp)
print("wrote", dst)
