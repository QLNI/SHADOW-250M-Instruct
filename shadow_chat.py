"""Start chatting with SHADOW. Picks the right binary for your system automatically.
    python shadow_chat.py
"""
import os, sys, platform, subprocess, pathlib
HERE = pathlib.Path(__file__).resolve().parent
osname = platform.system()
if osname == "Windows": k = HERE / "deployment" / "bin" / "windows" / "shadow.exe"
elif osname == "Linux": k = HERE / "deployment" / "bin" / "linux" / "shadow"
else: sys.exit("macOS build available on request: saikiranbathula1@gmail.com")
if osname != "Windows": os.chmod(k, 0o755)
sys.path.insert(0, str(HERE))
from shadow_runtime import Engine
eng = Engine(str(HERE / "deployment" / "shadow250m_instruct.shdw"), str(HERE / "deployment" / "fp131072.npy"), kernel=str(k))
print("SHADOW 250M. Type your message, 'quit' to stop.")
while True:
    try: q = input("you> ").strip()
    except EOFError: break
    if q in ("quit", "exit"): break
    if q: print("shadow>", eng.chat(q))
