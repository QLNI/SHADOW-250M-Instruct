import argparse, sys
from . import Engine
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--table", required=True)
ap.add_argument("--archive"); ap.add_argument("--ask"); ap.add_argument("--chat", action="store_true")
a = ap.parse_args()
eng = Engine(a.model, a.table, archive=a.archive)
if a.ask: print(eng.answer(a.ask))
elif a.chat:
    while True:
        try: q = input("you> ")
        except EOFError: break
        if not q.strip(): continue
        print("shadow>", eng.chat(q))
