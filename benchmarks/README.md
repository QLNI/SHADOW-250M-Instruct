# Benchmarks

`results.json` holds every number on the model card; `report.pdf` is the full report.
The evaluation archives are held out and not distributed.

`run.py` is the harness that produced the numbers. You can point it at your own archive
(a folder with a `tokens.u32` stream and a `bank_valid.jsonl` question bank in the same
format as `results.json` describes) to run the same evaluation on your own data.
