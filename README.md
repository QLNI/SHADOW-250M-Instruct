<h1 align="center">SHADOW 250M Instruct</h1>

![shadow](banner.jpg)

<p align="center">
  Instruct model of SHADOW 250M · 250M Parameters · 100M-Token Offline Context · 60 MB
</p>

<p align="center">
  License CC BY-NC 4.0 · <a href="https://huggingface.co/NODEMIND/SHADOW-250M">Hugging Face page</a>
</p>

## Get it running in one minute

The model is in this repository. Clone and chat:

    git clone https://github.com/QLNI/SHADOW-250M-Instruct
    cd SHADOW-250M-Instruct
    python shadow_chat.py

That picks the right prebuilt runtime for your system (Windows and Linux included, macOS
on request) and starts the chat. No framework, no downloads beyond the clone.

**SHADOW 250M Instruct** is a 250 million parameter language model built from scratch,
trained on 30 billion tokens of English text with about 0.7 billion further tokens of
instruction tuning. The complete deployment is 60 MB, vocabulary included. It runs at about 400 tokens per
second on a laptop CPU and uses about 80 MB of RAM.

Language modeling quality of the base model, measured on held-out English web text that
was never seen in training (educational web pages, 2,048 token windows): cross entropy
3.15 nats per token, perplexity 23.3, 0.99 bits per byte.

Next to its 2,048 token attention window, SHADOW can use an offline archive: a compressed
cache of up to 100 million tokens that stays on your hard disk. The model was trained from
step 1 to retrieve from this archive, all the way up to 100 million tokens. Due to a
limited budget it was not trained to reason over the tokens in the disk cache. It finds
facts and reads them back; it does not think across them.

For scale, the official released weight files of well-known small models, next to ours:

| model | parameters | weights on disk |
|---|---|---|
| GPT-2 | 124M | 548 MB |
| SmolLM2-135M-Instruct | 135M | 269 MB |
| SmolLM2-360M-Instruct | 360M | 724 MB |
| Qwen2.5-0.5B-Instruct | 494M | 988 MB |
| **SHADOW 250M Instruct** | **250M** | **60 MB, vocabulary included** |

Sizes are the main weight file of each official repository, as published. None of the
models above reads a 100 million token archive from disk.

*(The project was named after my first dog, Shadow, as it is my first big project.)*

---

## Some samples

Every sample shows the exact settings it was generated with, using the files in this repo.
Nothing on this page was made with settings other than the ones shown.

### 1.

Prompt: "Explain photosynthesis in two sentences."<br>
Settings: greedy decoding (temperature 0)<br>
Answer:
> Photosynthesis is a process in which plants convert sunlight into chemical energy, which
> is then used to produce oxygen and other chemicals. This process is called photosynthesis.

### 2.

Prompt: "Write a short poem about the sea."<br>
Settings: temperature 0.25 · top-k 30 · repetition penalty 1.15 · seed 2<br>
Answer:
> The waves had swept over, and they were crashing against each other like rocks on top of
> one another. The wind was strong but not very gentle as it tried to push them towards
> their destination without lifting any weight or pulling away from those who stood in
> front of her with a stiff smile that made the sea look even more majestic than before.

### 3. Reading its archive

Prompt: "What is the serial number of device Grus-189?" The fact sits 50.6 million tokens
deep in the 100M evaluation archive on disk.<br>
Settings: archive mode as shipped, retrieval k=16<br>
Answer:
> SN-442976

---

## Benchmarks

All numbers are our own, measured on held-out data with the harness shipped in
`benchmarks/`, together with the raw results. We compare with nobody on this page; the
harness runs on any archive of yours as well.

Standard, acc_norm, zero-shot, full sets, greedy scoring:

| PIQA | HellaSwag | ARC-Easy | ARC-Challenge |
|---|---|---|---|
| 0.60 | 0.34 | 0.41 | 0.24 |

Long context, answered from the archive on disk. Settings: archive mode as shipped,
retrieval k=16, exact match scoring, archives and facts held out from all training:

| | 1M tokens | 10M tokens | 100M tokens |
|---|---|---|---|
| Needle in a haystack (5 depths) | 0.98 | 0.98 | 0.98 |
| Needle with look-alike distractors | 1.00 | 1.00 | – |
| Multi-key needles | 1.00 | 1.00 | – |
| Two-hop variable tracking | 1.00 | 1.00 | – |
| Scattered story facts, latest wins | 1.00 | 1.00 | – |
| Fact QA, 6 task types with abstain | 0.97 | 0.95 | 0.83 |

The vocabulary table has its own benchmark. Every token carries a fixed 512-bit code;
`benchmarks/embedding_bench.py` scores those codes on human word-similarity ratings
(WordSim-353, 337 single-token pairs), offline, using only the files in this repo:

| codes | Spearman correlation with human ratings |
|---|---|
| the shipped vocabulary table | 0.619 |
| random 512-bit codes | 0.029 |

Classic 300-dimension float word vectors reach about 0.65 to 0.70 on the same test using
19 times more bits per word.

## Architecture

| Hyperparameter | Value |
|---|---|
| Hidden size | 1536 |
| Layers | 10 |
| Attention heads | 24 (GQA, 2 KV heads) |
| Head dim | 64 |
| Intermediate size (SwiGLU) | 4224 |
| Vocab size | 131,072 (frozen, 0 trainable parameters) |
| Positional encoding | RoPE θ=10,000 |
| Normalization | RMSNorm, ε=10-6 (incl. QK-Norm) |
| Tied embeddings | Yes (shared vocabulary table) |
| Attention window | 2,048 tokens + offline archive up to 100M |
| Body weight precision | under 2 bits per weight |
| Parameters | 250M |
| Runtime | bundled CPU kernel (AVX2/AVX-512), no framework needed |

![framework](framework.png)

## Performance

Measured on a laptop CPU with 8 physical cores, using the exact files in this repo. The
bundled kernel handles chat, the two-tier KV cache, and a live memory panel (`--status`).

| | |
|---|---|
| decode speed, 8 threads | 402 tokens/s |
| decode speed, 4 / 2 / 1 threads | 393 / 275 / 158 tokens/s |
| prefill speed | 409 tokens/s |
| RAM while chatting | ~80 MB |
| archive index build (once per archive, at load) | 2 s at 1M · 21 s at 10M · 3.2 min at 100M |
| retrieval per question | 37 ms at 10M · 435 ms at 100M |
| archive question, end to end | 0.45 s at 100M |

## The pieces, and what each one buys you

**The frozen binary vocabulary.** Every other language model spends a large trainable
matrix on its vocabulary; at this size it is often the single largest thing in the file.
SHADOW replaces it with a fixed table of 512-bit codes, one per token, 8.4 MB for all
131,072 tokens, zero trainable parameters. The same table serves the input and the output
of the network, so it cannot drift during training and never needs quantising afterwards:
the file you download is bit for bit the vocabulary the model was trained with. The codes
carry real meaning, and you can measure that yourself: `benchmarks/embedding_bench.py`
scores the table on human word-similarity ratings (WordSim-353) and reaches Spearman 0.619
with 512 bits per word, where random codes score 0.03 and classic 300-dimension float word
vectors score about 0.65 to 0.70 using 19 times more bits per word.

**The sub-2-bit body.** The transformer's weights are stored below 2 bits per weight and
are trained that way from the start, not compressed afterwards. That is why 250M
parameters fit in 52 MB with nothing lost relative to how the model trained, and why the
CPU kernel can stream the whole model through cache fast enough for 400 tokens per second.

**The two-tier KV cache and the offline archive.** The model keeps its last 2,048 tokens
at full precision in RAM like any transformer. Everything older is kept at 1 bit per value
and can live on disk, 320 bytes per token, which is what makes a 100 million token memory
cost 32 GB of disk instead of terabytes of RAM. Per question, the runtime pulls only the
few blocks that matter back into the window. The model was trained with this exact memory
layout from step 1, which is why it can read what comes back.

**The CPU kernel.** One small binary, no framework, no Python. It handles chat, both KV
tiers, and shows you a live panel of memory and speed with `--status`. AVX2 and AVX-512
paths are chosen automatically.

**The runtime.** `shadow_runtime/` adds archive question answering on top of the kernel:
a lexical index over the archive, retrieval, and an extraction layer that reads the
retrieved blocks. This is the pipeline behind every long-context number on this page.

**The fine-tuning kit.** Master weights, training script, and an exporter, so a style or
domain fine-tune runs on one gaming GPU and comes back out as your own 52 MB CPU model.
The pirate demonstration in [FINETUNING.md](FINETUNING.md) is the worked example.

## Fine-tuning

Everything needed is in `finetune/`; the 539 MB master weights download from the Hugging Face page (finetune/shadow250m_instruct.pt).

Yes, you can fine-tune it, on one GPU, and export your own 52 MB model for CPU. We did it
ourselves as a demonstration: 90 minutes on a laptop GPU turned SHADOW into a pirate
assistant, with benchmark scores unchanged. The full guide with the commands, the dataset,
and the before and after results is in [finetune/FINETUNING.md](finetune/FINETUNING.md).

> The capital of France be Paris. It is a UNESCO World Heritage Site... Yarr!

## Repository layout

    deployment/            the model: weights, vocabulary, and the runtime binaries
      shadow250m_instruct.shdw   52 MB weights
      fp131072.npy               8.4 MB vocabulary
      bin/windows/  bin/linux/   prebuilt CPU runtimes (macOS on request)
    tokenizer/             3 files, 5 MB
    finetune/              master weights, training script, exporter, guide, worked example
    benchmarks/            results, report, harness
    shadow_runtime/        archive question answering (Python)

## Usage

Easiest start, any system:

    python shadow_chat.py

Chat directly with the binary, no Python needed. Windows:

    deploymentin\windows\shadow.exe deployment\shadow250m_instruct.shdw deploymentp131072.npy --chat

Linux:

    deployment/bin/linux/shadow deployment/shadow250m_instruct.shdw deployment/fp131072.npy --chat

Add --status to either for a live memory panel. Ask a question against an archive (a folder
holding a tokens.u32 stream):

    python -m shadow_runtime --model shadow250m_instruct.shdw --table fp131072.npy \
        --archive path/to/archive --ask "your question"

Python:

    from shadow_runtime import Engine
    eng = Engine("shadow250m_instruct.shdw", "fp131072.npy", archive="path/to/archive")
    print(eng.answer("your question"))

### Chat template

    <start_of_turn>user
    {message}<end_of_turn>
    <start_of_turn>model
    {response}<end_of_turn>

## Intended use

Intended:

* Local assistants on CPU-only hardware, fully offline
* Question answering over large private text archives: logs, books, documentation
* Fine-tuning your own small assistant on one GPU
* Research and education on small models and long context

Not intended:

* Production or user-facing deployment without human review
* Factual question answering from the model's own memory, advice, or decision support
* Non-English text

## Limitations and bias

* Small. At 250M parameters, open facts, arithmetic, and long answers are weak. Expect
  mistakes outside the archive.
* The model retrieves and reads from its archive. It was not trained to reason across
  many archive documents; that needs a bigger training budget than this project had.
  Two-hop chains degrade at 100M tokens.
* Trained on public web text, so its outputs can carry the biases of that text.
* English only.

## Contact

Questions, results, or something you built with it: saikiranbathula1@gmail.com

---

*© NODEMIND 2026*
