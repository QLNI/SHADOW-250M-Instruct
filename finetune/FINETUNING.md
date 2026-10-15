# Fine-tune SHADOW 250M

You can fine-tune SHADOW on your own data with one GPU and one command, then export your model
as a 52 MB file that runs at hundreds of tokens per second on a plain CPU. This page shows the
full workflow with a real example we ran ourselves, results attached.

## What you need

* One GPU with 12 GB or more (a gaming laptop works; we used one)
* Python with torch, numpy, sentencepiece
* The files in this repo: `shadow250m_instruct.pt`, `fp131072.npy`, `modeling/`, `finetune.py`, `export_model.py`

## 1. Prepare your data

One conversation per line in a .jsonl file:

    {"messages": [{"role": "user", "content": "your question"}, {"role": "assistant", "content": "the answer you want"}]}

A few hundred conversations are enough for a style change. A few thousand for a domain.
`examples_pirate.jsonl` in this repo is the dataset used in the example below.

## 2. Train

    python finetune.py --data my_data.jsonl --steps 150 --out my_model

Defaults: learning rate 1e-5 with cosine decay, loss only on assistant tokens, batch of
32,768 tokens per step, quantisation kept in the loop so the exported model matches the
trained one. A small validation split is held out automatically and printed before and after.

| hardware | speed | 150 steps |
|---|---|---|
| laptop RTX 3080 (thermally limited) | ~0.9k tokens/s | ~90 min |
| A100 | ~7.7k tokens/s | ~12 min |

## 3. Export and run on CPU

    python export_model.py my_model/finetuned.pt my_model.shdw
    ./shadow my_model.shdw fp131072.npy --chat

The export prints a round-trip check against the trained model. Your fine-tuned model is now
a 52 MB file.

## The example we ran: pirate mode

472 conversations, assistant answers rewritten in pirate speech. 150 steps, 90 minutes on a
laptop GPU. Validation loss on the pirate data went from 2.570 to 2.512. PIQA on a 500-item
subset stayed level: 0.61 before, 0.63 after. Same prompts, greedy decoding, before and after:

**"What is the capital of France?"**

Before:
> The capital of France is Paris. It is the seat of the French Republic, and it is known for
> its historical landmarks, including the Eiffel Tower, the Louvre Museum...

After:
> The capital of France be Paris. It is a UNESCO World Heritage Site and a major
> international city. It has a rich history and a rich culture. Paris be home to the Louvre
> Museum...

**"Give three tips for staying focused while studying."**

Before:
> 1. Aim for a balance of time and energy. Aim for 10-15 minutes of focused work per day...

After:
> When confronted with a problem, staying focused means focusing on what needs to be done
> first. Blimey, to stay motivated, it's essential to set realistic goals and keep the
> momentum going...

The knowledge stays, the voice changes. Swap the pirate data for your support tone, your
domain, or your language style and the same 90 minutes applies.

## Tips

* Style: 100 to 500 pairs, 100 to 200 steps.
* Domain knowledge: thousands of pairs, 300 to 1000 steps, and consider mixing 30 to 50
  percent general chat data so the model does not forget how to converse.
* Learning rate: stay at or below 2e-5. This model trains with quantisation active and high
  rates make it unstable.
* Check before and after on prompts you care about, with greedy decoding, like we did above.
