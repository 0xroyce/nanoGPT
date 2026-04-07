# Post-Transformer nanoGPT Runbook

Author:

- Petr Royce
- GitHub: `0xroyce`


## Purpose

This file explains:

- how to test the modified `nanoGPT` code
- how to run baseline and local-attention experiments
- what source data is required
- how to prepare that data inside this repo

Canonical project path:

- [nanoGPT](/Users/0xroyce/WebstormProjects/Phoenix/nanoGPT)


## What Changed in the Code

The current `nanoGPT` fork includes:

- experiment flags in [model.py](/Users/0xroyce/WebstormProjects/Phoenix/nanoGPT/model.py)
- optional structured forward outputs
- local causal attention via `attention_mode='local'`
- optional experiment metric logging in [train.py](/Users/0xroyce/WebstormProjects/Phoenix/nanoGPT/train.py)


## Environment Setup

From inside [nanoGPT](/Users/0xroyce/WebstormProjects/Phoenix/nanoGPT), install the main dependencies:

```bash
pip install torch numpy transformers datasets tiktoken wandb tqdm requests
```

Notes:

- `requests` is needed by the Shakespeare data prep script
- `transformers` is needed if you want to initialize from GPT-2 checkpoints
- `datasets` and `tiktoken` are needed for OpenWebText preparation


## Test Levels

There are three useful levels of testing:

1. code smoke test with no dataset
2. tiny real training run using `shakespeare_char`
3. larger baseline run using `openwebtext`


## 1. No-Data Smoke Tests

These tests do not require any dataset files.

Run from [nanoGPT](/Users/0xroyce/WebstormProjects/Phoenix/nanoGPT):

### Syntax check

```bash
python -m py_compile model.py train.py
```

### Tiny forward-pass check for dense and local attention

```bash
python - <<'PY'
import torch
from model import GPT, GPTConfig

for mode, window in [('dense', 16), ('local', 4)]:
    cfg = GPTConfig(
        vocab_size=64,
        block_size=16,
        n_layer=2,
        n_head=2,
        n_embd=8,
        attention_mode=mode,
        attention_window=window,
    )
    model = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 16))
    targets = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, loss = model(idx, targets)
    info = model(idx, targets, return_info=True)
    print(mode, logits.shape, round(loss.item(), 4), info.metrics)
PY
```

What this checks:

- model construction
- dense attention path
- local attention path
- structured metric output


## 2. Quick Real Dataset Test

Recommended first real test:

- `shakespeare_char`

Why:

- tiny dataset
- fast to prepare
- useful for smoke-testing training after code changes

### Required source data

Source:

- Tiny Shakespeare text downloaded by [prepare.py](/Users/0xroyce/WebstormProjects/Phoenix/nanoGPT/data/shakespeare_char/prepare.py)

Preparation output:

- `nanoGPT/data/shakespeare_char/train.bin`
- `nanoGPT/data/shakespeare_char/val.bin`
- `nanoGPT/data/shakespeare_char/meta.pkl`

### Prepare the data

Run:

```bash
cd /Users/0xroyce/WebstormProjects/Phoenix/nanoGPT
python data/shakespeare_char/prepare.py
```

This downloads:

- `input.txt` from the tiny Shakespeare source used by Karpathy’s char-rnn examples

### Baseline training run

```bash
python train.py config/train_shakespeare_char.py --device=cpu --compile=False
```

If you have a GPU:

```bash
python train.py config/train_shakespeare_char.py
```

### Local attention training run

```bash
python train.py config/train_shakespeare_char.py \
  --attention_mode=local \
  --attention_window=128 \
  --log_experiment_metrics=True
```

Good conservative windows to try first:

- `128`
- `64`

### Sample from a trained checkpoint

```bash
python sample.py --out_dir=out-shakespeare-char
```

If using CPU:

```bash
python sample.py --out_dir=out-shakespeare-char --device=cpu
```


## 3. Serious Baseline / Benchmark Dataset

Recommended real benchmark dataset:

- `openwebtext`

Why:

- this is the standard dataset already wired into `nanoGPT`
- it is the right starting point for cost-to-target-loss comparisons

### Required source data

Source:

- Hugging Face `openwebtext` dataset

Preparation output:

- `nanoGPT/data/openwebtext/train.bin`
- `nanoGPT/data/openwebtext/val.bin`

Notes:

- no `meta.pkl` is required here because GPT-2 tokenization is fixed
- the prep script uses `datasets` and `tiktoken`
- the prep script downloads a large dataset and needs substantial disk space

Approximate size from the prep script comments:

- Hugging Face cache usage: about `54GB`
- `train.bin`: about `17GB`
- `val.bin`: about `8.5MB`

### Prepare the data

Run:

```bash
cd /Users/0xroyce/WebstormProjects/Phoenix/nanoGPT
python data/openwebtext/prepare.py
```

### Evaluate a GPT-2 baseline

```bash
python train.py config/eval_gpt2.py --compile=False
```

This is evaluation-only and useful for sanity checking the environment.

### Dense baseline training

Small single-GPU style baseline:

```bash
python train.py --dataset=openwebtext --compile=False --device=cuda
```

Full repo-style GPT-2 reproduction config:

```bash
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
```

### Local attention benchmark run

Start with explicit overrides:

```bash
python train.py \
  --dataset=openwebtext \
  --device=cuda \
  --compile=False \
  --attention_mode=local \
  --attention_window=256 \
  --log_experiment_metrics=True
```

Then compare against:

```bash
python train.py \
  --dataset=openwebtext \
  --device=cuda \
  --compile=False \
  --attention_mode=dense \
  --log_experiment_metrics=True
```

Conservative first local windows:

- `256`
- `128`


## Metrics to Watch

For every run, capture:

- train loss
- validation loss
- iteration time
- MFU
- `attention/active_fraction`
- `attention/window_tokens`

If possible also record:

- GPU memory use
- tokens/sec
- exact config used


## What Files Must Exist Before Training

### For `dataset=shakespeare_char`

Required:

- `data/shakespeare_char/train.bin`
- `data/shakespeare_char/val.bin`
- `data/shakespeare_char/meta.pkl`

### For `dataset=openwebtext`

Required:

- `data/openwebtext/train.bin`
- `data/openwebtext/val.bin`


## Fast Pre-Flight Checklist

Before launching a paid GPU run:

1. run `python -m py_compile model.py train.py`
2. run the no-data smoke test
3. run one short `shakespeare_char` training job
4. only then launch `openwebtext` experiments


## Suggested First Paid GPU Sequence

Once Prime Intellect GPU access is ready:

1. prepare `shakespeare_char` and run a short dense smoke train
2. run the same config with `attention_mode=local`
3. prepare `openwebtext` if not already prepared
4. run a dense baseline on your chosen budget
5. run local attention with `attention_window=256`
6. run local attention with `attention_window=128`
7. compare quality, speed, and memory


## Common Failure Modes

### Missing dataset files

Symptom:

- training fails because `train.bin` or `val.bin` is missing

Fix:

- run the appropriate `prepare.py` script first

### `torch.compile` issues

Symptom:

- compile-related errors or unstable behavior on some systems

Fix:

- add `--compile=False`

### CPU or low-memory environment

Symptom:

- training is too slow or runs out of memory

Fix:

- reduce `block_size`
- reduce `batch_size`
- reduce `n_layer`, `n_head`, and `n_embd`
- use `shakespeare_char` first


## Minimal Commands Reference

### Prepare Shakespeare char data

```bash
python data/shakespeare_char/prepare.py
```

### Baseline Shakespeare char train

```bash
python train.py config/train_shakespeare_char.py
```

### Local attention Shakespeare char train

```bash
python train.py config/train_shakespeare_char.py --attention_mode=local --attention_window=128 --log_experiment_metrics=True
```

### Prepare OpenWebText

```bash
python data/openwebtext/prepare.py
```

### Dense OpenWebText run

```bash
python train.py --dataset=openwebtext --device=cuda --compile=False
```

### Local OpenWebText run

```bash
python train.py --dataset=openwebtext --device=cuda --compile=False --attention_mode=local --attention_window=256 --log_experiment_metrics=True
```
