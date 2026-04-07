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
- sparse FFN / MoE via `ffn_mode='moe'`
- retrieval-first memory via `use_retrieval_memory=True`
- optional persistent-memory banking via `use_persistent_memory=True`
- optional memory-controller routing via `use_memory_controller=True`
- optional multi-timescale optimizer groups via `use_multiscale_optim=True`
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
3. MoE comparison runs using `shakespeare_char`
4. retrieval comparison runs using `shakespeare_char`
5. larger baseline run using `openwebtext`


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

### Tiny forward-pass check for retrieval memory

```bash
python - <<'PY'
import torch
from model import GPT, GPTConfig

cfg = GPTConfig(
    vocab_size=64,
    block_size=16,
    n_layer=2,
    n_head=2,
    n_embd=16,
    use_retrieval_memory=True,
    memory_slots=4,
    memory_topk=2,
)
model = GPT(cfg)
idx = torch.randint(0, cfg.vocab_size, (2, 16))
targets = torch.randint(0, cfg.vocab_size, (2, 16))
info = model(idx, targets, return_info=True)
print(round(info.loss.item(), 4), info.metrics)
PY
```


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

### MoE training run

Recommended first MoE comparison:

```bash
python train.py config/train_shakespeare_char.py \
  --max_iters=500 \
  --eval_interval=100 \
  --ffn_mode=moe \
  --num_experts=4 \
  --experts_topk=2 \
  --log_experiment_metrics=True \
  --compile=False \
  --out_dir=out-moe-top2 | tee moe_top2.log
```

Observed result so far:

- `4 experts, top-2` is the strongest architectural signal in the project yet
- it improved validation loss relative to dense on `shakespeare_char`
- it is still much slower than dense in the current implementation

Important note:

- for now, prefer `--compile=False` on MoE runs while benchmarking routing behavior

### Retrieval-memory training run

Recommended first retrieval comparison:

```bash
python train.py config/train_shakespeare_char.py \
  --max_iters=500 \
  --eval_interval=100 \
  --use_retrieval_memory=True \
  --memory_slots=16 \
  --memory_topk=4 \
  --memory_retrieval_weight=1.0 \
  --log_experiment_metrics=True \
  --compile=False \
  --out_dir=out-memory-topk4 | tee memory_topk4.log
```

What this first version is:

- sequence-local read-only retrieval
- pooled memory slots built from the current sequence
- top-k memory injection before the block stack

What to compare against:

- dense baseline
- `4 experts, top-2` MoE

Observed result so far:

- retrieval with `memory_slots=16` and `memory_topk=4` reached about `0.9976` validation loss at step `500`
- the same setup reached about `0.6100` validation loss at step `1000`
- retrieval with `memory_slots=32` and `memory_topk=4` reached about `0.6088` validation loss at step `500`
- the same `32/4` setup reached about `0.3142` validation loss at step `1000`
- this strongly outperformed both dense and MoE top-2 on `shakespeare_char`
- runtime stayed much closer to dense than to MoE

Recommended next retrieval ablations:

- `memory_slots=8`, `memory_topk=2`
- `memory_slots=16`, `memory_topk=2`
- `memory_slots=32`, `memory_topk=4`
- `memory_slots=16`, `memory_topk=8`

Current best retrieval setting:

- `memory_slots=32`
- `memory_topk=4`
- `memory_retrieval_weight=1.0`
- `use_persistent_memory=False` for the current winning validated runs

Weight-ablation note:

- `memory_retrieval_weight=0.5` underperformed badly
- `memory_retrieval_weight=1.5` stayed competitive at step `500`
- `memory_retrieval_weight=1.0` remains the safest best default because it also won the longer `1000` step run

Persistent-memory note:

- the codebase now supports `use_persistent_memory=True`
- this adds an EMA-updated memory bank that persists across training steps
- evaluation resets memory explicitly before loss estimation
- both persistent variants tested so far underperformed retrieval-only
- they should be treated as failed prototypes, not current best candidates

Controller-routing note:

- the codebase now supports `use_memory_controller=True`
- this adds an explicit token-level routing layer for retrieval access
- this prototype underperformed badly on `openwebtext`
- it should be treated as a failed branch, not the preferred next run

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
HF_HOME=/workspace/hf_cache HF_DATASETS_CACHE=/workspace/hf_datasets_cache TMPDIR=/workspace/tmp python data/openwebtext/prepare.py
```

Practical note:

- in containers, avoid the default Hugging Face cache under `/root/.cache/huggingface`
- that path can fill the smaller overlay filesystem even when `/workspace` has plenty of free space

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

### Retrieval OpenWebText validation

Best validated retrieval run so far:

```bash
python train.py --dataset=openwebtext --device=cuda --compile=False --batch_size=12 --block_size=256 --gradient_accumulation_steps=8 --n_layer=6 --n_head=6 --n_embd=384 --max_iters=5000 --lr_decay_iters=5000 --warmup_iters=200 --eval_interval=500 --eval_iters=50 --log_interval=10 --wandb_log=False --use_retrieval_memory=True --memory_slots=32 --memory_topk=4 --memory_retrieval_weight=1.0 --log_experiment_metrics=True --out_dir=out-owt-memory-s32-k4-5k | tee owt_memory_s32_k4_5k.log
```

Observed result so far:

- retrieval transferred strongly to `openwebtext`
- the `32/4` retrieval setting reached about `2.7048` validation loss at step `2000`
- the same retrieval setting reached about `0.8092` validation loss at step `5000`
- retrieval metrics became sparse and highly selective on the larger dataset too

### Persistent-memory benchmark

Recommended next benchmark:

```bash
python train.py --dataset=openwebtext --device=cuda --compile=False --batch_size=8 --block_size=256 --gradient_accumulation_steps=4 --n_layer=6 --n_head=6 --n_embd=384 --max_iters=2000 --lr_decay_iters=2000 --warmup_iters=100 --eval_interval=200 --eval_iters=50 --log_interval=10 --wandb_log=False --use_retrieval_memory=True --use_persistent_memory=True --memory_slots=32 --memory_topk=4 --memory_retrieval_weight=1.0 --persistent_memory_momentum=0.95 --log_experiment_metrics=True --out_dir=out-owt-memory-s32-k4-persistent-2k | tee owt_memory_s32_k4_persistent_2k.log
```

Observed result:

- both persistent-memory variants underperformed badly relative to retrieval-only
- do not continue spending GPU time on those versions

### Memory-controller benchmark

Recommended next benchmark:

```bash
python train.py --dataset=openwebtext --device=cuda --compile=False --batch_size=8 --block_size=256 --gradient_accumulation_steps=4 --n_layer=6 --n_head=6 --n_embd=384 --max_iters=2000 --lr_decay_iters=2000 --warmup_iters=100 --eval_interval=200 --eval_iters=50 --log_interval=10 --wandb_log=False --use_retrieval_memory=True --use_memory_controller=True --memory_controller_fraction=0.5 --memory_slots=32 --memory_topk=4 --memory_retrieval_weight=1.0 --log_experiment_metrics=True --out_dir=out-owt-memory-s32-k4-controller-2k | tee owt_memory_s32_k4_controller_2k.log
```

Observed result:

- controller-routed retrieval reached about `4.6696` validation loss at step `2000`
- retrieval-only remained far better at about `2.7048`
- this branch reduced memory utilization and increased retrieval entropy

### Multi-timescale retrieval benchmark

Recommended next benchmark:

```bash
python train.py --dataset=openwebtext --device=cuda --compile=False --batch_size=8 --block_size=256 --gradient_accumulation_steps=4 --n_layer=6 --n_head=6 --n_embd=384 --max_iters=2000 --lr_decay_iters=2000 --warmup_iters=100 --eval_interval=200 --eval_iters=50 --log_interval=10 --wandb_log=False --use_retrieval_memory=True --memory_slots=32 --memory_topk=4 --memory_retrieval_weight=1.0 --use_multiscale_optim=True --retrieval_lr_scale=2.0 --log_experiment_metrics=True --out_dir=out-owt-memory-s32-k4-multiscale-x2-2k | tee owt_memory_s32_k4_multiscale_x2_2k.log
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
- `ffn/active_fraction`
- `moe/router_entropy`
- `moe/expert_utilization`
- `moe/expert_load_std`
- `memory/active_fraction`
- `memory/local_slot_utilization`
- `memory/local_slots`
- `memory/persistent_active_fraction`
- `memory/persistent_enabled`
- `memory/persistent_slot_utilization`
- `memory/persistent_slots`
- `memory/persistent_valid`
- `memory/slot_utilization`
- `memory/retrieval_entropy`

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
4. run one short retrieval or MoE variant
5. only then launch `openwebtext` experiments


## Suggested First Paid GPU Sequence

Once Prime Intellect GPU access is ready:

1. prepare `shakespeare_char` and run a short dense smoke train
2. run `4 experts, top-2` MoE on the same setup
3. run the retrieval-memory variant on the same setup
4. compare validation loss, iteration time, and routing / retrieval metrics
5. run a small retrieval ablation sweep
6. keep the winning retrieval setting as the main branch
7. only after that decide whether to benchmark on `openwebtext`


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
HF_HOME=/workspace/hf_cache HF_DATASETS_CACHE=/workspace/hf_datasets_cache TMPDIR=/workspace/tmp python data/openwebtext/prepare.py
```

### Dense OpenWebText run

```bash
python train.py --dataset=openwebtext --device=cuda --compile=False
```

### Retrieval OpenWebText run

```bash
python train.py --dataset=openwebtext --device=cuda --compile=False --batch_size=12 --block_size=256 --gradient_accumulation_steps=8 --n_layer=6 --n_head=6 --n_embd=384 --max_iters=5000 --lr_decay_iters=5000 --warmup_iters=200 --eval_interval=500 --eval_iters=50 --log_interval=10 --wandb_log=False --use_retrieval_memory=True --memory_slots=32 --memory_topk=4 --memory_retrieval_weight=1.0 --log_experiment_metrics=True --out_dir=out-owt-memory-s32-k4-5k | tee owt_memory_s32_k4_5k.log
```

### Persistent-memory OpenWebText run

```bash
python train.py --dataset=openwebtext --device=cuda --compile=False --batch_size=8 --block_size=256 --gradient_accumulation_steps=4 --n_layer=6 --n_head=6 --n_embd=384 --max_iters=2000 --lr_decay_iters=2000 --warmup_iters=100 --eval_interval=200 --eval_iters=50 --log_interval=10 --wandb_log=False --use_retrieval_memory=True --use_persistent_memory=True --memory_slots=32 --memory_topk=4 --memory_retrieval_weight=1.0 --persistent_memory_momentum=0.95 --log_experiment_metrics=True --out_dir=out-owt-memory-s32-k4-persistent-2k | tee owt_memory_s32_k4_persistent_2k.log
```

### Local OpenWebText run

```bash
python train.py --dataset=openwebtext --device=cuda --compile=False --attention_mode=local --attention_window=256 --log_experiment_metrics=True
```

### MoE Shakespeare run

```bash
python train.py config/train_shakespeare_char.py --max_iters=500 --eval_interval=100 --ffn_mode=moe --num_experts=4 --experts_topk=2 --log_experiment_metrics=True --compile=False --out_dir=out-moe-top2 | tee moe_top2.log
```

### Retrieval Shakespeare run

```bash
python train.py config/train_shakespeare_char.py --max_iters=500 --eval_interval=100 --use_retrieval_memory=True --memory_slots=16 --memory_topk=4 --memory_retrieval_weight=1.0 --log_experiment_metrics=True --compile=False --out_dir=out-memory-topk4 | tee memory_topk4.log
```

### Retrieval ablation runs

```bash
python train.py config/train_shakespeare_char.py --max_iters=500 --eval_interval=100 --use_retrieval_memory=True --memory_slots=8 --memory_topk=2 --memory_retrieval_weight=1.0 --log_experiment_metrics=True --compile=False --out_dir=out-memory-s8-k2 | tee memory_s8_k2.log
python train.py config/train_shakespeare_char.py --max_iters=500 --eval_interval=100 --use_retrieval_memory=True --memory_slots=16 --memory_topk=2 --memory_retrieval_weight=1.0 --log_experiment_metrics=True --compile=False --out_dir=out-memory-s16-k2 | tee memory_s16_k2.log
python train.py config/train_shakespeare_char.py --max_iters=500 --eval_interval=100 --use_retrieval_memory=True --memory_slots=32 --memory_topk=4 --memory_retrieval_weight=1.0 --log_experiment_metrics=True --compile=False --out_dir=out-memory-s32-k4 | tee memory_s32_k4.log
python train.py config/train_shakespeare_char.py --max_iters=500 --eval_interval=100 --use_retrieval_memory=True --memory_slots=16 --memory_topk=8 --memory_retrieval_weight=1.0 --log_experiment_metrics=True --compile=False --out_dir=out-memory-s16-k8 | tee memory_s16_k8.log
```

### Retrieval weight ablations

```bash
python train.py config/train_shakespeare_char.py --max_iters=500 --eval_interval=100 --use_retrieval_memory=True --memory_slots=32 --memory_topk=4 --memory_retrieval_weight=0.5 --log_experiment_metrics=True --compile=False --out_dir=out-memory-s32-k4-w05 | tee memory_s32_k4_w05.log
python train.py config/train_shakespeare_char.py --max_iters=500 --eval_interval=100 --use_retrieval_memory=True --memory_slots=32 --memory_topk=4 --memory_retrieval_weight=1.5 --log_experiment_metrics=True --compile=False --out_dir=out-memory-s32-k4-w15 | tee memory_s32_k4_w15.log
```
