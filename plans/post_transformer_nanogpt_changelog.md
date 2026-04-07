# Post-Transformer nanoGPT Changelog

Author:

- Petr Royce
- GitHub: `0xroyce`


## Summary

This file records the repo changes made while preparing `nanoGPT` as an experiment harness for lower-cost, neuroscience-inspired language-model research.

Status:

- planning refactor complete
- Phase 0 harness refactor complete
- Phase 1 local causal attention prototype complete
- full training benchmark not yet run


## Plan Updates

Updated [post_transformer_nanogpt_plan.md](/Users/0xroyce/WebstormProjects/Phoenix/plans/post_transformer_nanogpt_plan.md) to a v2 plan with:

- a clearer mission tied to reducing cost to target loss
- explicit success metrics
- a staged path from `nanoGPT` experiments to later `LLaMA`-stack work
- early findings integrated into the roadmap
- a research backlog for higher-risk ideas
- concrete Phase 0 and Phase 1 checklists


## Code Changes

### 1. Experiment harness refactor

Updated:

- [model.py](/Users/0xroyce/WebstormProjects/Phoenix/nanoGPT/model.py)
- [train.py](/Users/0xroyce/WebstormProjects/Phoenix/nanoGPT/train.py)
- mirrored copies under [model.py](/Users/0xroyce/WebstormProjects/Phoenix/nanogpt/model.py) and [train.py](/Users/0xroyce/WebstormProjects/Phoenix/nanogpt/train.py)

What changed:

- added experimental config fields to `GPTConfig`
- added a structured `GPTForwardOutput`
- kept the original `(logits, loss)` behavior intact by default
- added room for named loss terms and named metrics
- refactored attention and FFN paths so dense mode remains the default reference path
- added concise comments at new extension seams

New config fields added:

- `attention_mode`
- `attention_window`
- `attention_topk`
- `ffn_mode`
- `num_experts`
- `experts_topk`
- `use_aux_losses`
- `aux_loss_weights`


### 2. Training-loop compatibility changes

Updated:

- [train.py](/Users/0xroyce/WebstormProjects/Phoenix/nanoGPT/train.py)

What changed:

- added support for unpacking either legacy tuple outputs or richer structured outputs
- added optional experiment-metric logging via `log_experiment_metrics`
- made checkpoint resume tolerant of newly added config fields
- preserved scalar-loss optimization behavior for default runs


### 3. Local causal attention prototype

Updated:

- [model.py](/Users/0xroyce/WebstormProjects/Phoenix/nanoGPT/model.py)

What changed:

- implemented `attention_mode='local'`
- added sliding-window causal masking
- kept `attention_mode='dense'` as the reference path
- added attention metrics

Current metrics exposed:

- `attention/active_fraction`
- `attention/window_tokens`
- `ffn/active_fraction`
- `model/attention_mode_is_dense`
- `model/ffn_mode_is_dense`

Current behavior:

- dense mode still runs as before
- local mode now restricts each token to a causal neighborhood of size `attention_window`
- generation also works through the local-attention branch


## Attribution Updates

Added fork attribution to the file headers of:

- [model.py](/Users/0xroyce/WebstormProjects/Phoenix/nanoGPT/model.py)
- [train.py](/Users/0xroyce/WebstormProjects/Phoenix/nanoGPT/train.py)
- [model.py](/Users/0xroyce/WebstormProjects/Phoenix/nanogpt/model.py)
- [train.py](/Users/0xroyce/WebstormProjects/Phoenix/nanogpt/train.py)


## Verification Completed

Completed checks:

- `python -m py_compile nanoGPT/model.py nanoGPT/train.py`
- small forward-pass construction test for dense mode
- small forward-pass construction test for local mode
- small `generate()` test for both dense and local modes

Observed smoke-test result:

- dense mode reported full attention activity
- local mode with `attention_window=4` reported reduced active attention fraction


## Not Yet Done

- no full dataset benchmark run yet
- no dense-vs-local training comparison yet
- no MoE FFN yet
- no retrieval memory module yet
- no recurrent state path yet
- no auxiliary training objectives yet


## Suggested Next Step

After GPU access is available:

1. run a dense baseline benchmark
2. run local attention with conservative windows such as `256` and `128`
3. compare throughput, memory, and validation loss
4. decide whether to continue into sparse FFN / MoE
